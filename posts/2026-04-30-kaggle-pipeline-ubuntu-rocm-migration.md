---
title: "KaggleパイプラインをWindows+MacからUbuntu+Macに移植した話（ROCm GPU対応）"
emoji: "🏆"
type: "tech"
topics: ["kaggle", "ubuntu", "ollama", "python", "llm"]
published: true
publication_date: "2026-04-30"
---

## TL;DR

- Windows+MacのKaggle自動化パイプラインをUbuntu+Macに移植した
- 接続先IP・Pythonパス・作業ディレクトリ・SSHコマンドをLinux向けに書き換え
- Coderモデルを `qwen2.5-coder:32b` → `qwen3.6:27b` に変更（Ubuntuに存在するモデルに合わせた）
- GPU指定は `device='cuda'` → ROCm環境でもそのまま動く（XGBoost/LightGBM）
- `run_code_ssh()` のWindowsコマンド（`cd /d`、Windows Pythonパス）をLinux向けに修正

---

## パイプラインの概要

`kaggle_optimizer.py` はStreamlitで動くKaggle自動化システム。
LLMチームがコード生成→GPU実行→CVスコア評価→提出判断を自動で回す。

```
Mac mini M4
  └─ qwen3:14b（Planner/Judge/Validator）  ← Mac Ollama
Ubuntu 24.04
  └─ qwen3.6:27b（Coder）                  ← Ubuntu Ollama
  └─ deepseek-r1:32b（Fixer）              ← Ubuntu Ollama
  └─ deepseek-r1:70b（高精度推論）          ← Ubuntu Ollama
  └─ RX 7900 XTX（ML実行）                 ← SSH経由
Claude API（最終手段・コードバリデーション）
```

パイプラインの流れ：

1. **Planner**（Mac/qwen3:14b）が戦略を立案
2. **Coder**（Ubuntu/qwen3.6:27b）が初期コードを生成
3. **Validator**（静的解析+LLM）がコードの問題を検出・修正
4. **GPU実行**（Ubuntu/SSH経由）でCVスコアを取得
5. **Judge**（Mac/qwen3:14b）が結果を分析、改善指示を出す
6. **Fixer**（Ubuntu/deepseek-r1:32b）がコードを改修
7. **意思決定エンジン**が提出判断（SUBMIT/IMPROVE/STOP/WAIT）

---

## 移植作業：定数の書き換え

元のコードはWindows向けに書かれていた。

```python
# 移植前（Windows向け）
WINDOWS_IP       = "192.0.2.20"
WINDOWS_OLLAMA   = "http://{}:11434".format(WINDOWS_IP)
WINDOWS_USER     = "gyaru"
WINDOWS_PYTHON   = "C:/Users/gyaru/miniconda3/envs/kaggle_env/python.exe"
WINDOWS_WORK_DIR = "C:/Users/gyaru/kaggle_work"
SSD_ROOT         = "/Volumes/UGREEN-SSD/kaggle_autmation"
```

`sed` で一括置換した。

```bash
# 定数の書き換え
sed -i 's|WINDOWS_IP       = "192.0.2.20"|LINUX_IP         = "192.0.2.2"|g' kaggle_optimizer.py
sed -i 's|WINDOWS_USER     = "gyaru"|LINUX_USER       = "ubuntu"|g' kaggle_optimizer.py
sed -i 's|WINDOWS_PYTHON   = "C:/Users/gyaru/miniconda3/envs/kaggle_env/python.exe"|LINUX_PYTHON     = "/home/ubuntu/ai-env/bin/python3"|g' kaggle_optimizer.py
sed -i 's|WINDOWS_WORK_DIR = "C:/Users/gyaru/kaggle_work"|LINUX_WORK_DIR   = "/home/ubuntu/kaggle_work"|g' kaggle_optimizer.py
sed -i 's|SSD_ROOT = "/Volumes/UGREEN-SSD/kaggle_autmation"|SSD_ROOT = "/home/ubuntu/kaggle_results"|g' kaggle_optimizer.py

# 変数名の参照も一括置換
sed -i 's|WINDOWS_IP|LINUX_IP|g' kaggle_optimizer.py
sed -i 's|WINDOWS_USER|LINUX_USER|g' kaggle_optimizer.py
sed -i 's|WINDOWS_PYTHON|LINUX_PYTHON|g' kaggle_optimizer.py
sed -i 's|WINDOWS_WORK_DIR|LINUX_WORK_DIR|g' kaggle_optimizer.py

# モデル変更
sed -i 's|MODEL_CODER   = "qwen2.5-coder:32b"|MODEL_CODER   = "qwen3.6:27b"|g' kaggle_optimizer.py
```

---

## run_code_ssh() のLinux対応

最大の書き換えポイントはSSH経由でコードを実行する関数。
Windows固有のコマンドをLinux向けに修正した。

```python
# 移植前（Windows向け）
ssh.exec_command('mkdir "{}" 2>nul'.format(WINDOWS_WORK_DIR))
cmd = 'cd /d "{}" && "{}" "{}" 2>&1'.format(
    WINDOWS_WORK_DIR, WINDOWS_PYTHON, os.path.basename(remote_path)
)

# 移植後（Linux向け）
ssh.exec_command('mkdir -p {}'.format(LINUX_WORK_DIR))
cmd = 'cd {} && {} {} 2>&1'.format(
    LINUX_WORK_DIR, LINUX_PYTHON, os.path.basename(remote_path)
)
```

sedで置換しようとしたが、クォートのエスケープが複雑で置換が壊れた。
特定行番号を直接書き換える方法で対処した。

```bash
# 673行目を直接書き換え
sed -i "673s|.*cmd =.*|        cmd = 'cd {} \&\& {} {} 2>\&1'.format(LINUX_WORK_DIR, LINUX_PYTHON, os.path.basename(remote_path))|" kaggle_optimizer.py
```

---

## GPU指定について

元のコードにはGPU指定が以下のように書かれていた。

```python
# XGBoost
tree_method='hist', device='cuda'

# LightGBM
device='gpu'

# CatBoost
task_type='GPU'
```

ROCm環境（RX 7900 XTX）でも `device='cuda'` がそのまま使えるケースが多い。
XGBoostはROCm対応ビルドであれば `device='cuda'` で動作する。
LightGBMの `device='gpu'` もROCm環境で動作する。

---

## 静的コード解析バリデーター

このパイプラインで面白い部分が `static_analyze_code()` 関数。
LLMが生成したコードの典型的なバグを実行前に機械的に検出する。

検出項目：
- `cross_val_score` を `sklearn.metrics` からimportしている（正しくは `sklearn.model_selection`）
- `scores.mean()` によるスコア出力がない
- `fillna(..., inplace=True)` の使用
- 目的変数のX_trainへの混入
- IDカラムの特徴量混入
- 高カーディナリティ文字列カラムの `astype(int)` 直接変換

```python
def static_analyze_code(code, dataset_info):
    issues = []

    # cross_val_scoreのimport先チェック
    if "cross_val_score" in code:
        if "from sklearn.metrics import cross_val_score" in code:
            issues.append(
                "cross_val_score が sklearn.metrics からimportされている。"
                "sklearn.model_selection に修正せよ"
            )

    # Score出力チェック
    if "scores.mean()" not in code:
        issues.append(
            "scores.mean() による Score出力が見つからない。"
            "スクリプト末尾に print(f'Score: {scores.mean():.4f}') を追加せよ"
        )
    # ...
    return issues
```

問題が検出された場合のみClaude APIを呼んで修正させる設計で、
問題なし → APIコスト0、問題あり → 最小限のAPI呼び出しになっている。

---

## データセット自動スキャン

`analyze_dataset()` でtrain.csvを自動解析してコンペ非依存の制約を動的生成する。

```python
def analyze_dataset(work_dir):
    info = {}
    # 目的変数を推定（testにないカラム）
    target_candidates = [c for c in train.columns if c not in test.columns]
    info["target"] = target_candidates[0] if len(target_candidates) == 1 else None

    # タスク種別を推定
    n_unique = target_col.nunique()
    if n_unique <= 20 or target_col.dtype == object:
        info["task_type"] = "classification"
    else:
        info["task_type"] = "regression"

    # 高カーディナリティ文字列（ドロップ推奨）
    high_card_cols = [col for col in str_cols
                      if train[col].nunique() / len(train) > 0.5]
    # ...
    return info
```

この結果をLLMへのプロンプトに埋め込むことで、
どのコンペでも適切な前処理コードが生成されるようになっている。

---

## 学んだこと

- **sedのクォートエスケープは複雑すぎる**。Windowsパスのダブルクォートを含む行の置換は行番号直接指定の方が確実
- **Coderモデルはそのサーバーに存在するモデルに合わせる**。`qwen2.5-coder:32b` がUbuntuになかったので `qwen3.6:27b` に変更した
- **ROCm環境でも `device='cuda'` 指定がそのまま使えることが多い**。XGBoost/LightGBMはROCmビルドであれば動作する
- **静的解析でLLMの典型的なバグを事前検出する設計は有効**。LLMはscikit-learnのimport元を間違える、inplace=Trueを使う、などのパターンが繰り返されるため

---

## 参考

- `kaggle_optimizer.py`：Streamlit製Kaggle自動化システム本体
- `decision_engine.py`：SUBMIT/IMPROVE/STOP/WAITを判断するルールエンジン
- `experiment_db.py`：SQLiteで試行履歴を管理するモジュール
- Titanic CVスコア 0.8857（KRS v8での実績）
