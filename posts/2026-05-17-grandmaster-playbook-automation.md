---
title: "KaggleグランドマスターのPlaybookをLLMパイプラインに自動化して組み込んだ話"
emoji: "🏆"
type: "tech"
topics: ["kaggle", "python", "機械学習", "llm", "自動化"]
published: false
publication_date: "2026-05-17"
---

## TL;DR

- NVIDIAのKaggleグランドマスターが2026年3月に1位を取った手法を調査して自動化に組み込んだ
- 「フルデータ再学習」「Adversarial Validation」「OOF保存」「Seed Averaging」を実装
- Hill ClimbingとPseudo Labelingは実装したが問題があり、正しい実装の方針も整理した
- パイプライン完了後に「Grandmaster Playbook」UIセクションが出現してワンクリックで実行できる

## なぜ今これをやるのか

Kaggle自動分析パイプライン（LLMが自動でコード生成→実行→改善するやつ）を開発していて、モデルの構造やエージェントの設計は整ってきた。でも「同じくらい賢いLLMを使っても上位kaggler の方がスコアが高い」という状況が続いていた。

2026年3月にNVIDIAのKaggle Grandmasterチームが複数のLLMエージェントを使って1位を取ったという事例を調べたところ、**スコアの差はモデルの賢さではなくテクニックの数と実行の徹底さ**にあることがわかった。

特に重要だったのがこの設計思想。

> 良し悪しに関わらず、全実験のOOF予測とテスト予測を必ずディスクに保存する

これが全後続テクニック（アンサンブル・Hill Climbing・スタッキング）の前提になっている。現行パイプラインは `best_code` と `best_cv` しか保存していなかった。

## 実装1: OOF予測の全試行保存

まずは基盤として全試行の予測ファイルを保存するようにした。実装はシンプルで5行。

```python
# ベストCV更新のたびに実行
try:
    _oof_dir = Path(work_dir) / "oof_predictions"
    _oof_dir.mkdir(exist_ok=True)
    _sub_path = Path(work_dir) / "submission.csv"
    if _sub_path.exists():
        import shutil
        _dst = _oof_dir / f"test_pred_trial_{trial_no:03d}_cv{current_cv:.4f}.csv"
        shutil.copy(_sub_path, _dst)
        add_log(f"💾 OOF保存: {_dst.name}")
except Exception as _e:
    add_log(f"⚠️ OOF保存失敗: {_e}")
```

これで `oof_predictions/` ディレクトリに `test_pred_trial_001_cv0.1108.csv` のようなファイルが蓄積されていく。

## 実装2: フルデータ再学習（最も確実に効く）

CVは `train` を分割して評価するが、最終的な提出モデルは **全 train データで学習したもの** を使うべき。これだけで LB スコアが 1〜3% 改善することが多い。

実装は「best_code の cross_val_score 呼び出し行をコメントアウトして、全データ fit だけ残す」という変換。

```python
def run_full_data_retraining(best_code: str, work_dir: str) -> str:
    lines = best_code.splitlines()
    new_lines = []
    for l in lines:
        # CVループ関連行をコメントアウト
        if re.search(r'cross_val_score|for\s+\w+.*in\s+\w*[Kk][Ff]old|\.split\(X', l):
            new_lines.append("# [full-retrain] " + l)
        else:
            new_lines.append(l)
    retrain_code = "\n".join(new_lines)
    # 全データ fit の明示
    retrain_code = retrain_code.replace(
        "model.fit(X, y)",
        "# [full-retrain] 全訓練データで最終学習\nmodel.fit(X, y)"
    )
    out = run_code_ssh(retrain_code, work_dir)
    # submission_full_retrain.csv として保存
    ...
```

これを **Kaggle 提出ボタン押下時に自動実行** する設計にした。提出フローがこうなった。

```
▶️ パイプライン起動
    ↓ (n試行)
decision: SUBMIT
    ↓
🔄 フルデータ再学習（自動）
    ↓
📤 submission_full_retrain.csv を提出
```

## 実装3: Adversarial Validation（train/test分布差の検出）

train と test のデータ分布が違う場合、CV スコアが良くてもLBスコアが出ないという典型的な問題がある。これを検出するのが Adversarial Validation。

仕組みはシンプル。「train か test か」を予測する二値分類器を学習して、AUC が高いほど分布差が大きい。AUC > 0.7 なら「分布ズレあり」と判定する。

```python
def run_adversarial_validation(work_dir: str, dataset_info: dict) -> dict:
    adv_code = f"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# target と ID 列を除外して結合
train["_adv_label"] = 0
test["_adv_label"]  = 1
combined = pd.concat([train[common], test[common]], ignore_index=True)
X = combined.drop(columns=["_adv_label"])
y = combined["_adv_label"]

# 二値分類で AUC 計測
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
auc = float(scores.mean())

# 重要な特徴量 = 分布ズレの原因カラム候補
clf.fit(X, y)
importances = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])
drift_cols = [col for col, imp in importances[:10] if imp > 0.05]

result = {{"auc": round(auc, 4), "drift_cols": drift_cols}}
print("ADV_RESULT:" + json.dumps(result))
"""
```

検出されたドリフトカラムは `dataset_info["drift_cols"]` に保存され、`build_dynamic_constraints()` で自動的に Coder への制約に反映される。

```python
if info.get("drift_cols") and info.get("adv_auc", 0) > 0.6:
    lines.append(
        f"- 【重要】train/testで分布が異なるカラム（Adversarial AUC={info['adv_auc']:.3f}）: "
        f"{info['drift_cols'][:5]} → 削除またはCV戦略を調整すること"
    )
```

**データDLボタン押下時に自動実行**される。コンペ参加直後に分布差を把握できる。

## 実装4: Seed Averaging

異なるランダムシードで複数回学習して予測を平均する手法。実装コスト最小で効果が確実。

```python
def run_seed_averaging(base_code: str, work_dir: str, n_seeds: int = 3) -> str:
    preds = []
    for seed in range(n_seeds):
        # random_state を書き換えて実行
        seeded = re.sub(r"random_state\s*=\s*\d+", f"random_state={seed * 42}", base_code)
        run_code_ssh(seeded, work_dir)
        sub_path = Path(work_dir) / "submission.csv"
        if sub_path.exists():
            df = pd.read_csv(sub_path)
            preds.append(df[pred_col].values)
    # 予測を平均して出力
    avg_pred = np.mean(preds, axis=0)
    ...
```

## 正直に言うと Hill Climbing の実装は間違っていた

実装した Hill Climbing はこうなっていた。

```python
# 現状の実装（間違い）
candidate = (best_pred * len(best_ids) + pred) / (len(best_ids) + 1)
if np.std(candidate) < np.std(best_pred) * 0.999:
    continue  # 分散が小さくなる方向に進む
```

本来の Hill Climbing は **OOFスコアで改善するかどうか** で判定する。でも現状のシステムはOOFの正解ラベルを保存していないため、正しい判定ができない。「分散が小さくなる方向」という代替基準は正しくない。

正しい実装のためには：
1. CV の各fold で `y_val` と `oof_pred` をファイルに保存する
2. Hill Climbing では保存した `y_val` で実際のスコアを計算して改善判定する

これは次のイテレーションで修正予定。今の実装は「OOFファイルを蓄積するだけ」として使うのが正しい。

## Grandmaster Playbook UI

パイプライン完了後にこのUIが表示される設計にした。

```
🏆 Grandmaster Playbook — 追加精度向上

[Adversarial AUC: 🟢 0.5234]  [⚡ Seed Averaging]  [🎯 Hill Climbing]  [🔬 Pseudo Labeling]
[蓄積予測ファイル数: 5本]     [3シード平均]         [2本以上で有効]     [高信頼度再学習]
[🔁 フルデータ再学習]
```

各ボタンは Streamlit の `st.button()` で実装されていて、押すと対応する関数が実行される。Seed Averaging と フルデータ再学習 は単体でも有効で、Hill Climbing は OOF が 2 本以上蓄積されると有効になる。

## 調査過程で「取り込まない」と判断したもの

今回 AutoKaggle・AIDE・MLZero・OpenHands 等を調査した。取り込まなかった理由も記録しておく。

| ツール | 取り込まない理由 |
|--------|----------------|
| AutoKaggle | 設計思想は参考にしたが、GPT-4o前提でローカルOllama構成と合わない |
| AIDE | ソリューションツリーの「設計思想」は将来実装予定。今は best_code 1本で十分 |
| MLZero | AWSクラウドサービス前提。ローカル構成と根本的に合わない |
| cuDF/cuML | CUDA専用。ROCmのRX 7900 XTXでは使えない |

「使えるツールを全部入れる」より「自分の環境に合うものを選ぶ」判断が大事だと改めて思った。

## 学んだこと

- **OOF保存は最初から設計に組み込むべき**。後付けだと保存形式が揃わなくて使いにくい
- **フルデータ再学習は確実に効く**。実装コストも低いのに後回しにしていた損失は大きい
- **「動く実装」と「正しい実装」は別**。Hill Climbingのように骨格だけあって実質機能しないコードは、ないよりまし程度だが誤解を生む
- Adversarial Validationの「ドリフトカラムを制約に反映する」設計は良かった。LLMへのフィードバックループとして自然に機能する

## 参考

- NVIDIA Kaggle Grandmasters Playbook（2025-2026）の勝利解法
- [AIDE: LLM Agents for Data Science](https://github.com/WecoAI/aider) のソリューションツリー設計思想
- 修正コミット: `feat: Phase1-3 refactor + Grandmaster Playbook`
- 環境: Ubuntu 24.04 / RX 7900 XTX 24GB / Ollama / Streamlit / Mac mini M4
