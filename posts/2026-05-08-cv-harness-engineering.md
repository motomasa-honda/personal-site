---
title: "LLM生成コードのCV未取得エラーをゼロに近づけるハーネスエンジニアリング実装記"
emoji: "🔬"
type: "tech"
topics: ["kaggle", "llm", "python", "機械学習", "自動化"]
published: false
publication_date: "2026-05-08"
---

## TL;DR
- LLMが生成したコードをそのままGPU実行するとCV未取得エラーが頻発する
- `sanitize_code`にハーネス層（型エラー自動修正）を追加することで頻出バグを潰せる
- さらにドライラン（先頭100行で事前実行）を挟むことで根本的にエラーを検出できる
- この2層構造でGPU実行前にエラーを弾けるようになった

## 背景：LLM生成コードの「実行して初めてわかる」問題

Kaggle自動化パイプライン（KRS）では、Planner→Critic→Coder→Fixer→GPU実行という流れでコードを生成・実行している。

問題は**実行して初めてエラーがわかる**構造になっていたこと。

```
LLMコード生成 → sanitize_code → GPU実行 → ❌エラー → Fixer → GPU実行 → ...
```

GPUでフル実行するたびにエラーが出て、Fixerに投げ直す。これが無駄なGPU時間とトークンを消費していた。

今回発生した典型的なエラーがこれ：

```
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'.
Did you mean: 'fill'?
```

`model.predict()`の返り値はnumpy配列なのに、LLMがpandasのメソッドを呼んでしまうパターン。LLMあるある。

## 解決策1：sanitize_codeにハーネス層を追加

既存の`sanitize_code`関数は簡単なinplace修正などをやっていたが、ここに「よくある型エラー自動修正」のレイヤーを追加した。

```python
def sanitize_code(code):
    import re as _re
    # 既存の修正...

    # ── ハーネス層: よくある型エラーを自動修正 ──────────────────────────────
    # (H1) ndarray.fillna().astype(int) → np.nan_to_num().astype(int)
    code = _re.sub(
        r"(\w+)\.fillna\((\d+)\)\.astype\(int\)",
        r"np.nan_to_num(\1, nan=\2).astype(int)",
        code,
    )
    # (H2) preds.fillna(0) 単体パターン
    code = _re.sub(
        r"(preds|y_pred|predictions)\.fillna\(([^)]+)\)",
        r"np.nan_to_num(\1, nan=\2)",
        code,
    )
    # (H3) np未importなら自動追加
    if "np.nan_to_num" in code and "import numpy as np" not in code:
        code = "import numpy as np\n" + code
    # (H4) model.predict().fillna() → pd.Series(model.predict()).fillna()
    code = _re.sub(
        r"(\w+\.predict\([^)]+\))\.fillna\(",
        r"pd.Series(\1).fillna(",
        code,
    )

    return code
```

ポイントは**LLMを使わずPythonのreモジュールで静的変換**していること。トークンを使わず、確実に修正できる。

## 解決策2：ドライラン（先頭100行で事前実行）

ハーネス層だけでは「想定外のパターン」は素通りしてしまう。より根本的な解決策として**ドライラン**を実装した。

```python
def dryrun_code(code, work_dir, n_rows=100, timeout=300):
    """先頭n_rows行のみで試し実行。エラーがあれば(False, エラー文字列)を返す。"""
    import shutil as _sh, subprocess as _sp, tempfile as _tf
    tmp_dir = _tf.mkdtemp(prefix="dryrun_")
    try:
        # train/test CSVを縮小コピー
        for fname in ["train.csv", "test.csv"]:
            src = os.path.join(work_dir, fname)
            if os.path.exists(src):
                df = pd.read_csv(src).head(n_rows)
                df.to_csv(os.path.join(tmp_dir, fname), index=False)
        # コードを書き込んで実行
        code_path = os.path.join(tmp_dir, "dryrun_code.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        result = _sp.run(
            [LINUX_PYTHON, "dryrun_code.py"],
            cwd=tmp_dir,
            capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, output
        if "Score:" not in output:
            return False, "Score出力なし:\n" + output[-500:]
        return True, output
    except _sp.TimeoutExpired:
        return False, "ドライランタイムアウト（{}秒）".format(timeout)
    finally:
        _sh.rmtree(tmp_dir, ignore_errors=True)
```

tmpディレクトリを作ってCSVを100行に絞り、そこで実行する。エラーが出たらFixerに渡して修正させてからフル実行する。

## パイプラインへの組み込み

フル実行の直前に差し込む形で組み込んだ：

```python
# (C-0) ドライラン（先頭100行で事前検証）
add_log("🔬 [ドライラン] 先頭100行で事前検証中...")
dr_ok, dr_out = dryrun_code(st.session_state.current_code, work_dir)
if not dr_ok:
    add_log("⚠️ [ドライラン失敗] → Fixerで修正: {}".format(dr_out[-300:]))
    fix_prompt = (
        "The following Python code failed a dry-run (100 rows). "
        "Fix it and return the complete code in Markdown(```python).\n\n"
        "Error:\n{}\n\nCode:\n{}".format(dr_out[-1000:], st.session_state.current_code)
    )
    st.session_state.current_code = sanitize_code(
        extract_python(chat_linux(LINUX_MODEL_FIXER, fix_prompt))
    )

# (C) Linux GPUでフル実行
output = run_code_ssh(st.session_state.current_code, work_dir)
```

## 結果的な2層防御構造

```
LLMコード生成
    ↓
🛡️ sanitize_code（ハーネス層）← 既知パターンを静的変換
    ↓
🔬 ドライラン（100行・最大5分）← 実際に動かして検証
    ↓ 失敗
🔧 Fixer（エラーメッセージを渡して修正）
    ↓ 成功
⚙️ フル実行（GPU）
```

ドライランの5分は長く見えるが、GPUフル実行でエラー→Fixer→再実行の手戻りと比べれば全然ペイする。

## 学んだこと
- LLMのコード生成エラーは「よくあるパターン」に偏る。パターンマッチで潰せるものは潰す
- 「実行して初めてわかる」構造はできるだけ前工程に引き上げる
- ドライランはシンプルだが効果的。tmpディレクトリにCSVを縮小コピーするだけで実現できる
- ハーネスエンジニアリングはLLMを使わなくてもPythonレベルで実装できる

## 参考
- パイプライン本体: `/home/motomasahonda/projects/kaggle/kaggle_optimizer.py`
- 関連関数: `sanitize_code`（726行目）、`dryrun_code`（新規追加）、`run_code_ssh`（821行目）
- Fixerモデル: deepseek-r1:32b（Ubuntu / ROCm）
