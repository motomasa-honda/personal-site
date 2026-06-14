---
title: "LLM生成コードをそのまま実行したら死んだので自動修正レイヤーを作った"
emoji: "🔬"
type: "tech"
topics: ["python", "llm", "kaggle", "ollama", "自動化"]
published: false
publication_date: "2026-05-03"
---

## TL;DR

- LLMが生成するPythonコードには典型的なバグパターンがある
- `sanitize_code()` で実行前に自動修正、`harness_kaggle.py` でOllamaを使ったレビューエージェントを実装した
- Fixer（修正エージェント）が短い応答を返す問題は、プロンプトに「完全なスクリプトを出力せよ」と明示することで解決した

---

## 背景

Kaggle自動化パイプラインでは、LLM（qwen3:27b / deepseek-r1:32b）がPythonコードを生成し、それをそのまま実行してCVスコアを取得する。

問題は、LLMが生成するコードが「動くコード」とは限らないことだ。毎回手で直すのは本末転倒なので、実行前に自動修正するレイヤーを作ることにした。

---

## 出てきたバグパターン一覧

実際に踏んだエラーたちをまとめると：

| パターン | エラー例 | 頻度 |
|---|---|---|
| markdownコードブロック残骸 | `NameError: name 'python' is not defined` | 毎回 |
| `.loc(` の括弧間違い | `SyntaxError: closing parenthesis ']' does not match '('` | 多い |
| `fillna(inplace=True)` | `FutureWarning` で死ぬ | 中程度 |
| `cross_val_score` のimport先 | `ImportError: cannot import from sklearn.metrics` | 中程度 |
| 未定義変数の参照 | `NameError: name 'df' is not defined` | 多い |

---

## sanitize_code()：静的自動修正

実行前に毎回通す関数。正規表現で機械的に直せるものを処理する。

```python
def sanitize_code(code):
    """LLMが生成したコードの典型的なバグを実行前に自動修正する。"""
    import re

    # ① inplace=True の fillna を安全な形式に変換
    code = re.sub(
        r"(\w+\[['\"][^'\"]+['\"]\])\.fillna\((.+?),\s*inplace=True\)",
        r"\1 = \1.fillna(\2)",
        code,
    )

    # ② cross_val_score / StratifiedKFold のimport先を強制修正
    code = re.sub(
        r"from sklearn\.metrics import ([^\n]*cross_val_score[^\n]*)",
        r"from sklearn.model_selection import \1",
        code,
    )
    code = re.sub(
        r"from sklearn\.metrics import ([^\n]*StratifiedKFold[^\n]*)",
        r"from sklearn.model_selection import \1",
        code,
    )

    # ③ importが抜けていたら先頭に追加
    if "cross_val_score" in code and \
       "from sklearn.model_selection import cross_val_score" not in code:
        code = "from sklearn.model_selection import cross_val_score\n" + code

    # ④ 先頭・末尾のmarkdown残骸を除去
    lines = code.splitlines()
    while lines and lines[0].strip() in ("python", "python3", "```python", "```", ""):
        lines.pop(0)
    while lines and lines[-1].strip() in ("```", ""):
        lines.pop()
    code = "\n".join(lines)

    # ⑤ .loc( → .loc[ の修正
    code = re.sub(r'\.loc\(', '.loc[', code)
    code = re.sub(r'\.iloc\(', '.iloc[', code)

    return code
```

これで静的に直せるものは処理できる。でも「未定義変数を使っている」とか「ロジックがおかしい」はこれだけでは無理だ。

---

## harness_kaggle.py：LLMによる動的レビュー

Ollamaで動く `qwen3:14b` をReviewer/Fixerとして使うモジュールを作った。

### アーキテクチャ

```
生成コード
    ↓
step_review_code()   ← qwen3:14bがJSON形式でレビュー
    ↓
{ approved, severity, issues, hallucinations }
    ↓ severity が "high" or "medium" の場合
step_fix_code()      ← qwen3:14bが完全なコードを出力
    ↓
修正済みコード → sanitize_code() → 実行
```

### Reviewer実装

LLMにJSONのみで返させるのがポイント。パースに失敗したら `approved: True` で通過させる（止まるよりマシ）。

```python
def step_review_code(code):
    prompt = (
        "You are a Kaggle code reviewer. Review the following Python code "
        "and respond ONLY in JSON format.\n\n"
        "Check for:\n"
        "1. Hallucinations (undefined variables, non-existent libraries)\n"
        "2. Data leakage (test data used in training)\n"
        "3. Critical bugs (shape mismatches, wrong target column)\n\n"
        "Respond ONLY with this JSON (no markdown, no explanation):\n"
        '{"approved": true, "severity": "none", "issues": [], "hallucinations": []}\n\n'
        "Code:\n" + code[:3000]
    )
    raw = _ollama_chat(REVIEWER_MODEL, prompt)
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
            result.setdefault("approved", True)
            result.setdefault("severity", "none")
            result.setdefault("issues", [])
            result.setdefault("hallucinations", [])
            return result
    except Exception:
        pass
    return {"approved": True, "severity": "none", "issues": [], "hallucinations": []}
```

### Fixer実装で詰まったこと

最初のプロンプトはこうだった：

```python
prompt = (
    "You are a Kaggle code fixer. Fix the following Python code.\n"
    "Respond ONLY with corrected Python code. No explanation, no markdown.\n\n"
    "Issues:\n" + issues_text + "\n\nCode:\n" + code[:3000]
)
```

これだと `qwen3:14b` が「了解しました。修正しました。」みたいな短い返答を返すことがあり、100文字未満でスキップされ続けた。

修正後のプロンプト：

```python
prompt = (
    "You are a Kaggle code fixer.\n"
    "OUTPUT RULES (STRICTLY FOLLOW):\n"
    "1. Output the COMPLETE fixed Python script only.\n"
    "2. Do NOT output any explanation, comments, or markdown.\n"
    "3. Do NOT use triple backticks or code fences.\n"
    "4. Start your response directly with: import\n"
    "5. The output must be the full script, longer than 200 characters.\n\n"
    "Issues to fix:\n" + issues_text + "\n\n"
    "Original code (fix and return the COMPLETE script):\n" + code[:3000]
)
```

`Start your response directly with: import` が効いた。最初のトークンを固定することで、説明文を前置きするパターンを防げる。

---

## 実際のログ

```
[10:06:55] 🔬 [Harness/Reviewer] コードをレビュー中...
[10:08:31] ⚠️  [Harness] ハルシネーション検知: ['df']
[10:08:31] 🔧 [Harness/Fixer] Reviewer指摘を修正中... severity=high
[10:10:31] ⚠️  [Harness] 修正結果が短すぎるためスキップ  ← 改善前
```

改善後：

```
[10:30:51] ✅ [Harness] コード品質OK severity=none
[10:30:51] ⚙️  [Win/GPU] SSH経由で実行中...  ← この表示は別途修正済み
```

---

## 使用モデルの選択

| 役割 | モデル | 理由 |
|---|---|---|
| Reviewer | qwen3:14b（Mac側Ollama） | JSON出力が安定、速い |
| Fixer | qwen3:14b（Mac側Ollama） | 同上 |
| Coder | qwen3:27b（Ubuntu側Ollama） | コード生成品質が高い |
| Fixer（重め） | deepseek-r1:32b（Ubuntu側Ollama） | 推論が必要な修正向け |

Reviewerは速度優先でMac側の軽いモデルを使っている。Ubuntu側はGPU実行中でVRAMを圧迫するため。

---

## 学んだこと

- **LLMへの出力指定は「最初のトークン」まで指定すると強い。** `Start with: import` は効果大。
- **Reviewerのパース失敗は通過扱いでいい。** 止めるより回す方が自動化としては正しい。
- **静的修正（sanitize）と動的修正（LLMレビュー）の二段構えが現実的。** 静的に直せるものは正規表現で、文脈が必要なものはLLMで。
- **コードのサニタイズは実行直前と、書き込み直前の両方に入れる。** Fixerが修正した後にも残骸が混入することがある。

---

## 参考

- `~/kaggle_pipeline/harness_kaggle.py`（今回新規作成）
- `~/kaggle_pipeline/kaggle_optimizer.py`（`sanitize_code` 関数）
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
