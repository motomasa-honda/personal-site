---
title: "自作マルチエージェントKaggleシステムをTitanicで実走テストした話【KRS-Core v0.1.0】"
emoji: "🚢"
type: "tech"
topics: ["kaggle", "langgraph", "llm", "python", "機械学習"]
published: true
publication_date: "2026-06-10"
---

## TL;DR
- LangGraphベースの自作Kaggleエージェント（KRS-Core）をTitanicで実走テスト
- `competition_url` / `workspace` をinitial_stateに渡し忘れるという凡ミスから始まり、テンプレートの構文エラー、提出制限超過まで連続でハマった
- 最終的にOptuna polish で **CV=0.88647**、Public LB **0.73684** を自律的に達成
- パイプラインが「一応動く」ところまで持っていくまでの泥臭いデバッグ記録

---

## システム概要：KRS-Coreとは

KRS（Kaggle Research System）を汎用マルチエージェント基盤として再設計したもの。7層アーキテクチャで構成されている。

```
core/
├── runtime/      ← Layer 1: BaseAgent ABC
├── router/       ← Layer 2: TaskType別LLMルーティング
├── tools/        ← Layer 3: Shell/Python実行
├── memory/       ← Layer 4: GrandmasterMemory
├── knowledge/    ← Layer 5: SkillLibrary (207 skills)
├── skills/       ← Layer 6: SkillExecutor
└── interfaces/   ← Layer 7: FastAPI endpoints

agents/
└── kaggle_agent/
    ├── kaggle_agent.py   ← KaggleAgent (BaseAgent継承)
    ├── graph.py          ← LangGraph 13ノードパイプライン
    └── nodes/            ← analyze, plan, critic, codegen, execute, submit, reflect...
```

LangGraphの13ノードグラフが自律的にKaggleコンペを回す設計。ローカルOllamaモデル（deepseek-r1:32b/70b, qwen3.6:27b）をタスク別にルーティングして使う。

---

## バグ①：`competition_url` KeyError

最初のテスト実行コードはこう書いた：

```python
agent = KaggleAgent()
result = agent.execute({
    "name": "titanic-test-001",
    "competition_slug": "titanic",
    "iterations": 2,
    "max_fix_retries": 2,
})
```

実行すると即死。

```
KeyError: 'competition_url'
File ".../agents/kaggle_agent/nodes/analyze.py", line 32, in run
    url = state["competition_url"]
```

`analyze.py` を読むと、URLからslugを導出する設計になっていた。`kaggle_agent.py` の `_build_initial_state()` を見ると `competition_url` も `workspace` も含まれていなかった。

修正は `_build_initial_state()` に2フィールドを追加するだけ：

```python
def _build_initial_state(self, task: Dict[str, Any]) -> Dict[str, Any]:
    import os
    slug = task.get("competition_slug") or task.get("name")
    url  = task.get("competition_url") or \
           f"https://www.kaggle.com/competitions/{slug}"
    workspace = task.get("workspace") or os.path.expanduser(
        f"~/projects/kaggle-research-system/workspaces/{slug}"
    )
    return {
        "competition_slug": slug,
        "competition_url":  url,
        "workspace":        workspace,
        # ...以下略
    }
```

設計上は `execute()` に渡す `task` dict にURLを入れれば良かったのだが、`_build_initial_state()` がURLを補完してくれないのが罠だった。

---

## バグ②：テンプレートの構文エラー

次のエラーはここ：

```
FileNotFoundError: agents/kaggle_agent/templates/tabular_classification/main.py.tmpl
```

テンプレートファイルが存在しなかった。`registry.py` を読むとタスク種別ごとにスケルトンを持つ設計で、`{{TARGET_COL}}` や `{{CV_CODE}}` などのプレースホルダーをLLMが出したJSONで置換する仕組みだった。

テンプレートを新規作成したが、今度は別の構文エラー：

```
SyntaxError: line 9, col 52: invalid syntax
>>> from sklearn.model_selection import StratifiedKFold(n_splits=5, ...)
```

`{{CV_CODE}}` をそのまま `import` 文に埋め込んでいたのが原因。`CV_CODE` はクラスの**インスタンス化式**なので、importとは分離しないといけない。

```python
# ❌ ダメなテンプレート
from sklearn.model_selection import {{CV_CODE}}

# ✅ 正しいテンプレート
from sklearn.model_selection import StratifiedKFold
# ...
cv = {{CV_CODE}}   # ← ここで初めて展開する
```

構文チェックはテンプレート作成後に毎回やるべきだった：

```python
import ast
src = open('main.py.tmpl').read()
src = src.replace('{{CV_CODE}}', 'StratifiedKFold(n_splits=5, shuffle=True, random_state=42)')
# ...他のプレースホルダーも置換
ast.parse(src)  # SyntaxErrorが出たら即わかる
```

---

## パイプラインが動き出した

テンプレート修正後、パイプラインが本格的に動き始めた。ログの流れはこう：

```
[analyze]    URL解析 → データDL → Top solutions mining (5 kernels, 14 skills抽出)
[autogluon]  import失敗 → skip (未インストール)
[leak_check] リーク検出完了: 問題なし
[cv_simulate] StratifiedKFold(n_splits=10) risk=0.10
[plan]       GrandmasterMemory 3721文字注入 → deepseek-r1:32b
[critic]     ok=False issues=5  ← 3回ループ
[codegen]    iter=1 strategy=baseline (2665 chars)
[execute]    exit=0 duration=1.6s ✅
[submit]     🎉 new best public=0.71531
```

LLMが計画→批評→コード生成を3回ループしてようやくcriticがOKを出し、生成されたコードが実行・提出された。

---

## Optunaでハイパーパラメータ最適化

iter=3でno_improve=2になったタイミングでreflectノードがOptuna探索を起動。25 trialsでLightGBMのハイパラを自動チューニング：

```
Trial 17: best_cv=0.8864666  ← これが最良
  n_estimators=2500, learning_rate=0.05595, num_leaves=52,
  max_depth=11, min_child_samples=5, subsample=0.60,
  colsample_bytree=0.66, reg_alpha=0.167, reg_lambda=0.00189
```

最終的に **CV=0.88647** を達成。テスト成功条件（>0.80）はクリアした。

---

## ハマりポイント：提出制限でループが止まらない

Titanicは**1日5回**の提出制限がある。iter=11を超えると毎回submitが静かに失敗するが、パイプラインはループを継続してしまう。

```
[submit] 失敗: kaggle failed: 
  100%|██████| 2.77k/2.77k [00:00<00:00, 3.58kB/s]
```

エラーメッセージが空（kaggle CLIが非ゼロ終了するが理由を出さない）なので、submit ノードが「失敗」と判定してもno_improveカウントが増えず無限ループに近い状態になった。

次のissue対応が必要：
- submit失敗時にエラー種別を判定（制限 vs ネットワーク vs その他）
- 制限エラーなら `should_continue = False` をstateにセットして終了

---

## 結果まとめ

| 指標 | 値 |
|------|-----|
| best_cv (Optuna) | **0.88647** |
| Public LB | **0.73684** |
| 実行イテレーション | 11回（正常） |
| パイプライン所要時間 | 約100分 |
| LLM呼び出し | deepseek-r1:70b(×6), 32b(×多数), qwen3.6:27b(×多数) |

成功条件の `result.success=True` / `best_cv > 0.80` はクリア。

---

## 学んだこと

- **`_build_initial_state()` は完全なstate仕様書**として管理すべき。`analyze.py` が何を期待しているか、stateのキー一覧をドキュメント化しておかないと追いにくい
- **テンプレートの構文チェックは `ast.parse()` で自動化**する。CI/CDに組み込むか、テンプレート保存時にフックを仕込むと良い
- **Kaggle提出制限はシステムレベルで対処**が必要。CLIエラーのパースが甘いと無限ループになる
- LangGraphのノード間でstateが流れる設計は、デバッグ時に「どのノードで何が起きたか」がログから追いやすくて良かった

## 参考

- KRS-Core v0.1.0 アーキテクチャ: `kaggle-research-system/` (private)
- 修正ファイル: `agents/kaggle_agent/kaggle_agent.py`, `agents/kaggle_agent/templates/tabular_classification/main.py.tmpl`
- 使用モデル: deepseek-r1:32b (PLAN), qwen3.6:27b (CODE/CRITIQUE), deepseek-r1:70b (Recurrent Reasoner)
