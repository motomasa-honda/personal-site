---
title: "LangGraph × ローカルLLMでKaggleを自動化するマルチエージェント設計の話"
emoji: "🤖"
type: "tech"
topics: ["langgraph", "ollama", "kaggle", "python", "llm"]
published: false
publication_date: "2026-06-10"
---

## TL;DR
- LangGraph 13ノードのパイプラインでKaggleコンペを自律実行するシステムを設計・実装した
- LLMのタスク別ルーティング（PLAN/CODE/CRITIQUE/JUDGE）をModelRouterで管理
- テンプレート駆動コード生成（LLMが直接コードを書くのは修正時のみ）という設計判断が肝
- ローカルOllamaとClaude APIをタスク特性で使い分ける構成

---

## なぜLangGraphを選んだか

KaggleのML実験ループは本質的に**グラフ構造**をしている。

```
plan → critic → (NG: plan に戻る) → codegen → execute → submit → reflect → (continue: plan に戻る / finish)
```

これをfor文で書くと条件分岐がすぐ複雑になる。LangGraphは「ノード＋エッジ＋条件付きエッジ」でこういう繰り返しループを宣言的に書けるのが利点だった。

実装した13ノードはこちら：

```python
# graph.py の構成（概略）
graph.add_node("analyze",           nodes.analyze.run)
graph.add_node("autogluon_baseline",nodes.autogluon_baseline.run)
graph.add_node("leak_check",        nodes.leak_check.run)
graph.add_node("cv_simulate",       nodes.cv_simulate.run)
graph.add_node("plan",              nodes.plan.run)
graph.add_node("critic",            nodes.critic.run)
graph.add_node("codegen",           nodes.codegen.run)
graph.add_node("branch_explorer",   nodes.branch_explorer.run)
graph.add_node("validate",          nodes.validate.run)
graph.add_node("execute",           nodes.execute.run)
graph.add_node("submit",            nodes.submit.run)
graph.add_node("judge",             nodes.judge.run)
graph.add_node("reflect",           nodes.reflect.run)

# plan→critic のループ (最大3回)
graph.add_conditional_edges("critic", route_critic, {
    "plan": "plan",
    "codegen": "codegen",
})
# reflect → continue or finish
graph.add_conditional_edges("reflect", route_reflect, {
    "plan": "plan",
    END: END,
})
```

---

## TaskType別LLMルーティング

タスクの性質によって最適なモデルは違う。PLANには推論力、CODEには実装力、CRITIQUEには批判的思考、JUDGEには信頼性が必要だ。

これをModelRouterで一元管理している：

```python
# core/router/model_router.py
DEFAULT_ROUTING = {
    TaskType.PLAN:      ModelSpec("deepseek-r1:32b",         Provider.OLLAMA),
    TaskType.CODE:      ModelSpec("qwen2.5-coder:32b",       Provider.OLLAMA),
    TaskType.CRITIQUE:  ModelSpec("gemma3:27b",              Provider.OLLAMA),
    TaskType.JUDGE:     ModelSpec("claude-sonnet-4-20250514",Provider.CLAUDE),
    TaskType.SUMMARIZE: ModelSpec("qwen3.6:27b",             Provider.OLLAMA),
    TaskType.EMBEDDING: ModelSpec("nomic-embed-text",        Provider.OLLAMA),
}

# ドメイン別オーバーライドも可能
default_router.override(
    TaskType.PLAN,
    domain="appgen",
    spec=ModelSpec("deepseek-r1:70b", Provider.OLLAMA)
)
```

JUDGEだけClaude APIを使うのがポイント。ローカルモデルは「評価」が甘くなりがちで、最終的なコードの質判定はClaudeの方が信頼できるという判断。

実走テストでは `qwen2.5-coder:32b` が未インストールだったので `qwen3.6:27b` で代替した。こういうフォールバックをRouterレベルで管理できるのが便利。

---

## テンプレート駆動コード生成の設計

「LLMにコードを直接書かせる」のは修正（Fixer）フェーズだけにする、というのが設計上の重要な判断だった。

初回・通常イテレーションはこの流れ：

```
Planner (LLM) → JSON仕様 → テンプレート + spec → コード
```

```python
# core/router/model_router.py の TemplateSpec
@dataclass
class TemplateSpec:
    task_type: str
    primary_model: str = "lightgbm"
    cv_code: str = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"
    target_col: str = ""
    id_col: str = ""
    n_classes: int = 2
    feature_engineering: list[str] = None
    hparams: dict[str, Any] = None

# registry.py がスケルトンにspecを差し込む
def render_template_from_spec(spec: TemplateSpec) -> str:
    skeleton = load_skeleton(spec.task_type)  # タスク種別ごとのtmpl
    replacements = {
        "{{CV_CODE}}":      spec.cv_code,
        "{{TARGET_COL}}":   spec.target_col,
        "{{PRIMARY_MODEL}}":spec.primary_model,
        # ...
    }
    return replace_all(skeleton, replacements)
```

テンプレートはタスク種別ごとに用意する（特化）が、プレースホルダーで可変部分を吸収する（汎用化）のハイブリッド設計。

```
agents/kaggle_agent/templates/
├── tabular_classification/main.py.tmpl  ← LightGBM/RF分類
├── tabular_regression/main.py.tmpl      ← 回帰
├── nlp/main.py.tmpl                     ← テキスト系
├── timeseries/main.py.tmpl
└── image/main.py.tmpl
```

**落とし穴**: `{{CV_CODE}}` はインスタンス化式（`StratifiedKFold(...)`）なので、import文に埋め込むと構文エラーになる。

```python
# ❌ ダメ
from sklearn.model_selection import {{CV_CODE}}

# ✅ 正しい
from sklearn.model_selection import StratifiedKFold
cv = {{CV_CODE}}
```

テンプレート作成後は必ず `ast.parse()` で検証する。

---

## GrandmasterMemoryによるプロンプト注入

planノードでは毎回「Grandmasterとしての記憶」をLLMプロンプトに注入している。

```python
# nodes/plan.py
mem = GrandmasterMemory()
memory_text = mem.recall(state)  # 3721文字
prompt = f"""
あなたはKaggle Grandmasterです。
以下の過去の知見を参考にして計画を立ててください：

{memory_text}

コンペ情報: {state['competition_meta']}
現在のスコア: {state['best_cv']}
...
"""
```

`GrandmasterMemory` は `ProjectMemory` を継承した実装で、過去のコンペ経験・スキルライブラリ（207スキル）・類似コンペの情報を優先度付きで詰め込んでいる。

```python
class GrandmasterMemory(ProjectMemory):
    def recall(self, state: dict) -> str:
        chunks = [
            self._chunk("episode_memory", self._episodes(state), priority=10),
            self._chunk("skill_library",  self._skills(state),   priority=8),
            self._chunk("similar_comps",  self._similar(state),  priority=6),
        ]
        return self._pack(chunks, self.max_chars)  # 文字数上限でトリミング
```

---

## Optuna自動チューニングとの連携

no_improveが閾値を超えるとreflectノードがOptuna探索を起動する。25 trialsで自動探索：

```python
# nodes/optuna_polish.py (抜粋)
def objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 15, 255),
        "max_depth":       trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # ...
    }
    model = lgb.LGBMClassifier(**params)
    cv_score = cross_val_score(model, X, y, cv=cv).mean()
    return cv_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)
```

Titanic実走では Trial 17 が最良で `CV=0.88647` を達成した。

---

## 実走ログで見えたシステムの動き

実際のログから読み取れるシステムの時系列：

```
13:57 analyze     URL解析・データDL・skills mining (14 skills from 5 kernels)
13:57 leak_check  リーク検出: 問題なし
13:57 cv_simulate StratifiedKFold(n_splits=10) を採択
13:57 plan×3      deepseek-r1:32b が計画 → critic(qwen3.6:27b)が否決×3
14:04 codegen     テンプレート展開 (2665 chars)
14:04 execute     exit=0, 1.6s
14:04 submit      🎉 LB=0.71531
14:07 optuna      25 trials → best_cv=0.88647
14:08 reflect     Recurrent Reasoner (deepseek-r1:70b ×6ループ)
14:17 codegen     改善コード生成
14:20 submit      🎉 LB=0.73684 (新ベスト)
```

critcノードが3回連続でplanを否決しているのが面白い。LLMが計画の甘さを自己批判して改善を促している。

---

## 残課題

実走テストで見つかった未実装・バグ：

| 項目 | 状況 |
|------|------|
| `core.playbook` | 未実装（stacking/pseudo-labeling がskip） |
| `CompetitionVectorizer` | importエラー（Qdrant連携） |
| 提出制限(5回/日)超過時の停止 | ループが止まらない |
| `runner` のCV_SCORE解析 | `cv=None` のまま（`_build_result` でbest_cvがN/A） |
| `discussions` サブコマンド | kaggle CLI非対応バージョン |

---

## 学んだこと

- **LangGraphの条件付きエッジ**はMLの反復ループと非常に相性が良い。「n回失敗したら別ルートへ」という制御が宣言的に書ける
- **LLMルーティング**はタスクごとに最適なモデルが違うので、Routerで一元管理するのが後から変えやすくて良い
- **テンプレート＋LLM JSON仕様**のハイブリッドは、LLMが完全にコードを生成するより安定する。プレースホルダーの型（式 vs 文）には注意が必要
- **AgentState（TypedDict）**をきちんと定義しておくと、LangGraphのノード間のインターフェースが明確になってデバッグしやすい

## 参考

- LangGraph: https://github.com/langchain-ai/langgraph
- Ollama: https://ollama.ai
- 実行環境: Ubuntu 24.04 / Ryzen 9 9950X / RX 7900 XTX (ROCm)
- リポジトリ: motomasa-honda/kaggle-research-system (private)
