---
title: "1 日で 4 機能が回り出した KRS-OS — 並列開発を支える疎結合な設計"
emoji: "🧱"
type: "tech"
topics: ["kaggle", "automl", "optuna", "agent", "krs-core"]
published: true
publication_date: "2026-06-23"
---

## TL;DR

朝起きた時点で 0 commit、夕方には 5 commit が origin/main に乗っていた。
内訳は

1. Planner プロンプトに meta-pattern を機械評価で注入 (Gap 1 完成)
2. Gap 2 AutoML 探索ループ — Optuna 統合本実装
3. cv_lb_log → LinearLBPredictor の factory 配線
4. TabPFN-v3 playbook (5-fold CV + GPU batch predict + HF DL hang 回避 helper)
5. 日報 + 1 日全体の整理

テストは 193 → 207 passed (+14)、5 skipped。全部 push + deploy 済。

これが**たまたま** 1 日でできたわけではなくて、KRS-OS の機能間結合を意図的に浅く
保ってきた成果なので、その設計について書く。

---

## 各 commit のサイズ感

`git log --oneline -5`:

```
2e31e61 feat(playbook): TabPFN v3 member for stack diversity (Foundation NN)
a31796f feat(automl): wire cv_lb_log → LinearLBPredictor via build_lb_predictor factory
2f651dc feat(automl): Gap 2 Optuna integration (TPE + warmup enqueue + LB-aware objective)
e031357 feat(planner): inject v8 meta-pattern recall (conditions/mechanism/expected_effect)
bfe3f36 docs(report): 2026-06-23 日報 — TabPFN s6e6 不適合 + 4 機能投入
```

| commit | LOC | テスト |
|---|---|---|
| Planner 注入 | +428 | +4 (boundary / lenient / strict / rendering) |
| Optuna 統合 | +372 | +4 (warmup enqueue / LB-aware / fallback / TPE 収束) |
| LB factory | +140 | +4 (empty / fitted / partial / SkillSeededSearch 統合) |
| TabPFN playbook | +384 | +6 (5 fast + 1 slow) |

平均で 1 機能 ≈ 330 行 + 5 テスト程度。ぎりぎり「半日仕事」のサイズ。

これが **同じ日に 4 個並列で進んだのは、互いに依存していないから**。
順番にやってもいいし、4 つのワークツリーで並列開発しても齟齬が出ない構造になっている。

---

## なぜ並列開発できたか

### 機能 1: Planner に meta-pattern 注入 (`e031357`)

入口は `agents/kaggle_agent/nodes/plan.py` で、`dataset_info` から構築した
`ctx_for_meta` (task / n_samples / n_features / has_text / ...) を
`get_skill_library().to_planner_prompt_meta(ctx)` に渡すだけ:

```python
ctx_for_meta = {
    "task":           state.get("task_type", ""),
    "metric":         state.get("evaluation_metric", "") or "",
    "n_samples":      di.get("n_rows"),
    "n_features":     len(di.get("all_columns", []) or []) or None,
    "has_text":       bool(di.get("text_cols")),
    "has_image":      "image" in (state.get("task_type") or "").lower(),
    "has_date":       bool(di.get("date_cols")),
    "target_nunique": di.get("target_nunique"),
}
meta_pattern_text = get_skill_library().to_planner_prompt_meta(
    ctx_for_meta, top_k=6, min_quality=0.4
)
```

SkillLibrary 側の新メソッド (`filter_by_context` / `to_planner_prompt_meta`) は **既存
`search()` と完全に直交した別経路** として追加した。既存呼び出し元は何も変えていない。
だから他機能の開発と衝突しない。

### 機能 2: Gap 2 Optuna 統合 (`2f651dc`)

`SkillSeededSearch.run()` を分岐:

```python
def run(self) -> SearchOutcome:
    start = time.time()
    accepted, rejected = filter_skills_for_context(self.seeds, self.ctx)
    warmups = self._initial_params_from_seeds(accepted)

    if self.param_space is not None:
        return self._run_optuna(accepted, rejected, warmups, start)
    return self._run_warmup_only(accepted, rejected, warmups, start)
```

`param_space=None` のときは旧 warmup-only スタブにフォールバックする。
既存テストは何もいじらず 100% green のまま。本実装の 4 テストを追加しただけ。

LB-aware objective も切替可能:

```python
def objective(trial):
    ...
    pred = self.lb_predictor.predict(cv_score)
    if pred.confidence >= self.lb_confidence_threshold:
        return pred.predicted  # 実機 cv-lb ペアが信頼できる時のみ
    return cv_score             # 普段は CV を最適化
```

`lb_predictor` の引き渡しは Strategy パターンになっていて、`NaiveLBPredictor` /
`LinearLBPredictor` のどちらを渡しても動く。

### 機能 3: cv_lb factory (`a31796f`)

機能 2 の `lb_predictor` を実機データで fit する手段。`experiment_db.cv_lb_log` から
過去提出ペアを読んで線形回帰する factory:

```python
def build_lb_predictor(competition, *, min_samples=5, db_path=None) -> LBPredictor:
    try:
        db = ExperimentDB(db_path=db_path) if db_path else get_db()
        pairs = [(float(p["cv"]), float(p["lb"])) for p in db.get_cv_lb_pairs(competition)]
    except Exception:
        return NaiveLBPredictor()
    if len(pairs) < min_samples:
        p = NaiveLBPredictor(); p.fit(pairs); return p
    p = LinearLBPredictor(min_samples=min_samples); p.fit(pairs); return p
```

実機 DB で `house-prices` (n=5) を読むと `a=0.754, b=0.0314, mse=0.000003, confidence=1.000`
を回復してくる。`SkillSeededSearch(lb_predictor=build_lb_predictor("house-prices"))` で
LB-aware objective が即動く。

これも **機能 2 から見ると interface 越しの依存** で、`LBPredictor` Protocol が満たせれば
中身が naive でも linear でも ridge でも動く。

### 機能 4: TabPFN playbook (`2e31e61`)

これは完全に独立。`agents/kaggle_agent/playbook/tabpfn_member.py` という新ファイルを
追加するだけで、既存のどこにも触らない。

[前の記事](/post.html?slug=2026-06-23-tabpfn-two-traps-foundation-nn-and-hf-anonymous-dl)
で書いた通り s6e6 では効かなかったが、wrapper 自体は汎用 (5-fold CV + batch predict +
prefetch helper) で次の tabular コンペで効く問題に当たったら即使える。

---

## 並列開発を阻まない 3 つの設計選択

これらが偶然 1 日で揃ったわけではなくて、過去 2 ヶ月の設計判断が効いている。

### 1. 既存パスを壊さず「並走パス」を追加する

`SkillLibrary.search()` (token-based, legacy) と `SkillLibrary.filter_by_context()`
(conditions-based, new) は別物として共存している。`search()` を使っている
箇所はそのまま動く。新パスは `filter_by_context` 経由で呼ぶだけ。

新機能を導入するときに既存呼び出し元を変更すると、テスト全部に影響が波及して
他機能の commit と衝突する。「**並走パスを追加する**」設計を選ぶと衝突しない。

### 2. interface (Protocol) で疎結合にする

`LBPredictor` は Python の `Protocol` で定義:

```python
class LBPredictor(Protocol):
    def fit(self, cv_lb_pairs: list[tuple[float, float]]) -> None: ...
    def predict(self, cv_score: float) -> LBPrediction: ...
```

`SkillSeededSearch` は具体型を一切知らない。`Naive` でも `Linear` でも `Ridge` でも
未来の `Bayesian` でも渡せる。これによって機能 2 (Optuna 統合) と 機能 3 (factory) が
独立に進められる。

### 3. 機能フラグ的なオプション引数

新機能を `param_space=None` の場合のみ起動する設計にしておくと、既存テストを 1 行も
変えずに新パスを追加できる:

```python
@dataclass
class SkillSeededSearch:
    ...
    param_space: SearchSpace | None = None  # ← 新規追加
    lb_confidence_threshold: float = 0.5    # ← 新規追加
    sampler_seed: int = 42                  # ← 新規追加
```

CLI/呼び出し側で何も指定しなければ旧動作のまま。新機能を使いたい人は引数を 1 つ追加する。
これによってデフォルト動作を破壊せずに機能を増やせる。

---

## TabPFN 側で起きた事故と並列開発の関係

[並列記事](/post.html?slug=2026-06-23-tabpfn-two-traps-foundation-nn-and-hf-anonymous-dl)
で書いた TabPFN の HF DL hang や CV-LB gap は、Planner 注入 / Optuna 統合 / LB factory
には**一切影響しなかった**。

TabPFN playbook は `playbook/` 配下の単独ファイルで、ほかの 3 機能から import されない。
だから TabPFN を s6e6 で取り下げる判断をしても、他の 3 機能はそのまま使える。
4 機能のうち 1 つが「s6e6 では効かなかった」となっても、他 3 機能の価値が消えるわけではない。

これも「並走パス」「Protocol」「機能フラグ」の設計が活きたケース。
**1 つの実験が失敗しても全体が止まらない** ように設計しておくのが、エージェント開発の
重要な性質だと思う。

---

## 数字で見る今日のコード

- commits: 5 (push + deploy 済)
- files added: 6
- files modified: 9
- tests: +14 (合計 207 passed / 5 skipped)
- 全テストの実行時間: ~12 秒 (slow を含む)
- LB 影響: なし (0.97226 維持)

LB は伸びなかったが、KRS-OS の能力曲線は今日 1 日で明確に上がった。
明日以降の Gold アタックは、この基盤の上で展開する。
