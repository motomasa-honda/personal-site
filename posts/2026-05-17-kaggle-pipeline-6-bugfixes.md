---
title: "Kaggle自動分析パイプラインの致命的バグを6本まとめて直した話"
emoji: "🐛"
type: "tech"
topics: ["kaggle", "python", "機械学習", "streamlit", "llm"]
published: false
publication_date: "2026-05-17"
---

## TL;DR
- CVスコアの比較方向が逆になっていてRMSLEタスクで永遠に改善されないバグがあった
- CVとLBのスケールが別単位で「CV: 0.0123 / LB: 3.5656」という比較不能な状態だった
- `StratifiedKFold` を回帰タスクに使っていてsklearnエラーが出ていた
- Titanicコンペ固有の文字列が汎用エンジンにハードコードされていた
- これらを `metric_registry.py` という1ファイルで根本解決した

## 問題の全体像

LLMエージェントがKaggleコードを自動生成・実行・改善するパイプラインを作っていた。Titanic（分類）では動いていたのに、store-sales（時系列回帰・評価指標RMSLE）に切り替えたら挙動がおかしくなった。

調べると致命的なバグが6本見つかった。

## バグ1: CVスコアの比較方向が逆

最も致命的だったやつ。ベストCV更新のロジックがこうなっていた。

```python
# 旧コード（2304行の中に埋まっていた）
if prev_best is None or current_cv > prev_best:
    st.session_state.best_cv = current_cv
    st.session_state.best_code = current_code
```

`current_cv > prev_best` つまり「大きい方が良い」固定。AccuracyやAUCなら正しいが、RMSLEは**小さい方が良い（minimize方向）**。

結果として「スコアが悪化したのにベスト更新」「スコアが改善したのに更新されない」という逆転現象が発生していた。

**修正**: `metric_registry.py` に `MetricSpec` クラスを定義し、`is_better()` メソッドで方向を考慮した比較に変更。

```python
@dataclass
class MetricSpec:
    canonical_name: str
    sklearn_scoring: str
    direction: str  # "maximize" | "minimize"
    cv_to_lb: Callable[[float], float]

    def is_better(self, candidate: float, reference: Optional[float]) -> bool:
        if reference is None:
            return True
        if self.direction == "maximize":
            return candidate > reference
        return candidate < reference  # minimize の場合は逆
```

呼び出し側はこうなった。

```python
_metric = st.session_state.get("resolved_metric", None)
_is_better = (
    _metric.is_better(current_cv, prev_best)
    if _metric else
    (prev_best is None or current_cv > prev_best)  # フォールバック
)
```

## バグ2: CVとLBが別スケールで比較不能

`sklearn` の `cross_val_score` が `neg_mean_squared_log_error` を返すとき、値は **負の二乗対数誤差**（例: `-0.0123`）になる。

一方KaggleのLBはRMSLE（例: `0.111`）。変換式は `sqrt(abs(cv_raw))`。

旧コードの `extract_score()` は `abs()` だけかけていたので `0.0123` を返し、LBの `0.111` と全く別物を比較していた。「CV改善してるのにLBが全然違う」という状態の根本原因がこれだった。

```python
# metric_registry.py に換算式を持たせる
MetricSpec(
    canonical_name="RMSLE",
    sklearn_scoring="neg_mean_squared_log_error",
    direction="minimize",
    cv_to_lb=lambda x: math.sqrt(abs(x)),  # ← ここが核心
    needs_log1p=True,
    clip_nonneg=True,
)
```

```python
# パイプライン側
_cv_raw = extract_score(output)
current_cv = apply_cv_to_lb_scale(_cv_raw)  # LBスケールに統一

def apply_cv_to_lb_scale(raw_val):
    _m = st.session_state.get("resolved_metric", None)
    if _m is None:
        return abs(raw_val) if raw_val is not None else None
    return _m.cv_to_lb(raw_val) if raw_val is not None else None
```

## バグ3: 回帰タスクにStratifiedKFoldを使っていた

`build_coder_prompt()` の else ブランチがこうなっていた。

```python
# 旧コード
else:
    cv_instruction = (
        "- CVは必ず StratifiedKFold(n_splits=10, shuffle=True, random_state=42) を使う\n"
        ...
    )
```

`StratifiedKFold` はラベルの層別化をするためのもので、**連続値の回帰タスクには使えない**。sklearnが実行時エラーを出す。タスク種別で分岐するよう修正した。

```python
_is_regression = (
    dataset_info.get("task_type") == "regression"
    or tpl_key == "tabular_regression"
)
if _is_regression:
    cv_instruction = "- CVは必ず KFold(n_splits=5, shuffle=True, random_state=42) を使う\n"
else:
    cv_instruction = "- CVは必ず StratifiedKFold(n_splits=5, shuffle=True, random_state=42) を使う\n"
```

## バグ4〜6: Titanicハードコード・スケールズレ・評価指標誤記

`decision_engine.py` にこんなコードが残っていた。

```python
# 旧コード（汎用エンジンのはずなのに...）
def _overfit_hint():
    return (
        "- Cabinカラムはhas_cabin = df['Cabin'].notna().astype(int) で...\n"
        "- Ticketカラムは必ずドロップする\n"
        "- NameカラムからTitle(称号)を抽出する...\n"
    )
```

store-salesには当然Cabin列もTicket列もない。毎回的外れなアドバイスがFixerに渡されていた。

また `fetch_comp_description()` でLLMに説明文を生成させると「評価指標はRMSE」と誤記することがあり、それを `lookup_metric()` が拾うとRMSEとRMSLEを間違えるという問題もあった。プロンプトに正式名称を強制する文言を追加して解消。

```python
"- 評価指標（RMSLE/RMSE/AUC/LogLoss/Accuracyなど正式名称で必ず明記すること）\n"
"重要: 評価指標は省略せず正確な名称（例: RMSLE, AUC, LogLoss）を使うこと"
```

## metric_registry.py が全ての根本解決策だった

結局、これらのバグは全部「メトリクスの知識がコード中に散在していた」ことが原因だった。修正後は `metric_registry.py` に全メトリクスの仕様を集約。

```python
METRIC_REGISTRY: List[MetricSpec] = [
    MetricSpec(
        canonical_name="RMSLE",
        sklearn_scoring="neg_mean_squared_log_error",
        direction="minimize",
        cv_to_lb=lambda x: math.sqrt(abs(x)),
        aliases=["rmsle", "root mean squared log error"],
        needs_log1p=True,
        clip_nonneg=True,
        task_hint="timeseries",
    ),
    MetricSpec(
        canonical_name="AUC",
        sklearn_scoring="roc_auc",
        direction="maximize",
        cv_to_lb=lambda x: x,
        aliases=["auc", "roc_auc", "area under curve"],
        task_hint="tabular_classification",
    ),
    # ...全12メトリクス
]
```

コンペの説明文から自動でメトリクスを検出して `session_state` に保存。以降の全処理がこれを参照する。

```python
# タスク判定ボタン押下時
_resolved = detect_metric_from_text(comp_description)
st.session_state["resolved_metric"] = _resolved
if _resolved:
    st.success(f"📏 評価指標を自動解決: **{_resolved.canonical_name}** (direction={_resolved.direction})")
```

実際にstore-salesで試すと「RMSLE (direction=minimize)」と正しく表示されるようになった。

## 学んだこと

- 「Titanicで動いた」と「汎用的に動く」は全然別物。最初から汎用設計にすべきだった
- メトリクスの `direction`（最大化/最小化）は1箇所で管理しないと必ずズレる
- CVとLBのスケールを早期に統一しないと「改善しているのか悪化しているのか」すら判断できない
- LLMが「RMSE」と書くか「RMSLE」と書くかはプロンプト次第。重要な固有名詞は強制する

## 参考

- 修正コミット: `feat: Phase1-3 refactor + Grandmaster Playbook`（kaggle-pipelineリポジトリ）
- 参考にした設計: [AutoKaggle](https://github.com/multimodal-art-projection/AutoKaggle) のML Tools Library思想
- 環境: Ubuntu 24.04 / RX 7900 XTX / Ollama / Streamlit
