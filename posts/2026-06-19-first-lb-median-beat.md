---
title: "Metric-as-SSOT 改修が、実 LB で『中央値超え』として実証された"
emoji: "🎯"
type: "tech"
topics: ["kaggle", "automl", "machinelearning", "validation", "mlops"]
published: true
publication_date: "2026-06-19"
---

## TL;DR

前日の `0/1 float を回帰として誤分類` 事件と同時に入れた **Metric-as-SSOT** 改修
(指標を単一の真実として task/提出形式/CV/方向の全ノードに一貫伝播する仕組み) を、
今日 quota が回復した瞬間に **playground-series-s6e5** で late-submit した。

結果: **public 0.94869 / private 0.94932** (修正前 0.94145 / 0.94268 から **+0.00724**)、
**中央値 0.94797 を超え**、**CV-LB gap は 0.00025** に収束した。机上で立てた改修が、
実 LB で 1 対 1 の結果として返ってきた。

## 修正前の状態 (前日のおさらい)

`playground-series-s6e5` (Predicting F1 Pit Stops, **二値分類 AUC**) を初めて end-to-end で
回したとき、二つの致命的な不整合が **互いに相殺して**たまたま動いていた状態だった:

1. **B1**: target が `0.0 / 1.0` の float だったため、`detect_task_type` が「float 型 = 回帰」と
   先に判定。タスクが **回帰として解かれた**。
2. **B2**: 分類テンプレは AUC でも `(pred > 0.5).astype(int)` で **ハードラベル**を提出していた。
   AUC はランク情報を見るので、ハードラベル提出はランク情報を失って致命的に悪化するはず。

ところが、B1 が「連続値で予測」していたおかげで提出は確率に近い数値になり、B2 が活きなかった。
結果、AUC = 0.94145 で偶然中央以下に着地。**B1 だけ直すと B2 が露出して悪化する**罠付き。

## 修正: Metric-as-SSOT を入れた

- `parser.metric_policy(metric)` を新設:
  - `(task_family, higher_is_better, submission, cv_metric)` を返す純関数。指標から逆算する。
  - 例: `auc → (classification, True, proba, roc_auc_score)`、`rmse → (regression, False, value, rmse)`。
- `analyze` で `task_type` を指標優先に決める (data ヒューリスティックは保険):
  ```python
  mp = metric_policy(state["evaluation_metric"])
  state["task_type"] = mp.get("task_type") or detect_task_type(...).get("task_type")
  state["metric_higher_is_better"] = mp.get("higher_is_better", ...)
  state["submission_format"] = mp.get("submission")
  ```
- テンプレ提出形式を `{{SUBMISSION_FORMAT}}` で render し、`proba | label` をテンプレが分岐する。
- これだけで B1/B2 が同時に正される (片方だけ直す悪化罠が消える)。

純関数なので Mac で pytest 16 ケース全 PASS。dev 再走で `metric_policy: task=tabular_classification
submission=proba` をライブログで確認、`submission.csv` の中身も確率になった。**机上の検証は完了**。

## 量子化された期待: 「中央値 0.94797 を超えるはず」

LB を眺めると、F1 PitNextLap の LB は **0.948〜0.955 の密集帯**で、1 位 0.955 / 中央 0.948 /
我々 0.94145。`AUC +0.007` で順位が大きく動くゾーン。

修正前の 0.94145 を「ハードラベル提出 = ランク情報喪失」由来と仮定すると、確率提出で
ランク情報が戻れば理論上は CV 値 (本番 CV(AUC) = 0.94894) に近いところに着地するはず。
中央 0.948 を狙えるかどうかが question。

## 結果: 改善幅 +0.00724、中央値超え

quota 復帰 (JST 9:00) を待って、修正後 submission を late-submit:

| 提出 | public | private |
|---|---|---|
| 修正前 (B1 と B2 が相殺) | 0.94145 | 0.94268 |
| **修正後 (確率提出)** | **0.94869** | **0.94932** |
| 改善幅 | **+0.00724** | **+0.00664** |

`public 0.94869 > 中央値 0.94797` で **実 LB で中央超えを達成**。さらに `CV 0.94894 vs LB 0.94869
= gap 0.00025` で、CV が LB を極めて正確に追従している。CV を信頼してチューニングできる状態。

## CV-LB gap が小さいことの含意

機械学習の現場で **CV だけ良くなる「CV-LB 乖離」**は最大の罠の 1 つ。原因はだいたい:

- CV のグルーピングが間違っている (Race/Year/Driver でグループ構造があるのに random KFold)
- target leak (test と相関する列を train で見てしまっている)
- sample distribution の違い (train と test のクラス比が違う)
- 評価指標と CV メトリックがズレている

我々は GroupKFold への切替や stratified の改善を **後回し**にしていたが、データで CV-LB gap
0.00025 まで詰まっているとわかれば、**GroupKFold は当面不要**と判断できる。データ駆動で
過剰設計を避けられる、という小さい収穫。

## 設計上の収穫: 「指標 = 単一の真実」が回路として通った

このセッションで一番大きいのは、Metric-as-SSOT が **机上の設計 → 実 LB での 1 対 1 結果**で
返ってきたこと。

```
parser.metric_policy(metric)
    ↓
analyze.py        task_type / submission_format / higher_is_better
    ↓
codegen.py        TemplateSpec.submission_format
    ↓
templates/...     {{SUBMISSION_FORMAT}} を埋め込み、proba/label を分岐
    ↓
submission.csv    確率 or ラベル (指標で自動切替)
    ↓
Kaggle LB         実 score
```

この **指標 → 各ノードへの一方向伝播**を貫いたことで、「タスクの種類は何か?」「CV 指標は?」
「提出形式は?」を **各ノードが独立に推測しない**ようになった。各推測がずれて B1/B2 のような
事故になっていたのが、SSOT 1 点で同期するように直った。

明日以降のメダル対象でも、Metric-as-SSOT がそのまま効くはず――と書いた翌日、3 クラス + 文字列
ラベルで [また 5 連続 ERROR を踏んだ](/post.html?slug=2026-06-19-multiclass-string-label-ssot)。
SSOT は「全部書ききった」と思ってからが本番。

## おまけ: dev / submit 分離も合わせて入った

`enable_submit` フラグを追加し、`False` の dev mode では Kaggle 自動提出を止めて CV だけ取れる
ようにした。再走で 10/day の quota を浪費しないため。これがあると **「修正したテンプレが
正しく動くか」だけを quota 0 で繰り返し確認**できて、開発速度が一気に上がる。

開発と本番で「動作させる人」の意図を分離するのは、機械学習システムでも普通の開発でも同じ
原則だった。

---

**関連**:
[0/1 が float で入っていただけで、二値分類が回帰として解かれていた](/post.html?slug=2026-06-19-binary-as-regression-metric-ssot) /
[5 連続 SubmissionStatus.ERROR が教えてくれた、多クラス文字列ラベルの SSOT 漏れ](/post.html?slug=2026-06-19-multiclass-string-label-ssot)
