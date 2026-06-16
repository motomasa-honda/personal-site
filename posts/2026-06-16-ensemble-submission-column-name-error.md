---
title: "CV で圧勝した ensemble が LB に出てこない — 犯人は出力 CSV の『列名 1 つ』だった"
emoji: "🏷️"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "debugging", "pandas"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- Kaggle Agent (KRS-Core) の hill climbing ensemble が、ローカル CV では single を大きく超えるのに **LB に一度も反映されていなかった**
- ジョブのログを見ると、ブレンドも提出も**全部成功**している。なのに Kaggle 上での採点だけが `SubmissionStatus.ERROR`
- 提出ファイルの中身は健全 (負値なし・NaN なし・件数も正しい)。**犯人は出力 CSV のヘッダが `Id,pred` だったこと**
- House Prices の正しい提出フォーマットは `Id,SalePrice`。ensemble だけが内部 OOF フォーマットの列名 `pred` を引き継いでいた
- `sample_submission.csv` から正しい列名を解決して出力するよう直し、ensemble が初めて COMPLETE で採点された (LB 0.12807)

## 症状: ログは全部緑、なのに LB が動かない

このシステムは複数の試行 (trial) の OOF 予測を貪欲ブレンド (Caruana の greedy hill climbing) して ensemble を作る。ローカル CV では確かに効いていた。

```
[hill_climb] step 0: +trial_1   score=0.12799
[hill_climb] step 1: +trial_121 score=0.12565
[hill_climb] step 2: +trial_132 score=0.12516
[hill_climb] step 3: +trial_111 score=0.12473
[hill_climb] step 4: +trial_1   score=0.12459
[hill_climb] step 5: +trial_112 score=0.12458
[hill_climb] saved: submission_hill_climbing.csv (CV=0.12458)
[submit] 📤 selected=submission_hill_climbing.csv (hill_climbing cv=0.12458)
[kaggle] submitted: submission_hill_climbing.csv
```

CV 0.12458 は single best (0.12808) を**大きく**上回る。submit ノードも正しく ensemble ファイルを選び、Kaggle への提出 API も成功している。ログ上の異常はゼロだ。

ところが提出履歴を見るとこうなっていた。

```
fileName                      status                     publicScore
submission_hill_climbing.csv  SubmissionStatus.ERROR     (なし)
submission.csv                SubmissionStatus.COMPLETE  0.12808
```

**single (`submission.csv`) は通るのに、ensemble (`submission_hill_climbing.csv`) だけが採点エラー**。ログは緑なので、エラーは Kaggle 側の採点フェーズで起きている。

## 切り分け: まず「値」を疑う

このコンペの指標は RMSLE で、`log(prediction + 1)` を取る。**予測に負値や NaN が混じると採点が即エラーになる**。ブレンドの過程で負値が出たのではないか、と最初は疑った。出力ファイルを直接検査する。

```
cols: ['Id', 'pred']   rows: 1459
min: 52607.81   max: 487184.04
NaN: 0   <=0: 0   inf: 0
```

値は完全に健全だった。住宅価格として妥当なレンジ、負値も NaN も inf もない。件数も提出に必要な 1459 行。

…が、ここで `cols: ['Id', 'pred']` が目に入る。**正しい提出フォーマットは `Id,SalePrice`** だ。

## 犯人: 内部フォーマットの列名がそのまま提出に漏れていた

正常に採点された single 側と並べると一目瞭然だった。

```
submission.csv                → Id,SalePrice   (COMPLETE)
submission_stacked.csv        → Id,SalePrice   (別経路、正常)
submission_optuna.csv         → Id,SalePrice   (別経路、正常)
submission_hill_climbing.csv  → Id,pred        (← これだけ違う)
```

ensemble の生成コードはこうなっていた。

```python
base_test = valid[0][1]["test_df"]          # 内部の test_pred CSV を読んだもの
id_col   = base_test.columns[0]             # "Id"
pred_col = "pred" if "pred" in base_test.columns else base_test.columns[1]
...
# 出力にも pred_col をそのまま使ってしまっている
sub_out = pd.DataFrame({id_col: base_test[id_col], pred_col: blended_test})
sub_out.to_csv("submission_hill_climbing.csv", index=False)
```

ブレンド対象の `test_pred_trial_*.csv` は**システム内部の中間ファイル**で、列名は `pred` で統一されている。読み取りに `pred` を使うのは正しい。問題は、**読み取り用の列名をそのまま出力の列名に流用していた**こと。Kaggle は `SalePrice` 列を探して見つけられず、エラーを返していた。

single / stacked / optuna が無事だったのは、これらが最終提出を作るときに `sample_submission.csv` の列名を引き継ぐ別経路を通っていたから。ensemble だけがこの作法から外れていた。

## 修正: sample_submission から「正しい列名」を解決する

このプロジェクトには、提出フォーマットの列名を `sample_submission.csv` から取るイディオムが既にあった (optuna 経路がそれを使っている)。ensemble の出力でも同じ作法に揃える。

```python
# 内部 test_pred は 'pred' 列だが、Kaggle は target 列名 (SalePrice 等) を要求する。
# ここを 'pred' のまま出すと採点が ERROR になる。
out_id_col, out_pred_col = id_col, pred_col
_sub_candidates = ["sample_submission.csv", "gender_submission.csv", "submission.csv"]
sample_path = next(
    (data_dir / f for f in _sub_candidates if (data_dir / f).exists()),
    None,
)
if sample_path is not None:
    _sample = pd.read_csv(sample_path, nrows=1)
    out_id_col, out_pred_col = _sample.columns[0], _sample.columns[1]

sub_out = pd.DataFrame({out_id_col: base_test[id_col], out_pred_col: blended_test})
```

ポイントは **読み取りの `pred_col` と出力の `out_pred_col` を分離**したこと。ブレンドは内部フォーマットの `pred` 列で行い、書き出すときだけ提出フォーマットの列名に翻訳する。sample が見つからない万一のときは従来挙動にフォールバックする。

## 検証: 同じ OOF から、今度は COMPLETE で採点される

コードを書き直させずに、残っていた実 OOF をそのまま再ブレンドして列名を確認する。

```
OUTPUT COLUMNS: ['Id', 'SalePrice']
   Id      SalePrice
 1461  123052.305...
 1462  157447.358...
ASSERT OK: 列名は Id,SalePrice
```

この修正版を Kaggle に手動提出したところ、

```
submission_hill_climbing.csv  SubmissionStatus.COMPLETE  0.12807
```

3 セッション越しで採点されなかった ensemble が、初めて LB に値を返した。

## 学び

- **ログが全部緑でも、外部採点系での失敗は別レイヤーで起きる**。提出 API の成功と採点の成功は別物
- バグの切り分けは「値 → 構造 (ヘッダ・件数・列名)」の順で潰すと速い。今回は値が健全だった瞬間に列名へ目が向いた
- **内部フォーマットの列名を出力に漏らさない**。読み取り用の列名と、外部に出す列名は明示的に分離する
- 同じ目的の処理が複数経路にあるとき、片方だけ作法から外れると静かに壊れる。**既存のイディオム (ここでは sample_submission からの列名解決) に揃える**のが安全
