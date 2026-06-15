---
title: "アンサンブルが効かない本当の理由は『コード生成の品質』だった — ローカル LLM を qwen3-coder に替えて Kaggle Agent が自己ベストを更新するまで"
emoji: "🧩"
type: "tech"
topics: ["kaggle", "machinelearning", "llm", "ollama", "ensemble"]
published: true
publication_date: "2026-06-15"
---

## TL;DR

- Kaggle Agent (自作の自動化システム KRS-Core) で**アンサンブルが一切効かない**問題を追った
- 表層の症状は「3 つの探索枝が全部同じ予測に収束 → blend しても単一モデルと同じ」
- 掘っていくと真因は **diversify が生成するコードの品質**だった。1 ワークスペースの 4 patch から **8 個以上の別々のバグ**が出てきた
- 決定的サニタイザで一般バグを潰し、**ローカル LLM を汎用/推論モデルからコード特化の `qwen3-coder:30b` に交換**、さらに**自己修復ループ**を足した
- 結果、アンサンブルが**初めて単一モデルを上回り** (CV 0.12779 → 0.12626)、**LB が自己ベストを更新** (0.12967 → 0.12824)。すべて **API 不使用・ローカル完結**

## 症状: 配線は完璧、効果はゼロ

少し前に、Kaggle Agent の OOF アンサンブル経路 (hill_climbing / stacking) を全部繋ぎ込みました。機構は end-to-end で発火する。なのに **LB が 1mm も動かない**。

原因は単純でした。**branch_explorer が生成する 3 つの探索枝が、全部ほぼ同じ予測に収束していた**。多様性ゼロのモデルをいくら blend しても、`ensemble = single` にしかならない。hill_climbing の重み最適化は賢いので、同一モデルを並べると「1 個だけ重み 1.0、残り 0」と正しく見抜く。

> アンサンブルの効果は、ベースモデルの多様性が上限を決める。

では、なぜ枝が多様にならないのか。

## 掘る: quota を使わない検証ハーネス

ここで効いたのが、**Kaggle に提出せずローカルで patch を検証するハーネス**を書いたことです。Kaggle の 1 日あたり提出上限 (10/日) を消費せずに、「生成されたコードが実行できるか・CV が妥当か」を回せる。

これを回した瞬間、真実が見えました。**diversify が生成した patch のほとんどがそもそもクラッシュしていた**。しかも理由がバラバラ:

| バグ | 内容 |
|---|---|
| `StratifiedKFold` を回帰に適用 | 連続値ターゲットで `Got 'continuous'` で即死 |
| XGBoost に category 列 | `enable_categorical=True` 未指定でクラッシュ |
| `Id` 列を特徴量に残す | train/test 列不整合 + リーク |
| `PolynomialFeatures` の誤 import | `sklearn.model_selection` から import (正しくは preprocessing) |
| `early_stopping_rounds` を fit に | LightGBM 4.x で廃止された引数 |
| **二重 log の偽 CV** | log1p 済みターゲットに `mean_squared_log_error` を再適用 → RMSLE=0.00997 の偽値 |

最後の「二重 log」が特に厄介でした。偽の極小 CV (0.00997) を出した patch が「最良」として選ばれ、システム全体を汚染する。これには **CV sanity guard** (信頼できるベースより極端に良い候補は誤計測として棄却) を入れて防御しました。

一般的なバグは決定的に書き換えるサニタイザで潰せます。でも、**潰しても潰しても次のバグが出てくる**。これは whack-a-mole だ、と気づいた瞬間が転機でした。

## 真因: コード生成モデルがコード特化じゃなかった

設定を見直すと、犯人が分かりました。

```
coder = 汎用モデル
fixer = 推論モデル (deepseek-r1 系)   ← diversify はこれが担当
```

**コード生成に、コード特化モデルを使っていなかった。** 推論モデルは「考える」のは得意でも、正しい API を吐くのは別の能力です。

ここで最新のローカルコーディングモデル事情をキャッチアップしました。2026 年時点の答えは明快で、**[`qwen3-coder:30b`](https://ollama.com/library/qwen3-coder:30b)** —— MoE で総 30B・アクティブ 3.3B、24GB VRAM に収まり、Apache 2.0、256K コンテキスト。「強いのに軽くて速い」。

coder と fixer をこれに交換しました (戦略立案・深い推論を担う planner / reasoner は推論モデルのまま)。

## 効果測定: quota ゼロのベンチ

検証ハーネスを「現行モデルで patch を生成して実行する」ベンチに拡張して、交換前後を比較しました。

```
旧 (推論モデル):  ほぼ全滅 + 偽 CV 0.0099
新 (qwen3-coder): patch 4-5/5 成功・CV 0.126〜0.128 で多様・全て base 超え
```

diversify 1 回あたりの生成時間も推論モデルより速い (25-28 秒)。**品質と速度が同時に上がった**。

残った ~20% の失敗は「`HistGradientBoosting` に存在しない `subsample` を渡す」みたいな**毎回違うランダムな hallucination**。これは個別に潰すと無限ループなので、**自己修復ループ**で汎用回収しました ── patch がクラッシュしたら traceback を fixer に渡して 1 回だけ修復・再実行する。

## ブレークスルー

すべてを繋いで、パイプライン全体を 1 本走らせました。ログがすべてを語っています。

```
🔧 自己修復成功 candidate 1: cv=0.1292        ← クラッシュを回収
🌾 OOF 収穫 2 件 (trials=[111, 112])          ← 多様な枝を保存
[hill_climb] step 0: +trial_1   score=0.12841
[hill_climb] step 1: +trial_111 score=0.12690
[hill_climb] step 2: +trial_121 score=0.12642
[hill_climb] step 3: +trial_1   score=0.12626
[hill_climb] weights={1: 0.5, 111: 0.25, 121: 0.25}   ← ついに重みが割れた
[submit] 📤 selected=submission_hill_climbing.csv      ← 初めて ensemble を選択
[submit] 🎉 new best public=0.12824                   ← 自己ベスト更新
```

セッション冒頭まで `{1: 1.0}` (= 単一モデル) だった hill_climbing の重みが、**3 つの異なる trial に割れた**。アンサンブル CV (0.12626) が単一 (0.12779) を**初めて上回り**、提出セレクタが初めてアンサンブルを選んだ。そして LB が **0.12967 → 0.12824** に更新された。

## 学び

1. **「機構が動く」と「効果が出る」は別物**。両方を別々に検証する価値がある。今回は配線が正しかったからこそ、ボトルネックがアンサンブル機構ではなく**コード生成品質**にあると断定できた
2. **外部 API に頼らなくても、ローカル LLM の品質ボトルネックは解ける**。鍵は「役割に合ったモデルを選ぶ」こと —— コード生成にはコード特化モデルを
3. **whack-a-mole に気づいたら一段上げる**。個別バグを潰し続けるより、(a) モデルを替える (b) 自己修復で汎用回収する、の方が筋がいい
4. **quota / コストを使わない検証ループ**は、反復速度を桁で変える。提出せずに「コードが動くか」を測れるだけで、1 日に試せる仮説の数がまるで違う

次は、この多様性が別ドメインのコンペでも再現するかを確かめます。アンサンブルがようやく「本物」になったので、ここから先の積み上げが楽しみです。

---

*Sources: [Ollama qwen3-coder](https://ollama.com/library/qwen3-coder:30b), [Best Local Coding Models 2026](https://www.promptquorum.com/power-local-llm/best-local-coding-models-2026)*
