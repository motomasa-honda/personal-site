---
title: "ensemble が効かない真因は『探索の多様性不足』だった — GBDT ばかりの自動探索に線形モデルを混ぜる"
emoji: "🌱"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "llm", "diversity"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- ensemble (hill climbing) が CV で圧勝しても LB で効かない問題を追っていったら、真因は重みの正則化ではなく **ブレンドする中身が似すぎていること**だった
- Kaggle Agent (KRS-Core) の探索方向 (`directions`) を見たら、モデル軸が **全部 GBDT 系** (lightgbm / xgboost / catboost / HistGradientBoosting) だった
- しかも方向リストは上位 `k` 個しか使われないので、実際の探索は「GBDT を少しいじる」バリエーションばかりになっていた
- そこで「**異系統モデル**」方向 (回帰 = Ridge / Lasso / ElasticNet、分類 = LogisticRegression、または ExtraTrees / RandomForest) を上位に追加した
- ローカル LLM (`qwen3-coder:30b`) が、その方向で実際に **Ridge を使うコードを生成**することを確認。ブレンドに低相関の多様性を入れる土台ができた

## ブレンドは「違う間違い方」をする学習器が要る

ensemble が効く理屈は単純で、**それぞれ違う方向に間違える学習器を混ぜると、誤差が打ち消し合う**から。逆に言うと、似た間違い方をするモデルをいくら混ぜても平均が少し滑らかになるだけで、汎化は伸びない。

別記事で、ensemble の重みを正則化 (均等化) したら LB が悪化した話を書いた。その実証から分かったのは、「重みをどういじっても効かないのは、**ブレンドする中身が似ているから**」ということだった。中身を多様にしない限り、ensemble は構造的に頭打ちになる。

## 犯人: 探索方向がぜんぶ GBDT だった

このシステムは LLM に「探索の方向 (`direction`)」を与えて、それぞれ違う試行コードを書かせる。その方向リストを見たらこうなっていた。

```python
directions = [
    "Feature Engineering 重視 — 派生特徴を追加。モデルは base と同じでよい。",
    "モデル切替 — lightgbm ↔ xgboost ↔ catboost ↔ HistGradientBoosting。",  # ← 全部 GBDT
    "正則化・ハイパラ厳格化 — learning_rate を半分、num_leaves を振る。",        # ← 同一モデル
    "前処理強化 — fillna、log1p、稀カテゴリ圧縮。",                              # ← 同一モデル
    "アンサンブル — LightGBM + HistGradientBoosting を 0.5/0.5 で blend。",      # ← 全部 GBDT
][:k]
```

モデル軸を動かす唯一の方向 (2 番目) すら、**GBDT の中での切り替え**でしかない。線形モデルも距離ベースも NN も、探索の選択肢に一度も入っていなかった。

さらに `[:k]` がある。実運用の `k=3` では上位 3 方向しか使われないので、実際の探索は「FE をいじる / GBDT を別の GBDT に変える / ハイパラを振る」の 3 つ。**どれも GBDT**だ。これでは ensemble に多様性が生まれようがない。

## 修正: 「異系統モデル」方向を上位に挿し込む

モデルの許可リスト自体は、もともと線形も RandomForest も ExtraTrees も含んでいた (環境に入っているので import できる)。足りなかったのは**それを使えと指示する方向**だけだった。そこで異系統方向を `k` の射程内 (2 番目) に追加する。

```python
directions = [
    "Feature Engineering 重視 — ... モデルは base と同じでよい。",
    "異系統モデル — GBDT と相関の低い系統に切替えてブレンドの多様性を稼ぐ。"
    "回帰なら Ridge / Lasso / ElasticNet、分類なら LogisticRegression、"
    "または ExtraTrees / RandomForest。線形・距離系は学習前に必ず "
    "X = X.fillna(X.median(numeric_only=True)) で NaN を埋め StandardScaler を通すこと。",
    "モデル切替 — lightgbm ↔ xgboost ↔ catboost ↔ HistGradientBoosting。",
    ...
    "アンサンブル — base の GBDT と異系統 (Ridge/ElasticNet/ExtraTrees 等) を blend。",  # ← 異系統 blend に更新
][:k]
```

ポイントは順序だ。`k=3` でも必ず「異系統モデル」が入るように **2 番目**に置いた。これで探索は最低でも `[FE 重視 / 異系統 / GBDT 切替]` の 3 系統に散る。base が GBDT なので、ブレンド候補に GBDT と線形が同居するようになる。

線形・距離系は GBDT と違って **NaN を渡すと即死**し、スケール非依存でもないので、`fillna` → `StandardScaler` のガードを方向の文言に埋め込んだ。LLM がここを忘れると実行時に落ちるためだ。

## 確認: LLM は実際に Ridge を書くか

方向を足しても、LLM がその方向で本当に異系統コードを書かなければ意味がない。ローカルの `qwen3-coder:30b` に、追加した異系統方向と GBDT のベースコードを渡して 1 回生成させた。

```
STRATEGY: diversify_iter1
  found: Ridge
```

生成コードには `Ridge` が含まれ、GBDT 系のキーワード (LGBM / XGB / CatBoost) は出てこなかった。**base の GBDT から、ちゃんと線形モデルへ系統を切り替えたコードを書いた**。ブレンドに低相関の多様性を入れる土台ができた。

## ここまでとこれから

end-to-end (実際にこの異系統 trial が OOF に収穫され、hill climbing が GBDT + 線形を混ぜて LB を更新するか) の検証は、Kaggle の提出枠を温存して次に回した。だが「探索の多様性こそが ensemble の効き目を決める」という筋は通った。

順番が大事で、

1. まず ensemble の機構を正しく動かす (重み分散・提出フォーマット)
2. 次に「CV が下がっても LB が動かない」= CV overfit を観測する
3. 正則化を試して「重みいじりでは直らない」と分かる
4. **真因は中身の多様性不足だと特定し、探索方向に異系統を足す**

という流れで、ようやく本丸に手が届いた。

## 学び

- ensemble の効き目は重みの付け方より **中身の多様性**で決まる。似た学習器の平均は伸びない
- 自動探索が偏るのは、たいてい**探索方向の定義そのものが偏っている**から。候補リストを疑う
- 方向リストに `[:k]` のような足切りがあるなら、**入れたい方向は射程内の順位に置く**。末尾に足しても使われない
- 許可リストにあるのに使われていない選択肢は「指示の不在」が原因のことが多い。能力ではなく方向を足す
- 線形・距離系を自動生成に混ぜるなら、NaN とスケールのガードを方向の文言に同梱する
