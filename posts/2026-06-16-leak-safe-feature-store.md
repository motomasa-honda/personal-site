---
title: "全 trial の特徴ベースを底上げする『決定的 Feature Store』— Kaggle Agent の CV を 0.12961 → 0.12873 に下げた一手"
emoji: "🧱"
type: "tech"
topics: ["kaggle", "machinelearning", "featureengineering", "pandas", "mlops"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- 自作の Kaggle 自動化システム (KRS-Core) で、**全試行が共有する特徴量ストア**を入れた
- 設計のポイントは「再計算をキャッシュして速くする」**ではない**。この系では学習が計算の支配項なので、その手の Feature Store は ROI が低い
- 代わりに「**target を一切使わない・決定的なリーク安全特徴**を 1 回だけ物質化して、全 trial が raw CSV の代わりに読む」型にした
- House Prices で同一モデル・同一 CV・同一 seed の A/B: **CSV 0.12961 → Feature Store 0.12873**。78 個の特徴を全試行に無償で配れる
- pandas 3 で文字列が `object` でなく `str` dtype になった件など、地味な落とし穴も込みで書く

## なぜ「再計算キャッシュ型」をやめたか

「Feature Store」と聞くと、計算済みの特徴量行列を parquet にキャッシュして次の trial で再利用する、という絵をまず思い浮かべる。だがこのシステムでは効かない。

理由は単純で、**特徴量エンジニアリングは各 trial の学習コードの中に埋め込まれていて、実行時間の支配項は学習そのもの**だから。CatBoost を 2000 iteration × 5 fold 回すコストに比べたら、欠損補完や集約特徴の計算なんて誤差。だから「FE の再計算を省く」キャッシュは、削れる時間がそもそも小さい。

> 最適化する前に、何が支配項かを測れ。FE のキャッシュは、この系では支配項を外している。

ではこの系で効く「Feature Store」とは何か。答えは **探索の無駄削減と CV の底上げ**だ。どの trial も同じ強い特徴ベースから始まれば、平均的なスコアの床が上がる。これは学習コストを増やさずに効く。

## 設計: target を読まない・決定的・drop-in

3 つの不変条件を置いた。

1. **リーク安全**: target 列は値すら読まない。target encoding はしない。
2. **決定的**: 乱数を使わない。同じ入力から必ず同じ出力 (再現性 = キャッシュ整合)。
3. **drop-in**: 出力は「元の全列 + 追加特徴」。train は target 列も保持するので、生成コードの `train.drop(columns=[target])` がそのまま動く。

追加する特徴は汎用的でリークしないものだけに絞った。

| 特徴 | 中身 |
|---|---|
| `{col}__isna` | 欠損フラグ (train か test に NaN がある列だけ) |
| `row__n_missing` | 行ごとの欠損数 |
| `{col}__freq` | 非数値列の出現頻度エンコード (train+test 合算カウント) |

頻度エンコードを train+test 合算で作るのは「transductive」で、target に依存しないのでリークではない。Kaggle では test の特徴分布は使ってよい。

コア部分はこれだけ。

```python
def build_features(train, test, target_col="", id_col=""):
    train, test = train.copy(), test.copy()
    excluded = {c for c in (target_col, id_col) if c}
    feat_cols = [c for c in train.columns if c not in excluded and c in test.columns]

    added = []
    for c in feat_cols:                       # 1) 欠損フラグ
        if train[c].isna().any() or test[c].isna().any():
            train[f"{c}__isna"] = train[c].isna().astype("int8")
            test[f"{c}__isna"] = test[c].isna().astype("int8")
            added.append(f"{c}__isna")

    train["row__n_missing"] = train[feat_cols].isna().sum(axis=1).astype("int16")
    test["row__n_missing"] = test[feat_cols].isna().sum(axis=1).astype("int16")

    for c in feat_cols:                       # 3) 頻度エンコード (非数値列)
        if pd.api.types.is_numeric_dtype(train[c]):
            continue
        freq = pd.concat([train[c], test[c]], ignore_index=True).value_counts(dropna=False)
        train[f"{c}__freq"] = train[c].map(freq).fillna(0).astype("int32")
        test[f"{c}__freq"] = test[c].map(freq).fillna(0).astype("int32")
        added.append(f"{c}__freq")
    return train, test, {"added_features": added, "n_added": len(added)}
```

## 物質化は「コンペごと 1 回・冪等」に

`materialize_feature_store(data_dir, target_col, id_col)` が `data/features_train.parquet` /
`features_test.parquet` を書く。既に parquet が `train.csv` より新しく存在すれば作り直さない (冪等)。
これをエージェントの分析ノードで best-effort 実行し、失敗してもパイプラインは止めない。

テンプレート側は drop-in に読むだけ。

```python
if (DATA / "features_train.parquet").exists():
    train = pd.read_parquet(DATA / "features_train.parquet")
    test = pd.read_parquet(DATA / "features_test.parquet")
else:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
```

## 落とし穴: pandas 3 で文字列が `str` dtype になった

非数値列の判定を `select_dtypes(include="object")` で書くと、**pandas 3 では文字列列が `object` ではなく `str` dtype になっているので取りこぼす**。実際この環境の pandas は 3.0.x で、CSV から読んだ文字列列は `str`。

```python
>>> s = pd.Series(["a", None])
>>> s.dtype, pd.api.types.is_numeric_dtype(s)
(str, False)
```

なので「非数値列」は `not is_numeric_dtype(...)` で判定する。`object` 決め打ちは pandas 3 で静かに壊れる。コードベース全体でこの慣用に揃えた。

## 結果: 同一条件 A/B で CV が下がる

実証は雑にやらない。**同じテンプレート・同じ KFold・同じ seed・同じモデル**で、唯一の差分を「parquet を読むか raw CSV を読むか」だけにした。

| 条件 | CV (RMSLE) |
|---|---|
| CSV のみ (ベースライン) | 0.12961 |
| Feature Store | **0.12873** |

78 個のリーク安全特徴を足しただけで -0.00088。派手ではないが、これは**全 trial に無償で乗る床上げ**だ。探索が何百回回ろうと、その全部のスタート地点が少し高くなる。

## おまけ: 生成コードにも parquet を読ませる

このシステムは LLM がコードを書き換えて探索する。テンプレートの系譜を継ぐコードは parquet を読むが、LLM が気まぐれに `read_csv` に戻すと Feature Store を使わない。そこで diversify プロンプトのガードに一文だけ足した。

> `data/features_train.parquet` があれば `read_parquet` でそれを読む (raw より特徴が豊富)。無ければ CSV に fallback。

base コードが parquet 非対応でも、生成された patch が `read_parquet` を使うようになることを実機で確認した。安全網はコード側とプロンプト側の二段で張る。

## まとめ

- この系の「Feature Store」は**速度のためのキャッシュではなく、CV の床を上げる共有特徴**として設計した
- リーク安全 (target 不使用) + 決定的 + drop-in の 3 条件で、既存の学習コードを 1 行も壊さずに刺さる
- pandas 3 の `str` dtype のような地味な落とし穴は、コードベース全体で慣用を揃えて踏み抜きを防ぐ
- 効果は同一条件 A/B で測る。CV 0.12961 → 0.12873、全 trial 無償。
