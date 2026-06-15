---
title: "CatBoost の cat_features に NaN を渡すと即死する — LLM の自己修復でも直せなかったバグを、決定的サニタイザで潰す"
emoji: "🐱"
type: "tech"
topics: ["kaggle", "catboost", "machinelearning", "llm", "pandas"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- Kaggle Agent (KRS-Core) の codegen ベンチで、CatBoost を使う patch が必ずクラッシュしていた
- 原因は `CatBoostError: cat_features must be ... NaN values should be converted to string`。**カテゴリ列に欠損があると CatBoost は学習を拒否する**
- 厄介なのは、LLM の自己修復ループ (traceback を渡して直させる) でも直らなかったこと
- さらに、astype の行だけ直しても **`X_test = test.drop(...)` で前処理ループ前にコピーした派生フレーム**に NaN が残り、予測時に再発した
- 最終的に「CatBoost を使うコードなら、**読み込み直後に train/test の非数値列 NaN を補完**する」決定的サニタイザで解決。実機で CV 0.12403 完走を確認

## 発見: quota を使わない codegen ベンチで炙り出す

このシステムは LLM (ローカルの `qwen3-coder:30b`) が探索コードを書く。Kaggle の提出上限 (10/日) を消費せずに「今のモデルが今どんなコードを書くか」を測るベンチを回したら、House Prices で 3 本中 1 本がこう落ちた。

```
_catboost.CatBoostError: Invalid type for cat_feature[...]=NaN :
cat_features must be integer or string, real number values and
NaN values should be converted to string.
```

LightGBM / XGBoost は NaN をネイティブに扱えるので、LLM はカテゴリ列を欠損のまま渡しがち。だが **CatBoost は cat_features に渡す列に NaN があると即死する**。House Prices には `PoolQC` や `Alley` のように「欠損 = 該当なし」のカテゴリ列が多く、これを素通しすると死ぬ。

## 自己修復 (LLM fixer) でも直らない

このシステムには、クラッシュした patch の traceback を fixer に渡して 1 回だけ直させる自己修復ループがある。ランダムな hallucination はだいたいこれで回収できる。だが**この CatBoost NaN は直せなかった**。エラーメッセージは明快なのに、fixer は的を射た修正を返せなかった。

LLM の自己修復は万能ではない。**再現性があり、原因が構造的なバグは、決定的に潰すのが正しい**。ここで sanitizer (生成コードに決定的な書き換えを掛ける層) の出番になる。

## 第一案 (astype の前に fillna) では半分しか直らない

最初はこう考えた。「`X[c].astype("category")` の前に `.fillna("missing")` を挟めばいい」。

```python
# X[c] = X[c].astype("category")
# → X[c] = X[c].fillna("missing").astype("category")
```

これで train 側は通って学習が走るようになった。ところが**予測でまた落ちた**。patch のコードをよく見るとこうなっていた。

```python
X_test = test.drop(columns=[id_col])      # ← ここで test のコピーを作る
...
for c in X.columns:                       # この後のループは X と test を変換するが
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = X[c].astype("category")
        test[c] = test[c].astype("category")
...
model.predict(X_test)                      # X_test は前処理を浴びていない別フレーム
```

`X_test` は前処理ループの**前**に `test.drop()` でコピーされた別オブジェクト。後段のループは `test` と `X` を変換するが、`X_test` には届かない。だから astype の行をいくら直しても、`X_test` には欠損カテゴリが残ったまま予測に流れる。これは LLM が書いたコードの**データフローのバグ**で、astype 行を狙い撃ちする修正では構造的に届かない。

## 解決: 読み込み直後に「源流」で埋める

データフローを追いかける代わりに、**源流で埋める**。CatBoost を使うコードなら、`train`/`test` を読んだ直後に非数値列の NaN を `"missing"` で補完する行を注入する。こうすれば `X = train.drop(...)` も `X_test = test.drop(...)` も、その後どんな順序でコピー・変換されても、埋まった値を引き継ぐ。

```python
def _catboost_fill_at_source(src):
    if "CatBoost" not in src:                 # CatBoost を使うときだけ
        return src
    if SENTINEL in src:                        # 冪等性: 二重注入しない
        return src
    # train = pd.read_csv(...) / test = pd.read_csv(...) の直後に注入
    block = (
        f"{indent}{SENTINEL}\n"
        f"{indent}for _df in (train, test):\n"
        f"{indent}    for _c in _df.columns:\n"
        f"{indent}        if not pd.api.types.is_numeric_dtype(_df[_c]):\n"
        f'{indent}            _df[_c] = _df[_c].fillna("missing")\n'
    )
    ...
```

ここでも pandas 3 の罠が出る。**文字列列は `object` でなく `str` dtype** なので、`select_dtypes(include="object")` では取りこぼす。`is_numeric_dtype` の否定で「非数値列」を拾うのが正解。

加えて、astype の前の fillna も「保険」として残し、誤爆防止のため**CatBoost を使わないコードには一切触らない**・**二重適用しない (冪等)** をテストで固定した。

## 検証: 落ちていた patch がそのまま完走する

修正版サニタイザを、以前クラッシュした patch のコードにそのまま適用して再実行した。

```
fold 0: RMSE=0.13071
...
CV RMSE = 0.12403
✅ done
```

`cat_features must be ...` で死んでいた patch が、コードを LLM に書き直させることなく完走した。House Prices の codegen 成功率は実質 2/3 → 3/3 に上がった。

## 学び

- エラーメッセージが明快でも、**LLM の自己修復が直せないバグはある**。再現性があるなら決定的に潰す
- バグの**症状の行**ではなく**データフローの源流**を直すと、コピーや変換の順序に依存せず効く
- pandas 3 では `object` 判定が静かに壊れる。`is_numeric_dtype` の否定で非数値列を拾う
- 決定的な書き換えは「対象外には触らない・二重適用しない」をテストで固定して初めて安全に運用できる
