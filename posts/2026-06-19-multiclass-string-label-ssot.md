---
title: "5 連続 SubmissionStatus.ERROR が教えてくれた、多クラス文字列ラベルの SSOT 漏れ"
emoji: "🌌"
type: "tech"
topics: ["kaggle", "automl", "machinelearning", "validation", "debugging"]
published: true
publication_date: "2026-06-19"
---

## TL;DR

二値分類で動いていた自律 Kaggle システムを、メダル対象の **3 クラス分類 (GALAXY/STAR/QSO)、
Balanced Accuracy** のコンペに投入したら、**5 連続 SubmissionStatus.ERROR** で順位がつかなかった。
解剖すると、Metric-as-SSOT の **多クラス + 文字列ラベル + 評価指標バリエーション**という 3 つの
盲点が同時に露呈していた。3 つを一気に直したら、`CV=0.954 / public LB=0.955` で valid な提出
ができるようになった。

## 状況: メダル対象初参戦が「全部 ERROR」

前日に二値分類の Playground Series で **Metric-as-SSOT** 改修を入れて、public LB 0.94869 で
中央超えを達成した。同じシステムで翌日のメダル対象 (3 クラス天体分類、teams ~1972) に投入。

```
metric_policy: task=tabular_classification higher=True submission=label
```

ここまではいい。ジョブも完走 (succeeded)。だが 5 提出全てこの状態:

```
ref      fileName        status                  publicScore  privateScore
53842651 submission.csv  SubmissionStatus.ERROR  (空)         (空)
53841298 submission.csv  SubmissionStatus.ERROR  (空)         (空)
53840598 submission.csv  SubmissionStatus.ERROR  (空)         (空)
53839224 submission.csv  SubmissionStatus.ERROR  (空)         (空)
53838472 submission.csv  SubmissionStatus.ERROR  (空)         (空)
```

スコアがつかない = 順位もつかない = **メダル対象なのに 0 点で 1 日が消費された**。

## 何が壊れていたか

`submission.csv` の中身を sample と並べる:

```
# 期待 (sample_submission.csv)
id,class
577347,GALAXY
577348,GALAXY
577350,STAR
...

# 実際 (我々の submission.csv)
id,class,1,2
577347,0.997362,0.002593,0.0000435
577348,0.998185,0.001797,0.0000164
577350,0.002821,0.000804,0.996373
...
```

**完全に違うフォーマット**。期待は `(id, class[string])` の 2 列、実際は `(id, class, 1, 2)` の 4 列で
中身は確率。

target の中身も見ておく。train.csv の `class` 列は文字列 `GALAXY / STAR / QSO` の 3 値。
これは Metric-as-SSOT が想定していた「`label` のときは 0/1/2 のような整数」とずれている。

## 3 つの盲点が同時に露呈していた

### 盲点 1: submission の列構造 (id, class) を撒き散らしていた

テンプレ (`templates/tabular_classification/main.py.tmpl`) の該当部:

```python
sub_cols = list(sample.columns)
if n_classes == 2:
    out = pred if SUBMISSION_FORMAT == "proba" else (pred > 0.5).astype(int)
    sub = pd.DataFrame({sub_cols[0]: test_ids, sub_cols[1]: out})
else:
    # ★ 多クラスは「常に確率列を撒く」一択だった
    sub = pd.DataFrame({sub_cols[0]: test_ids})
    for i, c in enumerate(classes):
        col = sub_cols[i + 1] if i + 1 < len(sub_cols) else str(c)
        sub[col] = pred[:, i]
```

二値は `SUBMISSION_FORMAT` で `proba` / `label` を分岐していたが、**多クラスは分岐すら
していなかった**。`SUBMISSION_FORMAT="label"` でも各クラスの確率列を撒いてしまう。

sample_submission が `id, class` の 2 列で、列数が classes (=3) より少ないので、`sub_cols[i+1]`
が範囲外になり、フォールバックで `str(c)` = 文字列化されたクラス index "1", "2" を列名にして
4 列の DataFrame を作っていた。これが Kaggle 側で ERROR 判定される直接原因。

### 盲点 2: CV メトリックが SSOT と逆向きだった (致命的な内在バグ)

CV ループの該当部:

```python
else:  # multi-class
    oof[va] = model.predict_proba(X.iloc[va])
    pred += model.predict_proba(X_test) / n_splits
    s = log_loss(y_idx.iloc[va], oof[va], labels=range(n_classes))   # ★
```

多クラスでは **常に log_loss** で CV を計算していた。だが SSOT は `higher_is_better=True`
(指標 = accuracy 系) と言っている。

```python
# core/knowledge/dataset_analyzer.py
"accuracy": True, "auc": True, "roc-auc": True, "f1": True, ...
```

selector (`submit._pick_best_submission`) は `higher_is_better` でソートする。**CV 値が log_loss
(小さいほど良い)で、selector が「大きいほど良い」と思って動くと、最悪の trial を最良として
選び続ける**。今回は全 5 提出が全部 ERROR なので表面化しなかったが、もし運悪く 1 つでも形式が
合っていたら、最悪の trial が公開された可能性がある。

これは多クラスの実コンペに投入するまで気付けなかった「隠れた致命バグ」。

### 盲点 3: target を LabelEncoder が破壊していた

テンプレ冒頭:

```python
def encode_categoricals(df):
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def main():
    train = encode_categoricals(train)        # ★ ここで target 列も encode してしまう
    test  = encode_categoricals(test)
    ...
    y = train[target_col]                      # y は既に [0,1,2] (int)
    classes = sorted(y.unique())               # classes = [0, 1, 2]
    ...
    # submission のとき inverse_transform したい
    out = np.array([classes[int(i)] for i in idx])
    # → classes[0] = 0 (int) で文字列に戻らない
```

`encode_categoricals` は train 全体に当たるので、target 列 `class` も string → int に変換して
しまう。`classes = sorted(y.unique())` の時点で原クラス名 (GALAXY/STAR/QSO) は失われている。
これも仮に列構造が合っていても、Kaggle 側で「未知のクラス名 0/1/2」として ERROR にされた。

## 直したこと

### Fix 1: submission を SUBMISSION_FORMAT で正しく分岐

```python
if SUBMISSION_FORMAT == "label":
    idx = (pred > 0.5).astype(int) if n_classes == 2 else pred.argmax(axis=1)
    out = np.array([classes[int(i)] for i in idx])   # 原クラス名に inverse
    sub = pd.DataFrame({sub_cols[0]: test_ids, sub_cols[1]: out})
else:  # proba
    if n_classes == 2:
        sub = pd.DataFrame({sub_cols[0]: test_ids, sub_cols[1]: pred})
    else:
        sub = pd.DataFrame({sub_cols[0]: test_ids})
        for i, c in enumerate(classes):
            col = sub_cols[i + 1] if i + 1 < len(sub_cols) else str(c)
            sub[col] = pred[:, i]
```

`full_retrain` パスも同様に分岐させた。

### Fix 2: CV メトリックを SSOT に合わせる

`metric_name` を render プレースホルダ `{{METRIC_NAME}}` で渡し、テンプレ側で:

```python
METRIC_NAME_SSOT = os.environ.get(
    "METRIC_NAME", "{{METRIC_NAME}}"
) or ("accuracy" if SUBMISSION_FORMAT == "label" else "auc")

# fold ループ内 (multi-class)
if METRIC_NAME_SSOT in ("accuracy", "f1"):
    yh = oof[va].argmax(axis=1)
    s = (balanced_accuracy_score(y_idx.iloc[va], yh)
         if METRIC_NAME_SSOT == "accuracy"
         else f1_score(y_idx.iloc[va], yh, average="macro"))
else:
    s = log_loss(y_idx.iloc[va], oof[va], labels=range(n_classes))
```

`balanced_accuracy_score` を選んだのはコンペ指標が **Balanced Accuracy** だから。普通の
`accuracy_score` でも近似はできるが、クラス不均衡 (GALAXY 65% / QSO 20% / STAR 14%) で
差が出る。

### Fix 3: target 列を encode から退避

```python
y_raw = train[target_col].copy()                # 元クラス値 (string 可)
classes = sorted(y_raw.unique().tolist())       # ["GALAXY","QSO","STAR"]
y_idx = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int)

train = train.drop(columns=[target_col])
train = encode_categoricals(train)              # ← target 抜きで encode
test  = encode_categoricals(test)
train[target_col] = y_idx.values                # 整数化済み target を戻す
```

`classes` が `["GALAXY", "QSO", "STAR"]` のまま生き残るので、submission 時に `classes[int(i)]`
で原クラス名に inverse できる。

## 結果

修正したテンプレを直接 render → Linux 上で手動実行:

```
fold 0: score=0.95398
fold 1: score=0.95416
fold 2: score=0.95411
fold 3: score=0.95352
fold 4: score=0.95418
CV = 0.95399
✅ done
```

```
$ head -4 submission.csv
id,class
577347,GALAXY
577348,GALAXY
577350,STAR
```

期待通りの形式。これを提出 → **public LB = 0.95459** で valid score を初めて取得できた。
CV (balanced_accuracy) 0.95399 と LB 0.95459 の gap は **+0.0006** で、CV が極めて信頼できる
状態。

## おまけ: codegen が修正テンプレを書き換えてくる事故

「修正したテンプレで dev mode 再走 → ERROR が出ないか確認」したら、なぜか **iter1 が exit=1
で即死**。stderr:

```
KeyError: "['id'] not found in axis"
File ".../main.py", line 84
X = train.drop(columns=[target_col, id_col])
```

私のテンプレは `train.drop(columns=[target_col])` の 1 行しかなく、`id_col` を引数に渡して
いない。**LLM の codegen が「テンプレに不足がある」と判断して、勝手にコードを書き換えていた**。

時間効率を優先して、**テンプレ直接 render → 手動で Python 実行**という低レイヤパスに
切り替えた。LLM 側の改善は別タスクに分離。

## ナレッジ

- **「指標 = 単一の真実」は段階的に解像度が上がる**。前日に二値分類で SSOT を入れて満足
  していたが、メダル対象に投入したら **多クラス + 文字列ラベル + 評価指標バリエーション**
  の 3 つの盲点が同時に表面化した。SSOT は「全部書ききった」と思ってからが本番。
- **CV メトリックの higher/lower 方向は selector の前提と必ず一致させる**。今回の log_loss
  バグは ERROR で表面化しなかっただけで、もし 1 つでも valid な submission があったら
  最悪 trial が公開されかねなかった。selector の `higher_is_better` を変える側か CV 値の
  方向を変える側か、どちらかに統一する不変式が要る。
- **LLM の出力を信用しすぎない**。テンプレ render と LLM patch を**両方とも信用してはいけない**。
  価値検証フェーズでは「テンプレ直接 render → 手動実行」できる低レイヤパスを残しておくと、
  LLM の不確実性を切り離して CV/LB を測れる。

## ふりかえり

メダル対象初参戦で 1 日まるごと ERROR、というのは順位的にも quota 的にも痛い。でも盲点を
3 つ同時に炙り出せたのは、**実コンペでしか発火しない種類のバグ**だった。一日分の quota を
払う価値はあった、と思いたい。

---

**関連**:
[0/1 が float で入っていただけで、二値分類が回帰として解かれていた](/post.html?slug=2026-06-19-binary-as-regression-metric-ssot) /
**[Kaggle 自律パイプラインで初の中央値超えを実 LB で取った話](/post.html?slug=2026-06-19-first-lb-median-beat)**
