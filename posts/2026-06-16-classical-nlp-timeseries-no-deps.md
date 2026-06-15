---
title: "torch が無くてもやれることはある — NLP は char+word TF-IDF、時系列はカレンダー特徴で弱モダリティを底上げする"
emoji: "📚"
type: "tech"
topics: ["kaggle", "nlp", "timeseries", "scikit-learn", "machinelearning"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- Kaggle Agent (KRS-Core) の弱点は NLP と時系列のテンプレートが貧弱なこと (NLP は TF-IDF + LogReg だけ、時系列は日付を unix 秒に潰すだけ)
- 深層学習で殴る前に、**依存を一切増やさず sklearn だけで底上げ**できる余地が大きい
- NLP: word n-gram に **char n-gram (char_wb 3〜5) を hstack** して LogisticRegression。タイポ・多言語・略語に強くなる定番
- 時系列: 日付を潰す代わりに **カレンダー特徴 (曜日・月・週・四半期・週末フラグ…) を展開**。リーク安全・決定的
- ついでに、元の NLP テンプレに `StratifiedKFold` の **import が抜けていた**バグも見つかって直した (= 一度も検証されていなかった証拠)

## 前提: その環境に重い依存は入っているか

「画像/NLP が弱い」と分かったとき、まず実行環境を調べたら torch も transformers も入っていなかった。深層学習を入れるのは別途やるとして (それはそれで GPU・ROCm の話になる)、**今この瞬間に、依存を増やさず・GPU 無しで上げられる分**を先に取りに行く。

sklearn と scipy はある。NLP も時系列 (tabular 寄り) も、ここで十分戦える。

## NLP: char n-gram を足すだけで効く

元のテンプレートは word の TF-IDF (1,2-gram) に LogisticRegression をかけるだけだった。ここに **char_wb の n-gram (3〜5)** を足して結合する。

```python
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

word_vec = TfidfVectorizer(analyzer="word",    ngram_range=(1, 2),
                           min_df=2, max_features=50000, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=2, max_features=50000, sublinear_tf=True)

word_vec.fit(all_text); char_vec.fit(all_text)
X = hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()
```

char n-gram は単語境界をまたいだ部分文字列を見るので、**タイポ・表記ゆれ・多言語・略語**に強い。「word だけ」より素直に上がるのは、テキスト系コンペでは古くからの定石。`sublinear_tf=True` で頻度を log スケールにするのも効く。

複数のテキスト列があるコンペもあるので、id/target 以外の非数値列を全部連結してから vectorize するようにした。

## 落とし穴 1: 元テンプレに `StratifiedKFold` の import が無かった

書き直していて気づいた。元の NLP テンプレートは本文で `cv = StratifiedKFold(...)` を使うのに、**その import がどこにも無かった**。つまり走らせれば `NameError` で即死する。これは「このテンプレートが一度も実機で検証されていなかった」ことの動かぬ証拠だった。

合成データでスモークを書いて回す習慣があると、こういう「存在するだけで壊れているテンプレート」をちゃんと検出できる。

## 落とし穴 2: sklearn 1.8 で liblinear が多クラス非対応に

二値だけでなく多クラスも 1 本のテンプレで扱いたい。`LogisticRegression(solver="liblinear")` は sparse に強くて速いが、**sklearn 1.8 では liblinear が多クラスを直接サポートしなくなった** (以前は暗黙に One-vs-Rest していた)。今は明示的に包む必要がある。

```python
from sklearn.multiclass import OneVsRestClassifier

base = LogisticRegression(C=4.0, max_iter=1000, solver="liblinear")
model = base if n_classes == 2 else OneVsRestClassifier(base)
```

OOF は二値なら 1 次元の確率、多クラスなら `(N, n_classes)` の 2 次元で保存する。これは後段のアンサンブル (hill climbing / stacking) が読む契約に合わせている。

## 時系列: 日付を潰さず展開する

元のテンプレートは日付列を `to_datetime().astype("int64") // 10**9` で **unix 秒に潰すだけ**だった。これだと「季節性 (曜日・月・週)」の情報を全部捨ててしまう。木モデルは unix 秒の連続軸からは周期を学べない。

なので日付列からカレンダー特徴を展開する。

```python
for c in date_cols:
    dt = pd.to_datetime(train[c], errors="coerce")
    if dt.notna().mean() < 0.5:        # 日付として解釈できない列はスキップ
        continue
    train[f"{c}__year"]    = dt.dt.year
    train[f"{c}__month"]   = dt.dt.month
    train[f"{c}__dow"]     = dt.dt.dayofweek
    train[f"{c}__week"]    = dt.dt.isocalendar().week.astype("int64")
    train[f"{c}__quarter"] = dt.dt.quarter
    train[f"{c}__is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    # ... is_month_start / is_month_end / dayofyear も
```

これらは **target を一切見ない・決定的**なのでリークしない。train/test に同じ処理を当てる。

### lag/rolling を「テンプレに固定しない」判断

時系列といえば target の lag / rolling 特徴だが、これは**あえてテンプレートに固定しなかった**。理由は、lag は系列キー (store × item など) に依存し、train/test の境界をまたぐ扱いを間違えると簡単にリークするから。Store Sales のようなコンペでは「店舗・商品ごとに過去の売上を test 側へ正しく merge する」必要があって、これは汎用テンプレートで安全に書けるものではない。

なので lag/rolling は、系列キーを知っている codegen (LLM) 側に委ねるとコメントで明示した。**汎用テンプレに無理に詰め込んでリークの温床を作るより、安全な特徴だけを固定し、危険な特徴は文脈を知る層に任せる**。

## 検証: 合成データで end-to-end スモーク

NLP も時系列も、手元に該当コンペのデータが無い。だから**合成データを作ってスモーク**した。NLP はポジ/ネガ語彙から二値・多クラスのテキストを生成、時系列はサイン波 + ノイズの売上を日付つきで生成。

3 本とも `rc=0` で、CV が出て、OOF と提出ファイルがアンサンブル契約どおりに生成されることを確認した。スコアの良し悪しより、まず **「壊れず最後まで走って正しい成果物を吐く」**ことを担保するのがテンプレートの責務だ。

## まとめ

- 深層学習を入れる前に、**依存ゼロ・GPU 無しで上げられる分**を先に取る
- NLP は word + char の TF-IDF を hstack するだけで素直に効く。sklearn 1.8 の liblinear 多クラスは OvR で明示的に包む
- 時系列は日付を潰さずカレンダー特徴に展開する。リーク安全な特徴だけ固定し、lag/rolling は系列キーを知る層に委ねる
- 「import 抜けで即死するテンプレート」は、合成データのスモークを習慣にすると確実に捕まる
