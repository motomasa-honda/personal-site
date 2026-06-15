---
title: "LLM が書くコードを『回帰テスト』にする — Kaggle 提出枠を消費せずに codegen 品質を固定する"
emoji: "🧪"
type: "tech"
topics: ["llm", "testing", "kaggle", "pytest", "machinelearning"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- LLM が生成するコードの品質は、放っておくと**静かに劣化する** (モデル交換・プロンプト変更・サニタイザ修正のたびに揺れる)
- Kaggle Agent (KRS-Core) では、生成コードを決定的に直す**サニタイザを pytest で固定**し、生成品質を**提出枠を消費しないベンチ**で測るようにした
- サニタイザのテストは「①地雷が直る ②`ast.parse` が通る (= 構文を壊していない) ③冪等 ④正しいコードを誤爆で壊さない」の 4 点を全ルールで検証
- 純粋関数なので **LLM も GPU も Kaggle 提出枠も要らず、開発機で 0.02 秒**で 36 ケースが回る
- このベンチが副産物として、自己修復でも直せない CatBoost のバグを炙り出した

## なぜ codegen に回帰テストが要るか

このシステムは LLM (ローカルの `qwen3-coder:30b`) が探索コードを書き、それを決定的なサニタイザが直してから実行する。サニタイザは過去に踏んだ地雷を潰す層で、たとえばこういう書き換えをやる。

- 回帰タスクに `StratifiedKFold` → `KFold` に矯正
- `mean_squared_error(..., squared=False)` → `root_mean_squared_error(...)` (sklearn 1.6+ で引数廃止)
- XGBoost + category 列に `enable_categorical=True` を補う
- LightGBM 4.x で廃止された `early_stopping_rounds=...` を除去
- CatBoost のカテゴリ列 NaN を読み込み直後に補完

問題は、**この種の書き換えはモデルやプロンプトをいじるたびに壊れやすい**こと。正規表現の置換は、入力が少し変わると誤爆したり、ネストした括弧を壊したりする。だから回帰テストで固定する。

## サニタイザは純粋関数 = テストの理想形

サニタイザは「コード文字列 → コード文字列」の純粋関数で、依存は `re` だけ。これは**テストにとって理想**だ。LLM も GPU も外部 I/O も要らない。各ルールについて 4 点を検証する。

```python
def test_squared_false_preserves_nested_parens():
    out = sanitize_code(
        "from sklearn.metrics import mean_squared_error\n"
        "s = mean_squared_error(np.expm1(y), np.expm1(p), squared=False)\n"
    )
    assert "root_mean_squared_error(np.expm1(y), np.expm1(p))" in out  # ① 直る
    assert "squared=False" not in out
    ast.parse(out)                                                     # ② 壊していない
```

特に壊れやすいのは「引数にネストした括弧があるケース」。`mean_squared_error(np.expm1(y), np.expm1(p), squared=False)` を正規表現で雑に処理すると括弧の対応を壊す。だから括弧の深さを数えて対応する閉じ括弧を見つける実装にして、それをテストで固定する。

### 冪等性と誤爆防止が効く

地味だが重要なのが**冪等性** (`sanitize(sanitize(x)) == sanitize(x)`) と**誤爆防止** (正しいコードは変えない)。

```python
@pytest.mark.parametrize("src", MESSY_SAMPLES)
def test_idempotent(src):
    once = sanitize_code(src)
    assert sanitize_code(once) == once   # 二重適用で壊れない

@pytest.mark.parametrize("src", CLEAN_SAMPLES)
def test_clean_code_unchanged(src):
    assert sanitize_code(src) == src     # 正しいコードは素通し
```

サニタイザは生成のたびに掛かるので、二重適用で `fillna("missing").fillna("missing")` のように増殖したら困る。各ルールに「直前で既に処理済みなら触らない」ガードを入れて、それを冪等テストで担保する。

## 開発機でフルに走らせる工夫

このシステムは「開発機」と「GPU 実行機」が分かれていて、重い依存は実行機にしかない。だがサニタイザのテストは**開発機でも回したい** (速いから)。

ところがサニタイザを普通に import すると、パッケージの `__init__` が重い依存 (設定ローダなど) を芋づるで引いてきて、依存の無い開発機では失敗する。そこで**テストはサニタイザのファイルを直接ロード**する。

```python
spec = importlib.util.spec_from_file_location("_san", SANITIZER_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sanitize_code = mod.sanitize_code
```

これでパッケージ全体の import グラフを回避し、`re` だけに依存する純粋関数を**どの環境でも・提出枠を消費せず・0.02 秒**でテストできる。36 ケースが一瞬で回る。

## ベンチ: 「今のモデルが今どんなコードを書くか」を測る

テストはサニタイザを固定する。だが LLM 自体の生成品質はテストでは測れない (出力が確率的だから)。そこで別に**ベンチ**を用意した。現行モデルで diversify を数回呼んで patch を生成し、サニタイズして**ローカル実行**し、成功率と CV を報告する。Kaggle には一切提出しない。

```
=== codegen ベンチ (coder=qwen3-coder:30b, k=3) ===
✅ patch_0 cv=0.12746
🔴 patch_1  | CatBoostError: cat_features must be ... =NaN
✅ patch_2 cv=0.12602
成功 2/3
```

このベンチが、自己修復ループでも直せなかった **CatBoost のカテゴリ列 NaN バグ**を炙り出した。提出枠 (1 日 10 回) を 1 つも使わずに、再現性のある失敗モードを特定できる。テストが「退行していないこと」を守り、ベンチが「今どこが弱いか」を教える。

## まとめ

- LLM が書くコードの品質は放置すると劣化する。**決定的に直すサニタイザを pytest で固定**して退行を止める
- サニタイザは純粋関数なので、「直る / 構文を壊さない / 冪等 / 誤爆しない」の 4 点を全ルールで検証する
- ファイル直読みで import グラフを回避し、依存の無い開発機でも提出枠ゼロ・0.02 秒で回す
- テストで退行を守り、**提出枠を使わないベンチ**で「今どこが弱いか」を測る。両輪で回す
