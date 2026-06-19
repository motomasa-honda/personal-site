---
title: "正規表現が、自分自身が挿入したヘルパ関数を再帰化させていた"
emoji: "♾️"
type: "tech"
topics: ["python", "regex", "testing", "machinelearning", "kaggle"]
published: true
publication_date: "2026-06-19"
---

## TL;DR

自律 Kaggle システムの 6 iter ジョブで、**iter1 は CV 0.949 で成功するのに iter2〜6 が
全て exit=1 (0.8 秒で即死)** という再現性ある奇病。stderr を読むと `RecursionError: maximum
recursion depth exceeded`。原因は **sanitizer の正規表現が、2 周目の sanitize で自分自身が
挿入したヘルパ関数の中身を書き換えていた**ことだった。

## 背景: sanitizer は LLM 生成コードの「事故防止柵」

自律 Kaggle システムは、LLM が出した `main.py` を **そのまま実行する前に sanitize** する。
LLM がよくやらかすパターン (pandas の `inplace=True`、`mean_squared_error(squared=False)`、
`X.iloc[idx]` を ndarray に当てる等) を `re.sub` で書き換える。純粋関数なので Mac で
pytest できる。

その中に「ndarray を `.iloc[idx]` する事故」対策がある:

```python
_ILOC_HELPER = (
    "def _iloc(_o, _i):\n"
    "    return _o.iloc[_i] if hasattr(_o, 'iloc') else _o[_i]\n"
)

def _fix_iloc_on_ndarray(src: str) -> str:
    pat = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\.iloc\[([A-Za-z_]\w*)\]")
    if not pat.search(src): return src
    new = pat.sub(r"_iloc(\1, \2)", src)
    if "def _iloc(" not in new:
        # import 群の直後にヘルパ定義を挿入
        ...
    return new
```

`X.iloc[tr]` を `_iloc(X, tr)` に書き換え、ヘルパ関数を 1 度だけ宣言する。ヘルパは
`.iloc` 属性があれば pandas 経由、無ければ普通の `[]` でアクセスする。

iter1 はこの sanitizer を通って正しく動いた。CV 0.949。問題ない。

## 異常: iter2 から全滅

iter2 の戦略は `catboost+target_encoding+histgb+polynomial_features`。期待のホープ。
が、stderr はこうだった:

```
File ".../main.py", line 20, in _iloc
    return _iloc(_o, _i)
           ^^^^^^^^^^^^^
File ".../main.py", line 20, in _iloc
    return _iloc(_o, _i)
           ^^^^^^^^^^^^^
[Previous line repeated 994 more times]
File ".../main.py", line 19, in _iloc
    if hasattr(_o, 'iloc'):
       ^^^^^^^^^^^^^^^^^^^
RecursionError: maximum recursion depth exceeded
```

`_iloc` ヘルパが自分自身を呼んでいる。中身を見にいくと:

```python
def _iloc(_o, _i):
    if hasattr(_o, 'iloc'):
        return _iloc(_o, _i)      # ★これが置換結果
    return _o[_i]
```

無限再帰だ。本来は `return _o.iloc[_i]` であるべき行が、`_iloc(_o, _i)` に置き換わっている。

## 原因: 自分自身を書き換えていた

`fix_by_judge` 戦略は **前 iter で成功した main.py を起点にパッチを当てる**。つまり sanitizer
を 2 回通る:

1. **1 周目** (iter1 で実行済み)
   - `X.iloc[tr]` を `_iloc(X, tr)` に置換
   - ヘルパ `def _iloc(_o, _i): return _o.iloc[_i] if hasattr(_o, 'iloc') else _o[_i]` を挿入
2. **2 周目** (iter2 の入力 = 1 周目の出力)
   - 正規表現 `([A-Za-z_]\w*)\.iloc\[([A-Za-z_]\w*)\]` が、ヘルパ本体の `_o.iloc[_i]` にもマッチする
   - 置換すると `return _iloc(_o, _i) if hasattr(_o, 'iloc') else _o[_i]` になる
   - ヘルパが自己再帰呼び出しに化ける

つまり、**sanitizer は冪等 (`sanitize(sanitize(x)) == sanitize(x)`) のはずだったのに、
今回の修正だけは冪等性を破っていた**。

## 修正: ヘルパ本体を正規表現にマッチしない形に書き換える

正規表現側を直す手もあるが、`(?<!\w)` のような negative lookbehind を増やしてもケースを
取りこぼす可能性がある。確実なのは **ヘルパ本体を「`name.iloc[name]` のパターンに見えない」
書き方にして、最初からマッチしないこと**。

```python
_ILOC_HELPER = (
    "def _iloc(_o, _i):\n"
    "    return getattr(_o, 'iloc')[_i] if hasattr(_o, 'iloc') else _o[_i]\n"
)
```

`getattr(_o, 'iloc')[_i]` は意味的に `_o.iloc[_i]` と等価だが、`name.iloc[name]` のパターンには
**マッチしない**。`getattr` は呼出式なので正規表現の `[A-Za-z_]\w*` に該当しない。これで
何回 sanitize しても変質しない。

## 既存テストを見たら、もっと早く気付けた

`tests/test_sanitizer.py` には既に冪等性パラメトリックテストがあった:

```python
MESSY_SAMPLES = [
    "df.fillna(0, inplace=True)\n...",
    "from sklearn.metrics import cross_val_score, ...",
    ...
]

@pytest.mark.parametrize("src", MESSY_SAMPLES)
def test_idempotent(src, is_reg):
    once = sanitize_code(src)
    twice = sanitize_code(once)
    assert once == twice
```

ちゃんと **`sanitize(sanitize(x)) == sanitize(x)` をパラメトリックで担保している**設計。

ところが、`_fix_iloc_on_ndarray` を導入したときに **`.iloc[idx]` を含むサンプル**を
MESSY_SAMPLES に追加していなかった。結果、ヘルパ自身が次の sanitize で壊れる事象を
テストが見抜けなかった。CI は緑のまま、本番で発火。

「冪等性のメタテストはあるが、サンプルが網羅されていないと意味が無い」――よく言われる話
だけど、自分でも踏んだ。

## 回帰テスト

修正と合わせて、冪等性サンプルに `.iloc` パターンを追加し、さらに **「ヘルパが自己再帰しない」
専用テスト**を入れた。`exec` してヘルパ自体を呼び、再帰スタックに落ちないことを直接検証する:

```python
def test_iloc_helper_not_self_recursive():
    src = "for tr, va in cv.split(X, y):\n    model.fit(X.iloc[tr], y.iloc[tr])\n"
    once = sanitize_code(src)
    twice = sanitize_code(once)
    assert "return _iloc(_o, _i)" not in twice
    assert "def _iloc(_o, _i):" in twice
    # 実際に exec して再帰しないか
    import re
    m = re.search(r"def _iloc\(_o, _i\):\n.*\n", twice)
    ns = {}; exec(m.group(0), ns)
    class _Iloc:
        def __init__(self, d): self.d = d
        def __getitem__(self, i): return self.d[i]
    class _DF:
        def __init__(self, d): self.iloc = _Iloc(d)
    assert ns["_iloc"]([10, 20, 30], 1) == 20     # iloc 無し → []
    assert ns["_iloc"](_DF([5, 6, 7]), 2) == 7    # iloc 有り → .iloc
```

`numpy` も `pandas` も import せず、最小スタブで挙動を確認する。test_sanitizer の方針
(純粋関数 + 重依存ゼロ) に沿わせた。

## ナレッジ

- **コード生成器のヘルパ関数は、生成後の文字列パターンを意識して書く**。`re.sub` の対象
  パターンと同じ表記を中身に書くと、再帰的に壊れる。
- **冪等性テストの存在 ≠ 冪等性の保証**。サンプル群が網羅していて初めて意味を持つ。
  「`re.sub` で書き換えるパターンを新規追加したら、そのパターン入りサンプルを冪等性テストに
  追加する」をルーチンにする。
- **「複数回適用されうるパス」を見落とさない**。今回 sanitizer は「1 回しか通らないつもり」で
  書かれていたが、`fix_by_judge` のような「前段の出力を再入力にする」フローでは 2 回通る。
  パイプライン全体を見て **多重適用される可能性のある関数は冪等性を契約に書く**。

---

iter1 の CV 0.949 が、iter2 以降の RecursionError で「次の戦略が試せない」のは痛かった。
特に iter2 の `catboost+target_encoding+histgb+polynomial_features` は CV 0.94 帯から
押し上げる本命だった。

修正後、s6e6 (別コンペ) ではこのバグは出ず、CatBoost 戦略が走る土台はできた。

---

**関連**:
[『パイプラインが固まった』の真犯人は深夜の知識収集タスクだった](/post.html?slug=2026-06-19-harvest-runaway-blocked-the-pipeline) /
[0/1 が float で入っていただけで、二値分類が回帰として解かれていた](/post.html?slug=2026-06-19-binary-as-regression-metric-ssot)
