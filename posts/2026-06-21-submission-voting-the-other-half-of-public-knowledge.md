---
title: "公開 OOF だけじゃない — score 付きハードラベル submission の山を加重多数決した"
emoji: "🗳️"
type: "tech"
topics: ["kaggle", "ensemble", "majorityvoting", "machinelearning", "metaensemble"]
published: true
publication_date: "2026-06-21"
---

## TL;DR

公開 OOF を活用する記事の続編。コンペによっては、OOF (確率) ではなく**ハードラベル submission
CSV だけを集めた Public Dataset** が存在する。ファイル名が **そのまま LB スコア**になっている
(`0.97220.csv` 形式) のがよくあるパターン。

これを活用するには **OOF stacker は使えない** (train 側の予測が無い)。代わりに **加重多数決**
(weighted majority voting) で test 確率を構成し、自分の stack 出力と blend する。
KRS-Core の Playbook に `submission_voting.py` として実装。検証コンペで LB
**0.97042 → 0.97197** (+0.00155) で **Silver 圏 (top 5%) 突入**。

## OOF と submission の違い、ここが効く

| 形式 | train 側 | test 側 | 使い道 |
|---|---|---|---|
| OOF + test predictions | あり (確率) | あり (確率) | stacker の **train 入力** になれる |
| submission CSV のみ | **無し** | あり (ハードラベル) | stacker に入れられない → **vote 用** |

Public Kernel の submission は一発提出のみ公開するケースが多く、OOF まで dump されない
ことが普通。だから「**最終 submission を集めただけの dataset**」が大量に流通する。
これは捨てるには勿体無いほどの情報密度がある (LB の真実値を保証された submission 群)。

## なぜ加重多数決か (素朴な等価重みではダメな理由)

20 voter のうち 2 voter が LB 0.97220、18 voter が LB 0.96400 だったとする。等価重みで
多数決すると 18 voter の判断が支配する → 結果は ~0.965 に収束する。

しかし**強 voter の判断**は LB 統計上**当たりやすい**ことが事前に分かっている。
重みを `weight ∝ (score - score_floor)` にすると:
- score=0.972 → weight = (0.972 - 0.96) × 1000 = 12
- score=0.964 → weight = (0.964 - 0.96) × 1000 = 4

3 倍重い。これだけで強 voter の判断が argmax に強く反映される。

### `score_floor` を 0.96 に置く意図

s6e6 で言えば、 LB **0.96 未満は中央値以下** (= ランダムよりは良いが、stack に混ぜるには
ノイズ寄り)。`score_floor = 0.96` で「中央値超え」を起点に weight を線形に振る。これによって
**「弱 voter は重みほぼ 0」**になる。

完全に切り捨てる (top_k で frequency 制限) より、**連続的に減衰**させる方が境界の人為性が
低い。

## 設計 — voter を集めて確率行列を吐く

```python
from pathlib import Path
import numpy as np, pandas as pd
import re

SCORE_FILE_PATTERN = re.compile(r"^(0\.\d{4,6})(?:\.[a-z])?\.csv$")

def discover_score_named_submissions(directory, score_min=0.0, score_max=1.0):
    """ <0.97220.csv>, <0.97223.b.csv> 等を score 付きで列挙 """
    out = []
    for p in directory.rglob("*.csv"):
        m = SCORE_FILE_PATTERN.match(p.name)
        if not m: continue
        s = float(m.group(1))
        if score_min <= s <= score_max:
            out.append((p, s))
    return out

def vote_submissions(records, n_test, n_classes,
                     score_floor=0.96, weight_scale=1000.0, top_k=None):
    """ 加重多数決 → (M, C) 確率行列 """
    if top_k: records = sorted(records, key=lambda r: -r.score)[:top_k]
    weights = np.array([max(1e-3, (r.score - score_floor) * weight_scale) for r in records])
    weights /= weights.sum()
    proba = np.zeros((n_test, n_classes))
    for w, r in zip(weights, records):
        proba[np.arange(n_test), r.labels_arr] += w  # one-hot accumulate
    return proba.astype(np.float32)
```

`labels_arr` は voter ごとの (M,) int 配列 (各 row のハードラベルを int index 化)。

## blend with own — 重要なのはここ

加重多数決だけで submit すると、それは「voter 群の最大公約数」になる。**自分の予測との
blend** を必ず噛ます。これは ToS 的にも「自分の予測」になる線引きの意味でも重要。

```python
def blend_with_own(own_test, vote_test, alpha=0.5):
    """ own_test * alpha + vote_test * (1-alpha) """
    out = alpha * own_test + (1 - alpha) * vote_test
    return out / out.sum(axis=1, keepdims=True)
```

alpha は **OOF が無いのでオフライン探索不可**。0.3 〜 0.5 の範囲で submit して LB から探る。
自分のモデルが voter より弱ければ alpha 小さめ (voter 寄り)、強ければ alpha 大きめ。

s6e6 では我々の stack 出力 LB 0.97042、voter 群の平均 LB が 0.971+ (top voter は 0.972+)。
voter 群がやや上 → alpha=0.4 (自分 40% / voter 60%) でスタート。

## 実装スケッチ — CLI 経由で

`scripts/external_oof_blend.py` に `--vote-dir` オプションを足した:

```bash
python scripts/external_oof_blend.py playground-series-s6e6 \
  --workspace ... \
  --vote-dir /tmp/krs_oof_cache/.../author_b_s6e6-submission/new \
  --vote-score-min 0.97 \
  --vote-top-k 25 \
  --vote-blend-alpha 0.4 \
  --out submission_voted.csv
```

内部フロー:
1. 既存の logit stacker で `own_test` (確率) を生成
2. `discover_score_named_submissions` で voter ファイルを列挙
3. `vote_submissions` で `(M, C)` 確率行列を構成
4. `blend_with_own(own_test, vote_test, alpha=0.4)` で blend
5. argmax → submission CSV

## 結果

| 構成 | LB |
|---|---|
| 内部 + 外部 OOF (前記事) | 0.97042 |
| **上記 + 25 voter (LB 0.97+) 加重多数決 blend (alpha=0.4)** | **0.97197** |

+0.00155 = top 17% → top 5% (Silver 圏到達)。1 quota で取れる lift としては破格。

## ToS との関係 — 「混ぜる」が線引き

念のため整理:
- **NG**: `0.97226.csv` をそのまま re-submit (これは submission farming に該当)
- **OK**: 25 件の voter を加重多数決 + 自分の stack と blend → 自分の予測として submit
- **OK**: voter 数も `score_floor` も `alpha` も自分の判断 → 「自分のモデル」と言える

Kaggle の rule で **集約と直接コピーは厳密に区別**される。我々の実装は前者なので問題なし。
(`vote_submissions` 単独では submit 不可、必ず `blend_with_own` を噛ます API 構造にして
**直接 voter を吐けないようにした**。)

## 知見

- **ハードラベルでも score 付きならメタアンサンブルできる**。OOF が無いから諦めるな。
- **score 重みは `(score - score_floor)` 線形**で十分。指数や対数より素直、tuning も少ない。
- **alpha は OOF が無い分 LB で探る必要があるので quota 食う**。最初は 0.4 で blind、次回以降
  別 alpha を試す。
- **自分の予測と blend を必須化**しておけば ToS の線引きが明確。**API 設計レベルで強制**する。

明日は **`tools/active_alpha_search.py`** を書く予定。LB feedback (CV-LB tracker) を見て
alpha を Bayesian 的に動的更新するモジュール。これで複数回 submit する場合の効率が上がる。

## まとめ

OOF (確率) と submission (ハードラベル) は別物なので、別の集約手段が要る。前者は stacker、
後者は加重多数決。両方を自前 stack と blend すれば、**「自分のモデルの限界 + 公開知の限界」**
の和を超える可能性がある。今回は 1 日で **top 49% → top 5%** までこのアプローチで圧縮した。
