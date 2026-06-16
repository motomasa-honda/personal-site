---
title: "CV で圧勝した ensemble が、LB では single と引き分けた — hill climbing が最適化していたのは『CV』だった"
emoji: "📉"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "crossvalidation", "overfitting"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- Kaggle Agent (KRS-Core) の hill climbing ensemble を、列名バグを直してようやく LB で採点させた
- ローカル CV では ensemble 0.12458 が single 0.12808 を**大きく**上回っていた。LB でも勝てると思っていた
- 実際の LB は **ensemble 0.12807 vs single 0.12808** — つまり**ほぼ引き分け**
- 機構 (重み分散・submit 選択・提出フォーマット) は完璧。それでも LB が動かないのは、**hill climbing が最適化しているのが LB ではなく CV だから**
- 「CV が上がった = LB が上がる」ではない。ensemble の効果はこの **CV-LB ギャップ**を込みで評価しないと幻になる

## 期待していた絵

このシステムは複数試行の OOF (out-of-fold) 予測を Caruana の greedy hill climbing でブレンドする。CV ベースで見ると、ブレンドは効いていた。

```
single best CV : 0.12808
ensemble    CV : 0.12458   (▲ 0.0035 改善)
weights = {trial_1: 0.33, trial_121: 0.17, trial_132: 0.17, trial_111: 0.17, trial_112: 0.17}
```

重みは 1 本に偏らず 5 試行に分散している。教科書どおりの「多様な学習器をブレンドして汎化を稼ぐ」絵だ。CV で 0.0035 も改善しているのだから、LB でも相応に勝てる — そう読んでいた。

## 現実: LB ではほぼ動かない

列名バグ (別記事に詳細) を直して、ようやく ensemble が Kaggle で採点された。並べるとこうなる。

```
              CV        public LB
single     0.12808      0.12808
ensemble   0.12458      0.12807
```

CV では 0.0035 もあった差が、LB では **0.00001**。誤差みたいなものだ。LB best はかろうじて更新したが、「ensemble が single を明確に超える」という期待した絵にはならなかった。

## なぜか: hill climbing は CV を直接最適化している

Caruana の greedy hill climbing は、各ステップで「どの試行を 1 票足すと**スコアが最も良くなるか**」を貪欲に選ぶ。ここでいうスコアは **OOF 予測に対する CV**だ。

```python
for step in range(max_iter):
    for no, d in valid:
        cand = blended + d["oof"]                  # この試行を 1 票足したら
        cand_norm = cand / (sum(weights.values()) + 1.0)
        s = _score(base_label, cand_norm, metric)  # ← CV (OOF) を直接評価
        # s が最良なら、この試行を採用
```

つまり重みは **CV を最大化するように選ばれている**。CV が下がるのは当たり前で、それはアルゴリズムが直接ねじ込んでいる目的関数そのものだからだ。問題は、**CV を下げる重みが、必ずしも未知データ (LB) を下げる重みではない**こと。

- OOF は学習データの分割に過ぎず、**fold をまたいだ情報の薄い漏れ**や、CV 分割固有のノイズを含む
- 貪欲に CV を最小化すると、その**CV 分割に固有のノイズにフィット**しやすい (同じ試行を複数回 pick して重みを盛るのは、その兆候)
- House Prices のように train が 1460 行と小さいコンペでは、CV の 0.003 は**統計的にほぼ誤差**で、LB に転写されない

要するに、hill climbing は「CV という名の検証セット」に対して軽く overfit していた。CV-LB ギャップは ensemble が作り出した幻だったわけだ。

## これは失敗ではなく「正しく測れた」こと

ネガティブな結果に見えるが、むしろ前進だと考えている。

- 機構は完全に動いた。重みは分散し、submit は ensemble を選び、提出フォーマットも正しくなった
- その上で「**ensemble の CV 改善は LB に転写されない**」という、コンペで最も重要な事実を**実スコアで観測できた**
- もし列名バグを放置したまま CV だけ見ていたら、「ensemble は効いている」と誤認し続けていた

Kaggle で勝つというのは、結局 **CV-LB ギャップをどう設計するか**の戦いだ。CV が下がって喜ぶのではなく、「その CV 改善は LB に乗るのか」を毎回疑う必要がある。

## 次にやること

CV に overfit しない ensemble にするための仮説をいくつか持っている。

- **重み付けに正則化を入れる** — 同一試行の重複 pick を抑える / 重みを均等寄りにする
- **試行の多様性を上げる** — 今回の 5 試行は似たモデル (GBDT 系) が多い。アルゴリズム/特徴量の系統を散らすと、ブレンドの汎化が効きやすい
- **ブレンド重みの決定に held-out をもう 1 段挟む** — OOF を直接最適化せず、さらに内側の検証で重みを選ぶ (nested)
- **CV と LB の相関そのものを記録する** — 1 コンペ 1 点ずつ、CV 改善が LB 改善に転写された比率を貯めて、信頼できる CV 設計を選ぶ

## 学び

- **CV が下がっても LB が下がるとは限らない**。特に貪欲最適化は、最適化対象 (CV) に overfit する
- ensemble の価値は「CV がどれだけ下がったか」ではなく「**LB がどれだけ下がったか**」でしか測れない
- 小さいコンペでの CV 0.003 は誤差の範囲。**差の大きさを統計的に割り引く**癖をつける
- ネガティブな実スコアは敗北ではない。**幻を幻と確認できた**のは、機構が正しく動いた証拠でもある
