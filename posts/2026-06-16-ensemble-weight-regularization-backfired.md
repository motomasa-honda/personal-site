---
title: "ensemble の重みを「均等寄り」にしたら LB が悪化した — CV を下げ切らない正則化の実証"
emoji: "⚖️"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "regularization", "overfitting"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- ensemble の hill climbing が CV に overfit していた (CV で圧勝 / LB で引き分け) ので、**重みの正則化**を入れた
- 入れた装置は 3 つ: ①同一試行の重複 pick 打ち切り ②微小改善 step で early stop ③最終重みを均等方向へシュリンク
- ③の uniform shrinkage を 0.25 で試したら、**CV はほぼ不変 (0.12459) のまま LB が 0.12807 → 0.12847 と悪化**した
- 原因は「このコンペでは重みが偏るのが正しかった」こと。試行が GBDT 系ばかりで多様性が低く、**best single 方向に重みを寄せるのが最適**だった
- 学び: ensemble の正則化は万能ではない。**均等化が効くのは試行が多様なときだけ**。shrinkage は default 無効にし、多様性が出てから再評価することにした

## 背景: CV で圧勝した ensemble が LB で引き分けた

別記事に書いたとおり、hill climbing ensemble は **CV 0.12458 で single 0.12808 を大きく上回る**のに、**LB は 0.12807 vs 0.12808 でほぼ同値**だった。Caruana の greedy hill climbing が OOF (= CV) を直接最適化するため、CV 分割固有のノイズに overfit していた。

対策として「CV を下げ切るのをやめ、LB への汎化を優先する」正則化を 3 つ入れた。

```python
def run_hill_climbing(..., max_pick_per_trial=3, min_improve=1e-4, shrink_to_uniform=0.0):
    ...
    for step in range(max_iter):
        prev_best = best_score
        for no, d in valid:
            if pick_count.get(no, 0) >= max_pick_per_trial:   # ①重複 pick 打ち切り
                continue
            ...
        improve = (best_score - prev_best) if higher else (prev_best - best_score)
        if step > 0 and improve < min_improve:                # ②early stop
            best_score = prev_best
            break
        ...
    # ③uniform shrinkage: 重みを均等方向へ戻す
    if shrink_to_uniform > 0 and len(weights_norm) > 1:
        uniform = 1.0 / len(weights_norm)
        weights_norm = {
            no: (1 - shrink_to_uniform) * w + shrink_to_uniform * uniform
            for no, w in weights_norm.items()
        }
```

直感的には、CV にフィットして尖った重みを均等方向へ戻せば、未知データ (LB) で汎化するはずだった。

## 実証: 均等化は CV を保つが LB を壊した

同じ OOF で、正則化前後の重みを比べる。

```
正則化なし: CV 0.12458  weights {1:0.333, 121:0.167, 132:0.167, 111:0.167, 112:0.167}
shrink=0.25: CV 0.12459  weights {1:0.363, 121:0.213, 132:0.213, 111:0.213}
```

early stopping が最後の overfit step (改善幅 0.00001) を削り、shrinkage が重みを均等方向に寄せた。CV はほぼ動いていない。狙いどおりに見える。

ところが LB に出してみると、

```
              CV        public LB
正則化なし   0.12458    0.12807
shrink=0.25  0.12459    0.12847   ← 悪化
```

**LB が悪化した**。CV を保ったまま重みを均したのに、未知データでの成績は落ちた。

## なぜ均等化が裏目に出たのか

理由はシンプルで、**このコンペでは重みが偏るのが正しかった**から。

- ブレンド対象の 5 試行は、LightGBM / XGBoost / CatBoost / HistGradientBoosting — **全部 GBDT 系**で、互いによく似ている
- その中で `trial_1` (LightGBM の素直なベースライン) が最も強く、LB でも単体で 0.12808 を出す主軸だった
- 似た者同士のブレンドでは、**弱い試行の比重を上げると主軸の足を引っ張る**。shrinkage はまさにそれをやってしまった

均等化が効くのは「多様な試行がそれぞれ違う方向の誤りを打ち消し合う」ときだ。試行が似ていると、均等化はただ主軸を薄めるだけになる。

## 対応: shrinkage は default 無効、多様性が出てから再評価

LB を悪化させた設定をデフォルトにするわけにはいかないので、`shrink_to_uniform` は **default 0 (無効)** にした。コメントに実測の根拠を残す。

```python
#    ※ shrink は default 0 (無効)。0.25 を試したところ CV 0.12459 維持のまま
#      LB が 0.12807→0.12847 と悪化した。試行多様性が低く best single 方向が
#      支配的なセットでは均等化は逆効果。多様性が確保されてから再評価する。
```

重複 pick 打ち切りと early stopping は、理論的に「CV の overfit step を削る」正しい方向で害が小さいので安全弁として残した。

## 本当の問題は「正則化」ではなく「多様性」だった

この実証で一番大きい収穫は、**CV-LB ギャップの真因が試行多様性の不足だった**と分かったことだ。重みをどう正則化しても、ブレンドする中身が似た GBDT ばかりでは LB は動かない。次にやるべきは正則化のチューニングではなく、**探索に異系統モデル (線形など) を混ぜて多様性そのものを作ること**だった。

## 学び

- ensemble の正則化 (均等化) は万能ではない。**均等化が効くのは試行が多様なときだけ**
- 似た試行のブレンドでは、重みが偏る (主軸に寄る) のが正しい。均等化は主軸を薄めて LB を壊す
- CV を保ったまま重みを動かしても、LB は別の動きをする。**最終判断はつねに LB で**
- ネガティブな結果は「対策の方向が違う」というメタ情報。今回は「正則化より多様性」と教えてくれた
