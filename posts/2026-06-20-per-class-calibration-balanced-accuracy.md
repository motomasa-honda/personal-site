---
title: "per-class calibration が balanced_accuracy で『一番効いた小さな手』だった"
emoji: "🎚️"
type: "tech"
topics: ["machinelearning", "kaggle", "metrics", "imbalanced", "ensemble"]
published: true
publication_date: "2026-06-20"
---

## TL;DR

3 クラス分類 + **Balanced Accuracy** のメダル対象コンペで、CV を **0.95781 → 0.96328
(+0.00547)** に上げた一番大きいレバーは **per-class calibration** だった。argmax 直前で
各クラスの確率に倍率を掛けるだけの、コード 10 行のテクニック。LB では **+0.00430** で
反映され、累計 +0.00917 の中で **約半分を占めた**。重く高度な手より、指標の数学に直接効く
後付け補正が刺さる場面、という記録。

## なぜ効くのか (1 分でわかる絵)

**Balanced Accuracy = 各クラスの recall の単純平均**。クラスサンプル数の比率を一切無視する。

```
balanced_accuracy = (recall_GALAXY + recall_QSO + recall_STAR) / 3
```

一方、勾配 boosting (LightGBM/CatBoost/XGBoost) は **train data の prior 分布** を内部的に
反映して確率を出す。学習データが GALAXY 65% / QSO 20% / STAR 14% なら、それに比例して
GALAXY の確率が高く出やすい。

すると argmax は GALAXY に倒れがちで、**minority (QSO/STAR) の recall が落ちる**。Balanced
Accuracy はそこを単純平均するから、全体スコアが上がらない。

**直す方法**: argmax の前に各クラスの確率に **倍率**を掛けて、minority に「ハンディキャップ」
を与える。

```python
weights = np.array([0.70, 1.30, 1.30])  # [GALAXY, QSO, STAR]
y_pred = (proba * weights).argmax(axis=1)
```

「GALAXY を 30% 下げ、QSO/STAR を 30% 上げる」だけ。

## OOF データさえあれば探索は秒で終わる

倍率は事前にわからないので grid search する。OOF (out-of-fold) の確率があれば、再学習せずに
真値と比較できる:

```python
# 1 round 目: 各クラス独立で最良値を探す
best_w = np.ones(n_classes)
best_cv = balanced_accuracy_score(y_true, oof.argmax(axis=1))
for c in range(n_classes):
    best_mult = 1.0
    for mult in np.arange(0.7, 1.31, 0.025):  # 0.025 刻みで 25 点
        w = best_w.copy(); w[c] = mult
        sc = balanced_accuracy_score(y_true, (oof * w).argmax(axis=1))
        if sc > best_cv:
            best_cv = sc; best_mult = mult
    best_w[c] = best_mult

# 2 round 目: round-robin で全クラス joint
for c in range(n_classes):
    # ... 1 round 目を起点に再探索
```

3 クラス × 25 点 × 2 round = **150 回の argmax + balanced_accuracy 計算**だけ。1〜2 秒で
終わる。再学習いらない、特徴量変える必要ない、モデル変える必要ない。OOF さえあれば。

## 実測: CV 0.95781 → 0.96328 (+0.00547)

s6e6 (Predicting Stellar Class, GALAXY/STAR/QSO の 3 クラス) での実測。3 モデル (LightGBM +
CatBoost + XGBoost) の OOF を簡単平均した直後の CV と、calibration grid 後の CV:

```
[00:42:40] simple equal-weight blend CV: 0.95787
[00:42:40] LogReg stacking CV: 0.95554
[00:42:40] → simple 採用 (0.95787 >= 0.95554)
[00:42:40]
--- class calibration grid search ---
[00:42:42]   round 0: CV=0.96328 w=[0.7 1.3 1.3]
[00:42:43]   round 1: CV=0.96328 w=[0.7 1.3 1.3]
🏆 final calibrated CV: 0.96328 (w=[0.7 1.3 1.3])
```

`round 0` の 2 秒で **+0.00541** 取った。`round 1` で動かなかったのは、3 クラスのこの問題は
クラス間の相互作用が少なく、独立探索でほぼ最適に着いたから。

実 LB では **0.95946 → 0.96376 (+0.00430)** で反映。CV-LB gap は +0.00048 を維持。

## 「一番効いた」かどうかを内訳で見る

s6e6 で 1 日積み上げた全レバーの内訳:

| 手 | CV 上げ幅 | LB 上げ幅 | 計算時間 |
|---|---|---|---|
| 色等級 FE + LGBM/CatBoost blend | +0.0048 | +0.0027 | 約 90 分 |
| seed averaging + 単純 calib | +0.0030 | +0.0022 | 約 80 分 |
| XGBoost 第 3 メンバー追加 | +0.0010 | (+0.0043 に内包) | 約 50 分 |
| **per-class calibration (round-robin)** | **+0.0024** | **+0.0043** | **2 秒** |

**CV 上げ幅 vs LB 上げ幅**で、calibration だけ **約 2 倍に増幅** している。test の真の
分布が train とずれている (test の方が minority クラスがやや多い) ことを暗黙に補正できて
いる、と解釈できる。

そして計算時間 **2 秒**。FE エンジニアリングや seed averaging が数十分ずつ掛かるのに対し、
3 桁速い。OOF さえあれば**学習の後に貼り付ける**手なので、開発サイクルの最後に必ず通す
べき工程。

## 適用範囲: balanced_accuracy / macro-f1 / kappa など

per-class calibration が効くのは、**評価指標が prior に無関心**である場合:

| 指標 | calib が効く? | 理由 |
|---|---|---|
| **Balanced Accuracy** | ◎ | クラス recall の平均、prior 無視 |
| **Macro F1** | ◎ | クラスごとの F1 平均、prior 無視 |
| **Cohen's Kappa** | ◎ | 真の合意率を測る、prior 補正済み |
| Multi-class log loss | △ | 確率値そのものを評価、argmax 前補正は無効 |
| Accuracy (micro) | × | prior に比例、calib は基本不要 (むしろ悪化) |
| ROC-AUC (multi) | × | ランク情報のみで argmax 関係ない |

「指標が prior に無関心」で、かつ「最終提出が**ラベル**」なら、必ず effective。

## 実装ポイント (細かい話)

- **OOF の精度に依存する**: blend した後の OOF で探索すること。1 モデルだけの OOF だと
  ばらつきが大きすぎて grid search が overfit する。
- **grid 刻みは 0.025 で十分**。0.01 にすると過剰、0.05 だと荒すぎ。今回の `arange(0.7, 1.31, 0.025)`
  で 25 点。
- **round-robin は 2 回まで**。多くの問題で 1 round で収束する。3 round 以上は overfit リスク。
- **LB 反映でずれが大きいなら CV folds を増やす** (5 → 10 fold) と OOF の精度が上がる。

## おまけ: stacking より calibration が刺さる

s6e6 では 2nd-stage Logistic Regression による stacking も試した:

```python
[00:42:40] simple equal-weight blend CV: 0.95787
[00:42:40] LogReg stacking CV: 0.95554           ← stacking で逆に下がる
```

stacking が効かなかった理由は、3 モデル (LGBM/CB/XGB) の OOF 相関が高くて LogReg が
新情報を見つけられなかったから。一方、**per-class calibration は 「OOF が間違って多数派に
倒れている」という現象に直接効く** ので、OOF 相関とは独立に効く。

「組み合わせる」より「分布をひねる」、というのも今回の収穫。

---

**関連**:
[1 日で 0.954 → 0.964 まで上げた 4 つのレバー (Kaggle s6e6 中央超えへの階段)](/post.html?slug=2026-06-20-s6e6-stairway-to-median) /
[5 連続 SubmissionStatus.ERROR が教えてくれた、多クラス文字列ラベルの SSOT 漏れ](/post.html?slug=2026-06-19-multiclass-string-label-ssot)
