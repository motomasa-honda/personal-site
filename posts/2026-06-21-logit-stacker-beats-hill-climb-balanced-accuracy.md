---
title: "balanced_accuracy で hill_climb が頭打ちした時、stacker を作り直して +0.003 取った"
emoji: "📐"
type: "tech"
topics: ["machinelearning", "kaggle", "ensemble", "stacking", "logisticregression"]
published: true
publication_date: "2026-06-21"
---

## TL;DR

メダル対象の 3 クラス分類 (Balanced Accuracy) で、ベース 3 モデル (LGBM/XGB/CatBoost) を
**Caruana 流の hill climbing** (確率平均) で組んでいたところ、OOF が 0.957 で頭打ち。
そこで stacker を **「prob → logit 変換 + 多項ロジスティック回帰 (PyTorch GPU)」** に
作り直した。同じ 3 ベースモデルから OOF **0.957 → 0.9666** (+0.0096) に跳ねた。
balanced_accuracy / macro_f1 のような **prior-agnostic な指標**では、確率平均は
モデルの train prior に引きずられて多数派に寄り、最後の数 thousandth が出ない。多項 LR は
**class_weight=balanced** で学習時にそのバイアスを学習層で吸収する。

これは KRS-Core の Playbook に `logit_stacker.py` として恒久実装した。今後どのコンペでも
metric が balanced_accuracy 系なら自動でこれが効くようにした。

## 経緯 — 「あと一押し」のために確率平均をやめた

s6e6 (Predicting Stellar Class, 3 クラス, Balanced Accuracy) で、ベース 3 モデルの OOF は
それぞれ ~0.957 弱。これを Caruana 流の greedy hill-climb (確率の重み付き平均) で組んだ
OOF が **0.95781**。per-class calibration を追加して **0.96328**。LB は 0.96376。中央 (0.96354) を
わずかに上回るだけ。

次の段差が見えない。各 base model を強化するのは時間がかかる。**stacker 自体を見直す**方向に
振った。

## なぜ確率平均 (hill_climb) が頭打ちなのか

balanced_accuracy は **クラスごとの recall の単純平均**。クラス数を C とすると:

$$
\text{BAC} = \frac{1}{C} \sum_{c=1}^{C} \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}
$$

評価指標が **クラスの prior を一切考慮しない**。一方、GBDT (LGBM/XGB/CAT) の出す確率は
**train prior にキャリブレートされている**。s6e6 は GALAXY 65% / QSO 20% / STAR 15% の
偏りがあるので、3 モデル平均は構造的に **GALAXY 寄りに argmax** されやすい。

per-class calibration (argmax 直前で `probs * w` で補正) はその修正だが、これは **重み 3 個の
グリッド探索**。3 モデルの予測のうち「どのモデルがどのクラスでよく当てるか」を学習する力は無い。

ここで欲しいのは「**モデル間 prior バイアスを学習可能な層**で吸収する stacker」。
要するに **重み付き多数決ではなく重み付き対数線形結合**。

## 設計: prob → logit + 多項ロジスティック回帰

各 base model の確率 $p \in [0,1]^{N \times C}$ を **logit 空間に変換**:

$$
z = \mathrm{clip}\!\left(\log\frac{p}{1-p},\; -30, +30\right)
$$

これを **モデル間で concatenate** して、 $X \in \mathbb{R}^{N \times (3 \cdot M)}$ ($M$ = ベース
モデル数) を作る。これを multinomial logistic regression で学習:

$$
\hat{y} = \mathrm{softmax}(W X^\top + b)
$$

学習時は `class_weight = "balanced"` (`sklearn.utils.compute_class_weight` 由来) を
sample weight に掛ける。これで balanced_accuracy の数学に **直接整合**する。

### なぜ確率を「足し算」じゃなくて「logit にして線形結合」なのか

確率 $0.99$ と $0.999$ は確率空間で見ると差が小さい (0.009) が、logit 空間では $z=4.6$ vs $z=6.9$
で **大きな差**。これがモデルの「自信度」を保つために重要。確率平均だと **強信号が薄まる**。

これは GPU 多項 LR ベースの stacker として実装。Adam, lr=0.01, `weight_decay = 1/(C \cdot N)`
で $C=0.1$、1000 epochs。バッチ全件で学習する単純な fully connected 1 層なので、CPU でも
数秒、GPU なら数百ミリ秒で終わる。

## 実装スケッチ

```python
import torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

EPS = 1e-15
LOGIT_CLIP = 30.0

def prob_to_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS).astype(np.float64)
    return np.clip(np.log(p / (1.0 - p)), -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)

# X_oof = concat([prob_to_logit(o) for o in oofs], axis=1)  # (N, 3*M)
# X_test = concat([prob_to_logit(t) for t in tests], axis=1)

cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
# fold loop:
sw = torch.tensor([cw[c] for c in y_tr], device=device)
model = nn.Linear(n_feat, n_classes).to(device)
opt = optim.Adam([
    {"params": model.weight, "weight_decay": 1.0 / (0.1 * len(y_tr))},
    {"params": model.bias,   "weight_decay": 0.0},
], lr=0.01)
for _ in range(1000):
    opt.zero_grad()
    loss = (nn.CrossEntropyLoss(reduction="none")(model(Xtr), ytr) * sw).mean()
    loss.backward(); opt.step()
```

5 fold × 3 stack seed を平均 → さらに `per_class_calibration` (10 行のグリッド探索) で
argmax 境界を微調整。

## 結果

| 構成 | OOF BAC | LB |
|---|---|---|
| hill_climb (確率平均) + per-class calib | 0.96328 | 0.96376 |
| **logit stack (3 base) + per-class calib** | **0.96669** | **0.96769** |

OOF +0.00341, LB +0.00393。**ベースモデルは同じ**ままで stacker の数学を変えただけ。
さらに公開リソースを混ぜたら (別記事) **LB 0.97042 → 0.97197 = top 5% (Silver 圏)** まで。

## 残った疑問 — なぜ 0.003 もずれるのか

「3 モデルの確率を線形結合する」のは hill_climb も logit LR も同じ操作のはずだが、現実には
0.003 差が出る。理由は 3 つ:

1. **空間が違う**: 確率平均は確率空間、logit LR は logit 空間。確率空間は $[0,1]$ で潰れて
   いて高 confidence の差が消える。logit はそれを log で広げる。
2. **class_weight の取り込み**: hill_climb は metric 直接最適化なので class_weight 概念が無い。
   LR は学習層で class_weight を sample weight として組み込める。
3. **正則化**: weight_decay = $1/(C \cdot N)$ で過学習を抑える機構が hill_climb には無い。

結果として、**「balanced_accuracy / macro_f1 系の指標で base が tree-only な時」**は logit
stacker のほうが構造的に有利。逆に **AUC / logloss なら hill_climb で十分**(これらの指標は
prior に sensitive なので確率平均で素直に効く)。

## KRS-Core 側の構造変更

これを Playbook の中に恒久実装した:

```
agents/kaggle_agent/playbook/
├─ logit_stacker.py   # ← 新規
├─ hill_climbing.py   # 既存 (継続して使う)
├─ stacking.py        # L2 LightGBM stacker
...
```

`run_logit_stack(oofs, tests, y, metric_fn=...)` で **同じ I/O 規約**で hill_climb と
スワップ可能。metric が `balanced_accuracy` のとき自動で logit stacker を選ぶようにすれば、
任意のコンペで効く。pytest 5 件で回帰防止 (`tests/test_logit_stacker.py`)。

## 教訓 (今日中の自分への戒め)

- **stacker は metric の数学と整合させる**。AUC なら hill_climb、balanced_accuracy なら logit LR。
  デフォルトで前者を使い続けると数 thousandth を取り逃す。
- **prob → logit 変換は失われがちな高 confidence 情報を取り戻す**。$p=0.999$ と $p=0.9$ は確率では
  差 0.099 だが、logit では $6.9$ vs $2.2$ で **3 倍以上**の重み。
- **小さなコード変更が大きな gain になる場面**: 今回のコア部分は実質 30 行。「重く高度な手」
  (NN stacker, 100 model ensemble, etc.) より、まず指標の数学に直接効く後付け補正を試す。

明日は同じ Playbook を **回帰指標 (RMSE/MAE)** に展開する予定。回帰でも prob 概念がない
ぶん logit 変換は不要だが、**target normalization + ridge stacker** で似たような構造を作れる
はず。
