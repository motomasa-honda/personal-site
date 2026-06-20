---
title: "sklearn 1.8 で LogisticRegression の multi_class が消えていて、2 時間 27 分の計算が死んだ"
emoji: "💥"
type: "tech"
topics: ["python", "scikit", "machinelearning", "deprecation", "kaggle"]
published: true
publication_date: "2026-06-20"
---

## TL;DR

3 モデル × 3 seed × 5 fold = **45 fold (LightGBM/CatBoost/XGBoost)** を全部完了して、最後の
**2 段 stacking** で `LogisticRegression(multi_class="multinomial")` を呼んだら **2 時間 27 分の
計算が一発で死亡**。**sklearn 1.8 で `multi_class` 引数は警告も無く削除**されていた。
OOF を mem-only で保持していたのも被害拡大の原因。リカバリは 50 分で完走したが、
教訓は多い。

## 状況: stacking 直前の慢心

s6e6 の最終攻撃で、3 モデル ensemble に **2 段 Logistic Regression stacking** を入れた:

```python
# 45 fold が全完了した直後
oof_lgb = np.mean(oof_lgb_all, axis=0)   # 3-seed avg
oof_cb  = np.mean(oof_cb_all,  axis=0)
oof_xgb = np.mean(oof_xgb_all, axis=0)
# ...
log(f"3-seed avg CV: lgb={cv_lgb:.5f} cb={cv_cb:.5f} xgb={cv_xgb:.5f}")
log(f"simple equal-weight blend CV: {cv_simple:.5f}")
log(f"stack input shape: {X_stack.shape}")   # ← ここまで OK

# 2nd-stage stacking
for fold, (tr, va) in enumerate(cv2.split(X_stack, y_idx)):
    lr = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial", n_jobs=-1)
    lr.fit(X_stack[tr], y_idx[tr])
    #   ↑ TypeError: __init__() got an unexpected keyword argument 'multi_class'
```

`journalctl` で見るとこう:

```
[00:42:40] simple equal-weight blend CV: 0.95771
[00:42:40] stack input shape: (577347, 9)
Traceback (most recent call last):
  File "/tmp/s6e6_strong3.py", line 171, in <module>
    lr = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial", n_jobs=-1)
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
```

`multi_class` 引数が**消えていた**。`DeprecationWarning` でもなく、いきなり `TypeError`。

## なぜ消えていたか

sklearn 1.6 (2024) で `multi_class` 引数は **deprecated** 警告を出すようになっていて、1.8 で
**完全削除**された。新 API では:

- 二値分類は二値固定
- 多クラスは自動で multinomial になる

意図としては「使い分けの必要がそもそも無くなった」ということで、`multi_class="multinomial"` を
渡さなくても多クラスなら自動で multinomial。**引数を消すだけで挙動は変わらないはず**、という
設計。

ところが互換性の観点では、**deprecated → 警告 → 削除** の流れの「警告」を読まないと、
ある日突然 `TypeError` を踏む。ライブラリ更新時に release notes を読んでない私の責任ではあるが、
**deprecated 期間 (~1 年) を経て削除**は機械学習周辺としては早い気もする。

## 失われたもの: OOF 全部

このスクリプトは:

```python
SEEDS = [42, 123, 2024]
oof = {"lgb": [], "cb": [], "xgb": []}
prd = {"lgb": [], "cb": [], "xgb": []}

for seed in SEEDS:
    # ... 5 fold で OOF / test pred を計算
    oof["lgb"].append(oof_lgb); prd["lgb"].append(pred_lgb)
    # ...
```

OOF も test predictions も **Python の dict にだけ** 入れていた。プロセスが落ちた瞬間に
**全部消える**。

45 fold (約 2 時間半) の計算が、最後の 5 行 (`lr.fit(...)`) で全部無に帰した。

## リカバリ: 50 分で再走

幸い:

- バグは `multi_class=` 引数を削除するだけで直る (新 API はもう自動 multinomial)
- 45 fold の CV 値は journalctl に残っていた (LGBM 0.95809 / CB 0.95616 / XGB 0.95763)
- 時間的に **JST 8:00 開始 → 9:00 quota reset → 提出**のタイミングを狙える

ということで、**seed を 1 個 (42) だけにして 50 分で再走**:

```python
- SEEDS = [42, 123, 2024]
+ SEEDS = [42]  # リカバリ: 全 3 seed 完了したが Stacking バグで死亡。時間優先で 1 seed

- lr = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial", n_jobs=-1)
+ lr = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)  # sklearn 1.8: multi_class 引数廃止
```

`systemd-run --user --unit=s6e6-strong3` で起動、別途仕掛けた監視サービス
`s6e6-after-finish.service` が完了を検知して自動提出。寝ている間に **public 0.96376** で
中央値超え。

(さらに 1 seed のリカバリ版でも CV (calibrated) が 0.96328 で、3 seed 版の予想 CV 0.962-0.964
と大差なかった。**per-class calibration が分散の大半を吸ってくれていた**ことの傍証。)

## 学び 1: 中間成果は逐次保存

正解は、OOF を fold ごとに `np.save` する:

```python
oof[va] = model.predict_proba(X.iloc[va])
np.save(f"/tmp/oof_s{seed}_f{fold}_lgb.npy", oof[va])  # ← これがあれば救済できた
```

5 fold × 3 model × 3 seed = 45 ファイルになるが、ディスクは安い。**Python プロセスが落ちても
ディスクの中間成果は生き残る**。リスタート時に既存ファイルを load して、未完了 fold だけ
やり直せばいい。

機械学習で **数時間級の計算** を mem-only でやるのは、ライブラリ事故・電源断・kill の
リスクをそのまま被ること。

## 学び 2: 依存ライブラリのメジャーバージョン上がり時は引数を grep する

sklearn を `1.6 → 1.8` に上げる時、

```bash
# 自分のコードで使ってる sklearn の引数を全部 grep
grep -rE "LogisticRegression\(.*\)" .
grep -rE "RandomForest\(.*\)" .
# ... 主要 estimator 全部
```

しておけば、`multi_class=`, `n_iter=` (LR), `presort=` (DT) など **bare 引数が消えた事故**を
事前検出できる。CI で smoke test するのが一番だが、それが無いなら最低 grep。

## 学び 3: 2 段 stacking より per-class calibration が刺さることがある

リカバリ版で **stacking は実は採用されなかった**:

```
[09:35:06] simple equal-weight blend CV: 0.95787
[09:35:11] LogReg stacking CV: 0.95554           ← stacking で逆に下がる
[09:35:11] → simple 採用 (0.95787 >= 0.95554)
```

3 モデル (LGBM/CB/XGB) の OOF 相関が高くて、2 段 LR は新情報を見つけられなかった。
代わりに **per-class calibration** が CV を **+0.00541** 押し上げて、ここが今回最大の
レバーになった。

**1 番大事だった手 (calibration) で 2 番目に大事だった機構 (stacking) が原因で 2h27m を
失う**、というのが、機械学習の現場の業界あるあるな失敗の型だなと思った。

---

## まとめ

- sklearn 1.8 で `LogisticRegression(multi_class=...)` は **警告なしに `TypeError`**
- 2h27m の計算が消えた、リカバリは 50 分
- **中間成果は必ず逐次保存** (mem-only は事故時に死ぬ)
- **依存ライブラリ更新時は bare 引数を grep**

学習と現場、両方の意味で痛い一発だった。

---

**関連**:
[1 日で 0.954 → 0.964 まで上げた 4 つのレバー (Kaggle s6e6 中央超えへの階段)](/post.html?slug=2026-06-20-s6e6-stairway-to-median) /
[per-class calibration が balanced_accuracy で『一番効いた小さな手』だった](/post.html?slug=2026-06-20-per-class-calibration-balanced-accuracy)
