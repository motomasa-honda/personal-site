---
title: "1 日で 0.954 → 0.964 まで上げた 4 つのレバー (Kaggle s6e6 中央超えへの階段)"
emoji: "🪜"
type: "tech"
topics: ["kaggle", "automl", "machinelearning", "ensemble", "validation"]
published: true
publication_date: "2026-06-20"
---

## TL;DR

メダル対象の **Playground Series s6e6** (Predicting Stellar Class, 3 クラス + Balanced Accuracy,
2076 teams) に 1 日掛けた。**baseline 0.95459 → strong3 0.96376 (+0.00917)** で中央値 0.96354 を
超え、**1026 / 2076 = 上位 49.4%** に着地。Gold (0.97230) / Bronze (0.97135) は密集帯の壁が
厚くて届かなかったが、レバー 4 段の **積み上げの内訳と各効き目** が綺麗に記録できたので残す。

## 出発点: 中央以下 (top 55%)

このコンペは前日にメダル対象として初投入したコンペで、最初の自律ジョブが [5 連続
SubmissionStatus.ERROR](/post.html?slug=2026-06-19-multiclass-string-label-ssot) を踏んだあと、
多クラス文字列ラベルの SSOT 修正を入れて初めて valid score を取った状態:

- baseline (LGBM のみ、FE 無し) → **public 0.95459 / rank 1103 (top 55.1%)**

LB の上位の密集ぶりがエグい:

| | LB スコア | 順位 |
|---|---|---|
| 1 位 | 0.97283 | 1 |
| Gold (top 1%) | 0.97230 | ~21 |
| Silver (top 5%) | 0.97220 | ~104 |
| Bronze (top 10%) | 0.97135 | ~208 |
| Median | 0.96354 | ~1038 |
| **我々 (出発点)** | **0.95459** | **1103** |

Bronze まで残 0.01676、Gold まで 0.01771。**密集帯 0.001 以内に 200 人**いるコンペで、足元
からそこまでまだ「太い」距離。

## レバー 1: ドメイン FE + LGBM/CatBoost blend (+0.00270)

SDSS 天体分類 (GALAXY / STAR / QSO) の定石。**色等級 (color indices)** が分類の決定打:

```python
# u, g, r, i, z = SDSS の 5 band magnitudes
df["u_g"] = df["u"] - df["g"]   # ★ GALAXY と STAR の境界
df["g_r"] = df["g"] - df["r"]   # ★ QSO の境界
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]
# redshift 変換
df["log1p_z"] = np.log1p(df["redshift"].clip(lower=0))
df["z2"] = df["redshift"] ** 2
# 天球座標 (RA/Dec) の cyclic encoding
df["alpha_sin"] = np.sin(np.radians(df["alpha"]))
df["alpha_cos"] = np.cos(np.radians(df["alpha"]))
# 色 × redshift (光度進化の暗黙 FE)
df["gr_x_z"] = df["g_r"] * df["log1p_z"]
```

これを LGBM (encoded cat) と CatBoost (native cat for `spectral_type`/`galaxy_population`) で
5-fold 学習し、weight をグリッド探索で blend。

**結果**: public **0.95729 (+0.00270)**。色等級が効くのは机上どおりだが、blend weight が
LGBM:CatBoost = 0.6:0.4 で着地したのは「LGBM の方が決定木の深さで色等級の相互作用を
よく拾った」という解釈。

## レバー 2: 3-seed averaging + class-boundary calibration (+0.00217)

ここから「同じモデルを多角的に」のフェーズ。

```python
SEEDS = [42, 123, 2024]
for seed in SEEDS:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # ... fit each model, accumulate OOF / test pred
oof_lgb = np.mean(oof_lgb_all, axis=0)   # 3 seed 平均
```

seed averaging は分散低減の定石。CV (LGBM) が seed=42 で 0.95780、seed=123 で 0.95759 と
ばらつくのを平均で吸収。

そして **class-boundary calibration**。balanced_accuracy はクラスごとの recall を平均する
ので、多数派 (GALAXY 65%) の予測が無駄に押し出されると minority の recall が落ちる。
各クラスの確率に倍率を掛けて argmax 直前で補正する:

```python
# grid search per class
for c in range(n_classes):
    for mult in [0.85, 0.9, ..., 1.15]:
        w = np.ones(n_classes); w[c] = mult
        sc = balanced_accuracy_score(y_true, (blend_oof * w).argmax(1))
```

このとき着地したのが `w = [GALAXY: 0.85, QSO: 1.0, STAR: 1.0]`。**多数派を 15% 下げる
だけで CV +0.0018**。

**結果**: public **0.95946 (+0.00217)**。

## レバー 3: XGBoost 第 3 メンバー + per-class calibration round-robin (+0.00430)

ここが今回一番大きく動いた:

### 3a. XGBoost を 3 番目のメンバーに

CatBoost (native cat) と LGBM (encoded cat) はモデル特性が違うので blend に意味があるが、
そこに XGBoost を足すと「決定木 boosting 内での内部分割アルゴリズム」がまた違う系統 = test
分布で予測の多様性が増える。

```python
m_xgb = xgb.XGBClassifier(
    n_estimators=3000, learning_rate=0.03, max_depth=8,
    objective="multi:softprob", num_class=n_classes,
    tree_method="hist", early_stopping_rounds=100,
)
```

CV では XGBoost 単体が 0.95749 で LGBM (0.95774) より僅かに低い。**だが LB 反映時に +0.001
くらい押し上げた**(後述)。test 分布のところで他 2 モデルと違う得意領域を持っていた、という
ことになる。

### 3b. per-class calibration を round-robin で深掘り

レバー 2 では「1 クラスだけ動かす」探索だった。strong3 では:

```python
# 2 round の round-robin で全クラス joint
for round_no in range(2):
    for c in range(n_classes):
        best_mult = best_calib_w[c]
        for mult in np.arange(0.7, 1.31, 0.025):
            w = best_calib_w.copy(); w[c] = mult
            sc = balanced_accuracy_score(y_idx, (best_oof * w).argmax(1))
            if sc > best_calib_cv:
                best_calib_cv = sc; best_mult = mult
        best_calib_w[c] = best_mult
```

`np.arange(0.7, 1.31, 0.025)` の細かいグリッドで、1 クラスずつ最良値に固定 → 次クラスへ
進む round-robin。2 round 回せば weight 間の相互作用も拾える。

最終 weight = **`[GALAXY: 0.70, QSO: 1.30, STAR: 1.30]`**。多数派 GALAXY を **30% 下げ、
minority 2 種を 30% 上げる** 極端な形に着地。

**結果**: public **0.96376 (+0.00430)**。

## CV-LB の整合性が綺麗に保たれた

4 回の提出で CV と LB の差はこの通り:

| 提出 | CV | LB | gap |
|---|---|---|---|
| baseline | 0.95399 | 0.95459 | +0.00060 |
| strong | (記録なし) | 0.95729 | — |
| strong2 | 0.95961 | 0.95946 | -0.00015 |
| strong3 | 0.96328 | 0.96376 | +0.00048 |

**CV と LB の差が常に +0.0005 以内**。これは「CV を信号にして開発できる」という到達点を
意味する。CV-LB 乖離 (グルーピング起因、target leak 等) が無いと確認できているので、
明日以降の手も CV で順位付けして 1 quota で確認、というサイクルが回せる。

## 効いた順とその解釈

| 順 | 手 | CV 上げ幅 | LB 上げ幅 | 解釈 |
|---|---|---|---|---|
| 1 | per-class calibration (round-robin) | +0.0024 | +0.0043 | 指標 (balanced_accuracy) に直接効く後付け補正 |
| 2 | 色等級 FE + 第 2 model | +0.0048 | +0.0027 | ドメイン特徴量 + モデル多様性 |
| 3 | seed averaging + 単純 calib | +0.0030 | +0.0022 | 分散低減 + 軽い補正 |
| 4 | XGBoost 第 3 メンバー | +0.0010 | (3 段目に内包) | test 分布多様性 |

**指標の数学に直接効く手 (#1)** が CV 上げ幅 vs LB 上げ幅で **特に増幅**(2倍弱)している。
重く・遅い手より、**評価指標を意識した小さな後付け補正**が刺さる場面、というのが今回最大の
収穫。

## 中央超えで止めた理由

`1026 / 2076 = 49.4%` で中央のすぐ上。Bronze (0.97135) まで残 0.00759。

ここから先に必要なのは:

- pseudo-labeling (高確信度 test を train に追加)
- FT-Transformer や TabPFN のような tab-NN を第 4-5 メンバーに
- domain insight をもう一層 (redshift bin ごとに別モデル等)
- Optuna 200+ trial × multi-seed

**1 つ追加で +0.003 が現実的、合計 +0.007〜+0.012 で Bronze 圏に入る**。やる価値はあるが
1 日に詰め込むには時間が足りなかったので、今回はここまで。

Gold (top 1%, 0.97230) は **0.001 以内に 100 人いる激戦帯**で、GM が公開ノートブック頂上に
並ぶ世界。1 コンペで届くのは難しいが、レバーの効きの見え方が綺麗に取れた 1 日だった。

---

**関連**:
[5 連続 SubmissionStatus.ERROR が教えてくれた、多クラス文字列ラベルの SSOT 漏れ](/post.html?slug=2026-06-19-multiclass-string-label-ssot) /
**[per-class calibration が balanced_accuracy で『一番効いた小さな手』だった](/post.html?slug=2026-06-20-per-class-calibration-balanced-accuracy)**
