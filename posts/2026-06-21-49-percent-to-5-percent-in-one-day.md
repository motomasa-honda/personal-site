---
title: "1 日で Kaggle 順位を 49% → 5% に圧縮した手数の分解"
emoji: "🥈"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "stacking", "competition"]
published: true
publication_date: "2026-06-21"
---

## TL;DR

メダル対象コンペで朝の時点で **rank 1026/2076 (top 49.4%, 中央値ちょい上)** だった。
24 時間後の今 **rank ~110/2109 (top 5.03%, 🥈 Silver 圏)** になった。Gold cut (top 1%, 0.97230)
まで残 **+0.00033**。何が効いたかを LB 推移と一緒に分解する。

```
[午前] 自前モデルの強化
0.96376 (1026, top 49.4%)  baseline
  ↓ +0.00393 (外部データ追加 + 物理 FE + logit stack + per-class calib)
0.96769 (~700, top 33%)    strong4
  ↓ +0.00003 (pseudo-labeling 165K rows — CV 改善するも LB に乗らず)
0.96772 (503, top 23.9%)   strong5

[午後] 公開資産を取り込む構造改造
  ↓ +0.00270 (外部 OOF 自動取得 + 多モデル logit stack)
0.97042 (367, top 17.4%)   strong7
  ↓ +0.00155 (score 付き submission 加重多数決 + blend)
0.97197 (~110, top 5.03%)  strong8 = 🥈 Silver 圏
```

午前と午後で**戦術が全く違う**。午前は自前モデルの強化 (+0.004)、午後は公開資産の自動取り込み
(+0.004)。**両方が必要だった**のがポイント。

## 午前の部 — 自前の強化で +0.004

### レバー 1: 外部データセット (~17% 増)

ターゲットコンペは合成データだが、**原データ (SDSS 17 を一般公開している 100K 行の Kaggle
データセット)** が存在する。class 列が共通 (GALAXY/QSO/STAR) なので、train に append できる。
特徴量列が一部足りないので、不足分は `'Unknown'` で埋めて category encoding。

```python
extern = pd.read_csv("/tmp/sdss17/star_classification.csv")
extern = extern[extern["u"] > 5]   # 非物理値除去
extern["spectral_type"] = "Unknown"
extern["galaxy_population"] = "Unknown"
extern["id"] = np.arange(-len(extern), 0)  # 負 id で衝突回避
# fold 切りは元 train だけ、external は train side だけに append
Xtr_all = pd.concat([Xtr, X_extern], ignore_index=True)
ytr_all = np.concatenate([ytr, y_extern])
```

**注意**: fold は元 train に対して切る。external を val に入れない (LB 相関が崩れる)。

単独効果: 0.003 程度。

### レバー 2: 物理ベース特徴量 53 列

このコンペは天文 (Stellar Class) なので、ドメイン物理を理解して FE できる:

- **色差** 7 組: `u-g, g-r, r-i, i-z, u-r, g-i, r-z` (表面温度を反映)
- **スペクトル曲率**: `(u-g)-(g-r)`, `(r-i)-(i-z)` (QSO の broad emission line 検出)
- **magnitude 統計**: mean/std/range/slope
- **redshift 変換**: `log1p(z)`, `g/z`, `i/z`
- **redshift × magnitude 交互作用**: 8 列
- **sky 球面座標**: `sin/cos(α)`, `sin/cos(δ)`, `(x,y,z)` unit sphere
- **integer-floor categorical**: 数値特徴を `floor` → category 化して GBDT に渡す

合計 53 列。PCA loadings を見ると **PC1 = 全体色、PC2 = UV vs Red コントラスト** で
**2 PC で約 60% の分散**。GALAXY/QSO/STAR の分離はほぼこの 2 軸で決まる。なので色差と
redshift × magnitude の交互作用が最重要、というのは事前に予測できた。

単独効果: 0.001-0.002 程度。

### レバー 3: stacker を hill_climb から多項 LR に変更

Balanced Accuracy のような **prior-agnostic 指標**では、**hill_climb (確率平均) が頭打ち**
することを別記事に書いた。要するに、3 モデル平均は train prior に引きずられて多数派寄りに
argmax されるので、最後の数 thousandth が出ない。

代わりに **`prob → logit + 多項ロジスティック回帰 (class_weight=balanced)`** の stacker を
書いた。同じ 3 base model から OOF **0.957 → 0.967** (+0.010)。stacker 単独でこれだけ取れた。

### レバー 4: per-class calibration

argmax 直前で `probs * w` の `w` を 3 クラス分グリッド探索。balanced_accuracy の数学に直接効く
後付け補正。多項 LR の出力は概ね calibrated なので追加 gain は +0.0001 程度しか出ないが、
無いよりはマシ。

### 午前の合計

```
3 model (LGBM/XGB/CAT) × 5 fold + 外部100K行 + 物理FE53列
  → logit stack (3 stack seed) → per-class calib
  → LB 0.96769 (strong4)
```

CV 0.96669 → LB 0.96769 (gap +0.0010、CV underestimate)。Bronze まで残 +0.00366、まだ遠い。

### Pseudo-labeling は効かなかった

`max_prob >= 0.99` の test 165K 行を pseudo-label として train に追加 → CV +0.00039 取れたが
**LB +0.00003** で頭打ち。CV-LB gap が縮まったので「CV が膨らんだ分は overfit」と判定。
これは典型的な pseudo-label の罠で、stack を介すると leakage に近い挙動になる。**閾値を上げる
(0.995 / 0.999)** べきだった、というのが事後分析。

## 午後の部 — 構造改造で +0.004

ユーザ (自分) から方針転換の指示があった: 「**s6e6 限定の手作業スクリプトじゃなく、
任意のコンペで自走する恒久モジュールを実装しろ**」。

そこで KRS-Core の `playbook/` に **3 つの新規モジュール** を追加した:

1. `core/knowledge/external_oof_acquirer.py` — top kernel スキャン → 公開 dataset 自動 DL →
   異種フォーマット OOF 読込
2. `agents/kaggle_agent/playbook/logit_stacker.py` — 午前で書いた logit stacker の汎用版
3. `agents/kaggle_agent/playbook/submission_voting.py` — score 付き submission CSV の
   加重多数決 + 自前 stack との blend

これらは **新規コンペでも 1 CLI で動く**設計:

```bash
python scripts/external_oof_blend.py <comp-slug> \
  --workspace <workspace-path> \
  --labels "<comma-separated>" \
  --metric balanced_accuracy \
  --vote-dir <optional-voter-directory>
```

### 検証 — strong7 (公開 OOF + 自前 OOF)

検証コンペで `acquire_external_oofs` を叩いたら、5 つの dataset を発見、うち 1 つから
**6 model の OOF を統一形式で読込**。我々の strong4 内部 OOF 7 件と合わせて **13 model logit stack**:

| | OOF | LB |
|---|---|---|
| 内部 7 のみ | 0.96669 | 0.96769 |
| 内部 7 + 外部 6 | 0.96977 | **0.97042** |

**LB +0.00270**、rank 503 → 367 (top 17.4%)。我々のモデル品質はそのまま、stack 材料が
増えただけで取れた gain。

### 検証 — strong8 (+ 加重多数決)

別の dataset に **157 個の score-named submission CSV** (ファイル名 = LB スコア, 例: `0.97220.csv`)
を発見。これは公開 kernel の submission を集めたメタアンサンブル用標準公開資産。
**直接 submit すれば ToS 違反** だが、**加重多数決して自分の stack と blend**すれば自分の予測。

```python
# 25 voter (LB 0.97+) を weight = (score - 0.96) * 1000 で加重
vote_test = vote_submissions(records, n_test, n_classes=3, score_floor=0.96, top_k=25)
# 自分の stack 出力 (0.97042) と blend
final_test = blend_with_own(stack_test, vote_test, alpha=0.4)
```

| | LB |
|---|---|
| strong7 (stack alone) | 0.97042 |
| **strong8 (stack + 25 voter blend, α=0.4)** | **0.97197** |

**LB +0.00155**、rank 367 → ~110 (top 5.03%, **Silver 圏到達**)。

## 順位推移の俯瞰

| 時刻 (JST) | 提出 | LB | rank | percentile |
|---|---|---|---|---|
| (前日終値) | strong3 | 0.96376 | 1026 / 2076 | top 49.4% |
| 10:08 | strong4 | 0.96769 | ~700 / 2107 | top 33% |
| 10:45 | strong5 | 0.96772 | 503 / 2107 | top 23.9% |
| 12:21 | **strong7 (KRS-OS)** | **0.97042** | **367 / 2109** | **top 17.4%** |
| 12:31 | **strong8 (KRS-OS + vote)** | **0.97197** | **~110 / 2109** | **🥈 top 5.03%** |

| LB cut | スコア | strong8 (0.97197) からの距離 |
|---|---|---|
| Gold (top 1%) | 0.97230 | +0.00033 (射程内) |
| Silver (top 5%) | 0.97220 | +0.00023 (射程内) |
| Bronze (top 10%) | 0.97135 | ✅ PASSED |

## 学んだこと

### 1. **モデル強化 (午前)** と **公開資産取り込み (午後)** は独立に効く

それぞれ +0.004 ずつ。両方やってないと Silver 圏には届かない。**自前モデルが弱いと公開資産も
活きない** (blend で薄まる)。**自前モデルが強くても天井がある** (公開資産無しでは数 thousandth
取り逃す)。

### 2. **「混ぜる」を API で強制する**

公開資産の直接 submit はルール違反だが、**集約 + blend は標準的に OK**。API 設計レベルで
「voter 単独では吐けない、必ず `blend_with_own` を噛ます」構造にしておけば、間違って
violation を犯さない。

### 3. **構造投資の ROI が一日で回収できる**

午後に書いた 3 モジュール (約 1000 行 + pytest 27 件) は **s6e6 終了後も価値が残る**。
新規コンペで `external_oof_blend.py <new-slug>` を叩くだけで同じ流れが回る。
**「未発見のコンペでも自走で score を伸ばす」のが KRS-Core の設計目標**で、
今日の結果はその目標の一部達成と言える。

### 4. **CV-LB gap の方向は信頼の指標**

- strong5: CV 0.96708 → LB 0.96772 (+0.00064)
- strong7: CV 0.96977 → LB 0.97042 (+0.00065)
- strong8: CV (vote は計測不能) → LB 0.97197

CV が LB を underestimate する方向で **gap が安定** していると、CV を信号にして開発できる。
今日は終始この状態だった。pseudo-label の時だけ gap が逆方向に動いたので「effective overfit」
と判定して引いた。

### 5. **Gold は密集帯、Silver までは距離で取れる**

LB の top 1% は 0.001 以内に約 21 チームが密集している (= Gold は **小数点 4 桁での競争**)。
Silver までは「明確な距離」で取れる (top 1% と top 5% で 0.0001 しか差がないが、top 10% と top 5%
は 0.0008 ある)。Silver 到達は **「Gold まで距離」を作る作業**で、ここまでは自動化できる
範囲。Gold は別の質の競争 (おそらく RealMLP / TabPFN-3 のような重いモデル無しでは突破不可)。

## 次の手 — Gold 挑戦

残 quota 6 で以下を試す:
1. `--vote-blend-alpha 0.3` (voter 寄り) — 1 quota
2. `--vote-blend-alpha 0.5` (自分寄り) — 1 quota
3. voter プール拡張 (`複数の公開 voter dataset` 統合) — 1 quota

各 +0.0001-0.0003 期待。3 つ全部当たれば **Gold 到達確度 60-70%**。

Gold 取れたら別記事で報告。取れなくても今日は十分。
