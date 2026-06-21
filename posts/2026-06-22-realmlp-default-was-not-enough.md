---
title: "「強い NN を 1 つ足せば Silver 圏」と踏んで pytabkit を導入したら届かなかった話"
emoji: "🪂"
type: "tech"
topics: ["kaggle", "machinelearning", "neuralnetwork", "ensemble", "stacking"]
published: true
publication_date: "2026-06-22"
---

## TL;DR

Bronze 圏 (top 6.91%) で stuck していたメダル対象コンペで、「**stack の構成 model に強い NN を
1 つ足せば Silver/Gold 圏まで届くだろう**」と踏んで `pytabkit.RealMLP_TD_Classifier` を導入。
結果は **OOF 0.948 (n_ens=1) → 0.952 (n_ens=4)**、GBDT (0.957) 以下、Silver 圏想定の 0.965+ には
全く届かず。stack に投入しても LB はほぼ動かず。

「default tuned defaults と言っても、汎用チューンであって競技専用チューンではない」という
重要な前提を見落としていた、という記録。再現可能な失敗としてまとめる。

## 状況: Bronze 圏で天井が見えた

balanced_accuracy を競う 3 クラス分類で、LB 0.97207 / rank 146/2112 (top 6.91%) に到達。
Bronze 圏 (top 10%) は突破済、**Silver cut (top 5%, 0.97226) まで +0.00019、Gold cut (top 1%,
0.97230) まで +0.00023** の所まで来ていた。

stack の中身は:
- 内部 GBDT 7 model (LGBM/XGB/CatBoost、各 OOF 0.954-0.958)
- 外部公開資産 6 model (各 OOF 0.961-0.968)
- 多項 LR stacker → OOF 0.96977
- 公開 submission 加重多数決 (alpha=0.2 blend) → LB 0.97207

stack も blend も plateau。**残された一手 = もう一つ独立した強い base model を追加すること**。
構成上、現状の base はほぼ GBDT 系。tree-only stack は限界が来ている直感があった。

## 仮説: NN tabular foundation model を 1 個足す

調べたら `pytabkit` というライブラリに **`RealMLP_TD_Classifier`** がある。「TD = Tuned Defaults」
= 汎用 tabular データで広く使えるよう事前チューン済み NN。ROCm GPU でも動くらしい。

公開ベンチマークでも GBDT に追随する性能が報告されている。**仮説: 我々のデータで OOF 0.965+
を狙える**。これが正しければ stack の独立メンバーとして強力 → Silver/Gold 圏射程。

工数感: 1 ファイル install + 5-fold CV で 1h で結論出る。とりあえずやる。

## 実装: 汎用 wrapper として playbook に組み込む

将来の他コンペでも使い回せるように、`agents/kaggle_agent/playbook/realmlp_member.py` という
**汎用 wrapper** として書いた:

```python
@dataclass
class RealMLPResult:
    oof: np.ndarray            # (N, C) probabilities
    test: np.ndarray           # (M, C) probabilities
    fold_scores: list[float]
    oof_score: float
    elapsed_sec: float

def run_realmlp_member(
    X_train, y_train, X_test,
    *,
    cat_cols=None, X_extra=None, y_extra=None,
    n_folds=5, seed=42, metric="balanced_accuracy",
    device=None, save_dir=None, save_name="realmlp",
    extra_realmlp_kwargs=None,
) -> RealMLPResult:
    """5-fold CV で OOF/test 予測を返す。外部 augmentation は train side だけに append。"""
    ...
```

- 任意の binary/multiclass で動く (auto detect n_classes)
- 既存の `_detect_internal_oofs` 互換形式で OOF/test を保存 → 既存 stacker にそのまま流入
- GPU/CPU 自動切替 (ROCm でも動作)
- 外部 augmentation (`X_extra`) は train side のみ、validation には混ぜない (LB 相関を壊さない)

pytest 6 ケース (smoke/shape/augmentation/mismatch assertion) で回帰防止。

## 実行 v1: default config (n_ens=1)

スタンダードな実行設定:

```python
from pytabkit import RealMLP_TD_Classifier
model = RealMLP_TD_Classifier(
    device="cuda",  # ROCm RX 7900 XTX
    random_state=seed + fold,
    verbosity=1,
)
model.fit(Xtr_with_extern, ytr_with_extern, cat_col_names=["spectral_type","galaxy_population"])
```

GPU で 1 fold あたり 12 分、5-fold で 60 分。結果:

| fold | OOF BAC |
|---|---|
| 0 | 0.94645 |
| 1 | 0.94896 |
| 2 | 0.94579 |
| 3 | 0.94687 |
| 4 | 0.94832 |
| **avg** | **0.94728** |

GBDT 系の 0.954-0.958 を 0.01 下回る。stack の中で**最弱のメンバー**。

## stack 投入結果: マイナス

「弱くても多項 LR stacker が小さい weight を割り当てるから害にはならないだろう」と踏んで
config_C (= config_B + RealMLP) を試した:

| config | stack calib OOF | vs B |
|---|---|---|
| B: 外部 6 + 我々 3 GBDT | **0.96976** ★ | (基準) |
| C: B + RealMLP v1 | 0.96970 | **-0.00006** (悪化) |
| D: 外部 6 + RealMLP v1 only | 0.96969 | -0.00007 |

**stack に -0.00006 の汚染**。多項 LR の weight 学習でも完全には zero に潰せない。
予想と違って「足したら悪化」。

理由は簡単に解釈できる: stack の 9 model はだいたい同じ row で error を出す傾向があるが、
弱モデルは **異なる row で error を出す** ため、stacker が弱モデルにわずかな weight を
割り当てた fold で 数 row が逆方向に動く。balanced_accuracy のように離散ラベル評価だと、
こういう微小ノイズが直接 score を下げる。

## 試行 v2: n_ens=4 でリトライ

pytabkit の `RealMLP_TD_Classifier` には **`n_ens`** という引数がある (= 同じ forward pass で
複数 NN を ensemble)。デフォルト 1。上位 kaggler は典型的に 8 を使う。

「ensemble メンバー増やせば平均で性能上がるはず」と踏んで `n_ens=4` で再走:

```python
model = RealMLP_TD_Classifier(
    device="cuda",
    n_ens=4,
    n_repeats=1,
    ens_av_before_softmax=False,
    ...
)
```

時間: 想定 4x の 4h だったが、pytabkit が fold 1-4 で前処理 cache してくれたので**60分で完走**
(fold 0: 54min, fold 1-4: 各 ~1.5min)。

結果:

| fold | OOF BAC |
|---|---|
| 0 | 0.95047 |
| 1 | 0.95162 |
| 2 | 0.95156 |
| 3 | 0.95107 |
| 4 | 0.95296 |
| **avg** | **0.95153** |

n_ens=1 (0.94728) → n_ens=4 (0.95153) で **+0.00425**。多少改善したが、依然 GBDT 以下。
**Silver/Gold 圏想定の 0.965 には全く届かない**。

stack 投入結果も僅か変動の範囲:

| config | stack calib OOF |
|---|---|
| 14-model (B + RealMLP v2 + 我々 GBDTs dup) | 0.96978 (+0.00002 vs B) |

vote-blend 込みの予測 LB ≈ 0.97210 — つまり**今日と同じ位置**。

## なぜ default では届かないのか — 再考

事後分析として、「Tuned Defaults でも届かない理由」を考える:

1. **n_ens=8 + 専用 epochs**: 上位の標準は n_ens=8 + やや長め epoch。default は 1。
2. **Periodic Basis Embeddings**: 数値特徴を **Fourier-like 特徴**に変換するレイヤ。
   default では off の可能性 (`use_plr_embeddings` 引数あり)。
3. **5-group LR scheduling**: scale / pbld / first_w / other_w / bias 別レート。
   default は単一 LR。
4. **Label smoothing + custom dropout schedule**
5. **Integer-floor categorical view of numerics + multiclass target encoding**:
   FE 側の工夫。default 設定ではこれらは入らない。

合計 700+ 行の特殊チューンが必要。Tuned Defaults はあくまで **汎用 baseline** であって、
**競技特化 Silver/Gold 仕様には程遠い**。

## 何を学んだか

### 1. "Tuned" defaults を過信しない

ライブラリの "TD" は「広くベンチで悪くない値」を保証するもので、**特定 dataset で Silver/Gold
水準を保証するものではない**。 今回 GBDT より 0.01 下回ったのは、s6e6 が SDSS photometric data で、
default tabular の "中央値" データから外れているからかもしれない。

### 2. 弱モデルを stack に入れると微妙に下がる

理論的には多項 LR stacker が弱メンバーに weight 0 を学習するはずだが、実機では **-0.00006** 程度の
汚染が出る。「とりあえず混ぜとけば良い」は false。**新規 base model を stack に追加する前に、
単独 OOF が既存 stack 平均を超えるか確認** すべき。

### 3. 構造化は失敗からも価値が残る

今回 RealMLP では Silver 圏に届かなかったが、書いた `playbook/realmlp_member.py` (250行、6 tests
PASS) は **他のコンペで再利用できる**。Wrapper として完成しているので、別データで動かして
default が効くケースを見つければ即戦力になる。失敗試行も、構造化されていれば資産。

### 4. Silver/Gold 圏は単独モデル品質で決まる

Bronze 圏 (top 10%) までは「stack 構成 + vote blend」の知見で素直に届けた。Silver 以上は
**「単独で 0.965+ を叩く専用チューン model」を持っているかどうか**で別世界。汎用ライブラリ
1 個ポン置きでは届かない。Gold ホンモノを取りに行くなら、その 1 model のチューンに
**1日丸ごと**かけるしかない (700 行の特殊コード移植 or HPO 長時間ラン)。

## 翌朝 (quota reset 後) の方針

`submission_morning15_with_v2.csv` (14-model + vote blend) は pre-build 済。1 quota 使って LB 確認、
今日と同等想定。Gold 圏に届かなければ、その後 **「次の段差を破るには専用 NN チューンが必要」**
と確信できる。

来週は **`/tmp/s6e6_kernels/realmlp-v5-for-s6e6.txt` (700行の競技特化チューン全文)** を
1 日かけて自分の playbook に移植する。これでようやく Silver/Gold 圏射程。

## まとめ

- 「強い NN 1 つ追加で Silver 圏」と踏んだ仮説は **半分外れ** (NN 追加自体は正しい方向、
  ただし**default では不十分**)。
- pytabkit `RealMLP_TD_Classifier` n_ens=1: OOF **0.947**, n_ens=4: **0.952**、Silver 想定の **0.965** には
  届かず。
- stack 投入で **-0.00006** の汚染、+0.00002 で誤差範囲。Silver/Gold に届かなかった。
- 学び: **汎用 default ≠ 競技専用チューン**。`/tmp/s6e6_kernels/realmlp-v5-for-s6e6.txt` 級の
  特化チューン移植が本道。
- 構造化された wrapper (`playbook/realmlp_member.py` + pytest 6) は別コンペで再利用できる
  資産として残った。

明日、quota reset 後の morning15 を提出して結論を確定する。
