---
title: "TabPFN を本番投入したら罠が 2 つあった — Foundation NN の domain mismatch と HF anonymous DL hang"
emoji: "🪤"
type: "tech"
topics: ["kaggle", "machinelearning", "tabpfn", "huggingface", "neuralnetwork"]
published: true
publication_date: "2026-06-23"
---

## TL;DR

s6e6 (3 クラス分類、astronomy data, 577K rows) に TabPFN-v3 を本番投入した。
5-fold OOF で **CV 0.96152**、しかし **単独 LB は 0.94691** で gap 0.0146 という異常な乖離。
ブレンドに混ぜると voters 本来 0.972+ が 0.952 に劣化。

技術的検証で見えた **2 つの罠**:

1. **HF anonymous DL hang**: `tabpfn` ライブラリは内部で `huggingface_hub.hf_hub_download` を
   呼ぶが、無認証だと **帯域 0 に絞られて永久 hang** する。同じ URL を `curl` で叩くと
   36 MB/s で 5 秒で完走する。
2. **Foundation NN の domain mismatch**: TabPFN は synthetic prior でメタ学習されており、
   astronomy real data (多バンド測光 + redshift) の分布構造に prior が合っていない仮説。
   subsample を増やしても本質的に解決しない可能性が高い。

「Foundation Tabular Model なら何でも上がる」という直感は危険、という記録。

---

## 背景

[前の記事](/post.html?slug=2026-06-22-bronze-is-autopilot-gold-needs-structure) で書いた通り、
s6e6 で Bronze (LB 0.97226) まで来て Gold cut (0.97230) まで +0.00004 short の状態だった。

ここで打てる手の中で「**stack に新しい誤差構造を持つメンバーを追加する**」のが最も筋がいい。
GBDT 系 + 既存 NN だけだと vote/blend が飽和してしまうため、根本的に違う inductive bias を
持つ Foundation NN = **TabPFN-v3** を投入する判断。

期待値:

- TabPFN は in-context learner、**事前学習された Bayesian prior を直接展開**して予測する
- GBDT / 通常 NN とは誤差構造が直交しやすい
- stack member として加えるだけで Gold cut 突破を狙える、という仮説

ところがこれが「**罠 2 連発**」だった、というのが今日の主題。

---

## 罠 1: HF anonymous DL が hang する

### 症状

`tabpfn==8.0.8` をインストールして、まず合成データで動作確認:

```python
from sklearn.datasets import make_classification
from tabpfn import TabPFNClassifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, random_state=42)
clf = TabPFNClassifier(device="cpu", random_state=42)
clf.fit(X, y)  # ← ここで永久 hang
```

`timeout 300` 付けて回しても、出力は `Warning: You are sending unauthenticated requests
to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`
の 1 行だけで終わる。

### 切り分け

`~/.cache/tabpfn/.cache/huggingface/download/` を覗くと、`.incomplete` ファイルが
**0 バイトのまま** 5 分間そこにある。HTTP 接続が確立した直後で実質帯域 0。

同じファイルを直接 `curl` で叩いてみる:

```bash
curl -L -o /tmp/test.ckpt --max-time 10 \
  "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-classifier-v3_default.ckpt"
# → HTTP=200 bytes=212804803 speed=36240809 time=5.87
```

**36 MB/s で 5.87 秒で完走**。202 MB の重みファイルが普通に取れる。
つまり帯域は問題なく、`huggingface_hub` の Python クライアントが anonymous 状態で
内部的に絞られている。

### 原因

`tabpfn/model_loading.py` を読むと、内部実装はシンプルに:

```python
from huggingface_hub import hf_hub_download
local_path = hf_hub_download(
    repo_id="Prior-Labs/tabpfn_3",
    filename="tabpfn-v3-classifier-v3_default.ckpt",
    local_dir=base_path.parent,
)
```

`hf_hub_download` は無認証だと
[一定のレートリミット](https://huggingface.co/docs/hub/api#authentication)
が掛かるが、明示的にエラーになるのではなく **TCP 接続は確立したまま帯域が絞られる** 仕様
らしい (`Warning: ... unauthenticated requests` だけ出す)。
`requests` セッションの read timeout が事実上効かない状態に陥り、無限 hang する。

### 回避

`HF_TOKEN` を設定すれば普通に動くはずだが、それなしで動かす方が再現性がある。
直接 `curl` で重みを取って TabPFN が期待するキャッシュ位置に置く helper を作った:

```bash
#!/usr/bin/env bash
# scripts/prefetch_tabpfn_weights.sh
set -euo pipefail
CACHE_DIR="${HOME}/.cache/tabpfn"
mkdir -p "${CACHE_DIR}"

TARGET="${CACHE_DIR}/tabpfn-v3-classifier-v3_default.ckpt"
if [ ! -s "${TARGET}" ]; then
  curl -L --max-time 180 -o "${TARGET}" \
    "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-classifier-v3_default.ckpt"
fi
```

これを `clf.fit()` の前に走らせると、tabpfn ライブラリは「既にキャッシュ済み」と判断して
HF DL をスキップする。1 回入れれば再利用される。

### 教訓

- **GPU/CPU/ROCm 周りより、HF 認証周りが本番運用の最初の壁になる**ことがある
- ライブラリの内部 DL を信用せず、**重要ファイルは別経路で取って事前配置できる設計** に
  しておくべき
- 同じ URL を `curl` で叩いて測ると、帯域問題と認証問題の切り分けが 30 秒で済む

---

## 罠 2: CV-LB gap 0.0146 (本質的)

### 数字

`prefetch_tabpfn_weights.sh` を入れて DL 問題を解決した後、s6e6 で 5-fold OOF を走らせた:

```python
# 設定
n_folds = 5
sub_size = 25000  # stratified subsample for fit per fold
n_estimators = 8  # TabPFN 内部 permutation 平均
device = "cuda"   # AMD RX 7900 XTX (ROCm 7.2, VRAM 25.8 GB)
ignore_pretraining_limits = True  # 10k 制限を抜く
predict_batch_size = 10000  # 247K test rows OOM 回避

# 各 fold
# - fit: 0.6s (subsample 25k なので速い)
# - val (115K rows) batched predict: 205s
# - test (247K rows) batched predict: 428s
# - 1 fold 634s × 5 fold = 52.8 min
```

結果:

| fold | val acc |
|---|---|
| 0 | 0.96145 |
| 1 | 0.96283 |
| 2 | 0.96120 |
| 3 | 0.96133 |
| 4 | 0.96076 |
| **OOF acc** | **0.96152** (fold_std 0.00070) |

fold 間の散らばりも極めて小さく、`accuracy_score(yva, p_va.argmax(1))` で直接計算した
honest な評価。提出 csv の format も問題なし (id 整列、class 分布 train 一致)。

Kaggle に出すと:

| 提出 | Public LB |
|---|---|
| TabPFN-v3 単独 | **0.94691** |
| α=0.1 blend (TabPFN + fachri top15 voters) | 0.95184 |
| (参考) 既存 LB best (fachri 単独) | 0.97226 |

**CV と LB の gap が 0.0146**。普通 Kaggle Playground 系は CV ≈ LB なので、これは異様に大きい。

### 何が起きているか

提出 format は完全に正しいので残る仮説は「train と test の分布が違う」か「TabPFN の prior が
domain に合わない」のどちらか。

両方の証拠が観察できる:

**(a) ブレンドで voters が劣化する**

α=0.1 で混ぜただけで voters 本来 0.972+ が 0.952 に落ちる。
TabPFN の確率分布 (smooth) が voters の near-one-hot を平滑化し、marginal なケースで
判定が **間違った方向に書き換わる** メカニズム。

```
voters_prob[i] = (0.93, 0.05, 0.02)  # 14/15 voters が GALAXY
tabpfn_prob[i] = (0.30, 0.50, 0.20)  # TabPFN は QSO 寄り

blended = 0.1 * tabpfn + 0.9 * voters
       = (0.867, 0.095, 0.038)        # argmax は GALAXY のまま、OK
```

これは平均的なケース。問題は voters が僅差 (0.40, 0.35, 0.25) のケースで、
TabPFN が間違ったクラスを強く push すると判定が翻る。それが 5997 行 (2.4%) 発生した。

**(b) Foundation NN の prior は astronomy にどう合わないか**

TabPFN は [PriorLabs の論文](https://arxiv.org/abs/2207.01848) によると、**synthetic
data の prior** (SCM ベースの構造的因果モデルのサンプリング) でメタ学習されている。
学習データは合成だが「実 tabular データに似せた」分布をカバーする設計。

s6e6 の特徴量は:

- 多バンド測光: `u, g, r, i, z` (5 バンドの光度、相関が強い)
- 物理量: `redshift` (スカラー、long-tail 分布)
- 角度座標: `alpha, delta` (2 次元の球面座標)
- カテゴリ: `spectral_type, galaxy_population` (low cardinality)

これらは「天文学的に意味のある **特定の物理関係**」(色-光度図 / Hubble 関係 / Tully-Fisher 等)
を反映しており、SCM ベースの合成 prior にない構造になっている可能性が高い。

実際に Chris Deotte の TabPFN-3 公開 kernel でも、s6e6 では単独で 0.96 帯にとどまっており、
スタックでも他 NN ほどの寄与にはなっていないという報告がある (公開コメント参照)。

### subsample を増やしても解決しない仮説

TabPFN は in-context model なので「学習データを増やす」= context window に詰め込む量を
増やすことに相当する。`ignore_pretraining_limits=True` で 25k を入れていたが、これを
100k / 200k に増やせば改善するか?

予想は **No**:

- TabPFN は in-context attention が事前学習された embedding 経由でしか働かない
- pre-training で見たことのない feature 関係には embedding 自体が表現力不足
- context を増やしても embedding が同じなら、attention の対象が増えるだけで
  「未知の特徴量関係を学習できる」わけではない

加えて TabPFN の inference は context size に対して **O(n²) メモリ**。
25.8 GB VRAM で 100k は OOM リスクが高く、200k はまず無理。
試す前から構造的限界が見えている、というのが今日の判断。

---

## まとめ

| 罠 | 性質 | 対処 |
|---|---|---|
| HF anonymous DL hang | 運用的 (技術的に解決可能) | prefetch helper で curl 直叩き、`HF_TOKEN` 設定 |
| CV-LB gap 0.0146 | 本質的 (prior と domain の不一致) | 諦めて別アプローチへ |

s6e6 では Foundation NN を主軸から外し、Chris の `RealMLP_v5` (LB 0.96979 単独実証) の
移植路線に切り替える、というのが今日の意思決定。

教訓を一行にまとめると、**「Foundation Tabular Model だから上がる」と思考停止せず、
最初の 1 fold で必ず LB を測る** こと。CV だけで判断すると 1 日半潰す。

書いたコード自体 (`agents/kaggle_agent/playbook/tabpfn_member.py` の wrapper + batch predict
+ prefetch helper) は次の tabular コンペで TabPFN が効くタイプの問題に当たったときに
再利用できるので、丸損ではない。だが「s6e6 で Gold を取る」目的には届かなかった、
という事実は事実として残す。

明日は RealMLP_v5 の移植に切り替える。
