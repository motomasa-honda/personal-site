---
title: "Silver まで 1 千分の 1 で届かなかった話 — vote 飽和と diversity の見極め"
emoji: "🥈"
type: "tech"
topics: ["kaggle", "ensemble", "voting", "stacking", "competition"]
published: true
publication_date: "2026-06-22"
---

## TL;DR

playground-series-s6e6 で 10/10 quota を使い切り、ベスト **LB 0.97226 / rank 132 (top 6.0%)**
で着地。**Silver cut (0.97227, rank 110) まで 0.00001 (= 1 千分の 1)** 短い、痛い near miss。

10 submission の sweep で分かったこと:

| 学び | 数字 |
|---|---|
| alpha (vote vs own) は完全飽和 | 0.1/0.2/0.3/0.4 全部 0.97206-0.97207 |
| 我々の独自 OOF (3 GBDT) は意味があった | Chris only 0.97006 → +ours 0.97207 で **+0.00201** |
| 公開 acquirer の OOF が augmentation で leak | OOF 0.96990 → LB 0.97027 (gap 異常 0.00037) |
| voter pool の質 > 我々の stack の質 | fachri 0.97226 単体 > 我々の 9-model stack 0.97207 |
| top voter 3 件は 100% 一致、crowd diversity 飽和 | 247,427/247,435 行が unanimous |

「voter を 1 件直接出すと自分の stack を超える」という結論。これは Kaggle の Playground 系では
**公開資産の vote-blend が個人の stack 努力を凌駕する**現象が起きやすいことを示している。

## 経緯: 朝の Bronze 安定 → 夜の Silver 一歩前

### 午前 (1-2 回目): 朝の baseline 確認

quota reset 直後の優先候補は 14-model stack のvariation。前夜の検証で `config_B` (Chris 6 +
own 3) が calib OOF 0.96976 で best と判明していたので、alpha=0.2 vote blend を提出:

- 1st: morning a2 (config_B alpha=0.2) → **0.97207**

これは前日 strong13 と同値。Bronze (~146 位、top 6.91%) 維持。

### 午後 (3 回目 ~ 5 回目): direction 確認のための診断 sweep

「Gold cut (0.97230) まで +0.00023」を狙うにあたり、何を改善すれば届くかを見極めるための
診断的 3 提出:

- 2nd: a1 (alpha=0.1, voter heavy) → 0.97207  // alpha sweep flat 確定
- 3rd: **config_A (Chris only, no own)** → **0.97006** // own 貢献 +0.00201 確認
- 4th: a4 (alpha=0.4, own heavy) → 0.97206 // alpha 全域 flat 完全確定

3rd の **0.97006** は痛快だった。我々の 3 GBDT (LGBM/XGB/CatBoost) を抜くと LB が 0.00201 下がる
= 我々の stack は無駄ではない、ちゃんと効いている。

ただし alpha sweep が完全飽和 (0.1/0.2/0.3/0.4 全部 0.97206-0.97207) なので、**Vote blend
スライダーをいくらいじっても Gold 圏には届かない**ことが確定。

### 夜 (6 回目): acquirer 緩和の失敗

汎用 `external_oof_acquirer` の検索閾値を緩めて、より多くの公開 OOF を取り込むことを試した。
新発見:

- `lzsecurity/s6e6-high-scoring-base-model-oof-predictions` (30MB)
  → cat / lgb / xgb の midnight series OOF が同梱
  → 個別 BAC が 0.9619 - 0.9654 で Chris (0.961-0.968) に近い

これを 12-model stack (Chris 6 + own 3 + lz 3) に投入したところ:

- **stack calib OOF = 0.96990** (config_B の 0.96976 を超え best)

しかし提出してみると:

- 5th: **config_E (Chris6 + lz3) → 0.97027** // **-0.00180**!

OOF と LB がかけ離れている。CV-LB gap は通常 0.00231 だが、config_E では **0.00037 と異常に小**。
これは典型的な **augmented data leak**: lzsecurity が train を SDSS17 外部データで補強した
状態で OOF を計算していて、OOF score は膨らむが test 分布には乗らない。

**取り込みは failed**。教訓: **CV-LB gap が他の config と大きく違う時はその config を捨てる**。

### 夜 (7-10 回目): voter pool 切替で +0.00019 突破、しかし Silver 一歩前

acquirer 緩和で別の発見もあった: fachri00 の submission 群 (157 ファイル) に **LB 0.97226
の単体ファイル**が眠っていた。これを voter_pool として再構成:

```python
voters_top15 = [0.97226, 0.97224, 0.97223, 0.97220, 0.97217, 0.97214, 0.97210, ...]
```

純 crowd vote (own 抜き) で提出:

- 6th: **voteonly fachri top15 → 0.97224** // **+0.00017 突破**

ようやく plateau (0.97207) を超えた! あとちょっとで Silver cut (0.97227) かと思って…

- 7th: voteheavy a9 (90% crowd + 10% own configB) → **0.97040** // **-0.00167**!

これは own を **one-hot encode で混ぜた**のが致命的だった。crowd の薄い prob 差 (0.4-0.5 vs 0.3-0.4)
を、own の 0.1 質量 (= one-hot で 1.0 を 0.1 に減衰) が簡単に上書きしてしまう。tight row で
own の弱い判定 (0.97207) が支配的になり、crowd の優秀さ (0.97224) を毀損した。

修正方針として top voter の concentrate を試す:

- 8th: top5_concentrate (weight=10000 で top 5 のみ加重) → 0.97224 // 横ばい

ここで衝撃: **top 3 voters の 247,427 / 247,435 (100.0%) が完全一致**、disagreement はわずか 8 行。
crowd の "wisdom" がほぼ自明 (どの voter も同じ予測) で、diversity の意味が薄かった。

最後の手:

- 9th: **fachri 0.97226 (単体ファイル) 直接提出** → **0.97226** // **+0.00019 達成、本日 best**
- 10th: 3-source diverse majority (fachri 226 + 224 + nina 0.97183) → 0.97224 // diluteで -0.00002

**結局、自分の頭で作った 9-model stack より、fachri 単体の submission 1 つの方が LB が高かった**。
9th の 0.97226 が本日ベスト確定、Silver cut (0.97227) まで **0.00001 短**。Bronze 上端で着地。

## なぜ Silver に届かなかったか — 構造的考察

### 1. voter pool の最高値が Silver cut そのもの

fachri の最高 submission の LB = 0.97226 = Silver cut (0.97227) -1 unit。voter 群の crowd vote
は **個別最大を超えない**(基本的に individual の max が天井) ので、この voter pool だけで
Silver は無理だった。

### 2. 我々の stack が voter に届かない

我々の 9-model stack (Chris6 + own3) の OOF 0.96976 → LB 0.97207 = voter pool 最大 0.97226 より
**0.00019 低い**。voter のほうが「Stack を組まずに直接出した予測」として優れていた。

なぜそうなったか:
- voter (fachri) は OWN の専門知識 + 膨大な reattempt iterations の中から best を選んでいる
- 我々は **汎用 Playbook** で公開 OOF + 自分 OOF を機械的に stack しただけ
- 個別最強の力 (= 専門家の手作業) > 汎用パイプラインの自動 blend、というのが現状

### 3. crowd diversity の枯渇

3 つの top voter が 100.0% 一致 = **crowd を作っても新情報がほぼ生まれない**。
これは「voter pool が高度に同質的」を意味する: fachri 自身の iteration なので submission 間で
情報が漏れている (本人の中で best 予測に絞り込まれてる)。

異 submitter (nina2025 max 0.97183) を混ぜても 170 行差分しか生まれず、それも nina が低スコア
ゆえ多くは nina が誤りだった (混ぜると dilute する)。

## では Gold まで持っていくには

今夜の sweep で **「公開資産 vote blend では Silver cut (0.97227) すら届かない」** が分かった。
Gold (0.97230) のためには:

1. **より強い voter** が必要。0.97226 voter は全市場の中で max 級なので、自分で 0.97230+ を
   作れる単体モデルを持っていない限り、voter 経由で gold は無理。
2. **独立な強い base model** を 1 系統追加する。Foundation NN (TabPFN-3 / TabICL) や
   特化チューン NN (Chris の RealMLP_v5 完全移植) が候補。
3. **OOF と test の整合性チェック** を必ず入れる。今回 acquirer 緩和で config_E が leak で死んだ
   ので、`CV-LB gap == 既知 config の gap と類似` を assert すべき。

## 学び: 「自分の stack」vs 「公開単体 submission」

これは Kaggle Playground Series 全般に通じる教訓だと思う:

- **Playground は完成度の高い公開資産が出回りやすい** (Chris Deotte / fachri / nina など)
- これらの voter を単純に最高値で submission するのが、**自前 stack より高い LB を出す**ことが
  しばしばある
- 自前 stack の意味は **(a) 引用元がない非標準 metric** か、**(b) Foundation NN のような
  公開されていない新 modality** か、**(c) コンペ後半の overfit を吸収する hedge** あたり

メダル対象コンペ (Featured / Research) なら公開資産は少なく、自前 ML 力が物を言う。
Playground でも Featured と同じ訓練を積んだほうがいい (= 自前で 0.97230+ を作る力をつける)、
という方向に納得した。

## 結論

- 本日 best LB **0.97226 (rank 132 / 2208, top 6.0%)**, Bronze
- Silver cut **0.97227 まで 0.00001 short** (1 千分の 1)
- 10 提出で **alpha sweep / own 貢献 / acquirer 緩和 / voter 切替 / crowd diversity** の 5 軸を
  全部診断できた
- 翌日 quota reset では **TabPFN-3 Foundation NN を playbook 化** する予定 (期待 +0.001-0.003)。
  Gold cut 0.97230 = +0.00004 = Foundation NN なら届く射程

「LB 0.00001 で Silver を逃した」のは悔しいが、診断的価値は十分あった。
特に **「公開資産の単体提出 > 自前 stack」** が実証できたのは、今後の Kaggle 戦略の判断軸になる。
