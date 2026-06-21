---
title: "Bronze は自走できる、Gold は構造投資が要る — 汎用 Kaggle OS の 2 つのギャップ"
emoji: "🪜"
type: "tech"
topics: ["kaggle", "machinelearning", "automl", "rag", "agent"]
published: true
publication_date: "2026-06-22"
---

## TL;DR

「過去コンペの経験を貯めて、新しいコンペで自走的にスコア改善する OS」を作っていて、
直近 1 か月で **Bronze 圏 (top 6.91%) を自走で取れる**ところまで来た。
ただし **Gold (top 1%) はこの設計の自然延長では届かない**ことが s6e6 の検証で確定した。

理由を 2 つのギャップに分解する:

1. **Skill 粒度ギャップ**: 抽出される skill が「kernel まるごとの塊」になっており、
   別コンペにそのまま転移しない。必要なのは「**条件 → 処方 → 反証**」の三項に抽象化された
   meta-pattern。
2. **AutoML ループギャップ**: skill を retrieve しても、新コンペ固有のハイパラ最適化を
   する仕組みがない。Chris の RealMLP_v5 (LB 0.969) は **PBLD embedding + 5-group LR +
   n_ens=8 + label smoothing** の**同時最適化**が肝で、1 つ欠けると 0.952 に落ちる。
   skill は「初期値の hint」止まりで、その先の探索ループが要る。

これは個別コンペのチューン作業を続けていてもいつまでも埋まらない構造ギャップなので、
KRS-OS 側の方針を「**コンペ専用チューンを書く**」から「**チューンを生成するエンジンを書く**」
に切り替える、という意思決定の記録。

---

## 背景: 自走で Bronze まで来た

ここ 1 か月、Playground Series s6e5 / s6e6 で KRS-OS (Kaggle Research System) の自走能力を
段階的に伸ばしてきた。s6e6 (3 クラス分類, balanced_accuracy) では人間がコンペ固有のチューンを
書かずに以下の経路で **LB 0.97207 / rank 146/2179 (top 6.91%) = Bronze 圏**に到達した:

- **External OOF Acquirer**: Kaggle 上位 kernel を自動スキャンして公開された OOF を取得
- **Logit Stacker**: 多項 LR で base model 群を stack
- **Submission Voting**: 公開高得点 submission との加重多数決
- **Branch Explorer + Knowledge Loop**: skill library から検索した上位陣テクを Planner に注入

これは「汎用 Kaggle OS」のプロトタイプとして十分手応えがある。**特定コンペの専門知識ゼロから
Bronze まで自走できる** — これは正直、想定より良い数字だった。

## 壁: Gold (top 1%) には届かない

Silver cut (0.97226) まで **+0.00019**、Gold cut (0.97230) まで **+0.00023** の所で
plateau。残された一手として「**強い単独 NN model を 1 つ stack に追加**」を試した。

採用したのは `pytabkit.RealMLP_TD_Classifier` (Tuned Defaults)。汎用 tabular で広く効くと
報告されている。だが結果は OOF **0.952** (vs Chris の RealMLP_v5 = 0.969、GBDT 群 = 0.957)。
stack に投入しても LB 改善は **+0.00002** (誤差範囲)。詳細は別記事
([「強い NN を 1 つ足せば Silver 圏」と踏んで届かなかった話](./2026-06-22-realmlp-default-was-not-enough)) に書いた。

つまり「**汎用チューン済み NN を使えば近づくだろう**」という仮定が崩れた。

## 何が違うのか — 2 つの構造ギャップ

Chris の RealMLP_v5 (公開 kernel) は 984 行の **s6e6 特化チューン**だ:

- **PBLD (Periodic Basis with Learned Decay) embedding** for numeric features
- **5-group LR scheduling** (scale / pbld / first_w / other_w / bias 別レート)
- **n_ens = 8** (我々は 4 まで)
- **Label smoothing + custom dropout schedule**
- **Integer-floor categorical view of numerics + multiclass target encoding**

これらの 1 つでも欠けると 0.969 が 0.952 に落ちる、というのが pytabkit default で実証された
事実だ (n_ens=4 でも +0.00425 しか動かなかった)。**これは「skill を持っていなかった」のでは
なく、「同時最適化の探索ループを持っていなかった」**。

ここで自分の頭の中で 2 つのギャップに分解した。

### ギャップ 1: Skill 粒度が「kernel の塊」になっている

現状の skill 抽出は、過去コンペの上位 kernel を LLM (Qwen3 70B) に流し込んで以下を抽出している:

```
- title: "TargetEncoding with smoothing"
- category: feature_engineering
- description: "なぜ効くか / どこで使うか" (150字)
- code_snippet: 30行以内のコード片
- when_to_use: 適用条件
```

問題はこの粒度だと **「Chris の v5 の塊」がそのまま 1 個の skill になる**こと。次のコンペ
(別ドメイン、別 metric、別 feature distribution) で retrieve しても、塊ごと適用はできない。

本来必要なのは **「条件 → 処方 → 反証」の三項に分解された meta-pattern** だ。例えば:

```
- title: "PBLD embedding for skewed numerics"
- conditions: {data_shape: tabular, n_numeric: >=4, skewness: ">1.0"}
- prescription: "PBLD(Periodic Basis with Learned Decay) を numeric column の入力 layer に挟む"
- mechanism: "通常の linear で表現しにくい非線形周期性を 学習可能 basis で吸収"
- anti_conditions: "n_numeric < 4 / 全て binary / 全て integer counts"
- expected_effect: "RealMLP backbone で +0.005-0.015 OOF (multiclass classifier)"
- code_snippet: "..."
```

この形なら、**新コンペで「numeric が多くて skew が大きい」と判定された瞬間に retrieve できる**。
かつ anti_conditions と expected_effect があるので、planner は「**期待効果がコストに見合うか**」を
構造化された判断ができる。

現実装の `when_to_use` 自由テキスト 1 行ではここまで届かない。

### ギャップ 2: AutoML 探索ループがない

仮にギャップ 1 を埋めて最適 skill を 10 個 retrieve できても、それらを **新コンペで正しく
組み合わせる**ところは依然手作業だ。Chris の 700 行は「PBLD + 5-group LR + n_ens=8 +
label smoothing + ...」の**同時最適化**で、1 個でも欠けると 0.969 が 0.952 に落ちる、
ということが実証された。

ここを自走させるには:

- skill を「**初期値の hint**」として渡す
- その上で **Optuna が新コンペ固有のチューン**をやる
- 1 trial = 5-10 分の **fast-CV proxy** で 200-500 trial 回す
- **「LB を予測できる較正済み内部 CV」**が必須 (これがないと Optuna が誤った方向に最適化する)

最後の「LB を予測できる内部 CV」が一番厄介だ。現状の CV-LB 相関 tracker (s6e5 → 0.70 Spearman)
は方向の正しさだけ確認している。**「実 LB ≈ a · OOF + b の較正済み回帰」**まで踏み込まないと、
Optuna に渡せる objective にならない。

## なぜ「skill を貯めても」自然に Gold に届かないか

ここまで書いて気付いたのは、「skill ストックを増やせばいつか届く」という素朴な仮説が
間違っているということだ。理由は 3 つある:

1. **コンペ固有のチューンは原理上 transferable ではない**。Chris の v5 の hidden_sizes は
   s6e6 用に決まっているし、5-group LR の比率は s6e6 の feature distribution に依存する。
   この値そのものを skill 化しても、別コンペでは使えない。
2. **何を残して何を捨てるかが事前に分からない**。「PBLD は効くが label smoothing は効かない」
   みたいな分岐は、新コンペで実測しないと決まらない。**実測 = 探索ループの仕事**であって、
   skill ストックの仕事ではない。
3. **Stack 段の汚染問題**。今回 14-model stack に弱 RealMLP v1 を入れたら **-0.00006 汚染**した。
   多項 LR は弱 model に小 weight を振るが、完全 zero にはならない。**L1 sparse stacker or
   member 自動 reject ゲート**が要るが、これは skill ではなく構造の仕事。

つまり「**Skill は問題空間の地図、AutoML ループは現地調査隊**」という分業が必要で、現 KRS-OS は
地図 (skill) だけで現地調査隊 (探索ループ) を持っていなかった。

## 方針転換: チューンを書くのではなく、チューンを生成するエンジンを書く

これらを踏まえて、今後 6-9 週間の構造投資の優先順位を切り替える:

| 機能 | 現状 | 必要レベル | 優先 |
|---|---|---|---|
| Skill 粒度 (meta-pattern: 条件/処方/反証/効果) | kernel 固有塊 | 抽象化原則 | **P0** |
| AutoML 探索ループ (Optuna × fast-CV proxy) | 無 | あり | **P0** |
| 内部 CV → LB 較正済み予測器 | tracker のみ | 較正済み回帰 | P1 |
| コンペ間 transfer 機構 (効果ペア記録 + 類似度推薦) | 無 | あり | P1 |
| Stack 自動 reject ゲート (L1 sparse / 弱 member) | 無 | あり | P2 |

これを進めると何が起きるか:

- **特定コンペで Gold を取る**ためのコードを書くのを **やめる** (今までやってきた手作業)
- 代わりに **「コンペが与えられたら Gold を取りに行くエンジン」**を書く
- 結果として s6e6 では Bronze 維持で割り切る可能性がある。短期 LB と長期能力のトレードオフ

これは正直、こだわって書いてきた個別チューンを手放す判断でもある。だが s6e6 で**「default
ハイチューン NN」というショートカットが幻想だった**ことが分かった以上、構造を直すしかない。

## 何ができれば Gold が射程に入るか (定量)

私の見立てでは、以下 4 つが揃ったとき「**コンペの 30-40% で Gold**」が射程:

1. **Meta-pattern 粒度の skill が 200+** (現 349 を再 distill + 新規追加)
2. **AutoML 探索ループが 200-500 trial / 6h で完走**
3. **内部 CV → LB の予測誤差 < 0.001 (絶対値)**
4. **コンペ間 transfer 機構が「過去 5-10 コンペから effect-validated skill だけ retrieve」**

合計工数感は **6-9 週間**。s6e6 締切 (2026-06-30) には間に合わないが、次のコンペには間に合う。

## 期待される副次効果: 「Kaggle Master ルート」のほうが先に達成される

最終目標 = Kaggle Grandmaster だが、「Gold を取る」を最終 KPI にしない戦略もある。

**「Top 10% を 10 コンペ連続で出せる OS」**のほうが、商品としても研究としても価値が高いし、
達成可能性も高い。Kaggle Master (Silver 多数 + Gold 1) はその先で自然と取れる。

しかも「Top 10% を 10 コンペ連続」のほうが汎用 GM-OS の検証としては強い: 1 コンペで Gold は
運の要素があるが、10 コンペ連続 Top 10% は **再現性のある能力**でなければ説明できない。

## 次のアクション

今日からやること:

1. **ブログを書いて方針宣言** (この記事) ← イマココ
2. **Skill 抽出 prompt v2 を書く** (meta-pattern 粒度 + 新フィールド)
3. **Chris の RealMLP_v5 で「新旧 prompt 比較実験」を実機検証**
4. **AutoML 探索ループの skeleton 設計** (Optuna interface 定義まで)

s6e6 の残り 8 日は、quota は朝の検証 1-2 本だけ使い、**残りは全部この構造投資に振る**。Bronze
維持を狙う最低限の提出だけ確保して、Gold lottery には付き合わない判断。

## おわりに

Kaggle に出会ってから「過去コンペの知見を貯めれば、自然に GM になれる OS が作れるはず」と
信じてやってきた。s6e6 で Bronze まで自走できたことで、その信念は正しかったと確認できた。
**ただし Bronze と Gold の間には、知識ストックの量では埋まらない構造ギャップがある** —
これも同時に確認できた。

「知識を増やす」から「知識の使い方を最適化する探索エンジンを作る」へ。今日が KRS-OS の一番
大きな pivot 点だと思う。記録として残す。
