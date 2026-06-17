---
title: "修理した知識ループを seed 9コンペで初めて1周させた — kernel +73 / discussion +27 skill"
emoji: "🌾"
type: "tech"
topics: ["kaggle", "mlops", "llm", "knowledge", "machinelearning"]
published: true
publication_date: "2026-06-17"
---

## TL;DR

- 数週間 no-op だった知識ハーベストと、403 で死んでいた discussion マイニングを直したあと、**seed 9コンペで初めて end-to-end の 1 周**を走らせた
- 結果: Kernel 由来 skill **+73**、Discussion 由来 skill **+27**、重複統合 -1、最終的に **349 件を PostgreSQL に sync**
- 抽出は終了済みの定番コンペ (titanic / house-prices / spaceship-titanic / digit-recognizer / nlp-getting-started / ieee-fraud / home-credit / amex / store-sales) を票数順にマイニング
- 所要 **約2時間 (7380s)**。ボトルネックは抽出 LLM が 70B 級で 1 要約あたり数分かかること。`skip_if_exists` で既マイニング分は即スキップされるので、再実行は安価
- 「直した」と「実データで結実した」は別。実際に 1 周させて初めて、書き込み側が機能していると言える

## 直してから、回す

ここ最近で知識ループの 2 つの詰まりを直した。ひとつは[ハーベストが数週間 no-op だった件](/post.html?slug=2026-06-17-knowledge-harvest-was-a-noop)、もうひとつは[古い CLI で discussions が 403 だった件](/post.html?slug=2026-06-17-discussions-403-pinned-cli)。どちらも「直した」とは言える。が、修正はユニットで確認しただけで、**全 seed コンペを通しで 1 周させたことはまだなかった**。

そこで実際に回した。対象は公開カーネルが豊富な終了済みコンペ 9 本。各コンペで上位カーネルと上位 discussion (write-up) をマイニングし、LLM で skill に要約 → 重複統合 → Obsidian インデックス再生成 → DB 同期、という一連を通す。

## 結果

```
[harvest] ✅ 完了 in 7380s
[harvest]   Kernel SKILL: +73
[harvest]   Discussion SKILL: +27
[harvest]   重複統合: -1
[harvest] skills→PG sync: 349件
```

discussion 側は、メダルコンペで `1st Place Solution` や `2nd place solution` の write-up 本文をきちんと拾えた。数週間ゼロだった経路が、実データで +27 skill を出した。これで知識ループの書き込み側 (収集 → 蒸留 → 保存) が、seed 全体に対して通しで動くことを確認できた。

## 所要時間という現実

正直に書くと、1 周に約 2 時間かかった。律速は明確で、**抽出に使っている LLM が 70B 級**で、1 つの要約に数分かかる。9 コンペ × (カーネル + discussion) で約 100 回の抽出 → 2 時間。

抽出は「ノートや write-up を要約して構造化する」タスクで、深い推論は要らない。ここに 70B は過剰だ。次の高速化は明らかで、**ハーベストの抽出ロールだけ軽量モデルに分離**すれば 2〜3 倍は速くなる。推論ロール (戦略立案) と抽出ロールでモデルを分けるのが筋がいい。

なお `skip_if_exists` のおかげで、既にマイニング済みのカーネルは再実行時に即スキップされる。途中で止めて設定を変えて回し直す、というイテレーションは安く回せる。

## 学び

- **「直した」と「実データで結実した」は別のマイルストーン**。ユニットの緑と、全体を 1 周通しての成果は分けて確認する
- バッチの評価軸は「貯まっているか」ではなく「**新しく良質なものが、通しで貯まり続けるか**」
- LLM パイプラインはロールごとに必要な能力が違う。要約/抽出に推論用の巨大モデルを使うのは時間の無駄。**ロール分離 = コストとレイテンシの設計変数**
- 冪等な再実行 (`skip_if_exists`) を最初から入れておくと、長時間バッチのチューニングが現実的になる
