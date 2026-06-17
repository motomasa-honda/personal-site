---
title: "自作の知識ハーベストが数週間 no-op だった — タイマーは回るのに何も学んでいなかった"
emoji: "🕳️"
type: "tech"
topics: ["kaggle", "machinelearning", "mlops", "llm", "debugging"]
published: true
publication_date: "2026-06-17"
---

## TL;DR

- 「過去コンペの上位解法を毎週マイニングして skill 化する」ハーベスト機構を持っていた。タイマーは毎日動いていた。が、ログを見たら **`competitions_found: 0` が数週間続いていた** — 一件もマイニングしていなかった
- 原因は連鎖していた。(1) コンペ発見が未来締切のアクティブコンペを返し「終了済」フィルタで全件落ちる、(2) CLI が返す ref がフル URL なのを slug 扱いして壊れたディレクトリを量産、(3) CLI の警告行が CSV ヘッダと誤認され全列が 1 行ズレて votes が常に 0、(4) その結果 votes=0 のチュートリアル演習を「上位解法」として拾う
- 4 つ全部直し、終了済コンペの seed + 実際に走らせたコンペを対象にしたら、ようやく titanic-tutorial(59170 票)/data-science-solutions(39979)/ensembling-stacking(15464) を**票数順にマイニング**するようになった
- 教訓: 「動いているように見える」自動化ほど疑え。タイマーの成功 ≠ 仕事の成功

## きっかけは「これ完成してるんだっけ?」

知識ループ (上位解法を蒸留 → skill ライブラリ → プランナーに供給) は一応動いている、と思っていた。skills.json には 232 件、Obsidian には 258 ノート、Qdrant も生きている。タイマーも毎日発火している。

念のためハーベストの実行ログを開いた。そこにあったのは:

```json
{ "competitions_found": 0, "competitions_processed": 0,
  "kernels_mined": 0, "discussions_mined": 0,
  "skills_before_consolidate": 207, "skills_merged": 0 }
```

`competitions_found: 0` が、何週間分も並んでいた。つまりハーベストは**毎回「対象コンペ 0 件」で、新規マイニングを一切していなかった**。既存 skill の重複統合だけを律儀に繰り返していたのだ。232 件の skill は、別経路 (ジョブ実行時の on-demand マイニング) が貯めたもので、目玉のはずの週次ハーベストは空回りしていた。

## 連鎖していた 4 つのバグ

### (1) 終了済コンペを探しているのにアクティブコンペが返る

発見コードは「最近終了したコンペ」を集める意図で、CLI をこう叩いていた。

```bash
kaggle competitions list --sort-by latestDeadline -p 1
```

返ってくる先頭は締切 **2030 年**の Getting Started 群。締切が未来なので「まだ終わっていない」と全件スキップされ、対象 0 件になる。CLI の並び順では終了済コンペがページ 1 に出てこない、という構造的なミスマッチだった。

→ 壊れた自動発見に頼るのをやめ、公開カーネルが豊富な**終了済コンペの seed** (titanic / house-prices / ieee-fraud / amex / home-credit など) と、**エージェントが実際に走らせたコンペ**を対象にした。

### (2) フル URL を slug 扱いしてディレクトリを汚染

CLI の ref 列は今や `https://www.kaggle.com/competitions/<slug>` というフル URL。これをそのまま保存パスに使い、`mined/https:/www.kaggle.com/competitions/...` という壊れた階層を量産していた。→ URL から最後のセグメントだけ取り出す正規化を入れた。

### (3) CLI の警告行が CSV ヘッダを乗っ取る

これが一番たちが悪い。kaggle CLI は標準出力の**先頭**にこの行を出す。

```
Warning: Looks like you're using an outdated `kaggle` version ...
ref,title,author,lastRunTime,totalVotes
alexisbcook/titanic-tutorial,Titanic Tutorial,Alexis Cook,...,59170
```

CSV パーサは 1 行目をヘッダとみなす。つまり**警告行がヘッダ扱いされ、本物のヘッダがデータ行に降格し、全列が 1 つズレる**。`totalVotes` 列が見つからず、votes は全件 0 になる。→ この警告行は CLI ラッパの出力段で除去した (全呼び出し元が一括で直る)。

### (4) votes=0 なので「人気のチュートリアル演習」を拾う

votes が全部 0 だと票数ソートが効かず、CLI が返した先頭をそのまま採用する。その先頭が `alexisbcook/exercise-arithmetic-and-variables` — Kaggle Python コースの**算術演習**で、ML 的価値はゼロ。学習者の fork で票だけは多い (513,186)。→ votes を自前で正しくパースして降順ソートし、`exercise-*` や votes=0 を除外した。

## 直した後

修正前は votes=0 の算術演習を「上位解法」として蒸留していた。修正後はこうなった。

```
titanic-tutorial                         votes=59170  → 2 skills
titanic-data-science-solutions           votes=39979  → 3 skills
introduction-to-ensembling-stacking      votes=15464  → 3 skills
```

ついでに読み出し側も直した。プランナーへの skill 供給がタスク非依存で、住宅価格 (回帰) の計画に Titanic (分類) の生存予測 skill が混ざっていた。タスク/メトリクスで関連付けるよう切り替えたら、回帰には Box-Cox 変換や target encoding が surfacing されるようになった。書き込みも読み出しも、ようやく意図通りに動く。

## 学び

- **「動いているように見える」自動化ほど疑え**。タイマーが緑でも、ログの中身 (`found: 0`) を見るまで no-op に気づけなかった。成功の定義を「実行した」ではなく「**有効な成果物が増えた**」に置く
- 外部 CLI のテキスト出力は契約が緩い。**警告行・フル URL 化・列順**は予告なく変わる。CSV は素朴な split ではなく csv パーサで、識別子は正規化して受ける
- 票数 0 で並ぶときの「先頭」は無意味。ソートキーが壊れていないかは、出力の中身で確認する
- パイプラインは「貯まっているか」ではなく「**新しく、かつ良質なものが貯まり続けているか**」で評価する
