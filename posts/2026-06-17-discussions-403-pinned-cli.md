---
title: "ピン留めした CLI が discussions を黙って 403 にしていた — 本体を上げずに一機能だけ逃がす"
emoji: "🔌"
type: "tech"
topics: ["kaggle", "mlops", "cli", "api", "knowledge"]
published: true
publication_date: "2026-06-17"
---

## TL;DR

- 知識ループの「上位解法 discussion (write-up) をマイニングして skill 化」する経路が、何週間も 0 件だった
- 原因は外部 CLI のバージョン。古い系列 (2.1.2) では discussions API が **403 Forbidden**。コンペ一覧やカーネル取得は通るのに、discussions エンドポイントだけが版固有で死んでいた
- かといって本体 CLI を上げるのは怖い。提出 / カーネル / コンペ取得が CLI に依存していて、メジャー更新は別のところを壊しうる
- 解にしたのは **「本体は据え置き、discussions だけ新版バイナリに逃がす」**。別 venv に新しい CLI を入れ、環境変数でそのパスを指す。該当機能だけがそこを使う
- 結果、メダルコンペの `1st Place Solution` / `2nd place solution` 等の write-up 本文を取得できるようになり、1 周のハーベストで discussion 由来の skill が +27 件

## 症状: 一覧は通るのに discussions だけ 403

知識ループの書き込み側を点検していて、discussion マイニングが一件も成果を出していないことに気づいた。手で叩くと、こうなる。

```
competitions list   → OK (コンペ一覧が返る)
kernels list/pull   → OK (カーネルが取れる)
competitions topics → 403 Forbidden
                      .../discussions.DiscussionApiService/ListTopics
```

認証は生きている (一覧は返る)。なのに discussions のエンドポイントだけが 403。しかも CLI 自身が「新しい版に上げては」と警告を出していた。つまり **discussions API が版を跨いで移設され、古い CLI からは触れなくなっていた**。サブコマンドの引数体系も版で変わっていて (`topics <slug>` → `topics list <slug>`)、見かけ上は「コマンドが壊れている」ようにしか見えない。

## 本体を上げたくない理由

素直なのは CLI をまるごと新版にすること。だが本番フローは CLI に深く依存している — コンペデータの取得、カーネルの一覧/取得、提出。メジャー更新は別の挙動 (出力フォーマット、廃止されたサブコマンド) を予告なく変える。実コンペ参戦を控えたこのタイミングで、土台の CLI を入れ替えてリグレッションを踏むのは割に合わない。

一方で、**提出は CLI バイナリではなく Python SDK 経由**だと分かった。つまりバイナリを差し替えても提出は影響を受けない。使われ方を切り分けると「discussions の取得だけが新版を必要としている」とはっきりした。

## 解: 一機能だけ別バイナリに逃がす

やったことはシンプル。

1. 別の隔離環境 (専用 venv) に**新版 CLI だけ**を入れる。本体環境の CLI は古いまま据え置き
2. `KAGGLE_DISCUSSIONS_BIN` という環境変数で、その新版バイナリのパスを指す
3. discussion マイニングのコードだけ、CLI ラッパをこのバイナリで生成する。未設定なら従来どおり既定の CLI にフォールバック

```python
# discussions だけ新版バイナリへ。明示 env > 設定ファイル > 既定 の順で解決
disc_bin = os.environ.get("KAGGLE_DISCUSSIONS_BIN") or settings.discussions_bin or "kaggle"
client = KaggleClient(kaggle_bin=disc_bin)
```

完全に可逆だ。環境変数を消せば元に戻る。本体 CLI は一切触っていないので、提出もカーネル取得も従来のまま。リスクを「discussions マイニングが動くか否か」だけに閉じ込められた。

## ついでに直した: 出力 CSV をちゃんとパースする

新版の topics 一覧はタイトル列にカンマを含む。素朴な `line.split(",")` だと列がズレて votes を取り違える。識別子の列 (id) は先頭で安全だが、票数は壊れる。素朴な split をやめて csv パーサで読むようにした。外部ツールのテキスト出力は契約が緩い、というのは前にも踏んだ教訓。

## 直した後

メダルコンペで実データ検証すると、こうなった。

```
[discussion] ✅ '1st Place Solution - Part 1' (votes=613) → 2 skills
[discussion] ✅ '1st Place Solution - Part 2' (votes=398) → 3 skills
[discussion] ✅ '1st Place Solution'        (votes=466) → 3 skills
[discussion] ✅ '2nd place solution'        (votes=122) → 1 skills
```

write-up の本文 (特徴量設計、後処理、CV 戦略、教訓) がきちんと取れている。1 周のハーベストで discussion 由来の skill が +27 件。数週間ゼロだった経路が、ようやく機能した。

## 学び

- **「依存をピン留めしている」ことが、静かな機能停止の原因になりうる**。401/500 と違って 403 は「権限がない」に見えるので、版の問題だと気づきにくい。一覧が通るのに一機能だけ落ちるなら、エンドポイント単位の差分を疑う
- 怖い更新は「全体に適用」ではなく「**必要な一機能だけに逃がす**」と可逆になる。使われ方 (バイナリ経由か SDK 経由か) を切り分けると、影響範囲を最小化できる
- 外部 CLI の出力は素朴な split ではなく、フォーマット (CSV) のパーサで受ける
