# personal-site

RM-Engineering（Recursive Motomasa Engineering）のサイト一式。
ビルドステップの無いプレーンな静的サイトで、Vercel にデプロイしている。

**本番:** https://personal-site-eight-delta.vercel.app/

## サイト構成

1つのデプロイに、意図的にトーンと配色を分けた2つのサイトが同居している。

| パス | 中身 | 配色 |
| --- | --- | --- |
| `/`（`index.html` ほかルート直下の各ページ） | RM-Engineering の事業サイト。企業向けAI-OS「KRS-OS」の導入を訴求する | 青 / 紫 |
| `/career/` | 転職用エンジニアポートフォリオ。Hero/About/Skills/Projects/Contact の1ページ構成 | 緑 / シアン |
| `/blog.html` + `/posts/*.md` | 上記2つの両方から参照される共有の技術ブログ | — |

公開前の残タスク（連絡先メールの差し替えなど）は [SITE-TODO.md](SITE-TODO.md) にまとまっている。

## 技術構成

- プレーンな HTML + CSS + JavaScript。**ビルド無し・npm 依存無し**（`package.json` も `vercel.json` も無い）
- CSS は手書き。`css/site.css` 冒頭の CSS 変数（背景スケール・テキスト4段階・アクセント6色）がデザイントークンになっている。**Tailwind は使っていない**
- JS は `js/main.js` の1本のみ。ナビのアクティブ表示、モバイルメニュー開閉、ブログのタグフィルタだけを担当する
- 外部依存は CDN 経由の2つだけ — Google Fonts と `marked@12`（Markdown 描画用。`post.html` でのみ読み込む）

```
├── index.html          # トップ（事業サイト）
├── product.html        # KRS-OS
├── services.html       # 料金パッケージ
├── approach.html       # 進め方・SLA
├── apps.html           # 実績・ポートフォリオ
├── about.html          # 代表者・事業者情報
├── contact.html        # 問い合わせ（現在フォームは無効化中）
├── privacy.html / terms.html / tokushoho.html   # 法務ページ
├── blog.html           # 記事一覧（カードは手書き）
├── post.html           # 記事ビューア（?slug= で posts/*.md を描画）
├── career/index.html   # 転職用ポートフォリオ
├── posts/*.md          # 記事本体（Markdown のまま配信）
├── css/site.css        # スタイル（全ページ共通・唯一のSSOT）
├── js/main.js
├── llms.txt            # LLM 向けサイト要約（手書き・静的）
├── robots.txt / sitemap.xml
└── SITE-TODO.md        # 公開前チェックリスト
```

## ローカルで動かす

リポジトリのルートを HTTP で配信するだけ。ビルドは不要。

```bash
python3 -m http.server 8000
# → http://localhost:8000/
```

`index.html` を `file://` で直接開くと正しく動かない。リンクや CSS が絶対パス（`/css/site.css`）で書かれている上、`post.html` が `fetch('/posts/<slug>.md')` で記事を読み込むため。必ず HTTP 経由で開くこと。

## ブログの仕組み

記事は `posts/*.md` に置いた Markdown ファイルそのもので、HTML への事前変換はしていない。

1. `blog.html` に記事カードが**手書きで**並んでいる
2. カードのリンク先は `/post.html?slug=<ファイル名から .md を除いたもの>`
3. `post.html` が `/posts/<slug>.md` を fetch し、YAML frontmatter を簡易パースして `marked` で描画する

つまり **記事一覧（`blog.html`）と記事本体（`posts/`）は自動では繋がらない**。`.md` を置くだけでは、URL を直接叩かないかぎり誰もその記事に辿り着けない。

### 記事を追加する手順

**1. `posts/` に Markdown を追加する**

ファイル名は `YYYY-MM-DD-英語スラッグ.md`（slug に使えるのは英数字・ハイフン・アンダースコアのみ。それ以外は `post.html` 側で弾かれる）。

```markdown
---
title: "記事タイトル"
emoji: "🧩"
type: "tech"
topics: ["llm", "python", "ollama"]
published: true
publication_date: "2026-07-14"
---

## TL;DR

...
```

| キー | 使われ方 |
| --- | --- |
| `title` | 記事見出しと `<title>` |
| `emoji` | 見出し上に表示されるアイコン |
| `topics` | 記事上部のトピック表示（配列） |
| `publication_date` | 記事上部の日付表示 |
| `type` / `published` | **`post.html` は読んでいない**。Zenn と同じ形式なので残っているだけで、`published: false` にしても非公開にはならない（記事を隠すならカードを消し、`.md` 自体を置かないこと） |

**2. `blog.html` にカードを追加する（忘れやすい）**

既存カードをコピーして、slug / タイトル / 抜粋 / 日付 / タグを差し替える。

```html
<a href="/post.html?slug=2026-07-14-your-slug" class="blog-post-card" data-tags="llm python debugging">
  <div class="blog-post-main">
    <div class="blog-post-meta">
      <span class="tag tag-blue">Infra</span>
      <span class="tag tag-cyan">LLM</span>
    </div>
    <h2 class="blog-post-title">記事タイトル</h2>
    <p class="blog-post-excerpt">抜粋…</p>
  </div>
  <div class="blog-post-side">
    <span class="blog-post-date">2026.07.14</span>
  </div>
</a>
```

タグは2系統あるので注意:

- `data-tags` … フィルタ用の**機能的な**文字列。`blog.html` 上部のフィルタボタン（`all` / `infra` / `kaggle` / `krs-core` / `langgraph` / `llm`）と部分一致で判定される。該当語が無いと、そのフィルタを押したときにカードが消える
- `<span class="tag tag-*">` … 見た目だけの表示用ラベル。`data-tags` とは連動していないので、両方に書く必要がある

## デプロイ

Vercel の静的ホスティング。ビルドコマンドも `vercel.json` も無く、リポジトリのルートがそのまま配信される。Git 連携により `main` への push で本番に反映される。

## 注意点 / 既知の状態

- **記事一覧に未反映の記事がある**（2026-07-17 時点で3本）。`.md` だけ追加して `blog.html` の更新を忘れたもので、サイト上のどこからもリンクされていない。以下で検出できる:

  ```bash
  # 左が blog.html のカード、右が posts/ の実ファイル。"> " が付いた行が一覧に出ていない記事
  diff <(grep -o 'post\.html?slug=[^"]*' blog.html | sed 's/.*slug=//' | sort -u) \
       <(ls posts/*.md | sed 's|posts/||; s|\.md$||' | sort -u)
  ```

- スタイルは `css/site.css` の1本のみ。旧 `css/style.css` は現行と矛盾するトークン（`--accent`/未読込の Syne 等）を持つ死蔵ファイルだったため削除済み
- `sitemap.xml` は手書きで、**記事の URL を1本も含んでいない**（トップレベルのページのみ）。記事を検索対象にしたいなら追記が必要
- ナビゲーションとフッターは13枚の HTML それぞれにコピーされている。ナビ項目を変えるときは全ファイルを直す必要がある
- 連絡先まわりは事業用メールアドレスの開設待ちで「準備中」表示のままガードしてある。`contact.html` のフォームは `<fieldset disabled>` で送信不可。解除手順は [SITE-TODO.md](SITE-TODO.md) を参照
