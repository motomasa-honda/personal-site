# Motomasa Honda — personal-site

本多基真の個人エンジニアサイトです。飲食店で働いた経験から作り始めた「だんどり」を代表作に、制作物と技術記事を紹介するプレーンな静的サイトです。Vercel にデプロイしています。

**本番:** https://personal-site-eight-delta.vercel.app/

## サイト構成

記事・制作記録・必要な事業者情報を、1つの明るいデザインにまとめています。

| パス | 中身 | 配色 |
| --- | --- | --- |
| `/` | 本人の自己紹介、代表作「だんどり」、技術記事、他の制作物 | Warm Paper |
| `/apps.html` | 個人で作ったもの・技術実験 | Warm Paper |
| `/career/` | 経歴詳細の補足 | Warm Paper |
| `/blog.html` + `/posts/*.md` | 技術記事一覧と記事本文。既存slugとURLを維持 | Warm Paper |

公開前の残タスク（連絡先メールの差し替えなど）は [SITE-TODO.md](SITE-TODO.md) にまとまっている。

## 技術構成

- プレーンな HTML + CSS + JavaScript。**ビルド無し・npm 依存無し**（`package.json` も `vercel.json` も無い）
- CSS は手書き。`css/site.css` 冒頭の CSS 変数（背景スケール・テキスト4段階・アクセント6色）がデザイントークンになっている。**Tailwind は使っていない**
- JS は `js/main.js` の1本のみ。ナビのアクティブ表示、モバイルメニュー開閉、ブログのタグフィルタだけを担当する
- 外部依存は Markdown 記事描画用の `marked@12`（CDN、`post.html` のみ）。フォントはOS標準フォントへフォールバックし、Google Fontsには接続しない。

```
├── index.html          # 本人・代表作・最近の技術記事
├── product.html        # KRS-OS
├── services.html       # 相談できるテーマ
├── approach.html       # 個人開発で意識すること
├── apps.html           # つくったもの（だんどりが先頭）
├── about.html          # 自己紹介・必要な事業者情報
├── contact.html        # 連絡先（フォームは準備中）
├── privacy.html / terms.html / tokushoho.html   # 法務ページ
├── blog.html           # 記事一覧（カードは手書き）
├── post.html           # 記事ビューア（?slug= で posts/*.md を描画）
├── career/index.html   # 経歴詳細の補足
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
