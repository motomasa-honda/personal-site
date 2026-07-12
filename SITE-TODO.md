# サイト公開前に記入・確認が必要な項目

RM-Engineering サイトを個人事業主向け事業サイトとして公開する前に、以下を実際の値に差し替えてください。
サイト内では `【◯◯を記入】` の形式でプレースホルダとして埋め込んであります(`grep -rn "記入" *.html` で全箇所を検索可能)。

## 優先度 ★★★ (特定商取引法・必須)

- [ ] 代表者氏名フルネーム(漢字表記必須、屋号・ローマ字・苗字のみ不可) → `tokushoho.html`, `about.html`
- [ ] 連絡先メールアドレス → `tokushoho.html`, `contact.html`, `about.html`, footer 全ページ
- [ ] 3パッケージの実際の価格(下限確定) → `services.html`, `tokushoho.html`
- [ ] 支払時期・引渡し時期の具体的日数 → `tokushoho.html`

## 優先度 ★★☆

- [ ] 所在地(または「請求により遅滞なく開示」運用にするか確定) → `tokushoho.html`
- [ ] 電話番号(または「請求により遅滞なく開示」運用にするか確定) → `tokushoho.html`
- [ ] 予約カレンダーURL(Cal.com 等) → `contact.html`
- [ ] 開業日 → `about.html`, `tokushoho.html`
- [ ] 現在の受注可能枠(残り◯社、更新日) → `index.html`, `services.html`

## 優先度 ★☆☆

- [ ] X (Twitter) / Zenn / Qiita の実URL(現状 zenn.dev / qiita.com のトップページのままプレースホルダ) → 全ページ footer, `about.html`
- [ ] Founder's Story 本文(現状は簡易版のまま) → `about.html`
- [ ] 代表者写真 or アバター画像 → `about.html`
- [ ] お問い合わせフォームの送信先(現状 mailto: リンクのみ。Resend 等のAPI連携は未実装) → `contact.html`
- [ ] コマンドセンター(KRS-OS運用画面)のスクリーンショット画像 → `apps.html` の Portfolio セクション(現状はテキストのみ、画像未挿入。LAN内IPは意図的に非公開)

## 実装メモ

- 今回の改修は既存のプレーン HTML/CSS/JS 構成のまま実施(Next.js 移行はせず)。
- `llms.txt` を追加。動的生成ではなく静的ファイル。
- 特商法ページは「請求により遅滞なく開示」のテンプレート文言を採用済み — 所在地・電話番号を公開するかは代表者の判断で `tokushoho.html` を編集。
- 旧記事のうち、企業向け営業サイトとして公開するには具体的すぎる内部実装の欠陥・脆弱性の記述を含むものは表現を調整済み(詳細は git log 参照)。
