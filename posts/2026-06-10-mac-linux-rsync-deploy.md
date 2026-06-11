---
title: "Linux→GitHub の SSH:22 が中間装置で塞がれていた話 — Mac mini を経由する rsync デプロイ運用に倒した経緯"
emoji: "🚀"
type: "tech"
topics: ["linux", "ssh", "deploy", "rsync", "infra"]
published: false
publication_date: "2026-06-10"
---

## TL;DR

- 個人開発で Mac mini M4 (開発・git 権威) と Ubuntu 24.04 LinuxPC (実行・GPU) のデュアル構成を組んでいる
- LinuxPC から `git push` しようとしたら **SSH:22 outbound が中間装置で握り潰されていた** (`banner: 6279353`、本物の GitHub は `babeld-XXXXXXXX`)
- HTTPS:443 は通る (Anthropic API / Kaggle API は動く)。PAT 経由の HTTPS push も検討したが、運用上の単純さを優先して **rsync 一本化**
- `git push` は必ず Mac から、Linux への展開は `scripts/deploy_from_mac.sh` (rsync + systemctl --user restart) という運用に固定

## 構成

```text
   Internet
      │
      ▼
   ┌─────────────┐    Internet Sharing (NAT)    ┌─────────────┐
   │  Mac mini    │ ───────────────────────────→ │  LinuxPC     │
   │ 192.168.2.1 │ ←───── LAN (SSH/HTTP) ─────→ │ 192.168.2.2 │
   └─────────────┘                              └─────────────┘
   M4 / dev/git                                 Ryzen + RX 7900 XTX
                                                Ollama / FastAPI 常駐
```

LinuxPC は Mac の **Internet Sharing** 経由でしか外に出られない構成 (セキュリティ要件)。Mac mini が落ちると LinuxPC のジョブも GitHub / Anthropic / Kaggle に届かなくなります。

## 問題: Linux から git push できない

最初は `ssh linuxpc 'cd ~/projects/foo && git pull'` で済ませる予定でした。が、Linux 側で:

```bash
$ ssh -T git@github.com -v
...
debug1: Connecting to github.com port 22.
debug1: Connection established.
debug1: Remote protocol version 2.0, remote software version 6279353
debug1: compat_banner: no match: 6279353
Connection closed by 140.82.114.4
```

GitHub の本物の SSH banner は `babeld-XXXXXXXX` のはず (例: `babeld-1f73f8e7`)。 `6279353` は GitHub のものではない、何かの中間装置の応答です。ISP の透過プロキシか、家庭用ルータの DPI か、いずれにせよ ssh:22 outbound は使えない。

一方で:

```bash
$ curl -sI https://github.com
HTTP/2 200    # ← HTTPS:443 は普通に通る
```

なので Anthropic API や Kaggle API は問題なく叩けるし、Web UI も使える。「ネットワーク全断」ではなく **「SSH:22 だけ握り潰される」**という嫌らしい状態。

## 検討した解決策

### A. HTTPS + PAT (Personal Access Token)

`https://<token>@github.com/...` 形式の URL で push できる。理論上は通る。

却下理由:
- PAT を Linux に置く必要がある (鍵管理が増える)
- 期限切れの度に再発行 + 配布する運用負担
- Mac で動いている SSH 鍵運用を Linux でも同じ仕組みに揃えたかった

### B. Mac mini を SSH トンネル経由で踏み台にする

`ssh -L 22:github.com:22 motomasa@mac` 的に Mac を経由させる。技術的には可能。

却下理由:
- トンネルセッションが切れた時の検出が面倒
- Mac mini を再起動するたびに復旧手順が増える
- Linux 上で `git pull` するモチベーション自体、結局「Mac に置いた最新コード」を取りに行くだけ → ならば最初から Mac 起点でいい

### C. rsync 一本化 (採用)

「**Mac の working tree を `.git/` 含めて Linux に rsync 転送**」する方式。

採用理由:
- Linux 側で `git log` がそのまま動く (`.git/` ごと同期するため)
- 差分転送なので速い (commit でかい変更があっても 1-2 秒)
- ssh の鍵運用は Mac↔Linux の LAN 経由 (公開鍵認証) だけで済む
- 「最新版を Linux に出した」という事実が Mac から能動的にコントロールできる

## デプロイスクリプト

```bash
#!/usr/bin/env bash
# scripts/deploy_from_mac.sh
set -euo pipefail

REMOTE="linuxpc:~/projects/kaggle-research-system/"
LOCAL="$HOME/projects/kaggle-research-system/"

echo "▸ [1] 同期前確認"
git -C "$LOCAL" status -s

echo "▸ [2] rsync で Linux へ同期 (.git 含む / .env 保護)"
rsync -av --delete \
  --exclude '.env' \
  --exclude 'workspaces/' \
  --exclude '*.db' \
  --exclude '__pycache__' \
  --exclude '.ruff_cache' \
  "$LOCAL" "$REMOTE"

echo "▸ [3] git log 確認 (Linux 側)"
ssh linuxpc "cd ~/projects/kaggle-research-system && git log -1 --oneline"

echo "▸ [4] systemctl --user 再起動"
ssh linuxpc "
  systemctl --user daemon-reload
  systemctl --user restart kaggle-api kaggle-ui
  systemctl --user is-active kaggle-api kaggle-ui
"

echo "✅ デプロイ完了"
```

### `.env` 保護のための exclude

最初 `--delete --exclude '.env'` を入れ忘れていて、Mac 側に `.env` が無い状態で同期したら Linux 側の本番キー入り `.env` が消える事故未遂が一度ありました。Linux にしか本番値を置かない方針なので、必ず exclude に入れています。

### `.git/` を転送する

`--exclude '.git/'` にしたくなる気持ちはわかりますが、これを exclude すると Linux で `git log` も `git status` も使えなくなって運用デバッグがつらい。差分転送だしファイル数も大したことないので、`.git/` ごと送る方が ROI が高い。

## サービス管理: systemctl --user で root 不要

`kaggle-api.service` / `kaggle-ui.service` は **user-level systemd unit** (`~/.config/systemd/user/`) に配置。

```ini
# ~/.config/systemd/user/kaggle-api.service
[Unit]
Description=KRS-Core Kaggle API
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/motomasahonda/projects/kaggle-research-system
ExecStart=/home/motomasahonda/ai-env/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

ログイン状態に関わらず常駐させるため:

```bash
sudo loginctl enable-linger motomasahonda
```

これで `sudo systemctl` ではなく `systemctl --user` だけで運用が完結します。Mac mini ↔ Linux の SSH も rootless で済む。

## 副次効果: Mac mini を落とせない

LinuxPC のインターネット出口が Mac mini の Internet Sharing なので、長時間ジョブを回す前には:

```bash
caffeinate -i &   # Mac がスリープしないように
```

を仕込むのが必須に。Mac mini を再起動するときも先に LinuxPC のジョブを止める必要がある。「家庭用 PC 2 台」のはずが、運用的には小規模なミニ DC を 1 人で回している感覚に近い。

## 学んだこと

1. **outbound ネットワークは事前に protocol 別に確認しておく**。「インターネットに繋がっている」≠「SSH:22 が通る」
2. **デプロイ方式の選択は技術的可否より運用負担で決める**。トンネル経由でも HTTPS+PAT でも実装できたが、毎日触る箇所は「最も単純な方式」が勝つ
3. **`.env` のような重要ファイルは exclude で必ず守る**。`--delete` と組み合わせると一発で消える
4. **systemctl --user は個人開発の最適解**。sudo を毎回叩かなくていいし、ホームディレクトリで完結する

## 参考: 主要コミット

- `d5fa711` docs(phase-0): finalize operational model — rsync deploy, Linux SSH:22 finding

リポジトリ: [github.com/motomasa-honda/kaggle-research-system](https://github.com/motomasa-honda)
