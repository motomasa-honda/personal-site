---
title: "Mac mini M4 + Ubuntu RTX機でKaggle自動化AIを動かすデュアルノード構成の設計と落とし穴"
emoji: "🖥️"
type: "tech"
topics: ["kaggle", "ubuntu", "macmini", "ollama", "自動化"]
published: true
publication_date: "2026-05-30"
---

## TL;DR
- Mac mini M4（制御ノード）+ Ubuntu PC（GPU実行ノード）の2台構成でKaggle自動化システムを構築
- MacはNotebook取得・Git管理のみ、UbuntuはGPU推論・パイプライン実行に特化
- systemd user-mode + SSH kickでMacからUbuntuの蒸留を自動起動
- Kaggle CLI 2.2.0はlegacy `kaggle.json`を認識しない。`~/.kaggle/access_token`に新形式トークンを置く必要あり

---

## なぜデュアルノード構成にしたか

Kaggle自動化システム（KRS: Kaggle Research System）を作っていて、最初はMac miniだけで全部やろうとしていた。

問題は**deepseek-r1:70bを動かすにはVRAMが足りない**こと。Mac mini M4は統合メモリで優秀だが、ローカルLLMをフル活用するには別途GPU機が欲しい。

そこで手元にあったRyzen 9 9950X + RX 7900 XTX（24GB VRAM）のUbuntu PCを実行ノードとして追加した。

```
Mac mini M4 (192.0.2.1)  ←→  Ubuntu PC (192.0.2.2)
  制御・Harvest・Git管理         GPU推論・パイプライン実行
  256GB SSD (容量少ない)         2TB SSD + 24GB VRAM
```

Ethernet直結で192.0.2.x/24のプライベートネットワーク。

---

## 役割分担の決め方

最初は**MacがUbuntu OllamaにリモートLLM呼び出し**してSKILL抽出する設計だった（v7.1）。これには問題が2つあった。

1. **MacのSSDが圧迫される**: KaggleのNotebookファイル（mined/）がMacに蓄積し続ける
2. **ネットワーク越しのLLM呼び出しが非効率**: LAN経由とはいえ大量テキストをやり取りする

v7.2で設計を変えた：

```
【旧 v7.1】
Mac: Notebook DL → Ubuntu Ollama呼び出し → SKILL抽出 → mined/がMacに残る

【新 v7.2】
Mac: Notebook DLのみ → Ubuntu に rsync で転送 → Mac側のmined/を削除
Ubuntu: 受け取ったNotebookをローカルOllamaで直接SKILL抽出 → 容量解放
```

これでMac 256GB SSDの圧迫問題が解決した。

---

## SSH双方向設定とエイリアス

MacからUbuntu、UbuntuからMacの双方向SSHが必要。

```bash
# Mac → Ubuntu (デフォルト鍵を使用)
ssh ubuntu@192.0.2.2

# Ubuntu → Mac (専用鍵を生成)
ssh-keygen -t ed25519 -C "ubuntu-to-mac" -f ~/.ssh/id_ed25519_mac -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_mac.pub dev@192.0.2.1
```

エイリアスは `~/.ssh/config` に登録：

```
# Ubuntu側の ~/.ssh/config
Host macmini
    HostName 192.0.2.1
    User dev
    IdentityFile ~/.ssh/id_ed25519_mac

Host linuxpc
    HostName 192.0.2.2
    User ubuntu
```

落とし穴：`check_network.sh` が `macmini` と `linuxpc` の両方にSSH疎通確認をかけるが、`linuxpc`はUbuntu自身へのSSH（ループバック）なので常にNGになる。実害はないが最初は混乱した。

---

## systemd user-modeでMacからkick

MacがHarvestを完了した後、UbuntuのLLM蒸留を自動起動したい。

```bash
# Mac側 harvest_runner.sh から
ssh linuxpc "systemctl --user start krs-distill-oneshot.service"
```

systemd **user-mode**（`--user`）を使うのが重要。sudoなしで動く。

```ini
# systemd/krs-distill-oneshot.service
[Unit]
Description=KRS v7.2 One-shot Meta-Skill Distillation

[Service]
Type=oneshot
WorkingDirectory=/home/ubuntu/projects/kaggle-research-system
ExecStart=/bin/bash scripts/boot_distill.sh --force
TimeoutStartSec=18000

[Install]
WantedBy=default.target
```

再起動後も動くようにlingerを有効化：

```bash
loginctl enable-linger $USER
```

サービスが `masked` になってしまうことがあった。その場合：

```bash
systemctl --user unmask krs-distill-oneshot.service
# ※ unmaskするとファイルが削除されることがある
# その場合は再登録
bash scripts/install_systemd_services.sh
systemctl --user daemon-reload
```

---

## Kaggle CLI 2.2.0の認証問題

`pip install kaggle` でインストールすると2.2.0が入る。これが厄介だった。

従来の `~/.kaggle/kaggle.json`（username + key形式）が**認識されない**。

```json
// 旧形式 (2.2.0では動かない)
{"username":"<your-username>","key":"<your-kaggle-api-key>"}
```

新形式のAPIトークン（KaggleのSettings → API Tokens → 新しいトークンを生成）は文字列で発行される：

```
KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

これを**`~/.kaggle/access_token`**（`kaggle.json`ではない）に保存する：

```bash
echo "KGAT_xxxxx..." > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
kaggle competitions list  # これで通る
```

また `kaggle competitions list` のデフォルトソートが古いコンペを返すので、最新コンペを取るには：

```bash
kaggle competitions list --sort-by recentlyCreated --csv
```

さらにURLが `ref` カラムにフルURLで入ってくるので slug 抽出が必要：

```bash
kaggle competitions list --sort-by recentlyCreated --csv \
  | tail -n +2 \
  | cut -d',' -f1 \
  | sed 's|https://www.kaggle.com/competitions/||'
```

---

## GPU電力制限の永続化

RX 7900 XTXでdeepeseek-r1:70bを動かすと400W近く消費する（公称TDP 355W）。

`rocm-smi`で300Wに制限できるが**再起動で消える**。systemdで永続化：

```bash
sudo tee /etc/systemd/system/rocm-powercap.service << 'EOF'
[Unit]
Description=ROCm GPU Power Cap (300W)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/rocm-smi --setpoweroverdrive 300
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable rocm-powercap.service
sudo systemctl start rocm-powercap.service
```

GPU[1]（iGPU）がエラーを出すが無視してOK。GPU[0]（RX 7900 XTX）への制限は正常に適用される。

---

## deploy.shでワンコマンド同期

Mac編集 → Ubuntu反映 → GitHub pushを毎回手動でやるのはつらい。`scripts/deploy.sh`を作った：

```bash
# フルデプロイ（これだけでOK）
bash scripts/deploy.sh "feat: 新機能追加"

# 内部で実行される処理:
# 1. Ubuntu → Mac rsync (Ubuntuの変更を引き取り)
# 2. git add & commit & push → GitHub
# 3. Mac → Ubuntu rsync (Macの変更をUbuntuに配布)
# 4. systemctl --user restart kaggle-api kaggle-ui
```

部分実行も可能：

```bash
bash scripts/deploy.sh --pull-only      # Ubuntu→Mac引き取りのみ
bash scripts/deploy.sh --push-only "msg" # Git pushのみ
bash scripts/deploy.sh --deploy-only    # Ubuntu反映+再起動のみ
bash scripts/deploy.sh --dry-run "msg"  # 確認のみ
```

---

## 学んだこと

- **Mac mini 256GBは制御専用と割り切る**: LLM推論・大容量ファイルは全部Ubuntuに
- **systemd user-modeはsudoなしで自動起動できる**: サービス管理が圧倒的に楽
- **Kaggle CLI 2.2.0は破壊的変更あり**: `~/.kaggle/access_token`が正しい保存先
- **GPU電力制限はsystemdで永続化**: `rocm-smi`は再起動で消える
- **デプロイスクリプトは最初から作る**: 同期ミスによるバグを防げる

## 参考
- `scripts/harvest_runner.sh` - Mac側のHarvestスクリプト
- `scripts/boot_distill.sh` - Ubuntu側の蒸留スクリプト
- `scripts/deploy.sh` - ワンコマンドデプロイ
- `systemd/krs-distill-oneshot.service` - SSH kickで起動するサービス
