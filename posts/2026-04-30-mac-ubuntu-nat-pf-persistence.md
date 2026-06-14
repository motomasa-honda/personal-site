---
title: "Mac mini M4とUbuntu直結環境でインターネット共有をpfctlで永続化するまでの3時間"
emoji: "🔥"
type: "tech"
topics: ["mac", "ubuntu", "network", "pf", "linux"]
published: false
publication_date: "2026-04-30"
---

## TL;DR

- Mac mini M4とUbuntu（Ethernet直結）でインターネット共有をしようとしてハマった
- macOSのインターネット共有は `192.168.2.x` サブネットを自動割り当てする仕様
- 独自サブネット（10.10.10.x）を使っていたためNATが機能しなかった
- pfctlで手動NATを設定するも再起動で消える問題が発生
- launchdの永続化もタイミング問題で失敗し続けた
- 最終的にUbuntu側のIPを `192.168.2.2` に変更してmacOSの仕様に合わせて解決

---

## 構成と背景

Mac mini M4とUbuntu PCをEthernetケーブルで直結してローカルLLM開発環境を構築している。

```
Mac mini M4（192.168.2.1）
  └─ Ethernet直結
Ubuntu 24.04（192.168.2.2）
  └─ Ryzen 9 9950X / RX 7900 XTX 24GB
```

MacのWi-Fi（インターネット）をUbuntuに共有する構成。
UbuntuはヘッドレスでRDP接続（gnome-remote-desktop、ポート3390）で運用している。

---

## 最初のハマり：NATルールが当たらない

もともとUbuntuのIPを `10.10.10.2`、MacのEthernetを `10.10.10.1` に設定していた。
macOSのシステム設定でインターネット共有を有効にしたが、UbuntuからのPingが通らない。

```bash
# Ubuntu側
ping -c 3 8.8.8.8
# → 100% packet loss
```

原因を調査するとpfのNATルールを確認できた。

```bash
sudo pfctl -s nat -a com.apple.internet-sharing/shared_v4
# → nat on en1 inet from 192.168.2.0/24 to any -> (en1:0)
```

**macOSのインターネット共有は `192.168.2.0/24` に対してのみNATをかける仕様だった。**
`10.10.10.x` サブネットはNATの対象外なので当然通らない。

---

## pfctlで手動NATを追加してみた

`10.10.10.x` を維持したまま解決しようとpfに手動でNATルールを追加した。

```bash
# NATルールファイルを作成
sudo bash -c 'cat > /etc/pf.anchors/nat-linux << EOF
nat on en1 from 10.10.10.0/24 to any -> (en1)
pass all
EOF'

# pf.confにanchorを追加
sudo bash -c 'cat >> /etc/pf.conf << EOF
anchor "nat-linux"
load anchor "nat-linux" from "/etc/pf.anchors/nat-linux"
EOF'

# 適用
sudo pfctl -F all
sudo pfctl -e
sudo pfctl -f /etc/pf.anchors/nat-linux
```

これでPingが通るようになった。

```bash
# Mac側のtcpdumpで確認
# en1に 192.168.1.8（MacのWi-Fi IP）→ 8.8.8.8 へのNATが確認できた
sudo tcpdump -i en1 -n icmp
# → 192.168.1.8 > 8.8.8.8: ICMP echo request ✅
# → 8.8.8.8 > 192.168.1.8: ICMP echo reply ✅
```

---

## 再起動で消える問題：launchdで永続化を試みる

Mac再起動後にpfルールが消えることが分かった。
launchdのplistを作成して自動適用を試みた。

```bash
sudo bash -c 'cat > /Library/LaunchDaemons/com.user.pf-nat.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.pf-nat</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>sleep 10; /sbin/pfctl -F all; /sbin/pfctl -e;
          /sbin/pfctl -f /etc/pf.anchors/nat-linux</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>/var/log/pf-nat.log</string>
    <key>StandardOutPath</key>
    <string>/var/log/pf-nat.log</string>
</dict>
</plist>
EOF'
sudo launchctl load /Library/LaunchDaemons/com.user.pf-nat.plist
```

`sleep 10` でネットワーク起動を待つようにしたが、**再起動後も疎通しない**状態が続いた。

ログを確認するとpfctlは実行されているが、macOSのインターネット共有サービスが
後から起動してpfルールを上書きしていることが分かった。

---

## 根本解決：UbuntuのIPをmacOSの仕様に合わせる

3時間格闘した末に「macOSの仕様に合わせる」という方針に切り替えた。

UbuntuのIPを `192.168.2.2` に変更する。

```bash
# Ubuntu側でNetworkManagerを使ってIP変更
sudo nmcli con modify "有線接続 3" \
  ipv4.addresses 192.168.2.2/24 \
  ipv4.gateway 192.168.2.1 \
  ipv4.dns "8.8.8.8 8.8.4.4" \
  ipv4.method manual
sudo nmcli con up "有線接続 3"
```

Mac側のSSH configも更新。

```
# ~/.ssh/config
Host linuxpc
  HostName 192.168.2.2
  User motomasahonda
  IdentityFile ~/.ssh/id_ed25519_linuxpc
```

これで再起動後も自動的にNATが機能するようになった。
macOSのインターネット共有が `192.168.2.x` に対してNATをかける仕様だったので、
そこに乗っかるのが一番シンプルな解決策だった。

---

## Ubuntu側のDNS永続化

IP変更後もDNSが再起動で消える問題があった。
`systemd-resolved` の設定ファイルを追加して永続化した。

```bash
sudo bash -c 'cat > /etc/systemd/resolved.conf.d/dns.conf << EOF
[Resolve]
DNS=8.8.8.8 8.8.4.4
Domains=~.
EOF'
sudo systemctl restart systemd-resolved
```

さらに起動時にデフォルトゲートウェイを確実に設定するスクリプトを用意した。

```bash
# /etc/network/if-up.d/set-gateway
#!/bin/sh
ip route replace default via 192.168.2.1
```

---

## 学んだこと

- **macOSのインターネット共有は `192.168.2.x` サブネット専用**。独自サブネットを使う場合はpfctlの手動設定が必要だが、再起動後の永続化が難しい
- **pfctlをlaunchdで永続化する場合、macOSのインターネット共有サービスが後勝ちする**。タイミング制御（sleep）だけでは解決しない
- **仕様に抗うより仕様に合わせる方が早い**。今回は3時間格闘した末に「IPを変える」という5分の作業で解決した
- `tcpdump -i en1 -n icmp` でNATの動作をパケットレベルで確認するのが原因特定に有効だった

---

## 参考

- macOS `pfctl` manpage
- `/etc/pf.anchors/com.apple` の中身を読むとインターネット共有の仕組みが分かる
- `sudo pfctl -s nat -a com.apple.internet-sharing/shared_v4` でNATルールを確認できる
