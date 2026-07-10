---
title: "systemdは「稼働中」と言い張っていたが、ソケットファイルは消えていた"
emoji: "🎭"
type: "tech"
topics: ["linux", "ubuntu", "infra", "mlops", "debugging"]
published: true
publication_date: "2026-07-10"
---

## TL;DR

同じマシンにDockerが2種類(パッケージマネージャ違い)で二重にインストールされている
状態を整理しようと、片方を削除した。削除自体はきれいに終わったが、直後に残った方の
Dockerが動かなくなった。`systemctl status`は「稼働中」と表示し続けていたが、実際には
肝心のソケットファイルが消えていた——ステータス表示を信じて「動いているはず」と思い込むと
気づけない類の壊れ方だった話。

---

## 二重インストールに気づく

開発用のLinux機で、Dockerがパッケージマネージャ経由(APT)とアプリストア経由(Snap)の
両方でインストールされ、両方とも起動している状態になっていた。どちらも同じソケット
ファイルの場所を使う設計のため、共存自体は珍しくないが、片方だけを本来使うつもりで
いたので、使っていない方を削除して整理することにした。

```
$ snap list docker
Name    Version  Rev   Tracking       Publisher
docker  29.3.1   3505  latest/stable  canonical
```

## 削除は成功。ただし副作用が

```
$ sudo snap remove docker
docker removed (snap data snapshot saved)
```

削除コマンド自体は問題なく完了した。だが直後に、残しておいたはずのAPT版Dockerに
コマンドを投げると失敗するようになった。

```
$ docker ps
failed to connect to the docker API at unix:///var/run/docker.sock:
dial unix /var/run/docker.sock: connect: no such file or directory
```

不思議だったのは、systemdに聞くと「動いている」と答えたことだ。

```
$ systemctl status docker.socket
● docker.socket - Docker Socket for the API
     Active: active (running) since ...
     Listen: /run/docker.sock (Stream)
```

「稼働中」で「そのパスをリッスンしている」と表示されているのに、実際にそのパスに
ファイルは存在しなかった。

```
$ ls -la /run/docker.sock
ls: cannot access '/run/docker.sock': No such file or directory
```

## なぜこうなったか

2つのDockerは同じソケットパス(`/run/docker.sock`)を共有する前提で動いていた。
片方(Snap版)を削除したときのクリーンアップ処理が、その共有パスのファイル自体を
一緒に削除してしまったと見られる。もう片方(APT版)のsystemdユニットは、削除処理の
影響を直接受けたわけではないので、自分の内部状態としては「有効化されていて、稼働中」
のままだった。だが実際にリッスンしていたはずのソケットは、外部から消されたことに
systemd自身は気づいていなかった——だから「有効」だが実体のないソケットという、
矛盾した状態がそのまま表示され続けた。

## 直し方は単純だった

```
$ sudo systemctl restart docker.socket docker.service
$ ls -la /run/docker.sock
srw-rw---- 1 root docker 0 ... /run/docker.sock
```

ソケットとサービスを再起動させれば、systemdは実体のないソケットに気づいて正しく
作り直す。既に動いていたコンテナが再起動で巻き込まれて止まることはあったが(元々
テスト用に一時的に立てていたものだけだったので実害はなかった)、その後は`docker
compose up`から実際のコンテナ起動、内部からの応答確認、`down`での後片付けまで、
一連の流れが問題なく通った。

## 今日の教訓

`systemctl status`が返す「active (running)」は、そのユニット自身の管理下にある
プロセスや状態についての自己申告であって、外部から見た実際のファイルシステムの
状態までは保証しない。今回のように、他のプロセスの削除処理が横から実体を消して
しまうケースでは、systemd自身はそれに気づく手段を持たない。

「ステータスは正常」という表示と、「実際に使おうとしたら失敗する」という現象が
食い違ったときは、ステータス表示の方を疑い、実体(この場合はソケットファイルの
存在そのもの)を直接確認しにいく方が早い。ステータスは症状の一部でしかなく、
診断そのものではない。
