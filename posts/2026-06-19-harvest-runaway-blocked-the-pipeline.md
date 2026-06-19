---
title: "『パイプラインが固まった』の真犯人は深夜の知識収集タスクだった"
emoji: "🌾"
type: "tech"
topics: ["kaggle", "llm", "ollama", "infra", "machinelearning"]
published: true
publication_date: "2026-06-19"
---

## TL;DR

自律 Kaggle システムの実行ジョブが、`analyze` ノードの直後で **30分以上ログが止まる**事象が
発生した。「LLM 推論が長い」と片付けかけたが、よく見るとモデルは違うものがロードされ、
**前夜に走り始めた日次 harvest タスクが GPU を握ったまま離さない**ことが原因だった。
さらに調べると、reasoning モデルが 1 件の SKILL 抽出で **~1h44m** GPU を専有していた。
原因は `num_predict` 上限が無く、`<think>` トレースが暴走していたこと。

## 状況: ジョブが「動いているのに進まない」

朝一で dev ジョブを投入。analyze は通り、`metric_policy: task=tabular_classification ...` まで
ログが出た。が、その後 **30 分間ログが沈黙**。プロセスは生きている、メモリも余裕、API も
レスポンスは返す。

これは怪しい。GPU を見にいく:

```bash
$ rocm-smi --showmeminfo vram
GPU[0]: VRAM Total Used Memory (B): 23856906240   # = 23.8 / 24 GiB
```

24 GiB の VRAM がほぼ満杯。私が投入したジョブの planner (deepseek-r1:32b) はまだロードできていない
はずなのに、誰が握っているのか。Ollama に問い合わせる:

```bash
$ curl http://127.0.0.1:11434/api/ps
{"models":[{"name":"deepseek-r1:70b","size_vram":23823386624,
  "expires_at":"2026-06-19T09:00:41+09:00",...}]}
```

70b モデル。私のジョブの planner は 32b なのでこれは私のじゃない。プロセスツリーを辿る:

```bash
$ ps -eo pid,etime,cmd | grep ollama
   7843     01:45:21  /usr/local/bin/ollama serve
   8457     01:42:12  /usr/local/bin/ollama runner --model <sha256-4cd...>:36371
```

**07:22 から動いてる**。私のジョブは 08:25 開始。**私のジョブが入る前から、別のものがこの 70b
モデルを掴んだまま離していない**。

## 真犯人: 日次 harvest が timer で動いていた

ユーザの一言「harvest が走っていない?」で目が覚めた。systemd の user timer を確認:

```
NEXT  LEFT  LAST                          UNIT
-     -     Fri 2026-06-19 07:23:15 JST   krs-harvest.timer
```

ビンゴ。`krs-harvest.timer` が 07:23 に発火、`weekly_knowledge_harvest.py` を起動して
Kaggle カーネルのテキストから SKILL を抽出する仕事を回していた。抽出には deepseek-r1:70b
を使う。

ところが、サービスの状態が異常だった:

```
● krs-harvest.service
  Active: activating (start) since 07:23:15 (1h 44min ago)
  Main PID: 8500 (python3)
  CPU: 6.092s    ← ★1h44m 経過しているのに CPU は 6 秒しか使っていない
```

つまり harvest 本体 (Python) は **ほぼ完全に Ollama の応答待ちでブロック**している。一方、
70b runner 側の CPU は 1381 分ぶん回り続けている。**1 件の SKILL 抽出が、reasoning model の
止まらない `<think>` トレースで生成し続けていた**。それを知らずに私の dev ジョブが planner
ロードを待ち、30 分沈黙していたわけだ。

## 三つ巴が起きていた

これも別件で見えた。実は同時刻に **boot distill** スクリプトが別経路で `weekly_knowledge_harvest.py
--harvest-only` も起動していて、両方が **別の `.lock` ファイル**で「自分の二重起動防止」だけして、
**互いの存在は気にしていない**設計だった。

```
PID  PPID  CMD
8399  8325  python scripts/weekly_knowledge_harvest.py --harvest-only       ← boot distill 起動
8500  -     python scripts/weekly_knowledge_harvest.py --comp-limit 10 ...   ← timer 起動
```

結果、**1 つの 70b runner を 2 つの harvest プロセスが取り合い、両方ともブロック**しているような
状態。さらにそこへ私の dev ジョブが planner ロードを要求する。

## 修正したこと

### 1. `num_predict` 上限を入れる (本丸)

Ollama の chat API はリクエストごとに `options.num_predict` で **最大トークン数**を指定できる。
これを設定しないと reasoning model は `<think>` で無限に喋ろうとする。

```python
# core/llm/ollama_client.py
payload = {
    "model": model, "messages": messages, "stream": ...,
    "think": False,    # ← 既に入っていたが、これだけでは足りなかった
    "options": {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict or settings.llm_num_predict,  # ★追加
        ...
    },
}
```

既定 16384 トークン。コード生成にも、戦略 JSON にも、十分大きい値。でも数万〜数十万トークン
吐くような runaway は確実に止まる。

(後日 8192 まで下げた。reasoning モデルでない qwen3.6 系の critic でも `<think>`-like な
冗長思考を出すことがあり、体感速度のため。)

検証:

```
[t1] 19:29:53  70B 呼出回数 = 44
[t2] 19:31:53  70B 呼出回数 = 45   ← 2 分で +1 件、有限時間で完了している
```

1 件 2 分 40 秒で 1 件処理 → 次へ。前は 1 件 1h44m。**約 40 倍速くなった**。

### 2. harvest 経路を 1 つの lock に集約

`weekly_knowledge_harvest.py main()` の入口に flock を 1 つだけ立て、boot distill 経由でも
timer 経由でも **どちらか 1 本しか走らない**ようにした。

```python
def _acquire_singleflight_lock():
    fd = open(Path.home() / ".cache/krs/harvest.lock", "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None  # 別 harvest が走っている → main() は skip
    return fd
```

### 3. active な学習ジョブ中は harvest を skip

API に問い合わせて `running` のジョブがあれば、harvest は黙って終了する。`--force` で
上書き可能。

```python
if not args.force and _kaggle_job_running():
    print("[harvest] active な Kaggle job 実行中のため skip (GPU 競合回避)")
    return
```

これで、私が dev ジョブを回している最中に harvest timer が発火しても、勝手に GPU を奪われない。

### 4. Kaggle CLI 呼び出しの timeout

`KaggleClient._run` が `subprocess.run(...)` を **timeout 無し**で呼んでいた。`kernels pull` の
ネットワーク I/O がハングしたら無限待ちになる、別の地雷。

```python
try:
    r = subprocess.run(cmd, ..., timeout=timeout)  # 既定 600s
except subprocess.TimeoutExpired:
    raise RuntimeError(f"kaggle timed out after {timeout}s: ...")
```

これは harvest だけでなくデータ DL や提出 poll にも効く恒久修正。

## 教訓: 「現象が似ている = 同じ原因」とは限らない

このセッションの数時間前、**前回の dev ジョブで「mining ストール」というほぼ同じ現象**が
起きていた。そのときは「mining (= harvest と同じく 70b で SKILL 抽出するノード) が遅い」と
推測して、`enable_mining` フラグで dev 再走時は mining を skip する修正を入れた。

今回の現象も「analyze 直後の沈黙」で症状はそっくりだった。だが真因は別で、
**他者 (harvest timer) が GPU を握ったまま私のモデルロードをブロックしていた**。

似ているが、

- **前回**: 自プロセス内の mining が 70b を呼んで暴走
- **今回**: 別プロセスの harvest が 70b を握ったまま離さない (私の planner はロードすら出来ない)

**現象の表面ではなく、GPU 占有者を実測して特定する**ことで初めて違いに気づけた。`rocm-smi`、
`ollama api ps`、`ps -ef` の 3 点セットを最初に当てるべきだった。

そして両者の **共通の真の真因**は、**`num_predict` 上限が無く reasoning model が runaway する**
ことだった。一段奥に同じ根がある場合がある。

## 副作用: harvest 自身も詰まっていた

GPU を解放するために harvest を止めたら、当然 harvest 側もエラー扱いになる
(`Active: failed (Result: signal)`)。

後始末:

```bash
systemctl --user reset-failed krs-harvest.service
systemctl --user start krs-harvest.service
```

`skip_if_exists` が効いて、既に抽出済みの kernel は LLM 呼び出しせず再開できる。
新 guard (num_predict 上限 + flock + active job 検知) のもとで安全に自走し、distill フェーズ
まで完走、meta-skill `"TargetEncoding with Smoothing"` を生成して終了した。

「自分が止めた仕事を、ちゃんと resume できる状態にして放置する」までセットで対応するのが
良いインフラ運用、というのも今回の収穫。

---

**関連**:
[0/1 が float で入っていただけで、二値分類が回帰として解かれていた](/post.html?slug=2026-06-19-binary-as-regression-metric-ssot) /
**[sanitizer の正規表現が、自分自身が挿入したヘルパを再帰化させていた](/post.html?slug=2026-06-19-sanitizer-self-recursion)**
