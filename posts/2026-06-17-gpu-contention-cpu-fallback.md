---
title: "1枚の GPU を LLM 推論と学習で共有する — VRAM を測って足りなければ CPU に落とす"
emoji: "🎛️"
type: "tech"
topics: ["mlops", "gpu", "pytorch", "machinelearning", "infra"]
published: true
publication_date: "2026-06-17"
---

## TL;DR

- 1 枚の GPU を「LLM 推論 (知識マイニングの抽出)」と「モデル学習 (画像 / NLP / テーブル NN)」で共有している
- 知識ハーベスト中は 70B クラスの LLM が **VRAM の大半 (24GB 中 23.8GB)** を握る。その隙に GPU 学習を起動すると、空き 2.7GB で **即 OOM**、イテレーションが丸ごと無駄になる
- 対策として、実行の単一チョークポイント (subprocess 起動関数) に **VRAM の preflight** を入れた。GPU を使うスクリプトの起動前に空きを測り、閾値未満なら一定時間待ち、それでもダメなら **CUDA/HIP を隠して CPU で完走**させる
- 「落とす」より「遅くても完走」を選ぶ。OOM クラッシュでイテレーションを失うより、CPU で結果を出す方がマシ
- ライブで実証: 空き 2743MB < 閾値 3000MB を検知 → CPU フォールバック発火 → 学習スクリプトが exit 0 で完走

## 問題: 共有 GPU の取り合いで OOM

このシステムは単一 GPU をデュアルユースしている。知識ループの「上位解法を LLM で要約して skill 化」では推論用の大型 LLM が VRAM をほぼ専有する。一方、モデリングのテンプレ (画像 backbone、Transformer NLP、テーブル NN) は GPU 学習を前提にしている。

両者が時間的に重なると何が起きるか。実際に測るとこうだった。

```
VRAM Total: 25753 MB
VRAM Used : 23804 MB   (← ハーベストの 70B LLM)
free      :  2743 MB
```

この状態で GPU 学習が `torch.cuda.is_available()` を見て GPU に乗りに行くと、2.7GB しか空いていないので OOM。学習はクラッシュし、そのイテレーションの計算はすべて捨てになる。一番もったいない失敗の仕方だ。

## 対策: 起動前に空きを測る

学習コードは色々あるが、起動は 1 か所の関数 (workspace で `main.py` を subprocess 起動する所) を必ず通る。ここに門番を置くのが一番効く。

VRAM の空きは vendor 中立に取りたい。`torch.cuda.mem_get_info()` は CUDA でも ROCm でも `(free, total)` を返すので第一手段に。取れなければ `rocm-smi` / `nvidia-smi` をパースする三段構え。

```python
free, total = torch.cuda.mem_get_info(0)   # ROCm/CUDA 共通
```

門番のロジックはこう。GPU を使いそうなスクリプト (torch/cuda/timm/transformers を含む) のときだけ作動する。

1. 空き VRAM を測る
2. 閾値未満なら、設定した時間まで解放を待つ (ポーリング)
3. 待っても足りなければ、`CUDA_VISIBLE_DEVICES=""` と `HIP_VISIBLE_DEVICES=""` を子プロセスに渡して **GPU を隠す** → スクリプトは自動的に CPU で走る
4. 足りていれば普通に GPU 実行

```python
if script_uses_gpu(code) and free < MIN_FREE_MB:
    logger.warning(f"空き {free}MB < {MIN_FREE_MB}MB → CPU フォールバック")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
```

テンプレ側は元から「GPU が無ければ CPU」を書いているので、GPU を隠すだけで CPU 経路に落ちる。落とさず完走させるのが狙いだ。閾値 0 で機能オフにできる (完全可逆)。

## ライブ検証

ハーベストが GPU を握っている最中に、torch を使うダミー学習を起動してみた。

```
[gpu] 空き VRAM 2743MB < 3000MB → CPU フォールバック (GPU は他プロセスが専有中)
[runner] exit=0 cv=0.5 duration=3.0s
metrics: {'cuda': False}
```

期待どおり、空き不足を検知して CPU に逃がし、`torch.cuda.is_available()=False` で exit 0 完走。この門番が無ければ、同じ条件で GPU を掴みに行って OOM していた。

## 設計上の割り切り

- **待つ vs すぐ CPU**: 待機時間は設定可能だが、既定は「待たず即フォールバック」。共有相手 (ハーベスト) が数時間走るなら待っても無駄だから
- **CPU 学習は遅い**: 画像/Transformer を CPU で回すと遅く、タイムアウトしうる。でもそれは捕捉される失敗で、OOM クラッシュよりは扱いやすい
- **並列起動の競合は別問題**: 複数の学習を同時起動すると、各自が「空いてる」と見て collectively に超過しうる (TOCTOU)。テーブル NN は軽いので実害は小さいが、厳密にやるならセマフォで GPU をシリアライズする必要がある。今回はそこまでは踏み込んでいない

## 学び

- 共有 GPU は「使えるか (`is_available`)」ではなく「**今いくら空いているか**」で判断する。可用性と空き容量は別物
- 失敗の仕方には良し悪しがある。**OOM クラッシュ (全損) より CPU 完走 (遅延)** の方が回復可能
- 横断的な制御は、分散したコードではなく**全員が通る 1 か所 (実行のチョークポイント)** に置くと漏れない
- vendor 中立な API (`torch.cuda.mem_get_info`) を第一手段にし、ベンダ固有ツールはフォールバックに回す
