---
title: "RX 7900 XTX に ROCm 版 PyTorch を入れて Kaggle Agent の画像/NLP を GPU 解禁する — hf-xet が NAT 越しでハングする地雷つき"
emoji: "🔥"
type: "tech"
topics: ["rocm", "pytorch", "amd", "gpu", "huggingface"]
published: true
publication_date: "2026-06-16"
---

## TL;DR

- Kaggle Agent (KRS-Core) は今まで tabular (LightGBM/XGBoost/CatBoost) しか強くなく、画像と transformer NLP が手薄だった
- 根本原因は単純で、**実行環境に torch も timm も transformers も入っていなかった**
- RX 7900 XTX (gfx1100) + ROCm 7.2 の箱に、**システム ROCm に合わせた torch wheel** を入れて GPU を解禁した
- 一番ハマったのは torch 本体ではなく、**huggingface_hub が新採用した `hf-xet` が NAT 越しのダウンロードでハングする**こと。49 分間 0% CPU・0 バイトで沈黙した
- 解決はシンプルで `pip uninstall hf-xet`。あと iGPU 混在で `device_count=2` になる話も書く

## まず事実確認: 何が入っていて何がないか

「画像が弱い」の真因を確かめる。実行環境の Python で深層学習スタックを調べたら、見事に何もなかった。

```
torch        : NO
torchvision  : NO
timm         : NO
transformers : NO
sklearn      : YES 1.8.0
scipy        : YES 1.17.1
```

画像 (CNN/ViT) や transformer NLP には PyTorch が要る。つまり弱さは「実装をサボっていた」のではなく「依存が無かった」。ここを埋める。

GPU と ROCm の状態も先に確認する。

```
GPU      : gfx1100 (AMD Radeon RX 7900 XTX)
ROCm     : 7.2.2  (/opt/rocm-7.2.2)
/dev/kfd : present
groups   : ... video ... render ...   # GPU を掴むのに必要なグループに入っている
```

`render` / `video` グループに入っていて `/dev/kfd` があるので、ユーザ空間さえ入れれば GPU は使える状態。

## wheel 選び: システム ROCm に合わせる

ROCm 版 torch の wheel は、自前で ROCm ランタイムを同梱する。とはいえシステムの ROCm 7.2 に素直に合わせるのが安全。PyTorch の wheel インデックスを叩いて、Python 3.12 (cp312) 用の wheel が実在するか確認する。

```bash
for v in rocm6.2 rocm6.3 rocm6.4 rocm7.0 rocm7.2; do
  curl -s -o /dev/null -w "$v -> %{http_code}\n" \
    https://download.pytorch.org/whl/$v/torch/
done
```

`rocm7.2` のインデックスに `torch-2.12.0+rocm7.2-cp312` が居たので、これにした。torchvision はバージョン整合が必須で、**`torchvision_minor = torch_minor + 15`** の関係になっている (torch 2.12 ↔ torchvision 0.27)。

## インストール前に dry-run で爆風半径を測る

ここが大事。この箱は本番サービス (FastAPI / エージェント) が同じ venv で動いている。torch を入れて numpy や pandas が巻き込まれて壊れたら最悪だ。だからまず `--dry-run` で何が変わるか見る。

```bash
pip install --dry-run --index-url https://download.pytorch.org/whl/rocm7.2 \
    torch==2.12.0 torchvision==0.27.0
# Would install: filelock fsspec mpmath networkx setuptools sympy
#                torch-2.12.0+rocm7.2 torchvision-0.27.0+rocm7.2 triton-rocm-3.7.0
```

`numpy` も `pandas` も `sklearn` も変更リストに無い = 既存の科学計算スタックは無傷。安心して本インストールに進む。

```bash
pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
    torch==2.12.0 torchvision==0.27.0
```

入ったら、必ず**実テンソル演算で GPU を確認**する。`is_available()` が True でも実演算でコケる ROCm はあるので。

```python
import torch
print(torch.cuda.is_available())          # True
print(torch.cuda.get_device_name(0))      # AMD Radeon RX 7900 XTX
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
print((a @ b).shape)                      # matmul 4096^3 が 0.09s で通る
```

ROCm は `torch.cuda` の API をそのまま喋る (HIP がバックエンド)。gfx1100 は素のままで動いて、`HSA_OVERRIDE_GFX_VERSION` のような小細工は要らなかった。

## 本題の地雷: hf-xet が NAT 越しでハングする

timm/transformers を PyPI から足して、`timm.create_model("resnet18", pretrained=True)` で動作確認しようとしたら、**49 分間沈黙した**。プロセスを覗くとこうだった。

```
PID      ELAPSED  %CPU  STAT  WCHAN
...      48:49    0.1   Ssl   futex_do_wait
```

ダウンロードの一時ファイルは **0 バイトの `.incomplete` のまま増えない**。なのに `curl https://huggingface.co` は 200 を 0.08 秒で返す。HTTP は通るのに本体 DL だけ固まる。

犯人は **`hf-xet`**。最近の huggingface_hub (1.x) は、デフォルトで Xet という別プロトコル・別エンドポイント (CAS サーバ) からファイルを取りに行く。この箱はインターネットに NAT 越しでしか出られない構成で、**xet の通信がその NAT を越えられずにハングしていた**。

解決はあっけない。

```bash
pip uninstall -y hf-xet      # → huggingface_hub が通常の HTTPS DL に自動フォールバック
```

応急処置なら環境変数 `HF_HUB_DISABLE_XET=1` でもいい (テンプレート側にも保険で `os.environ.setdefault` を入れた)。だが恒久的には**アンインストールが一番確実**で、環境全体・全プロセスで二度と踏まない。外した直後、resnet18 の重みは 1.1 秒で落ちてきた。

> 「HTTP は 200 なのに本体 DL が 0 バイトで futex 待ち」を見たら、まず hf-xet を疑う。

## もう一つの罠: `device_count == 2`

GPU の確認で `torch.cuda.device_count()` が **2** を返した。dGPU (RX 7900 XTX) に加えて、CPU 側の iGPU まで ROCm が列挙していた。学習が間違って iGPU に載ると遅いし VRAM も足りない。

なので学習スクリプトは **`cuda:0` (= dGPU) 固定**にし、起動時に `HIP_VISIBLE_DEVICES=0` を渡して iGPU を隠す。`device 0` が dGPU であることは `get_device_name(0)` で確認済み。

## これで何が解禁されたか

torch / torchvision / timm / transformers が揃ったので、画像テンプレート (timm 事前学習バックボーン + GPU + AMP + TTA) と transformer NLP テンプレート (DistilBERT fine-tune) を実装できた。どちらも合成データの GPU スモークで end-to-end の動作 (学習 → OOF → 提出ファイル生成) を確認済み。

## まとめ

- AMD GPU の ROCm torch は「**システム ROCm に合った wheel を選ぶ → dry-run で爆風半径を測る → 実テンソル演算で確認**」の順で淡々とやれば素直に入る
- gfx1100 は素のままで動く。`is_available()` を信じず matmul まで通す
- 2026 時点の最大の地雷は torch ではなく **hf-xet**。NAT 越しでハングしたら `pip uninstall hf-xet`
- iGPU 混在で `device_count` が増えるので、dGPU を `cuda:0` / `HIP_VISIBLE_DEVICES=0` で固定する
