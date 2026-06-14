---
title: "OllamaのApple Metal（MPS）クラッシュをリトライ＋Linuxフォールバックで乗り越えた話"
emoji: "🍎"
type: "tech"
topics: ["ollama", "appleSilicon", "llm", "python", "kaggle"]
published: false
publication_date: "2026-05-07"
---

## TL;DR
- Mac mini M4上のOllamaで`gemma4:26b`を呼ぶと`command buffer 1 failed (status 1)` でクラッシュする
- 原因はApple Metal（MPS）のコマンドバッファ枯渇。長いプロンプトを一発送信すると再現する
- **プロンプト短縮 → 最大3回リトライ → 失敗時はLinux側LLMにフォールバック**という3段構えで解決
- Mac単体では動くのに、Pythonから呼ぶと落ちる理由も解説する

---

## 背景：Mac mini M4をCriticエージェントとして使いたかった

KaggleのAI自動化パイプライン（通称KRS）では、Ubuntu（Ryzen 9 9950X / RX 7900 XTX）をメインの推論マシンとして使っている。ただし「同じアーキテクチャのLLMだけでレビューすると盲点が生まれる」という理由で、Mac mini M4上の`gemma4:26b`をCriticエージェント専用に割り当てていた。

構成はこんな感じ：

```
Ubuntu (192.168.2.2)          Mac mini (192.168.2.1)
─────────────────────         ──────────────────────
Planner: deepseek-r1:32b      Critic: gemma4:26b  ← ここが落ちる
Coder:   qwen3.6:27b
Fixer:   deepseek-r1:32b
```

Pythonコードでは`ollama.Client`を使ってMacのOllamaに接続している：

```python
MAC_OLLAMA_HOST = "http://192.168.2.1:11434"
client_mac = Client(host=MAC_OLLAMA_HOST)

def chat_mac(model, prompt):
    res = client_mac.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]
```

---

## エラーの全貌

パイプライン実行中に以下のエラーが出てクラッシュした：

```
ollama._types.ResponseError: an error was encountered while running the model:
error: command buffer 1 failed with status 1
signal arrived during cgo execution
GGML_ASSERT([rsets->data count] == 0) failed
(status code: 500)
```

**Mac側のターミナルで単体テストすると普通に動く：**

```bash
ollama run gemma4:26b "hello"
# → Hello! How can I help you today?  ← 正常
```

つまり「モデルが壊れている」わけではなく、**Pythonから長いプロンプトを送ったときだけ落ちる**。

---

## 原因：Apple MetalのコマンドバッファはPythonからだと枯渇しやすい

`GGML_ASSERT([rsets->data count] == 0)`はllama.cppのMetal実装で発生するアサーション。Metal APIはGPUへの命令をコマンドバッファにキューイングするが、以下の条件が重なると枯渇する：

1. **プロンプトが長い**（今回は計画テキスト＋データセット情報＋評価観点で2000字超）
2. **Pythonプロセスからネットワーク越しに呼ぶ**（ollama CLIと異なりHTTPレイヤーのオーバーヘッドがある）
3. **連続リクエスト**（パイプライン内で前のレスポンスを受けてすぐ次のリクエストを送る）

CLI経由だと内部でバッファが適切にリセットされるが、HTTPサーバー経由だと状態が残りやすいらしい。

---

## 解決策：3段階の防御

### ① プロンプトを短縮する

元のプロンプトは日本語で丁寧に書かれた2000字超のf-string。これを1000字以内に圧縮した：

```python
# Before: 丁寧な日本語プロンプト（〜2000字）
prompt = f"""あなたはKaggleコンテストのコード品質・データ分析の批評家です。
以下の実装計画を批判的かつ具体的に評価してください。
【タスク】
{task}
...（長い）
"""

# After: 要点だけ（〜800字）
plan_short = plan[:1500] if len(plan) > 1500 else plan
task_short = task[:300]  if len(task)  >  300 else task
prompt = (
    "Kaggleコード批評家として以下を評価してください。\n"
    "タスク: " + task_short + "\n"
    "データ: " + str(dataset_info.get('n_rows','?')) + "行/"
              + str(dataset_info.get('n_cols','?')) + "列"
              + " target=" + str(dataset_info.get('target','?')) + "\n"
    "計画:\n" + plan_short + "\n"
    "評価観点: 1)データリーク 2)特徴量 3)CV設計 4)train/test整合性 5)落とし穴\n"
    "問題なし→1行目に「承認」/ 問題あり→1行目に「問題あり」+箇条書き"
)
```

### ② 最大3回リトライ（3秒インターバル）

```python
import time
for attempt in range(3):
    try:
        feedback = chat_mac(LINUX_MODEL_CRITIC, prompt)
        approved = not any(kw in feedback for kw in CRITIC_NG_KEYWORDS)
        return approved, feedback
    except Exception as e:
        add_log("⚠️ [Critic] Mac試行" + str(attempt+1) + "失敗: " + str(e))
        time.sleep(3)  # Metalバッファのリセット待ち
```

### ③ Linux側LLMへのフォールバック

3回とも失敗したら`deepseek-r1:32b`（Ubuntu側）で代替する：

```python
add_log("⚠️ [Critic] MacクラッシュのためLinux(deepseek-r1:32b)フォールバック")
try:
    feedback = chat(LINUX_MODEL_FIXER, prompt)
    approved = not any(kw in feedback for kw in CRITIC_NG_KEYWORDS)
    return approved, feedback
except Exception as e:
    add_log("❌ [Critic] Linuxフォールバックも失敗: " + str(e))
    return True, "Criticスキップ（両方失敗）"
```

「別アーキテクチャでレビューする」という設計思想は維持しつつ、パイプライン全体が止まらないようにした。

---

## 結果

- Macが落ちてもパイプラインが継続するようになった
- プロンプト短縮後はMacクラッシュ自体も激減した
- フォールバック先をFixer用モデル（`deepseek-r1:32b`）にしたのは「すでにLinuxで動作確認済みの`chat()`関数がそのまま使えるから」という実用的な理由

---

## 学んだこと
- Apple Metal（MPS）はプロンプト長に敏感。CLIで動いてもHTTP経由で落ちることがある
- `GGML_ASSERT([rsets->data count] == 0)`が出たらまずプロンプトを短くしてみる
- マルチマシン構成でLLMを使うなら「片方が落ちても全体が止まらない」フォールバック設計は必須
- `time.sleep(3)`のインターバルは「Metalバッファのリセット待ち」として有効に機能する

## 参考
- 対象ファイル: `~/projects/kaggle/kaggle_optimizer.py` の `run_critic()` 関数
- Ollama issue: Metal backend crashes with long prompts (llama.cpp upstream)
- 構成: Mac mini M4 (192.168.2.1:11434) + Ubuntu RX7900XTX (192.168.2.2:11434)
