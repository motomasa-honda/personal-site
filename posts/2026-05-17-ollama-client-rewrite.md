---
title: "ローカルLLMパイプラインのOllama呼び出しを全面刷新した話 — think:falseバグとMac依存の排除"
emoji: "🧠"
type: "tech"
topics: ["ollama", "python", "llm", "mac", "ubuntu"]
published: true
publication_date: "2026-05-17"
---

## TL;DR

- Ollama Python ライブラリ v0.21.2 に `thinking` フィールドのバグがあり `content` が空になる
- `/api/generate` エンドポイントを使っていたが `/api/chat` + `think: false` に変更が必要
- Critic エージェントが Mac mini の Ollama を叩く設計になっていて Ubuntu が単独で動けなかった
- `OllamaClient` クラスを自作して全 LLM 呼び出しを統一し、両問題を一気に解消した

## 問題の背景

Mac mini M4（IP: 192.0.2.1）と Ubuntu 24.04 PC（Ryzen 9 9950X / RX 7900 XTX）の2台構成でKaggle自動分析パイプラインを動かしている。重いLLM推論はUbuntu側のOllamaで実行し、MACはGitHub pushとKaggle提出の司令塔という役割分担だ。

ところがコードを読み返すと、LLM呼び出しが3種類の関数に散らばっていた。

```python
ask_linux_ssh(prompt, model)       # /api/generate エンドポイント
ask_linux_ssh_stream(prompt, model, placeholder)  # streaming版
chat_linux(model, prompt)          # streaming + ログ保存付き
chat_mac(model, prompt)            # Mac Ollama へ直接
```

しかも `chat_linux()` の実装が `/api/generate` を使っていた。

```python
def chat_linux(model: str, prompt: str, timeout: int = 900) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,   # ← /api/generate 形式
        "stream": True
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",  # ← 旧エンドポイント
        ...
    )
```

## Ollama v0.21.2 の thinking バグ

Ollama Python ライブラリの v0.21.2 には、`thinking` フィールドに関するバグがある。deepseek-r1 や qwen3 シリーズの thinking モデルを使うと、レスポンスの `content` フィールドが空文字列になるケースがある。

原因は Ollama の REST API が thinking トークンを別フィールドで返す仕様なのに、Python ライブラリ側がそれを正しくハンドリングできていないこと。

回避策は2つ。

1. REST API を `urllib.request` で直叩きして `think: false` をトップレベルに指定する
2. `/api/chat` エンドポイントを使う（`/api/generate` は非推奨化されつつある）

```bash
# 動作確認コマンド（think:false が必須）
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen3.6:27b",
  "stream": false,
  "think": false,
  "messages": [{"role": "user", "content": "Say OK"}]
}' | python3 -m json.tool | grep content
```

## OllamaClient クラスを自作した

`base.py` に `OllamaClient` を実装して全呼び出しを統一した。

```python
class OllamaClient:
    """
    Ollama を REST API 直叩きで呼び出す。
    think: false をトップレベル指定して thinking バグを回避。
    """
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _build_payload(self, model, prompt, system, stream, **kw) -> bytes:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        body = {
            "model": model,
            "messages": msgs,
            "stream": stream,
            "think": False,          # ★ thinking バグ回避
            "options": {
                "temperature": kw.get("temperature", 0.2),
                "num_predict": kw.get("max_tokens", -1),
                "num_ctx": kw.get("num_ctx", 8192),
            },
        }
        return json.dumps(body).encode("utf-8")

    def chat(self, model, prompt, system="", **kwargs) -> ChatResponse:
        data = self._build_payload(model, prompt, system, stream=False, **kwargs)
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
        content = payload.get("message", {}).get("content", "") or ""
        return ChatResponse(content=content, model=model, ...)

    def chat_stream(self, model, prompt, system="", **kwargs):
        data = self._build_payload(model, prompt, system, stream=True, **kwargs)
        # ... streaming実装
```

インスタンスはモジュールレベルで生成。

```python
# kaggle_optimizer.py の先頭付近
from base import OllamaClient, ChatResponse

_ollama_linux = OllamaClient(LINUX_OLLAMA_URL)   # "http://localhost:11434"
_ollama_mac   = OllamaClient(MAC_OLLAMA_HOST)     # "http://192.0.2.1:11434"
```

## chat_linux() を OllamaClient ベースに刷新

既存の `chat_linux()` はログ保存と Streamlit への streaming 表示という実用的な機能を持っていたので、そこだけ残して内部を差し替えた。

```python
def chat_linux(model: str, prompt: str, timeout: int = 900) -> str:
    """OllamaClient(/api/chat + think:false)を使用してバグを回避。"""
    placeholder = st.session_state.get("stream_placeholder", None)
    role_label = st.session_state.get("stream_role", model)
    full_text = ""

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n\n=== {datetime.now()} [{role_label}] ===\n")
        # ← ここを chat_stream() に変更
        for token in _ollama_linux.chat_stream(model, prompt, timeout=timeout):
            full_text += token
            logf.write(token)
            logf.flush()
            if placeholder is not None:
                lines_buf = full_text.split("\n")
                display = "\n".join(lines_buf[-30:])
                placeholder.code(f"[{role_label}]\n{display}", ...)
    return full_text
```

動作確認は一発で通った。

```
OllamaClient テスト: OK
```

## Critic の Mac 依存を排除した

コードを読んでいて一番驚いたのがここ。

```python
def run_critic(plan, task, dataset_info):
    """Gemma4 Critic: プロンプト短縮+リトライ+Linuxフォールバック"""
    for attempt in range(3):
        try:
            feedback = chat_mac(LINUX_MODEL_CRITIC, prompt)  # ← Mac を叩いている！
```

コメントには「Linux Critic」と書いてあるのに実装は `chat_mac()` でMac Ollamaを叩いていた。しかも `LINUX_MODEL_CRITIC`（= `qwen3.6:27b`）というLinux用のモデル名をMacのOllamaに渡している。MacにそのモデルがあればたまたまCI動作するが、Macが落ちているとCriticが全滅する。

フォールバックもひどかった。

```python
    except Exception:
        feedback = chat(LINUX_MODEL_FIXER, prompt)  # ← chat() は未定義！
```

`chat()` という関数は存在しない。フォールバックするたびに `NameError` が出る状態だった。

修正後：

```python
def run_critic(plan, task, dataset_info):
    """Ubuntu Ollama (LINUX_MODEL_CRITIC) で実行。Mac依存を排除。"""
    for attempt in range(3):
        try:
            resp = _ollama_linux.chat(LINUX_MODEL_CRITIC, prompt)
            if resp.error:
                raise RuntimeError(resp.error)
            feedback = resp.content
            ...
        except Exception as e:
            add_log(f"⚠️ [Critic] 試行{attempt+1} 失敗: {e}")
    # フォールバックも chat_linux() で統一
    feedback = chat_linux(LINUX_MODEL_FIXER, prompt)
```

これでUbuntu単体でパイプライン全体が動くようになった。

## OLLAMA_KEEP_ALIVE=0 の設定

ついでに VRAM 節約のための設定も追加した。エージェント切り替え時にモデルを毎回アンロードする設定。

```bash
sudo systemctl edit ollama.service
```

```ini
[Service]
Environment="ROCR_VISIBLE_DEVICES=0"
Environment="OLLAMA_KEEP_ALIVE=0"
```

`ROCR_VISIBLE_DEVICES=0` はマザーボード内蔵の iGPU を除外して RX 7900 XTX だけ使う設定。これを忘れると iGPU に推論が割り当たってパフォーマンスが激落ちする。

## 学んだこと

- Ollama Python ライブラリは最新版でもバグがある。重要な箇所は REST API 直叩きが安定
- `think: false` はトップレベルに指定する必要がある。`options` の中に入れても効かない
- 「LinuxモデルをMacで動かす」という設計の矛盾は、変数名に `LINUX_` と書いてあっても気づきにくい。コードレビューが大切
- フォールバック処理に未定義関数を書いてしまう凡ミスは、静的解析かテストがないと気づきにくい

## 参考

- Ollama `/api/chat` ドキュメント: https://github.com/ollama/ollama/blob/main/docs/api.md
- 修正コミット: `feat: Phase1-3 refactor + Grandmaster Playbook`
- 環境: Ubuntu 24.04 / RX 7900 XTX 24GB / Ollama / Mac mini M4
