---
title: "Kaggleスコアが上がらない本当の理由——同じLLMに全部任せるのをやめた話"
emoji: "🤖"
type: "tech"
topics: ["ollama", "llm", "kaggle", "python", "機械学習"]
published: true
publication_date: "2026-05-07"
---

## TL;DR
- Kaggle自動化パイプラインで「AIに任せすぎ」問題が発覚
- 同一モデルがPlanner・Coder・Fixerを兼任すると、同じ思考パターンでバグを再生産する
- モデルを役割ごとに分離し、**Criticを別アーキテクチャで追加**することで盲点を検出できるようにした
- Mac mini（軽量・監視）+ Linux PC（重量処理）の2台構成でコスト最適化

## なぜスコアが上がらなかったのか

Kaggle自動化パイプラインを組んで数ヶ月。CVスコアはそこそこ出るのにLBスコアが伸びない、改善が同じところをぐるぐる回っている——という状態が続いていた。

原因を振り返ると明確だった。

```
Planner(qwen2.5:14b) → 計画立案
    ↓
Coder(qwen2.5-coder:32b) → コード生成
    ↓
Fixer(deepseek-r1:32b) → コード修正
    ↓
Validator → 静的解析
```

このパイプラインの問題は「**Plannerが間違えた方向性をCoderが忠実に実装し、Fixerも同じ思考パターンで直したつもりのバグを量産する**」こと。ループが閉じていて、外部からの批判的視点がない。

## 新しい設計: 役割の分離とCriticの追加

解決策は単純で、**各ロールを得意なモデルに分け、Plannerとは異なるアーキテクチャのCriticを追加する**こと。

```
| 役割      | モデル              | 場所      |
|-----------|---------------------|-----------|
| Planner   | deepseek-r1:32b     | Linux     |
| Critic    | gemma4:26b          | Mac mini  |  ← 新設
| Coder     | qwen3.6:27b         | Linux     |
| Fixer     | deepseek-r1:32b     | Linux     |
| Reviewer  | deepseek-r1:32b     | Linux     |
| Reasoner  | deepseek-r1:70b     | Linux     |  ← 停滞時のみ
| Validator | claude-sonnet-4-6   | Claude API|
| Judge     | claude-sonnet-4-6   | Claude API|
```

**ポイントはCriticをPlannerと別アーキテクチャにすること。** 同じモデルに批評させると同じ盲点を持つ。deepseek-R1系とGemma4系では推論の傾向が異なるため、片方が見落とした問題をもう片方が検出できる。

## Criticの実装

CriticはPlannerの計画を受け取り、承認/NG を返す。NGの場合はPlannerに差し戻す。

```python
CRITIC_NG_KEYWORDS = [
    "問題あり", "要修正", "NG", "リーク", "データリーク",
    "問題があります", "修正が必要", "懸念があります",
]

def run_critic(plan: str, task: str, dataset_info: dict) -> tuple[bool, str]:
    prompt = f"""あなたはKaggleコンテストの批評家です。
以下の実装計画を批判的に評価してください。

【タスク】{task}
【提案された計画】{plan}

確認観点:
1. データリークのリスク
2. 特徴量エンジニアリングの妥当性
3. train/testの整合性

問題がなければ「承認」、問題があれば「問題あり」と最初の行に書いてください。
"""
    feedback = chat_mac(LINUX_MODEL_CRITIC, prompt)  # Mac mini側で実行
    approved = not any(kw in feedback for kw in CRITIC_NG_KEYWORDS)
    return approved, feedback
```

差し戻しのループ（最大2回）：

```python
for critic_retry in range(CRITIC_MAX_RETRY + 1):
    approved, feedback = run_critic(plan_text, comp_description, dataset_info)
    
    if approved:
        add_log(f"✅ [Critic] 計画を承認（{critic_retry}回目）")
        break
    else:
        add_log(f"⚠️ [Critic] NG → Plannerに差し戻し")
        # Plannerに修正させる
        plan_text = chat_linux(
            LINUX_MODEL_PLANNER,
            f"Criticから指摘がありました。修正してください。\n\n指摘: {feedback}\n\n元の計画: {plan_text}"
        )
```

## 70Bモデル(Reasoner)の位置付け

deepseek-r1:70bは**常時使うのではなく、改善が停滞した時だけ投入する**設計にした。

理由はシンプルで、**速度がボトルネックになるから**。Kaggleで重要なのは試行回数であり、70Bで深い1回より32Bで速い3回の方が現実的にスコアが上がる。

```python
REASONER_TRIGGER_COUNT = 3  # 改善なしがこの回数続いたら70Bを起動

if (
    use_reasoner_auto
    and st.session_state.no_improve_count >= REASONER_TRIGGER_COUNT
    and not st.session_state.reasoner_used
):
    add_log("🧠 [Reasoner/70B] 改善停滞を検知。深い推論を開始...")
    reasoner_output = run_reasoner(...)
    st.session_state.reasoner_used = True
```

ベスト更新時にReasonerフラグをリセットすることで、次の停滞まで再起動しない。

## Mac mini + Linux PCの2台構成

もう一つのポイントは**役割による物理マシンの分離**。

- **Mac mini M4**: Critic（軽量判断）、監視・UI表示
- **Linux PC（Ryzen 9 9950X / RX 7900 XTX）**: Coder、Fixer、Planner（重量処理）

Criticは判断タスクなので軽量モデルで十分。Mac miniのgemma4:26bがちょうどいい。一方でコード生成・修正は大量のトークンを消費するのでLinux側に任せる。

Ollama同士の接続はHTTPで直結：

```python
# Mac mini側（192.0.2.1）
MAC_OLLAMA_HOST = "http://192.0.2.1:11434"
client_mac = Client(host=MAC_OLLAMA_HOST)

# Linux側（localhost、Streamlit自体がLinuxで動作）
def ask_linux(prompt, model, timeout=900):
    import urllib.request, json
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"]
```

Streamlit自体はLinux上で動かし、Mac miniのブラウザからアクセスする構成。

## 学んだこと
- **同一モデルに複数ロールを担わせると同じバイアスでループする**。役割分離は設計の基本
- CriticはPlannerと**別アーキテクチャ**であることが重要。同じ系統だと同じ盲点を持つ
- 70Bは「常時稼働」より「停滞時の切り札」として使う方がKaggleの試行回数的に有利
- Streamlitの接続テストボタンは早期に実装しておくと環境起因のバグ切り分けが楽
- Mac mini側のOllamaはデフォルトでは外部アクセス不可。`launchctl setenv OLLAMA_HOST "0.0.0.0"` が必要

## 参考
- Ollama公式ドキュメント: https://ollama.com
- qwen3.6:27b SWE-bench Verified 77.2% (2026年4月リリース)
- deepseek-r1:32b / 70b: MIT License、ローカル実行可能
- gemma4:26b: Google、Mac mini M4で快適動作
