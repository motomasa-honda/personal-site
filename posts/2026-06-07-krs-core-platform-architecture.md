---
title: "Kaggle自動化ツールを『汎用AIエージェント基盤』に育てる設計判断の記録"
emoji: "🧠"
type: "tech"
topics: ["ai", "llm", "kaggle", "postgresql", "qdrant"]
published: true
publication_date: "2026-06-07"
---

## TL;DR

- Kaggle専用だったKRSをKRS-Core（汎用マルチエージェント基盤）へリファクタリングしている
- 「汎用化」と「今動くものを作る」の両立は難しく、方針を3回変えた
- 最終的に「v8を動かしてから抽象化する」ではなく「Core Skeleton先に作る」に落ち着いた
- PostgreSQL + Qdrant + GrandmasterMemoryの構成はKaggle専用概念を排除すれば汎用基盤になる

---

## なぜKaggle専用ツールを汎用基盤に育てようとしているのか

もともとKRSはKaggleコンペを自動分析するツールとして作り始めた。LangGraphで13ノードのパイプラインを組み、deepseek-r1:32bやqwen3:27bといったローカルLLMがコードを生成・実行・提出するというシステムだ。

しばらく動かしてみると、「このパイプライン、Kaggle以外でも使えるんじゃないか」と気づいた。Goal → Plan → Execute → Review → Learn というサイクルは、データ分析に限らずコーディング支援にも、業務自動化にも応用できる。

というわけで、Kaggle専用だった設計を汎用化するKRS-Core構想が生まれた。

---

## 方針決定まで3回ブレた話

正直に書く。この設計判断、3回変わった。

**1回目: v8をそのまま完成させてから後でKRS-Core化**

「動くものが先」という判断。Kaggle専用のまま実装して、後で共通部分を抽出するつもりだった。

問題点: v8の実装が全部Kaggle専用の命名になる。`competition_vectors`、`GrandmasterMemory`、`cv_score`。これを後で汎用化するコストが大きい。

**2回目: 最初からKRS-OSとして設計する**

Karpathyが提唱するLLM WikiやSynapse AIを参考に、「AIを動かすOS」として最初から設計しようとした。

問題点: 壮大すぎて手が動かない。実際に動くものが1つもない状態でOS設計をしても、使われないレイヤーを作り込むだけ。

**3回目 (確定): v8パッチを適用しつつ、Core Skeletonを先に作る**

- v8のパッチ（バグ修正・記憶基盤・Error Analysis等）は全部適用する
- 同時にCore Skeletonだけ先に作っておく
- Kaggle専用コードは`agents/kaggle_agent/`に分離する
- 命名だけ汎用化する（`competition_vectors` → `project_vectors`等）

これが今の方針。

---

## KRS-Coreの7層アーキテクチャ

確定した構成はこうなった。

```
core/
├── memory/       # Memory Layer — 経験を保存
├── knowledge/    # Knowledge Layer — 抽象化された知識
├── skills/       # Skill Layer — 再利用可能な手順
├── runtime/      # Agent Runtime — 共通ライフサイクル
├── tools/        # Tool Runtime — Shell/Python実行
├── router/       # Model Router — LLMへの振り分け
└── interfaces/   # Interface Layer — FastAPI/CLI

agents/
└── kaggle_agent/ # KRS v8のKaggle専用コードをここへ集約
    ├── nodes/
    ├── playbook/
    ├── templates/
    └── kaggle_client/
```

重要な設計原則が2つある。

**Memory / Knowledge / Skillを混同しない**

- Memory: 実際に起きた出来事（実行履歴、エラー履歴、実験結果）
- Knowledge: Memoryから抽象化されたもの（「LightGBMは欠損値に強い」等）
- Skill: 実行可能な手順（「Kaggle分析手順」「FastAPI構築手順」等）

KnowledgeとSkillを別レイヤにするのがポイント。KnowledgeはKnow、SkillはKnow Howだ。

**AgentはCoreの利用者であって、主役ではない**

```python
# こういう設計にする
class BaseAgent:
    def __init__(self, goal: str, memory: ProjectMemory):
        self.goal = goal
        self.memory = memory  # ← Core共有の記憶層

    def run(self) -> dict:
        raise NotImplementedError

class KaggleAgent(BaseAgent):
    def run(self):
        return run_kaggle_pipeline(self.goal, self.memory)
```

将来CodingAgentやResearchAgentを追加する時も、Core側は変更しない。これが「新Agentを追加する時にCore変更が不要」という設計成功条件。

---

## v8で実装した記憶基盤がそのままCoreになる

v8で実装したPostgreSQL + Qdrantの構成は、Kaggle専用の命名さえ変えればそのままCore層になる。

**PostgreSQLスキーマ（汎用化後）**

```sql
-- Kaggle専用だった名前を汎用化
-- competitions → projects
-- experiments  → tasks (実験もタスクの一種)

CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    title TEXT,
    task_type TEXT,        -- kaggle_tabular_cls / coding / research ...
    domain TEXT,           -- kaggle / github / custom
    project_vector REAL[], -- 類似プロジェクト検索用
    created_at TIMESTAMPTZ DEFAULT now()
);
```

**Qdrantコレクション（汎用化後）**

```python
# competition_vectors → project_vectors
# 32次元の設計思想はそのまま
COLLECTIONS = [
    ("project_vectors",  32,  "プロジェクト類似度検索"),
    ("feature_vectors",  384, "特徴量・手法の意味検索"),
    ("research_vectors", 384, "知識・Discussion検索"),
]
```

**GrandmasterMemory → ProjectMemory**

```python
# core/memory/project_memory.py
class ProjectMemory:
    """
    全記憶への統合アクセス。
    Kaggle専用だったGrandmasterMemoryを汎用化したもの。
    """
    def recall_for_planner(self, state: dict) -> str:
        # 類似プロジェクト + ベスト戦略 + Error Analysis
        # + 推奨SKILL + 失敗パターン
        ...
```

---

## Kaggle固有概念をCoreに混入させないルール

リファクタリングで一番大事なのはここだ。以下のものはCoreに入れない。

| Kaggle専用 | 判定理由 |
|-----------|---------|
| `competition_vector.py`の32次元設計 | KaggleのメタデータAPIに依存 |
| `backfill_competition_vectors.py` | Kaggle CLI依存 |
| `weekly_knowledge_harvest.py` | Kaggleカーネルのスクレイピング |
| `cv_score` / `lb_score` | Kaggle特有の評価指標 |
| `submit.py` | Kaggle提出API依存 |

逆にそのまま使えるもの（Core候補）:

| ファイル | 理由 |
|---------|------|
| `db/connection.py` | 汎用PostgreSQL接続 |
| `llm/team.py` | どのタスクでも使えるLLMチーム |
| `memory/grandmaster_memory.py` | 命名変更だけでOK |
| `error_analysis.py` | OOF分析はデータ分析全般で使える |
| `model_config.py` | Model Routerとして汎用化済み |

---

## 今後のロードマップ

```
【完了】KRS v8パッチ群の適用
  PostgreSQL + Qdrant基盤
  GrandmasterMemory
  Error Analysis Agent
  Review対応パッチ4本

【次】KRS-Core化
  Step KRS-0: Core Skeleton作成
  Step KRS-1: BaseAgent追加
  Step KRS-2: 命名リファクタリング
  Step KRS-3: Memory/Knowledge/Skill 3層分離
  Step KRS-4: Agent Runtime実装
  Step KRS-5: Tool Runtime (Shell/Python)
  Step KRS-6: Interface Layer整理

【将来】KRS-OS
  KRS-CoreにCodingAgent/ResearchAgentが乗った時点で
  自然にOS的な形になる想定
```

---

## 学んだこと

- 「汎用化」を最初から設計しようとすると手が動かない。まず1つのドメインで動くものを作ることが汎用基盤への最短経路
- Memory / Knowledge / Skillを同一レイヤに詰め込むと後で分離できなくなる。最初から3層に分けておく
- Kaggle専用の命名（`competition_slug`、`GrandmasterMemory`等）をCoreに混入させると、2つ目のAgentを作る時に必ず詰まる
- 設計方針が3回変わることは失敗ではない。動くものが増えるにつれて「何が本当に共通か」が見えてくる

## 参考

- Karpathy LLM Wiki: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- KRS GitHub: https://github.com/motomasa-honda/kaggle-research-system
- 関連: KRS v8 PostgreSQL+Qdrant+GrandmasterMemory実装セッション (2026-06-07)
