---
title: "Kaggle 専用 v8 を 7 層の汎用マルチエージェント基盤 (KRS-Core) にリファクタした全工程"
emoji: "🏗️"
type: "tech"
topics: ["python", "llm", "kaggle", "langgraph", "refactor"]
published: true
publication_date: "2026-06-08"
---

## TL;DR

- Kaggle 専用に育ってきた `kaggle-research-system v8` を、汎用マルチエージェント基盤 **KRS-Core v0.1.0** へリファクタした
- 設計の中心は **7 層アーキテクチャ** (`core/knowledge` から `core/interfaces` まで) + **BaseAgent ABC** + **ModelRouter (TaskType → LLM 自動選択)**
- 「Kaggle Agent」は数ある Agent の 1 つとして `agents/kaggle_agent/` 配下に隔離
- 新しい Agent を `BaseAgent` 継承 + `AgentRegistry.register()` だけで載せられる状態にした
- リファクタ後の **Phase 0 衛生化** で CI 復旧 / `.env` 履歴除去 / Linux→GitHub SSH:22 ブロック問題まで一気に解決

## 動機: なぜ Kaggle 専用のままだと詰むのか

これまで動かしていたのは `kaggle-research-system v8` という、ひとつの巨大な LangGraph パイプラインに Planner / Coder / Validator / Judge / Reasoner ... を全部詰め込んだモノでした。

実コンペで結果が出始める一方、次の野望が見えてきます。

- 自然言語からアプリを生成する **APPGENAgent**
- 作業ログから技術記事を書く **BlogGenAgent**
- 将来的に複数 Agent が協調する **KRS-OS**

これらを v8 にそのまま増築すると、コンペ用に最適化した state や judge ロジックが他ドメインに漏れ出して破綻します。「Kaggle Agent はあくまで 1 つの Agent」という構造を作る必要がありました。

## 7 層アーキテクチャ

リファクタ後の `core/` は以下の 7 層に分かれています。下から上へ依存方向が一方通行になっています。

```text
07  interfaces/   FastAPI · AgentRunRequest                ← 公開 API
06  skills/       SkillExecutor · prompt injection
05  runtime/      BaseAgent ABC · lifecycle mgmt           ← Agent の骨格
04  tools/        ShellTool · PythonTool
03  router/       TaskType → LLM 自動選択                  ← モデル決定
02  memory/       ProjectMemory 汎用基底
01  knowledge/    SkillLibrary · EpisodeMemory             ← 知識の井戸
```

Kaggle 固有の処理 (analyze / leak_check / cv_simulate / branch_explorer / playbook など) は全部 `agents/kaggle_agent/` 配下に隔離。`core` から `agents` を import するのは禁止 (CI で `from agents` 検出を入れる予定)。

### BaseAgent ABC

すべての Agent はこの ABC を継承します。

```python
class BaseAgent(ABC):
    domain: str

    @abstractmethod
    def build_graph(self) -> CompiledGraph: ...

    @abstractmethod
    def build_initial_state(self, params: dict) -> dict: ...

    def run(self, params: dict) -> dict:
        graph = self.build_graph()
        state = self.build_initial_state(params)
        return graph.invoke(state)
```

KaggleAgent はこれを継承して LangGraph の 13 ノードを返すだけ。APPGENAgent も同じ ABC を継承するだけで、`AgentRegistry` 経由で公開されます。

### ModelRouter — TaskType でモデルを決める

ノード側で `model = ollama_or_claude_based_on_complexity()` のようなアドホックなロジックが散らばっていたのを、`TaskType.PLAN / DIVERSIFY / FIX / JUDGE / REVIEW` のように抽象化して、Router が `.env` の設定を読んで割り当てるようにしました。

```python
team = get_team()              # 内部で router を解決
plan = team.plan(ctx)          # PLAN は deepseek-r1:32b
code = team.fix(...)           # FIX は qwen2.5-coder:32b
verdict = team.judge(...)      # JUDGE は claude-sonnet (ENABLE_CLOUD=true 時)
```

設定変更は `.env` 1 ファイルで済むようになりました。

## Phase 0 — 衛生化で何を直したか

新しい棚卸しが終わった後、放置していた負債を一気に片付けたのが Phase 0 です。

### 1. CI 復旧

`tests/test_orchestrator_graph.py` は 1 世代前の `core.orchestrator` を import していて、リファクタ後に存在しないモジュールを呼んでいました。pytest 全滅状態。

→ `agents.kaggle_agent.graph.build_graph` 経由に書き換え。GitHub Actions に `ruff check + pytest` を入れて緑化。

### 2. setuptools の flat-layout 問題

`core/` と `agents/` の 2 パッケージ並列が `pyproject.toml` から見えていなくて wheel が壊れていました。

→ `packages = ["core", "agents"]` を明示してビルドを正常化。

### 3. `.env` を `.gitignore` に追加 + 履歴除去

`.env` (Anthropic / Kaggle のキー入り) がコミット `900a9e2` 起源で履歴に残っていました。Private リポジトリでも履歴経由で漏れるので即対応:

```bash
# Mac mini で
git filter-repo --invert-paths --path .env
git push --force origin main
```

API キーは Anthropic 側 / Kaggle 側で両方再発行済み。

### 4. Linux→GitHub SSH:22 が中間装置で塞がれていた

これは Phase 0 で最も衝撃的だった発見。LinuxPC で `ssh -T git@github.com -v` すると:

```text
debug1: Remote protocol version 2.0, remote software version 6279353
debug1: compat_banner: no match: 6279353
```

GitHub の本物の banner は `babeld-XXXXXXXX` のはず。中間装置 (ISP の透過プロキシか何か) が SSH を握り潰していたわけです。HTTPS:443 は通る (Anthropic / Kaggle API は動く) ので「ネットワークがおかしい」とは気づきにくい。

結果として、**Linux からの `git push` は不可能 → push は必ず Mac mini から、Linux への展開は rsync ベース**という運用に固定しました。

```bash
# scripts/deploy_from_mac.sh
rsync -a --delete \
  --exclude '.env' \
  --exclude 'workspaces/' \
  ~/projects/kaggle-research-system/ \
  linuxpc:~/projects/kaggle-research-system/
ssh linuxpc 'systemctl --user restart kaggle-api kaggle-ui'
```

`.git/` ごと rsync するので Linux 側でも `git log` でデプロイ済みコミットが追えます。

## リファクタの副産物

新基盤に乗り換えた直後から、以下が「自然に」見えるようになりました。

- **AgentRegistry**: `registry.register("kaggle", KaggleAgent)` の 1 行で公開 API が生える
- **state_factory**: 旧 2 系統だった初期化 (jobs.py 経由 vs /core 経由) を 1 つに統合
- **共有 memory (`core/memory/`)**: GrandmasterMemory を全 Agent 横断で使える基底に
- **systemd --user**: `kaggle-api` / `kaggle-ui` を root 不要で常駐 (`loginctl enable-linger`)

## 次に向かう先

このリファクタは KRS-Core という土台を作っただけで、ここから先が本番です。

1. Knowledge ループの完成 (Episode 書き戻し + Planner recall を実コンペで実証)
2. Kaggle で実際に LB best を更新する Agent 動作
3. APPGENAgent / BlogGenAgent を載せる
4. 複数 Agent が共有 Memory 経由で連携する KRS-OS

次の記事では、この基盤の上で **「KRS が過去の自分から学ぶ Knowledge ループ」を閉じるまで**の話を書きます。

## 参考: 主要コミット

- `5371b83` feat: KRS-Core v0.1.0 — 7-layer generic multi-agent platform
- `70d0953` chore(phase-0): hygiene — fix broken test, add CI, document operational policy
- `d5fa711` docs(phase-0): finalize operational model — rsync deploy, Linux SSH:22 finding
- `97598f1` fix(ci): explicitly declare setuptools packages

リポジトリ: [github.com/motomasa-honda/kaggle-research-system](https://github.com/motomasa-honda)
