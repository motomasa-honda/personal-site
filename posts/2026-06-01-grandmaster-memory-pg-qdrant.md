---
title: "「賢いAIより記憶するAI」— KaggleグランドマスターAIの設計方針とPostgreSQL+Qdrant移行の意思決定"
emoji: "🧠"
type: "tech"
topics: ["kaggle", "llm", "postgresql", "qdrant", "設計"]
published: false
publication_date: "2026-06-01"
---

## TL;DR
- Kaggle自動化AI（KRS）を「Grandmaster Workflow AI」に進化させる設計を考えた
- 結論：AgentをたくさんよりDBを深く。「推論能力」より「経験蓄積能力」が本質
- SQLite（既存）→ PostgreSQL + Qdrantへの移行を決定。二度手間を避けるため最初から本番設計で行く
- Competition Similarity SearchとError Analysis Agentが最もROIが高い

---

## 現状のKRSと「Grandmaster」のギャップ

KRS v7.2はLangGraphベースの13ノードパイプラインで、コンペURLを投げるとデータ分析→コード生成→実行→submitまで自動でやってくれる。技術的には「優秀なAIコーディングAgent」に近づいている。

しかしKaggle Grandmasterの本質を考えると、決定的に足りないものがある。

Grandmasterが強い理由は**過去の経験の再利用**だ。

```
問題を見る
↓
「これ、去年の〇〇コンペと同じパターンだ」
↓
有効だった手法をすぐに試す
↓
失敗した手法は最初からスキップ
↓
誤差分析で「高齢者層に弱い」を発見 → 特徴量追加
```

現状のKRS v7.2の問題はここだ：

```
Agentを実行
↓
経験を構造化して保存できない
↓
次のコンペでゼロから考え直す
```

207 SKILLが蓄積されているが、「このSKILLはどのコンペで効いたか」「どの特徴量がどれだけCVを改善したか」という**実験レベルの知識**が記録されていない。

---

## SQLiteのままスキーマ拡張するか？ → しない

最初はSQLiteにテーブルを追加して段階的に拡張しようとした。しかし考え直した。

**SQLiteの限界:**
- 並行書き込みに弱い（Agentが並列実行されると詰まる）
- 全文検索が貧弱
- ベクトル検索は別途Qdrantが必要なので結局2システム管理になる

**二度手間を避けるため、最初からPostgreSQL + Qdrantで設計する。**

---

## 新しいアーキテクチャ設計

```
PostgreSQL (事実を保存する)
    competitions    -- コンペメタ情報 + 類似度ベクトル
    experiments     -- 実験記録（旧SQLiteのtrialsを移行）
    features        -- 特徴量知識ライブラリ
    experiment_features -- 実験×特徴量の貢献度記録
    sample_errors   -- サンプルレベルの誤差（Error Analysis基盤）
    research        -- 外部知識（Discussion/GitHub/arxiv）

Qdrant (思い出す装置)
    competition_vectors  -- コンペ類似度検索
    feature_vectors      -- 特徴量類似度検索
    experiment_vectors   -- 過去実験の類似検索
    research_vectors     -- 研究知識の検索

        ↓
Grandmaster Memory（統合クエリ層）
        ↓
Plannerへの一括注入
```

Obsidianの位置付けも整理した：**ObsidianはDBではなく可視化ツール**。PostgreSQL + QdrantのデータをMarkdownに書き出してObsidianで見る。

---

## Competition Similarity Search の設計

新規コンペが来たときに「これは過去のどのコンペに似ているか」を自動検索する。

コンペを28次元ベクトルで表現する：

```python
def compute_competition_vector(meta: dict) -> np.ndarray:
    # データ構造 (6次元)
    structural = [
        np.log1p(meta["n_rows"]),
        np.log1p(meta["n_cols"]),
        meta["missing_ratio"],
        meta["n_cat_cols"] / max(meta["n_cols"], 1),
        meta["n_num_cols"] / max(meta["n_cols"], 1),
        meta["target_imbalance"],
    ]

    # タスク種別 (5次元 one-hot)
    task_types = ["tabular_classification", "tabular_regression",
                  "timeseries", "nlp", "image"]
    task_vec = [1 if meta["task_type"] == t else 0 for t in task_types]

    # 評価指標 (8次元 one-hot)
    metrics = ["auc", "accuracy", "f1", "rmse", "mae", "logloss", "map", "other"]
    metric_lc = meta["eval_metric"].lower()
    metric_vec = [1 if m in metric_lc else 0 for m in metrics]

    # ターゲット種別 (3次元 one-hot)
    target_types = ["binary", "multiclass", "regression"]
    target_vec = [1 if meta["target_type"] == t else 0 for t in target_types]

    return np.array(structural + task_vec + metric_vec + target_vec)
```

これをQdrantに保存して、新規コンペ投入時にcosine similarityで検索する。

**Plannerへの注入イメージ:**

```
## 類似コンペの知見 (Competition Similarity Search)

【1位: spaceship-titanic (類似度0.91)】
- 有効特徴量: CabinDeck分割, GroupSize, TotalSpend合計
- 有効CV: StratifiedKFold(5)
- 有効モデル: LightGBM + CatBoost ensemble
- 注意点: 欠損値が意味を持つ (Cold Sleep中は欠損)
- 失敗事例: NNは過学習した

【2位: tabular-playground-s03e04 (類似度0.84)】
...
```

これがあるとPlannerの初手の質が劇的に変わる。「ゼロから考える」から「過去の知見を踏まえて考える」になる。

---

## Error Analysis Agentが最優先の理由

ロードマップ検討で「Error Analysis AgentはなぜV18なのか」という疑問が出た。

Grandmasterが初日にやることを思い出してほしい：

```python
# OOF予測と正解を突き合わせる
errors = true_labels - oof_preds
high_error_idx = np.argsort(np.abs(errors))[-100:]
df_train.iloc[high_error_idx]  # 高誤差サンプルを見る
```

これで「60歳以上の男性に弱い」「欠損値があるサンプルに弱い」がすぐわかる。

CVスコアという1つの数字だけ見て次の施策を考えるのはアマチュアの発想だ。Error Analysisがあれば：

```
CV: 0.842 → 0.851（+0.009）
Error Analysis: 高齢者層（60歳以上）の誤差が全体の40%を占める
→ 次の施策: 年齢帯別の特徴量を追加
```

という具体的な改善ループが回る。V18ではなくV9〜V10で実装すべき機能だ。

---

## 実装順序の最終決定

```
v8a: Docker環境構築 (PostgreSQL + Qdrant)
v8b: PostgreSQLスキーマ設計・マイグレーション
v8c: analyze.pyでコンペメタを自動保存
v8d: execute.pyでfeature importanceを自動保存

v9a: compute_competition_vector()実装
v9b: Qdrant collection: competition_vectors 作成
v9c: find_similar_competitions()実装
v9d: Plannerへの注入テキスト生成

v10a: Error Analysis Agentの実装
v10b: LangGraphへの組み込み (execute → error_analysis → submit)
v10c: sample_errorsテーブルへの保存

v11: Feature Library (Qdrant: feature_vectors)
v12: Grandmaster Memory統合クラス
```

---

## Qdrantを今すぐ入れない理由もあった

最初「Qdrantは後回しでNumPyのcosine similarityでプロトタイプ」という案もあった。

```python
# プロトタイプ案（Qdrantなし）
from sklearn.metrics.pairwise import cosine_similarity
all_vecs = load_all_competition_vectors()
sims = cosine_similarity([new_vec], all_vecs)
```

これで動くには動く。でも最終的にQdrantに移行するなら、データの移行コストより**最初から正しい設計で進む方がトータルコストが低い**という判断でQdrant採用に踏み切った。

コンペ数が100を超えたあたりでNumPy配列の全件スキャンがボトルネックになるのは目に見えている。

---

## ハードウェア方針の補足

長期的にはRAM増設を検討している。現在：
- RAM 64GB
- VRAM 24GB (RX 7900 XTX)

deepseek-r1:70b（Q4量子化で約40GB）を動かすにはVRAMが足りないのでRAMオフロードが発生している。

**推奨方針（コスパ順）:**
1. RAM 128GBに増設（Ryzen 9 9950XはDDR5対応、比較的安価）
2. RX 7900 XTXを2枚に（VRAM 48GB、ROCm multi-GPUは課題あり）
3. CUDA Linux Server（ROCm縛りから解放される）

Mac Studio Ultra 256GBは魅力的だが、ROCm + Ollamaのエコシステムから外れるのでPyTorch系ライブラリとの相性が未知数。

---

## 学んだこと

- **「賢いAgent」より「記憶するDB」を先に作れ**: 実験のたびにゼロスタートでは進歩しない
- **SQLiteは個人プロジェクトの最初だけ**: 並行処理・ベクトル検索が必要になったら即座に移行
- **Competition Similarity Searchは技術的核心**: ここが差別化になる
- **Error AnalysisはROI最大の機能**: CVスコアだけ見るな
- **二度手間の恐怖より、正しい設計の価値が上**: 最初から本番設計で進む

## 参考
- `core/tracker/experiment_db.py` - 現在のSQLiteスキーマ
- `core/knowledge/episode_memory.py` - 既存のEpisode Memory設計
- `core/orchestrator/nodes/analyze.py` - Plannerへの知識注入の現状
- docs/KRS_v8_handoff.md - v8設計詳細
