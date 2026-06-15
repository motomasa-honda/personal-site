---
title: "LangGraph で Knowledge ループを閉じる — Kaggle Agent が過去の自分から学ぶまで"
emoji: "🔁"
type: "tech"
topics: ["python", "llm", "kaggle", "langgraph", "agent"]
published: true
publication_date: "2026-06-11"
---

## TL;DR

- Kaggle Agent (KRS-Core v0.1.0) で **Knowledge ループの閉鎖**を実機実証
- `reflect → EpisodeMemory 書き戻し` (write 側) と `plan → GrandmasterMemory recall` (read 側) を全部繋いだ
- 同一コンペを 3 回走らせて「**前回 episode の successful 戦略を Planner が完全踏襲して LB best を更新**」する観察に成功
- Phase 1 から Phase 7 まで合計 9 段階の修正を経て、概念実証としての KRS-OS マイルストーンを達成

## なぜ Knowledge ループが必要なのか

LLM ベースの自律 Agent は黙っていると **同じ過ちを毎ジョブ繰り返します**。前回 catboost で爆勝ちしても、次の起動では「とりあえず lightgbm から」と言い出す。記憶を持たないからです。

これを解決するのが Episode Memory + Skill Library を組み合わせた **Knowledge ループ**:

```text
analyze → recall (過去の episode を引いてくる)
   ↓
plan → Planner プロンプトに記憶を注入
   ↓
codegen → execute → submit → judge
   ↓
reflect → EpisodeMemory に書き戻し (write side)
   ↓
[次ジョブで analyze が recall する]
```

これは LangGraph のグラフ上では普通の state 遷移ですが、ループの 1 周がジョブを跨ぐ点が肝です。

## 設計上の難所

### 1. condition key の決め方

Episode は「次に類似コンペを見たときに引ける」状態じゃないと意味がない。索引キーを何にするかが全て。

最終的に採用したのは:

```python
condition = f"{task_type}+{evaluation_metric}+{target_type}"
# 例: "tabular_classification+accuracy+binary"
#     "tabular_regression+rmsle+"
```

`target_type` は分類タスクでのみ `binary` / `multi` を付与。回帰で `nunique=663` を `multi` 扱いすると condition が汚れるので除外。

### 2. reflect で書き戻すタイミング

`should_continue=False` (= ジョブ終了) のすべてのパスで `_persist_episode()` を呼ぶ必要があります。早期 return 経路を見落とすと「最後の iter の成功エピソードが書き戻されない」が起こります。

```python
# 全体タイムアウト
if elapsed_total > 8 * 3600:
    state["should_continue"] = False
    _persist_episode(state); return state

# 連続同一エラー
if all(e == last3_errors[0] for e in last3_errors):
    state["should_continue"] = False
    _persist_episode(state); return state

# 目標達成
if best >= target:
    state["should_continue"] = False
    _persist_episode(state); return state

# 最大イテレーション
if iteration >= max_iterations:
    state["should_continue"] = False
    _persist_episode(state); return state
```

4 ヶ所すべてで `_persist_episode` を呼ぶように地道に修正。

### 3. ノイズ抑制ガード

analyze が Kaggle ダウンロード失敗等で dataset_info を埋め損ねると、`condition = "unknown+unknown+"` の empty episode が書き戻され、次ジョブの recall でゴミが混入します。

```python
empty_condition = task_type in ("", "unknown") and metric in ("", "unknown")
no_score        = state.get("best_cv") is None and state.get("best_public") is None
no_signal       = not judge_summaries and not cv_history
if empty_condition and no_score and no_signal:
    logger.warning(f"[reflect] episode skip ({slug})")
    return
```

これを入れて初めて、Planner プロンプトの S/N が許容範囲に収まりました。

### 4. AgentState TypedDict の地雷

LangGraph で `StateGraph(AgentState)` を使うとき、`TypedDict` に**宣言されていないキーは silently drop されます**。

最初これに気づかず、`state["max_iterations"]` がドロップされて常に default の 30 になってしまい、3 iter で止めたかったのに 30 周回り続けるバグを踏みました。

新しい state フィールドを追加するたびに `state.py` の `AgentState` にも宣言を増やす、というルールを CLAUDE.md / memory に固定化。

## Phase 6 — write 側を成立させる

ここまで書き戻しは動いていたものの、`successful_strategies` がほぼ空のまま蓄積される問題が残っていました。

原因は「Judge LLM が `successful_strategies: [...]` のような構造化出力を返したときだけ抽出する」設計だったこと。Judge は文章で帰ってきがちで、構造化に失敗していたわけです。

**Phase 6 の改修**: Judge 出力に依存せず、**CV 改善実績ベース**で `successful_strategies` を直接埋める。

```python
# 直近 history の最良 CV を更新した試行を successful とする
best_cv = -inf
successful = []
for h in history:
    if h["cv_score"] is not None and h["cv_score"] > best_cv:
        successful.append(h["strategy"])
        best_cv = h["cv_score"]
```

これで Knowledge ループの「書き戻し側」が安定して動くようになりました。

## Phase 7 — recall 側を実機実証

Phase 6 で書き戻しが動いている状態で、3 回目の House Prices ジョブを投げた smoke `12e1c261` で以下を観察:

| # | 観察 | 意味 |
|---|------|------|
| 1 | `[plan] GrandmasterMemory: 5009文字注入` × 3 回 | Planner プロンプトに episode + GM が届いている |
| 2 | iter=1 strategy = `baseline:Basic Feature Engineering with Random Forest` | **前回 episode の successful を完全踏襲** |
| 3 | iter=2 strategy = `catboost+target_encoding+groupkfold5+l2` | Ollama が catboost を初選択 (Phase 6-2 解禁の効果) |
| 4 | episode `failed: ['catboost+target_encoding+groupkfold5+l2']` | 失敗側も正しく記録 |
| 5 | best_public 0.13059 → **0.12967** | **recall した戦略がそのまま LB best 更新** |

これが KRS-OS の核心マイルストーン: 「**Agent が過去の自分の経験を見て次の戦略を踏襲し、結果が改善する**」最初の実証。

## 学んだこと

1. **メモリ機構は recall できなければ無価値**。書くだけの memory は dev のメモ帳になりがち。read 経路を実コンペで観察するまで「動いている」と言わない方が良い
2. **condition key は idempotent でなければならない**。同じコンペを 2 回走らせて key が変わると recall 不能
3. **LangGraph state schema は厳密に**。silently drop されるので CI でチェックしたい (これは TODO)
4. **Knowledge ループは小さく始める**。最初から weekly Harvest + Obsidian + Qdrant 全部繋ぐと、どこで詰まったか分からなくなる

## 次に向かう先

ここまでで「Agent が記憶を持つ」が成立しました。次は

- Knowledge を持った Agent の出力品質を、Ensemble (OOF Blending) で LB に押し上げる (Phase 8-2)
- 単一コンペ内で完結していた recall を、コンペ横断 (NLP / multiclass / timeseries) で機能させる (Phase 8-3)
- 複数の Agent が同じ Knowledge を共有する KRS-OS へ

特に Phase 8-2 では「OOF を保存しているのに ensemble submission が一切 LB に届いていなかった」という致命的なバグを発見します。それは次の記事で。

## 参考: 主要コミット

- `4bf7c66` feat(phase1-1b): write back episodes on pipeline termination
- `6bade05` fix(reflect): skip empty-condition episodes to avoid recall noise
- `8756396` fix(state): declare control fields in AgentState
- `d7ef928` feat(phase6): CV-based strategy attribution in episodes
- `9ac78f9` fix(prompt): ban train_test_split + force fold-wise target encoding

リポジトリ: [github.com/motomasa-honda/kaggle-research-system](https://github.com/motomasa-honda)
