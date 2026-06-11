---
title: "OOF を保存しても submit していなかった話 — Kaggle Agent の ensemble 経路を蘇生させた Phase 8-2"
emoji: "🔧"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "python", "debugging"]
published: false
publication_date: "2026-06-11"
---

## TL;DR

- KRS-Core (Kaggle Agent) で **playbook 6 関数中 4 つが未配線**、繋がっている 2 つも結果が submit に届いていない致命的バグを発見
- `submit.py` が常に `submission.csv` だけを提出していたため、**stacking / hill_climbing / pseudo_labeling が生成した派生 submission が一切 LB に届いていなかった**
- TRIAL_NO の連番化 (env 経由注入)、regression OOF の log 空間保存、adversarial validation の Id 列除外などを含む 8 ファイル/200 行修正で「OOF → LB」経路を初めて閉じた
- 3 trial で CV が全て同一だったため LB 改善効果は明日の Kaggle quota reset 後に検証予定

## 発見の経緯

Kaggle Agent (KRS-Core) は数日前 Phase 7 で「Knowledge ループ閉鎖」というマイルストーンを達成。Planner が過去の episode を recall して LB best を更新する経路は確認済みでした。

ここから次のフェーズに進む前に、システム全体の **アーキテクチャ的振り返り**をやってみたところ、致命的な gap が見えてきました。

> なぜ 1 つのモデルで CV 0.89 出ているのに LB は 0.7488 で頭打ちなのか？
> 普通の Kaggle GM の戦法は **ensemble (stacking / blending)** のはず。

実装を見にいくと、`agents/kaggle_agent/playbook/` には以下が並んでいます:

| 関数 | 配線 | 結果が submit に届く? |
|---|---|---|
| `hill_climbing` (Caruana blending) | ❌ 未配線 | — |
| `multi_level_stacking` | ✅ reflect から呼出 (1 回限り) | ❌ 届かない |
| `pseudo_labeling` | ✅ reflect から呼出 | ❌ 届かない |
| `adversarial_validation` | ❌ 未配線 | — |
| `seed_averaging` | ❌ 未配線 | — |
| `full_retrain` | ❌ 未配線 | — |

そして `submit.py` の該当箇所:

```python
sub = workspace / "submission.csv"   # ← これだけしか提出していない
if not sub.exists():
    last["error"] = "submission.csv 未生成"
    return state
client.submit(slug, sub, msg)
```

つまり `submission_stacked.csv` も `submission_hill_climbing.csv` も `submission_pseudo_labeled.csv` も、**全部死コード**。Phase 7 で達成したと思っていた LB 改善は、**単一モデルの best_code を提出した結果**でしかなかったわけです。

## 修正ステップ

### Step 1: `submit.py` を best-submission selector に

`workspace` 内の `submission*.csv` 候補から OOF CV best を選んで提出する関数に書き換え:

```python
def _pick_best_submission(workspace, state, last):
    higher = state.get("metric_higher_is_better", True)
    candidates = []

    base = workspace / "submission.csv"
    base_cv = last.get("cv_score")
    if base.exists() and base_cv is not None:
        candidates.append((base, float(base_cv), f"single cv={base_cv:.5f}"))

    for fname, key in (
        ("submission_hill_climbing.csv",  "hill_climb_cv"),
        ("submission_stacked.csv",        "stacking_cv"),
        ("submission_pseudo_labeled.csv", "pseudo_label_cv"),
    ):
        p = workspace / fname
        cv = state.get(key)
        if p.exists() and cv is not None:
            candidates.append((p, float(cv), f"{fname} cv={cv:.5f}"))

    if not candidates:
        return base, "fallback"
    candidates.sort(key=lambda t: t[1], reverse=higher)
    return candidates[0][0], candidates[0][2]
```

### Step 2: hill_climbing を reflect に配線 + stacking を再実行可能に

```python
# Phase 8-2: Hill Climbing (Caruana) — OOF が 2 個以上で発火、再実行可
if len(history) >= 2:
    hc = run_hill_climbing(...)
    if hc and not hc.get("error"):
        state["hill_climb_cv"] = float(hc["cv"])

# Stacking — OOF が増えたら再実行 (旧仕様は「1 度のみ」だった)
n_oof = sum(1 for h in history if h.get("cv_score") is not None)
if n_oof >= 3 and n_oof >= state.get("stacking_n_trials_last", 0) + 2:
    result = run_multi_level_stacking(...)
    state["stacking_cv"] = float(result["stacked_cv"])
    state["stacking_n_trials_last"] = n_oof
```

### Step 3: adversarial_validation を analyze にフック

```python
# Phase 8-2: train/test 分布差を検出 → Planner プロンプトに hint として注入
adv = run_adversarial_validation(workspace)
state["adv_validation"] = adv
```

Planner プロンプトに `## Adversarial Validation\nAUC=0.51 (low_drift)` のように差し込まれるようにしました。

## ここからが本当の地獄: 構造バグ 3 連発

最初の smoke を投げると、なんと **iter=1, 2, 3 全部で OOF が 1 個しか保存されない**。`trial_001.npy` が毎 iter 上書きされていました。

### バグ 1: TRIAL_NO が常に 001

`codegen.py:91` で `trial_no=iteration` を template に注入していたが、**diversify (iter=2 以降) は LLM 生コード経路**で、LLM は `TRIAL_NO = 1` をそのまま引き継ぎがち。

**解決**: post-hoc rename ではなく、**実行前に環境変数で TRIAL_NO を渡す方式**に変更:

```python
res = run_local(workspace, script="main.py",
                env={"TRIAL_NO": str(iteration)})
```

そして全 template (tabular_classification / tabular_regression / nlp / timeseries):

```python
import os
TRIAL_NO = int(os.environ.get("TRIAL_NO", "{{TRIAL_NO}}"))
```

LLM が無視して `TRIAL_NO=1` を書いた場合の保険として、post-hoc rename も併設しています。

### バグ 2: Regression OOF が raw 価格スケールで保存されていた

最初の修正後の smoke で `hill_climbing CV = 27160` という非常識な値が。RMSE が 27160 ドル ? と思いきや、`OOF` を **expm1 して raw 価格空間で保存**していて、Hill Climbing 側は RMSLE (log 空間) で評価していたため、計算がメチャクチャになっていました。

```python
# 修正前
oof_out = np.expm1(oof) if TARGET_TRANSFORM == "log1p" else oof
np.save(OOF_DIR / "...", oof_out)        # raw space
np.save(OOF_DIR / "...", y_raw.values)    # raw space

# 修正後 — CV と同じ log 空間で保存
np.save(OOF_DIR / "...", oof)             # log space
np.save(OOF_DIR / "...", y.values)        # log space
```

Submission の `test_pred` は raw space (`expm1` 後) のまま — Kaggle 提出は raw 必須なので。

### バグ 3: Adversarial Validation で adv_auc=1.0000

adv_val を有効化したら House Prices で `adv_auc=1.0000 (high_drift)` という警告が。これは「train と test が完全に分離できる = データセットがそもそも別物」を意味する非常事態のはずだが、よく見ると **Id 列**が train/test で範囲が重ならない (train: 1-1460, test: 1461-2919)。

つまり LightGBM は Id 列だけで完璧に train/test を分類していて、それを「分布シフト」と誤検知していたわけ。

```python
# 修正: Id / 連番系の列を除外
likely_id = {c for c in common if c.lower() in ("id", "index", "row_id", "rowid")}
for c in list(common):
    s = pd.concat([train[c], test[c]], ignore_index=True)
    if (pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing
            and s.is_unique and s.iloc[0] >= 0):
        likely_id.add(c)
common = [c for c in common if c not in likely_id]
```

結果 `adv_auc=0.5114 (low_drift)` という正しい値に。

## 最終 iter でも playbook を走らせる

reflect.py の `max_iterations 早期 return` が playbook 配線より先にあったため、最終 iter で hill_climbing / stacking が走らない問題も残っていました。

```python
# 修正前
if iteration >= max_iterations:
    state["should_continue"] = False
    _persist_episode(state)
    return state    # ← ここで playbook をスキップしていた

# 修正後
is_final_iter = iteration >= max_iterations
if is_final_iter:
    state["should_continue"] = False  # フラグだけ立てる

# ... playbook 処理を実行 ...

if is_final_iter:
    _persist_episode(state)
```

## 結果

smoke `47d1dd59` で以下を確認:

- ✅ trial_001 / 002 / 003 全部揃った (TRIAL_NO env 経由が機能)
- ✅ adv_auc=0.5114 (Id 除外で正常値)
- ✅ hill_climb candidates=[1, 2] で複数 OOF 認識
- ✅ submit.py が single (0.12961) vs hill_climb (0.13029) で正しく single を選択

LB 改善効果は Kaggle daily submit quota (10/day) を消化済みのため明日 (JST 9:00 以降) に持ち越し。

## 学んだこと

1. **「実装されているコード」と「実際に LB に届いているコード」は別物**。Phase 7 で「Knowledge ループ閉鎖」と言っていたのは単一モデルの結果で、ensemble は最初から繋がっていなかった。アーキテクチャ的振り返りをしないと永遠に気づけない
2. **OOF / blending は CV と同じ空間で保存・評価する**。RMSLE では log 空間、AUC では確率。submission のためのスケール変換は最後にやる
3. **adversarial validation は Id 列を必ず除外**。これがないと常に "high_drift" 判定になる
4. **LLM 生コード経路では「設定値の引き継ぎ」を期待しない**。プロンプトで指示しても破る。env 変数や設定ファイルなど **コード外側から強制**する方が robust

## 次に向かう先

LB 改善が明日確認できたら、次は **branch_explorer の多様性向上**。今回の smoke では 3 trial の CV が全部 0.12961 で同一だったので、ensemble 効果が出る余地が無い状態でした。Caruana が weight={1:1.0} で正しく「best single だけ採用」する挙動を示しているのが、その証拠。

trial 間の多様性が確保できれば、ensemble の本領発揮 + Knowledge ループから learn された戦略の幅も広がるはず。

## 参考: 主要コミット

- `ec41221` feat(phase8-2): OOF blending を LB に接続 + adversarial validation 配線
- `f9bab3a` fix(phase8-2): TRIAL_NO 連番化 + regression OOF を log 空間で保存
- `cff1f12` fix(phase8-2): TRIAL_NO を env 経由で注入 + adv_val の Id 列誤判定対策
- `e606c82` fix(phase8-2): 最終 iter でも playbook を走らせる

リポジトリ: [github.com/motomasa-honda/kaggle-research-system](https://github.com/motomasa-honda)
