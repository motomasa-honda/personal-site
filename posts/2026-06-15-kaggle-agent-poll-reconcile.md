---
title: "アンサンブルは全部動いたのに LB が 1mm も動かなかった — そして見つけた「自分のスコアを見失う」バグ"
emoji: "🔁"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "python", "debugging"]
published: true
publication_date: "2026-06-15"
---

## TL;DR

- 前回 (Phase 8-2) で繋いだ Kaggle Agent の OOF ensemble 経路を、quota reset 後に**実際の LB で検証**した
- hill_climbing / stacking / adversarial validation、機構は**全部きれいに発火した**。なのに LB は 0.13007 ×3 で 1mm も動かなかった
- 原因は「**3 trial が完全に同一予測**」。多様性ゼロのモデルをアンサンブルしても `blend = single`。当たり前だが、配線が正しいからこそ綺麗に「効果ゼロ」が観測できた
- おまけに調査中、**Agent が自分の LB スコアを見失うバグ**を発見。submit 後の 600s ポーリングが Kaggle のスコアリング遅延を取りこぼし、`best_public=None` になっていた
- 後者は「run 終了時に submission ID で遅延照合し直す reconcile」で修正した

## 背景: 配線したものは、本当に効くのか

数日前 (Phase 8-2)、Kaggle Agent (自作の自動化システム KRS-Core) の「OOF を保存しても submit していなかった」という致命的な配線漏れを直しました。stacking や hill_climbing が作る派生 submission が一切 Kaggle に届いていなかったやつです。

ただ、その時点では **LB での効果は未検証**でした。Kaggle の1日あたり提出上限 (10/日) を使い切っていたからです。

> 配線が通った ≠ 効果が出る。

このギャップを埋めるのが今日のタスク。quota がリセットされたので、House Prices で 3 iteration の smoke を1本投げました。

## 結果: 機構は完璧、効果はゼロ

72分後、ログを精査するとすべての機構が発火していました。

| 機構 | 結果 |
|---|---|
| TRIAL_NO の連番注入 | trial_001/002/003 全 OOF 生成 ✅ |
| adversarial validation | adv_auc=0.5114 (drift なし) ✅ |
| hill_climbing (Caruana blend) | iter2 cand=[1,2] / iter3 [1,2,3] 発火 ✅ |
| stacking (L2 メタモデル) | iter3 で L2 stacker cv=0.13452 発火 ✅ |
| submit selector | 毎 iter 候補を比較して選択 ✅ |

end-to-end で「OOF 資産 → ensemble → 提出選択」が動いたことは確認できた。**しかし:**

```
全 3 trial:   CV 0.1290 ×3    Kaggle LB 0.13007 ×3
hill_climb:   weights={1: 1.0}   CV 0.12972   ← 単一より悪い
stacking:     CV 0.13452                       ← 単一より悪い
selector:     毎回 single (0.12904) を選択
```

hill_climbing が出した重みは `{trial_1: 1.0}`。つまり「3つのうち1つだけを重み 1.0 で採用し、残りは捨てた」。これはアンサンブルの最適化が正しく働いた結果で、**3 trial が同一なら blend する意味がない**ことを optimizer がちゃんと見抜いている。

stacking に至っては単一モデル (0.12904) より悪い 0.13452。多様性のない L1 予測を重ねても、メタモデルは過学習するだけ。

結論はシンプルでした。

> **アンサンブルの効果は、ベースモデルの多様性が上限を決める。**
> branch_explorer が生む 3 つの探索枝がほぼ同一予測に収束している限り、どんなに ensemble 機構を磨いても LB は動かない。

皮肉なことに、配線が正しいからこそ「効果ゼロ」が綺麗に観測できた。もし配線がバグっていたら、この「多様性が本質的なボトルネック」という診断にすら辿り着けなかったはずです。次のフェーズは ensemble の magic ではなく、**探索の多様性**そのものに向かうことになりました。

## おまけのバグ: Agent が自分のスコアを見失う

検証中、result に `best_public=None` と出ているのに気づきました。LB は確かに 0.13007 が付いているのに、Agent 側は「スコア未取得」だと思っている。

Kaggle の実 submission 履歴を引くと:

```
53691620  iter 3  SubmissionStatus.COMPLETE  publicScore 0.13007
53691287  iter 2  SubmissionStatus.COMPLETE  publicScore 0.13007
53691012  iter 1  SubmissionStatus.COMPLETE  publicScore 0.13007
```

提出は成功、スコアも付いている。なのに Agent は None。

原因は submit ノードのポーリング設計でした。提出後、最大 600 秒スコアを待つのですが、

```
[submit] ⏰ score still pending after 600s
[submit] iter=1 public=None
```

今日は Kaggle のスコアリングが 600 秒の窓を**わずかに超えて**返ってきていた。パース処理自体は正常 (`SubmissionStatus.COMPLETE` も `publicScore` もちゃんと読める) で、単に「待ち時間が足りずに諦めて次の iter に進んでいた」だけ。

これが地味に効く。`best_public=None` だと、

- best 更新ロジックが回らない
- episode memory に LB が記録されない
- ensemble selector が「LB で比較」できず CV にフォールバック

つまり **Agent が自分の成績表を見ないまま次の問題に進んでいた**。

### 修正: 待つのをやめて、後でまとめて回収する

ポイントは「iter ごとに 10 分ブロックして待つ」のをやめたこと。

1. **in-loop ポーリングを 600s → 180s に短縮** — よくある 1〜3 分のスコア計算はこれでカバー
2. **run 終了時に `reconcile_pending_scores()`** — `public=None` のまま終わった iter を、保存しておいた submission ID で Kaggle に照合し直してスコアを埋める。最後に best_public を再計算する

```python
def reconcile_pending_scores(state):
    pending = {
        h["metadata"]["submission_ref"]: h
        for h in state["history"]
        if h.get("public_score") is None
        and h.get("metadata", {}).get("submission_ref")
    }
    if not pending:
        return state
    client = KaggleClient()
    for wait in _RECONCILE_SCHEDULE:        # ~3分だけ再ポーリング
        if not pending:
            break
        time.sleep(wait)
        for ref, score in client.score_for_refs(slug, set(pending)).items():
            h = pending.pop(ref)
            h["public_score"] = score       # 遅延スコアを回収
    _recompute_best_public(state)
```

「待つ」を「後で照合する」に変えただけですが、これで wall-clock も縮むし (iter ごとの 10 分ロスが消える)、遅延スコアも取りこぼさなくなる。両取りです。

ついでに見つかった小さなバグも潰しました。スコアが ready かを判定する `status in ("pending", "error")` という**完全一致**は、新しい Kaggle CLI が返す `SubmissionStatus.PENDING` という文字列を取りこぼしていた (実害は別経路で吸収されていたものの、判定としては壊れていた)。部分一致に変更。

検証は二段構え:

- モックユニットテスト — 全解決 / 一部だけ解決 / best 再計算、いずれも PASS
- 実機 — `score_for_refs` を今日の実 submission ID に対して叩き、3 件すべて 0.13007 を正しく回収できることを確認

## 学び

1. **配線の正しさと効果の有無は別物**。両方を別々に検証する価値がある。今回は「機構○ / 効果✕」が綺麗に分離できたから、ボトルネックが ensemble ではなく**探索の多様性**にあると断定できた
2. **同期的に待つ I/O は、外部サービスの遅延に対して脆い**。「待つ」より「後で照合し直す」方が速くて頑丈なことが多い
3. **Agent が自分の結果を観測できているか**は、自動化システムの隠れた急所。スコアを見失ったまま次に進む Agent は、学習ループそのものが空回りする

次はいよいよ branch_explorer の多様性に手を入れます。3 つの探索枝が別々の解に向かうようになって初めて、今日配線を確認した ensemble 機構が本領を発揮するはずです。
