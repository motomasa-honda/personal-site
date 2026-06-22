---
title: "Obsidian Skill ライブラリの品質管理 — 重複排除・品質スコア・MOC 自動生成の 3 層機構"
emoji: "🧹"
type: "tech"
topics: ["kaggle", "obsidian", "llm", "automation", "rag"]
published: true
publication_date: "2026-06-22"
---

## TL;DR

過去 1 か月で蓄積した 721 件の skill カード (Kaggle 上位 kernel から LLM で抽出した
「再利用可能な技術カード」) を眺めていたら、**「TargetEncoding with smoothing」が 38 件**
重複していたり、Python 演習レベルのチュートリアルカードが混入していたり、品質がかなり
ばらついていた。

Obsidian の skills/ vault も同様に荒れていたので、3 層の品質管理機構を導入した:

1. **正規化タイトル dedup** (CamelCase 対応): "TargetEncoding" / "Target Encoding" / "Smoothing TargetEncoding" を同キーに
2. **quality_score** (0-1): votes + v8 フィールド充実度 + コード品質 + 抽象度の合成
3. **MOC (Map of Content) 自動生成**: by_category / by_quality / meta_patterns / by_competition

実機で 721 件中 120 件が dedup で archived 状態 (削除はしない、履歴保全)、
quality_score の mean が **0.36 → 0.68 に倍増**、meta_pattern 比率が **1% → 82% へ**。

## 状況: 721 件の skill が「使い物にならない」問題

KRS-Core (Kaggle 自動化システム) では、毎週の知識ハーベストで Kaggle 上位 kernel を
LLM (Qwen3 70B / DeepSeek-R1 70B) に流し込んで、再利用可能な技術要素を JSON で抽出している。
1 年弱で 721 件まで増えていた。

ところがいざ Planner に「skill を検索して inject せよ」と命じても、

- 同じタイトルの skill が大量にヒットする
- 重要なメタ情報 (適用条件 / 反証 / 期待効果) が空欄
- コードが付いてない skill が混じる
- 過去の Python 演習レベルの kernel から拾った価値ゼロ skill が紛れる

という状況だった。

数字で見ると:

```python
# 計測前
total = 721
exact title duplicates: 15 タイトルが 2-38 件ずつ
v8 構造化フィールド (mechanism / conditions ...): 6/721 (0.8%)
mean quality_score (後述の合成指標): 0.36
```

特に **TargetEncoding with smoothing が 38 件**。これでは Planner に 6 個 retrieve したら
半分以上 TargetEncoding で埋まってしまう。

## なぜ既存の Jaccard dedup は機能していなかったか

Skill ライブラリには `consolidate_duplicates(similarity_threshold=0.85)` という Jaccard ベースの
重複統合関数があった。これを 721 件に対して走らせると…

```
th=0.95:  merged=0  remaining=721  reduction=0.0%
th=0.85:  merged=0  remaining=721  reduction=0.0%
th=0.75:  merged=2  remaining=719  reduction=0.3%
th=0.65:  merged=6  remaining=715  reduction=0.8%
```

**全く効いていない**。Jaccard 0.85 では「TargetEncoding with smoothing」と
「Target Encoding with Smoothing」の token Jaccard が 0.85 に届かないらしい
(description が違うため)。

これは典型的な「naive な類似度では同義表現を統合できない」問題。

## 機構 1: 正規化タイトルでバケット dedup

embedding を使わずに済む方法として、**title の段階で同義表現を吸収するキー正規化**を実装した。

```python
def normalize_title(title: str) -> str:
    """- 記号を空白に
       - CamelCase / PascalCase を分割 ("TargetEncoding" → "target encoding")
       - lowercase
       - stopword 除去
       - 単語ソート (順序違いの同義タイトルを揃える)
    """
```

ポイントは **CamelCase 分割**。これがないと:

```
"TargetEncoding with smoothing"  -> "smoothing target..."  (CamelCase 残る)
"Target Encoding with Smoothing" -> "encoding smoothing target"  (分割される)
```

別キーになってしまう。CamelCase 正規表現で:

```python
_CAMEL_RE = re.compile(r"([A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|\d+)")
```

これで "TargetEncoding" を ["Target", "Encoding"] に分解、結果:

```
"TargetEncoding with smoothing"  -> "encoding smoothing target"
"Target Encoding with Smoothing" -> "encoding smoothing target"  ← 同一!
"Smoothing TargetEncoding"        -> "encoding smoothing target"  ← 同一!
```

これでバケット dedup できるようになった。実機 721 件で **120 件を archived 状態に**、
51 個のバケットが形成された。最大バケットはもちろん TargetEncoding 系。

## 機構 2: quality_score で「使える skill」を可視化

dedup だけだと「同名 5 件のうち 1 件を残す」までは決まるが、**どれを残すか** の判断軸が要る。
そのために 0-1 の合成 quality_score を定義した:

```python
def compute_quality(self) -> float:
    votes_norm    = min(self.source_votes / 1000.0, 1.0)         # 上位陣 kernel 由来か
    v8_filled     = sum(bool(getattr(self,f)) for f in v8_fields) / 4.0  # 構造化メタ充実度
    code_quality  = (50-1500 chars が最適、それ以外は減衰)
    abstraction   = 1.0 if meta_pattern else 0.5 if kernel_specific else 0.0

    quality_score = 0.4*votes_norm + 0.3*v8_filled + 0.2*code_quality + 0.1*abstraction
```

意図:

- **votes (40%)**: 上位陣由来は本当に効くものが多い (人気バフ)
- **v8 メタ充実度 (30%)**: 適用条件 / 反証 / 期待効果が埋まっているほど機械可読
- **コード長 (20%)**: 50-1500 字が最適、ベタコピーの長文や 1 行スニペットを減点
- **abstraction_level (10%)**: meta_pattern (転移可能) > kernel_specific > unspecified

代表選択時はバケット内で quality_score 最大を rep として残し、他は archived。
**archived は物理削除しない**: 履歴保全 + Obsidian の skills/ vault は残ったまま frontmatter に
`status: archived` を付ける運用にした (誤判定で重要 skill を消したくない)。

## 機構 3: archived 状態と planner search の連携

Skill library の検索関数 `search(task_type, metric)` で、

```python
for s in self.skills:
    if s.status == "archived":
        continue                # ← archived は検索結果から除外
    ...
    score += s.quality_score * 2.0   # ← quality_score もスコアに加算 (max +2)
```

これにより、Planner に注入される skill は:
1. archived ではない (重複の代表のみ)
2. haystack 一致が高い (条件マッチ)
3. quality_score が高い (品質保証)

の 3 条件を満たすものに絞られる。

## 機構 4: MOC (Map of Content) 自動生成

Obsidian 側は人間が眺めて使えるよう、`_index/` に 4 つの index Markdown を毎ジョブ自動生成
する `write_moc_indexes(skills)` を追加:

```
_index/by_category.md       # FE / model / cv / postprocess ... ごとに quality 降順
_index/by_quality.md        # 全体 Top 50 (表形式)
_index/meta_patterns.md     # abstraction_level=meta_pattern のみ、category 別
_index/by_competition.md    # コンペ別の n_skills / n_meta / avg_quality
```

これにより Obsidian で `_index/by_quality.md` を開けば、**今この瞬間に最も使える 50 skill が
一覧で見える**。

## 効果: 9.4 時間バッチで 595 件全 v8 化

dedup + quality はインフラ整備の話で、実体は中身が空欄のまま。本丸は「既存 skill 595 件
すべてに v8 メタフィールドを LLM で補完する」夜間バッチ:

```bash
nohup python -u scripts/upgrade_skills_to_v8.py \
    --consolidate-first --save-every 30 > /tmp/skill_upgrade_batch.log 2>&1 &
```

軽量 upgrade prompt (kernel 全文ではなく既存 skill のテキスト 1.5KB だけ入力) で
1 件 30-90 秒。実機平均 **56.7 秒/skill**、9.4 時間で **595/595 OK, 0 fail**。

結果:

| 指標 | バッチ前 | バッチ後 |
|---|---|---|
| total skills | 721 | 721 |
| active | 601 | 601 |
| archived | 120 | 120 |
| **meta_pattern** | **6 (1%)** | **493 (82%)** ✓ |
| kernel_specific | 0 | 108 |
| unspecified | 595 | **0** ✓ |
| **mean quality** | **0.36** | **0.68** (倍) |
| median quality | 0.345 | 0.718 |

82% が meta_pattern (= 別コンペにも適用可能な原理単位) になった。これが効くかは
**「次の Kaggle コンペで自動 retrieve した skill が実際にどれくらいスコアに寄与するか」**
で判断するしかない。今後の検証ポイント。

## 学び: dedup は title 段階でやれ

embedding を使った類似度判定は重そうに見えるが、実は **title の段階で適切に正規化すれば
naive bucket dedup で十分機能する**。CamelCase 分割と stopword 除去だけで TargetEncoding 38 件
問題が解けた。

逆に Jaccard 0.85 のような description ベース類似度は、表現ゆれが大きいテキストでは
**ほぼ何も merge しない** ことが実機で確認できた。Jaccard を上げると false positive が増え、
下げると false negative が増える、という古典的な閾値ジレンマ。

最初から「同概念 = 同 title 正規形」と決めてバケット切るほうが、運用上も理解しやすい。

## 学び: archived ≠ deleted

過去の自分は「重複は deduplicate (= 物理削除)」と思っていた。だが LLM 抽出は完璧ではなく、
誤判定で重要 skill が消える事故が起きうる (実際、似て非なる「BFS with State Hashing」と
「StateHashingBFS for ARC」のような 2 件を merge してしまったら片方の文脈が永久に失われる)。

**archived 状態で残す + frontmatter で見えにくくする** が運用上の妥協点。Obsidian の
graph view ではグレーアウトして表示できる (今後の TODO)。

## おわりに

「LLM 抽出ベースの knowledge base は、量を増やすほどノイズが増える」という現実を
1 か月で経験した。質の悪い 700 件より、質の高い 200 件 + 構造化メタの方が、最終的に
Planner / AutoML loop の判断品質を上げる。

3 層機構 + 9.4h バッチで、それなりに使える状態まで持ってこられた。本日のコミット:
[3994ae3](https://github.com/motomasa-honda/kaggle-research-system/commit/3994ae3)
[389cc78](https://github.com/motomasa-honda/kaggle-research-system/commit/389cc78)
[b778a2e](https://github.com/motomasa-honda/kaggle-research-system/commit/b778a2e)
[5b4032e](https://github.com/motomasa-honda/kaggle-research-system/commit/5b4032e)
