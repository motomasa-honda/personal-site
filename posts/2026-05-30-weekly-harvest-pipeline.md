---
title: "Kaggle上位NotebookをLLMで自動SKILL化する週次Harvestパイプラインを作った"
emoji: "🌾"
type: "tech"
topics: ["kaggle", "llm", "ollama", "python", "自動化"]
published: true
publication_date: "2026-05-30"
---

## TL;DR
- 毎週日曜03:00にMacがKaggle上位Notebookを自動DL、UbuntuのLLMで「SKILL」として知識化
- SKILLをさらにdeepeseek-r1:70bで蒸留して`meta_skill`（汎用ベストプラクティス）を生成
- `meta_skill`はKaggleパイプラインのPlannerプロンプトに自動注入される
- 新形式Kaggle APIトークン（KGAT_xxx）対応・Notebook DLをMac専任にして容量節約

---

## なぜ「SKILL」という概念を作ったか

Kaggle Grandmasterがなぜ強いかを考えたとき、答えは**経験の蓄積**だと思った。

`LightGBMにTargetEncodingを組み合わせるとtabular分類で強い`

こういう再利用可能な技術知識を、コンペをまたいで蓄積・活用できるシステムが欲しかった。それがSKILL Libraryだ。

```python
@dataclass
class Skill:
    title: str           # "TargetEncoding with smoothing"
    category: str        # "feature_engineering"
    model: str           # "lightgbm"
    description: str     # なぜ効くか
    code_snippet: str    # そのままコピペできるコード
    when_to_use: str     # 適用条件
    competition: str     # 元コンペ
    source_kernel: str   # 元Notebook
    source_votes: int    # 投票数（品質指標）
```

---

## Harvestパイプラインの全体像

```
【Mac mini - 毎週日曜03:00 cron】
harvest_runner.sh
  ↓
Kaggle CLI で最新コンペ取得 (--sort-by recentlyCreated)
  ↓
各コンペの上位Notebook DL → ~/kaggle-kb/mined/
  ↓
Ubuntu起動中？ YES → rsync で送信 → Mac側 mined/ 削除 → SSH kick
            NO  → mined/ を Mac に保持 → 手動: --send-only

【Ubuntu - kick受信後】
boot_distill.sh --force
  ↓
PHASE 1: weekly_knowledge_harvest.py --harvest-only --mined-dir ~/kaggle-kb/mined/
  ↓ deepseek-r1:32b でSKILL抽出
  ↓
mined/ 削除（容量解放）
  ↓
PHASE 2: weekly_knowledge_harvest.py --distill-only
  ↓ deepseek-r1:70b でmeta_skill生成
  ↓
sync_kb.sh push → Mac に skills.json 逆送り → Obsidian で閲覧可
```

1回のHarvestで90ファイル（10コンペ × 10Notebook）を処理し、207 SKILLが生成された。

---

## Notebookからのコード抽出

`.ipynb`は単なるJSONなので、コードセルだけ抽出する：

```python
def _extract_code_from_ipynb(path: Path, max_chars: int = 30000) -> str:
    nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        if text.strip():
            chunks.append(text)
    code = "\n\n# ---- cell ----\n\n".join(chunks)
    return code[:max_chars]
```

---

## LLMによるSKILL抽出プロンプト設計

抽出プロンプトはシンプルに「このコードから再利用可能なSKILLをJSONで返して」：

```python
_SUMMARIZE_PROMPT = """あなたは Kaggle Grandmaster です。
以下は Kaggle コンペ "{slug}" の上位投票カーネル ({ref}, 投票数 {votes}) のコード抜粋です。

このコードから「再利用可能な技術要素 (SKILL)」を JSON で抽出してください。
1カーネルにつき 1〜3個の SKILL を返してください。

各 SKILL は以下のフィールドを持ちます:
- title: 30字以内の技術名
- category: feature_engineering | model | cv | postprocess | ensemble | other
- model: 主要モデル名
- description: 150字以内、なぜ効くか
- code_snippet: 30行以内の最重要コード片
- when_to_use: 適用条件

出力 JSON:
{{"skills": [{{"title": "...", ...}}]}}
---コード抜粋---
{code}
"""
```

v7.2では `mine_from_mined_dir()` という関数を追加して、**Kaggle CLIなしで既にDLされたNotebookを直接読む**ようにした。これでUbuntuにKaggle認証が不要になった。

```python
def mine_from_mined_dir(mined_dir: Path, *, skip_if_exists: bool = True) -> list[MinedKernel]:
    """
    mined/<slug>/<author>__<kernel-ref>/<notebook>.ipynb
    の構造を直接読んでSKILL抽出する
    """
    for slug_dir in sorted(mined_dir.iterdir()):
        slug = slug_dir.name
        for kdir in sorted(slug_dir.iterdir()):
            ref = kdir.name.replace("__", "/", 1)  # ディレクトリ名からref復元
            code = _find_kernel_payload(kdir)
            # LLMでSKILL抽出 ...
```

---

## meta_skill蒸留: SKILLをさらに昇華する

生のSKILLは「このコンペではこの手法が効いた」という個別事例。  
それをdeepseek-r1:70bで蒸留して**コンペをまたいで使える汎用ベストプラクティス**にするのが`meta_skill`。

```
[SKILL A] Titanic: TitleExtraction (votes=245)
[SKILL B] Spaceship Titanic: TitleExtraction (votes=189)
[SKILL C] Tabular Playground: TextFeature (votes=130)
    ↓ Reasoner LLM (deepseek-r1:70b) で蒸留
[meta_skill] テキストカラムからのTitle/Prefix抽出は
             passenger系コンペで安定的に+0.01〜0.02効く
```

類似度0.55以上・5件以上のクラスタで自動蒸留が走る。蒸留後の元SKILLは`archived`に移動。

---

## sync_kb.shの双方向同期設計

Mac↔Ubuntuのskills.json同期は単純なrsyncではなく、**merge_skills.py**経由でマージする。

理由：単純上書きだとUbuntu側で生成した`meta_skill`がMacのpushで消えてしまう。

```bash
# sync_kb.sh の push_mined モード（v7.2で追加）
do_push_mined() {
    rsync -avz --stats --remove-source-files \
        "$MINED_DIR/" "$KRS_REMOTE_HOST":"~/kaggle-kb/mined/"
    # --remove-source-files: 送信済みファイルをMac側で自動削除
    find "$MINED_DIR" -type d -empty -mindepth 1 -delete
}
```

`--remove-source-files`がポイント。rsyncの転送完了と同時にMac側のファイルを削除できる。

---

## Streamlitで進捗モニタリング

Harvest中の進捗をブラウザで確認できるページを作った。

主な表示内容：
- **プロセス状態**: `pgrep -f weekly_knowledge_harvest` でリアルタイム確認（点滅ドット）
- **mined/状況**: コンペ別Notebook数をプログレスバーで表示
- **SKILL生成進捗**: コンペ別SKILL数をプログレスバーで表示
- **最新ログ**: `~/logs/distill/*.log`の直近30行をリアルタイム表示
- **自動更新**: トグルONで10秒ごとにリロード

---

## 学んだこと

- **SKILLの粒度設計が重要**: 1Notebookから3SKILLまでという上限が品質を保つ
- **mined/はキャッシュと割り切る**: 正規データはskills.jsonのみ。Notebook生ファイルは使い終わったら消す
- **meta_skill蒸留はdeepseek-r1:70b専用**: 32bだと質が落ちる。蒸留だけ重モデルを使う設計が正解
- **`--remove-source-files`**: rsyncでファイル移動（コピー＋削除）が一発でできる
- **Kaggle CLIの認証は`access_token`ファイルで**: 新形式トークンの保存先が変わっている

## 参考
- `scripts/harvest_runner.sh` - Mac側Harvestスクリプト
- `scripts/boot_distill.sh` - Ubuntu側harvest+蒸留スクリプト
- `core/knowledge/top_solutions_miner.py` - `mine_from_mined_dir()`実装
- `scripts/weekly_knowledge_harvest.py` - `--mined-dir`対応
- `ui/pages/8_Knowledge_Harvest.py` - 進捗モニターUI
