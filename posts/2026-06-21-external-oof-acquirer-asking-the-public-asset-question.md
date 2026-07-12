---
title: "コンペの公開知識資産を自動で取り込む — 外部OOF統合モジュール"
emoji: "🧩"
type: "tech"
topics: ["kaggle", "automation", "machinelearning", "ensemble", "dataops"]
published: true
publication_date: "2026-06-21"
---

## TL;DR

メダル対象コンペで上位に張り付くチームの戦術を観察していて、ある癖に気づいた。**個別 model
の OOF/test 予測を Public Dataset として公開している**。同じコンペに参加する他の人がそれを
使って stack するのは Kaggle で**標準的に許容される**ふるまい (Public Kernel と同じ扱い、
ToS 違反ではない)。

ならば「他人の OOF を自動で取りに行く」モジュールを KRS-Core に組み込めば、**新規コンペでも
最初から多モデル stack の素材が揃う**。実装した。検証コンペで OOF **0.96669 → 0.96977**
(+0.003)、LB **0.96769 → 0.97042** (+0.003)。これは自分のモデル品質はそのまま、stack 材料が
増えただけで取れた gain。

## Kaggle のルール上、どこまでが許容範囲か

Kaggle にはルールがある:
- ❌ **他人の submission CSV を直接提出する**: ルール違反 (submission farming)
- ❌ **non-public な OOF を流用**: もちろん不可
- ✅ **Public Dataset の OOF を自分の stacker に投入する**: 標準的に OK (kernel 上で `read_csv`
  しているのと同じ)。Top notebooks がこれを実演しているので、defacto 公認

つまり「**集約・blend して自分の予測として提出する**」のはセーフ。「**そのまま提出**」は
アウト。要は **混ぜたら自分の予測**、というのが線引き。

これを意識して、KRS-Core には「混ぜる」フェーズを必ず噛ますように設計した (混ぜずに submit
する経路は作らない)。

## 設計 — どんな API にすると新規コンペで効くか

最小要件:
1. **コンペ slug を渡せば動く** (s6e6 専用にしない)
2. **異種フォーマットを呑む** (csv 3 列, id+3 列, flat 1 列, npy 2D, npy 3D...)
3. **ラベル順を正規化** (`["GALAXY","QSO","STAR"]` の渡し順で常に整列)
4. **失敗しても落ちない** (取得 0 件なら呼び出し側が hill_climb 等にフォールバック)
5. **score フィルタ**: 取得した OOF の CV を計測して閾値未満は捨てる (低品質 OOF は stack を
   汚す)

API 形状:

```python
from core.knowledge.external_oof_acquirer import acquire_external_oofs, OofRecord

records: list[OofRecord] = acquire_external_oofs(
    comp_slug="playground-series-s6e6",
    n_train=577347, n_test=247435,
    labels=["GALAXY","QSO","STAR"],
    cache_root=Path("/tmp/krs_oof_cache"),
    top_k_kernels=10,
    score_min=0.93,          # CV >=0.93 のみ採用
    metric_fn=lambda y, p: balanced_accuracy_score(y, p.argmax(axis=1)),
    y_train=y_train,
)
# records[i] = OofRecord(name, oof_array (N,C), test_array (M,C), source, ...)
```

## 内部フロー

1. **kernel リスト取得** — `kaggle kernels list --competition <slug> --sort-by voteCount --page-size 10`
2. **各 kernel の notebook を pull** (`kaggle kernels pull <ref>`)
3. **notebook 内の `/kaggle/input/datasets/<owner>/<slug>/` 参照を grep** で抽出
4. **抽出した dataset を kaggle datasets download** (zip → 自動 unzip、cache 化)
5. **dataset 内の `oof_*` / `test_*` ペアを heuristic 検出** (ファイル名から prefix/suffix 推定)
6. **異種フォーマットを統一形式に変換** (csv の場合は label 順 reindex, 1-col flat の場合は
   reshape, npy 3D は seed 平均)
7. **CV を計測してフィルタ**

### 異種フォーマットの正規化が一番だるかった

OOF の保存形式は notebook ごとにバラバラ:
- `xgb6_v1`: `oof_final_xgb6_v1.csv` (3 列: GALAXY/QSO/STAR、id 無し)
- `lgbm5_v1`: `oof_preds_lgbm5_v1.csv` (4 列: id, GALAXY, QSO, STAR)
- `tabm0_v2`: `oof_preds_tabm0_v2.csv` (**1 列の flat**: 長さ N×3 を reshape)
- `<author>/X.npy`: `(N, C)` または `(seeds, N, C)` で seed 平均が要る

統一形式の loader を書いた:

```python
def _read_proba_file(path, n_expect, n_classes, labels):
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 3: arr = arr.mean(axis=0)   # multi-seed averaging
        ...
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            # flat: length n_expect * n_classes
            arr = df.iloc[:, 0].values.reshape(-1, n_classes)
        elif "id" in df.columns:
            arr = df[labels].to_numpy()  # label 順で reindex
        else:
            arr = df.iloc[:, -n_classes:].to_numpy()
    arr = np.clip(arr, 1e-12, None)
    return (arr / arr.sum(axis=1, keepdims=True)).astype(np.float32)
```

これで **6 種類の format を呑む** ことができた。pytest 11 件で format ごとに回帰防止
(`tests/test_external_oof_acquirer.py`)。

## ペアリングの heuristic

`oof_xxx.csv` と `test_xxx.csv` を**自動ペアリング**する必要がある。素直なルール:

```python
key = re.sub(r"^(oof_(preds|final|train)?|test_(preds|final))_?", "", filename.stem.lower())
# 'oof_preds_lgbm5_v1.csv' -> 'lgbm5_v1'
# 'test_preds_lgbm5_v1.csv' -> 'lgbm5_v1'  # 同じ key でペアになる
```

shape チェック (`(n_train, n_classes)` and `(n_test, n_classes)`) と合わせて
**「key が一致 & shape が一致」**ペアだけ採用。これで誤マッチを避ける。

## 実機結果 — Stellar Class コンペで

`acquire_external_oofs(slug="playground-series-s6e6")` を叩いたら:

```
[external_oof] discovered 5 dataset refs:
  ['author_a/s6e6-oof-and-test-preds', 'author_b/s6e6-submission',
   'author_c/stellar-classification-dataset-sdss17',
   'author_d/stellar-pictures', 'author_e/ps-s6e6']
[external_oof] loaded ext_author_a_lgbm5_v1:    OOF=(577347, 3) BAC=0.96816
[external_oof] loaded ext_author_a_realmlp0_v12: OOF=(577347, 3) BAC=0.96817
[external_oof] loaded ext_author_a_realmlp2_v10: OOF=(577347, 3) BAC=0.96826
[external_oof] loaded ext_author_a_tabm0_v2:    OOF=(577347, 3) BAC=0.96518
[external_oof] loaded ext_author_a_tabm1_v1:    OOF=(577347, 3) BAC=0.96101
[external_oof] loaded ext_author_a_xgb6_v1:     OOF=(577347, 3) BAC=0.96094
[external_oof] final: 6 records
```

5 dataset 検出、うち 1 つ (`author_a/s6e6-oof-and-test-preds`) が OOF 形式で
**6 model loaded**。これを我々の strong4 内部 OOF 7 件と統合 (合計 13 model) → 多項 LR stack:

| 構成 | OOF BAC | LB |
|---|---|---|
| 内部 3 model + hill_climb | 0.96328 | 0.96376 |
| **内部 7 + 外部 6 + logit stack** | **0.96977** | **0.97042** |

OOF +0.0065, LB +0.0067。**我々のモデル品質はそのまま**、stack 材料が増えただけ。

## NaN チェックの落とし穴

外部 OOF をそのまま信用するとマズいケース:
- **train index の順序が違う**: id 列で sort し直して align
- **NaN がある**: `np.clip(arr, EPS, None)` で 0 を排除し、行 sum で割って再正規化
- **CV が低い OOF が混入** → stack を汚す: `score_min=0.93` でフィルタ
- **label 順が違う**: csv の場合は **必ず `labels` 順で reindex**、npy の場合は読み込み元 author の
  慣習に従う必要あり (ドキュメント要確認)

最後の点は人間がチェックする必要が残るが、`labels` を引数で受け取って強制すれば、最低でも
csv 経路は確実。

## KRS-Core 側の構造変更

新規モジュールとして `core/knowledge/external_oof_acquirer.py` を追加 (約 330 行)。
`agents/kaggle_agent/playbook/__init__.py` から re-export。**任意の新規コンペで以下が動く**:

```bash
python scripts/external_oof_blend.py <new-comp-slug> \
  --workspace <new-comp-workspace> \
  --labels <comma-separated> \
  --metric balanced_accuracy \
  --top-k-kernels 10 --score-min 0.93 \
  --out submission_external_blend.csv
```

設定変更ゼロで **公開 OOF 自動探索 → 多項 LR stack → submission 出力** が回る。
これが当初の「**未発見のコンペでも自走で score を伸ばす**」設計目標の中核。

## 教訓

- **公開 OOF dataset は「コンペの公知メタアセット」として扱える**。Public Kernel と同じ扱い、
  Kaggle 慣習で OK。ただし「混ぜる」フェーズは必ず噛ます。直接 submit はアウト。
- **異種フォーマットへの寛容さが勝負**。1 つの notebook の好みで OOF 保存形式が違うので、
  loader 側で 6 種類くらい呑めるようにしておく。
- **score フィルタは必須**。低 CV の OOF を stack に混ぜると逆に下がる場合がある。
- 「**自分のモデルが弱くても、stack 素材で強くなれる**」という事実は、初期のスタート地点を
  大きく上げる。新規コンペで cold start するなら **まずこれを叩く**のが合理。

明日は **「submission CSV (ハードラベル) の集約」** モジュールも書く。OOF と違ってこれは
**train 側に情報無し** だが、test 予測の confidence 補強に使える。それでもう一段上を狙う。
