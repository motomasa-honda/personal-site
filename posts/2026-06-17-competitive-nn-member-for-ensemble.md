---
title: "GBDT と拮抗する NN を ensemble に効かせる — 弱い多様性は一票も入らない、そして float 精度の罠"
emoji: "🧠"
type: "tech"
topics: ["kaggle", "machinelearning", "ensemble", "pytorch", "diversity"]
published: true
publication_date: "2026-06-17"
---

## TL;DR

- 前回「多様性は精度が拮抗して初めて効く」と学んだ。今回その続きで、GBDT と拮抗する **PyTorch MLP メンバー**を自動探索に組み込んだ
- 最初の NN は CV 0.50 と惨敗。原因は**ターゲット未標準化**で、ネットワークが平均値に張り付く典型的な underfit だった。fold 内でターゲットを標準化して逆変換したら CV 0.138 まで回復し、GBDT (0.129) と拮抗
- それでも hill climbing は NN に重みを **0** しか与えなかった。犯人は ensemble ロジックではなく、**OOF ラベルの float 精度**だった
- GBDT は float64、NN は float32 でラベルを保存していて、同じ y なのに 1e-7 ずれる。一致判定が `np.array_equal`（完全一致）だったため、NN trial が blend 候補から**黙って除外**されていた
- `allclose` に直し NN ラベルも float64 に揃えたら、NN が **0.25 の重み**を獲得。GBDT-only の CV 0.12462 が GBDT+NN で **0.12275** に改善した

## 競争力のある多様性メンバーが欲しい

ブースティング木 (GBDT) ばかりを混ぜても ensemble は頭打ちになる。誤差の打ち消しには「違う間違い方」をするモデルが要る。前回、線形モデル (Ridge) を混ぜたが「単体精度が GBDT に大きく劣ると、低相関でも貪欲ブレンドに採用されない」と分かった。多様性は**精度が拮抗して初めて**効く。

そこで、木と根本的に系統が違い (滑らかに予測する) かつ拮抗しうる **ニューラルネット (MLP)** を、固定テンプレとして自動探索 (`branch_explorer`) に 1 候補注入することにした。LLM 生成は壊れやすく暴走しやすいので、検証済みの固定実装で堅くする方針だ。

## 罠1: ターゲットを標準化しないと NN は平均に張り付く

最初の実走で CV が 0.50。回帰ターゲット (住宅価格の log) は 11〜13 のレンジで、これを MSE 損失でそのまま学習させると、ネットワークは初期の出力 0 付近からなかなか抜け出せず、結局「全部だいたい平均」を予測して止まる。RMSE ≒ ターゲットの標準偏差、という underfit の典型形だ。

直し方はシンプルで、**fold ごとにターゲットを標準化して学習し、予測を逆変換**する。

```python
ymean, ystd = y[tr].mean(), y[tr].std() + 1e-8
va_std, te_std = train_fold((y[tr] - ymean) / ystd, ...)
va_pred = va_std * ystd + ymean   # 元の空間へ戻す
```

これで MSE の条件数が整い、CV は 0.50 → **0.138**。GBDT の 0.129 に十分拮抗する水準になった。

## 罠2: NN が「一票も入らない」— 犯人は float の精度

拮抗精度になったので、当然 hill climbing がいくらか重みを割り当てると思った。が、重みは GBDT だけで NN は 0。手で計算すると、NN を 1/4 混ぜるだけでブレンド CV は 0.12462 → 0.1224 に**改善する**のに、貪欲法は NN を一度も選ばない。

ログを掘ると、NN trial は候補リストには出るのに `valid` 集合から消えていた。原因はラベル一致チェックだった。

```python
base_label = trials[0].label          # GBDT: float64 で保存
valid = [t for t in trials if np.array_equal(t.label, base_label)]
# NN は float32 で保存 → 同じ y でも 1e-7 ずれ array_equal が False
```

GBDT テンプレはラベルを `float64` で、私の NN は `float32` で保存していた。**同じ正解ラベルなのに dtype 違いで 4.7e-7 ずれる**。完全一致を要求する `np.array_equal` がこれを「別物」と判定し、NN trial を blend から静かに弾いていた。候補表示には出るのに重みが付かない、という分かりにくい症状になる。

これは NN に限らず、**精度の違う dtype で OOF を保存する任意の trial が黙って除外される**潜在バグだ。直し方は二段で:

```python
# 1) 一致判定を許容誤差つきに
valid = [t for t in trials
         if t.label.shape == base_label.shape
         and np.allclose(t.label, base_label, rtol=1e-4, atol=1e-5)]

# 2) NN 側もラベルを float64 で保存して揃える (torch には fold 内で float32 にキャスト)
```

## 結果: NN が 0.25 の重みを取る

直したら hill climbing の重みはこうなった。

```
weights = {1: 0.5(GBDT base), 999: 0.25(NN), 111: 0.25(FE-GBDT)}
GBDT-only blend CV 0.12462  →  GBDT+NN blend CV 0.12275
```

単体では GBDT に負ける NN (0.138) が、**木と低相関ゆえに 1/4 の重みを獲得**し、ブレンド全体の CV を押し下げた。これが「拮抗して初めて効く多様性」の実物だ。さらに同じ評価プールに seed 平均メンバーを足すと CV は 0.121 まで伸びた。

## 学び

- 表形式の NN は **ターゲット標準化**をしないと平均張り付きで簡単に死ぬ。木の感覚で放り込むと underfit する
- 「ensemble に入らない」とき、まず疑うのは重みロジックではなく **OOF の整合性** (空間・行順・dtype)。同じ y でも保存 dtype が違うと完全一致判定をすり抜けて落ちる
- 一致判定は `array_equal` ではなく `allclose`。浮動小数を完全一致で比べる設計は、いつか必ず別経路の精度差で壊れる
- 多様性は低相関 *かつ* 拮抗精度の両方が要る。貪欲 ensemble は「弱いメンバー」を weights=0 で正直に教えてくれる — 重みを読めば多様化が効いたか一目で分かる
