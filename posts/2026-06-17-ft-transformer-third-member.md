---
title: "FT-Transformer を ensemble の第3系統にする — 木でも MLP でもない予測のクセを足す"
emoji: "🧬"
type: "tech"
topics: ["kaggle", "ensemble", "pytorch", "transformer", "machinelearning"]
published: true
publication_date: "2026-06-17"
---

## TL;DR

- テーブルデータの ensemble に「GBDT」と「MLP」を入れていたが、両者だけだと予測のクセが 2 系統しかない
- 第3系統として **FT-Transformer (Feature Tokenizer + Transformer)** を純 PyTorch で実装し、ブランチ探索が候補として自動注入するようにした
- 各特徴量を 1 トークンに射影 (`x_j → x_j·W_j + b_j`)、CLS トークンを足して Transformer に通し、CLS 出力をヘッドへ。依存追加なし
- 合成データのスモークで、回帰は FT-Transformer (RMSE 0.173) が MLP (0.207) を上回った。木とも MLP とも相関が違うので、貪欲ブレンドが重みを割り当てられる
- 既定は MLP のまま (後方互換)。`NN_ARCH=ft_transformer` で切替

## なぜ第3系統が要るのか

ensemble は「精度が拮抗していて、かつ予測が低相関なメンバー」を混ぜると効く。GBDT (木) と MLP (全結合) は確かにクセが違うが、それでも 2 系統。貪欲ブレンドや stacking に「もう一つ違う見方」を渡せれば、伸びしろが出る。

FT-Transformer はテーブルデータ向けの Transformer で、Kaggle でも強メンバーとして定番化している。木の「軸並行の分割」とも、MLP の「滑らかな全結合」とも違う、**特徴量間の attention**という別の帰納バイアスを持つ。第3系統にうってつけだ。

## 実装: 依存を増やさず純 PyTorch で

外部ライブラリを増やすと環境の再現性が痛む。FT-Transformer の肝は「各特徴量を独立にトークン化する」ところだけなので、これは素の PyTorch で書ける。

```python
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, dim))
        self.bias   = nn.Parameter(torch.empty(n_features, dim))
        nn.init.normal_(self.weight, std=0.02); nn.init.zeros_(self.bias)
    def forward(self, x):                       # x: (B, n_features)
        return x.unsqueeze(-1) * self.weight + self.bias   # → (B, n_features, dim)
```

あとは CLS トークンを先頭に足し、`nn.TransformerEncoder` に通し、CLS の出力を小さなヘッドに渡すだけ。

```python
t   = self.tok(x)                       # (B, F, d)
cls = self.cls.expand(t.shape[0], -1, -1)
t   = torch.cat([cls, t], dim=1)        # (B, F+1, d)
t   = self.enc(t)
return self.head(t[:, 0])               # CLS → 出力
```

one-hot 後の高次元でもトークン数 = 列数で素直に動く。attention は O(n²·d) なのでトークン次元 d は小さく (32) 保ち、CPU でも実用域に収めた。

## 既存資産との接続を壊さない

ensemble パイプラインは OOF / submission / metrics の「保存契約」で各メンバーを束ねている。新メンバーもこの契約 (同じ空間で OOF を保存し、ラベルの dtype を他メンバーと揃える) を満たさないと、ブレンド時に黙って弾かれる。FT-Transformer メンバーも既存 NN テンプレと同じ保存規約に乗せ、ブランチ探索が MLP に**加えて**もう 1 候補として注入する形にした。既定の挙動 (MLP) は変えていない。

## スモークの結果

合成データで分類/回帰 × MLP/FT-Transformer の 4 通りを CPU で回し、全部が OOF/submission/metrics を生成して CV が有限になることを確認した。

```
classification / mlp            AUC 0.993
classification / ft_transformer AUC 1.000
regression     / mlp            RMSE 0.207
regression     / ft_transformer RMSE 0.173   ← MLP より良い
```

合成データなので絶対値に意味はないが、**FT-Transformer が MLP と別のクセで、かつ拮抗 (or 上回る) 精度を出す**ことは確認できた。これが低相関の第3系統として効く前提条件になる。

## 学び

- ensemble の伸びしろは「もう一つ違う帰納バイアス」にある。木 / 全結合 / attention は別物
- 強メンバーでも依存を増やさず素の PyTorch で書ける部分は多い。再現性のためにそうする価値がある
- 新メンバーを足すときは精度より先に「**保存契約を満たすか**」。ブレンドに乗らなければ精度は無意味
- 既定の挙動は変えず、新アーキは opt-in の切替にしておくと安全に増やせる
