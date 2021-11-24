## 振り返り

### トップsolutionsから得たもの

#### スケーリング後の分布を確認するべし
- 多くのノートブックでRobust Scalerが利用されていたが、デフォルトの(25, 75)ではスケーリング後でも-200~200の値域の特徴量が存在していた．そのため、(10, 90)などに調整する必要があった．

#### multi-taskで正則化するべし
- targetのdiffやdiffのdiffを加えたmulti-taskで学習
- targetが離散値だったので分類問題へ

#### foldを変えて検証するべし
- fold数やseedを変えて複数検証する

#### ライブラリを変えるべし
- kerasとpytorchでスコアは変わる
- 初期化の方法を揃える必要あり

#### 学習率スケジューラを変えるべし
- ReduceOnPlateau以外も試す
- CosineAnnealingWarmupRestartが良かったらしい

#### バッチサイズを変えるべし
- 学習率と関係するため、何も考えず固定するのは危険

#### 訓練データとの差分を予測するべし
- 人口データだったため、データ間に不自然なくらいの類似性があった
- 訓練データの中で最も類似するデータとの差分を予測する
- top5からランダムに取り出すことでTTAも可能
