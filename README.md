# 必殺評価人

## 概要
ユーザーの何かしらのレビュー(テキスト)を読み込み、星評価(1~5)を判断し返すアプリです。

## クイックスタート
1, 任意のディレクトリで
```
git clone https://github.com/mehuu2000/hissatsu-shigotonin.git
cd hissatsu-shigotonin
```

2, アプリを起動
```
make run
```

3, 指示に従いレビュー ⇒ 評価

## 学習手順
1, 任意ディレクトリでコードをクローン
```
git clone https://github.com/mehuu2000/hissatsu-shigotonin.git
cd hissatsu-shigotonin
```

2, データセットの取得
```
make datasets
```

3, モデルの学習
```
make train
```

4, アプリを起動
```
make run
```

(1, 2, 3, 4が面倒くさいと思う人)
(
```
make all
```
)

## 利用データセット
### messengers-reviews-google-play
- **Hugging Face URL**： (https://huggingface.co/datasets/UniqueData/messengers-reviews-google-play)
- **作成者**： UniqueData
- **ライセンス**： CC-BY-NC-ND-4.0

### amazon_reviews_multi_ja
- **Hugging Face URL**： (https://huggingface.co/datasets/SetFit/amazon_reviews_multi_ja)
- **作成者**： SetFit