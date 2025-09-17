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

(1, 2, 3, 4が面倒くさいと思う人。上記を実行済みの場合はする必要はありません。)
```
make all
```

## 利用データセット
### messengers-reviews-google-play
- **Hugging Face URL**： (https://huggingface.co/datasets/UniqueData/messengers-reviews-google-play)
- **作成者**： UniqueData
- **ライセンス**： CC-BY-NC-ND-4.0

### amazon_reviews_multi_ja
- **Hugging Face URL**： (https://huggingface.co/datasets/SetFit/amazon_reviews_multi_ja)
- **作成者**： SetFit
- **概要** : アマゾン商品のレビュー
- **言語** : 日本語

### review_helpfulness_prediction 
- **Hugging Face URL**： (https://huggingface.co/datasets/tafseer-nayeem/review_helpfulness_prediction)
- **作成者** : tafseer-nayeem
- **ライセンス** : CC-BY-NC-SA-4.0
- **概要** : レビューがほかのユーザーにとってどれほど有用であるかを扱うデータ

### yelp_review_full
- **Hugging Face URL**： (https://huggingface.co/datasets/Yelp/yelp_review_full)
- **作成者** : yelp
- **ライセンス** : other
- **概要** : テキストからどの程度の評価なのかを扱うデータ

### drug-reviews
- **Hugging Face URL**： (https://huggingface.co/datasets/Mouwiya/drug-reviews)
- **作成者** : Mouwiya
- **ライセンス** : odbl
- **概要** : 薬のレビュー

