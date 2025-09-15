# データセット: messengers-reviews-google-play
# 作成者: UniqueData
# ライセンス: CC BY-NC-ND 4.0
# 利用に関する注意: このデータセットは非商用利用に限られます。

import os
from datasets import load_from_disk, load_dataset, ClassLabel
from transformers import AutoTokenizer
import random

isUpdate = True  # データセットを再ダウンロードする場合はTrueに設定

# ディレクトリが空かどうかを確認する関数
def is_dir_empty(path):
    with os.scandir(path) as it:
        if any(it):
            return False
    return True

# データセットの基本情報を定義
datasets_name = "UniqueData/messengers-reviews-google-play"
raw_dataset_path = "../store/raw"
processed_dataset_path = "../store/processed"
Nickname = "google_play"

# データセットの保存先パスを定義
train_path = os.path.join(raw_dataset_path, "train", Nickname)
validation_path = os.path.join(raw_dataset_path, "validation", Nickname)

tokenized_train_path = os.path.join(processed_dataset_path, "train", Nickname)
tokenized_validation_path = os.path.join(processed_dataset_path, "validation", Nickname)

# 削除する不要な列を定義
def columns_remove(dataset):
    new_columns = []
    necessary_columns = ['text', 'label']  # 必要な列

    for col in dataset.column_names:
        if col not in necessary_columns:
            new_columns.append(col)
    return new_columns

# データセットをダウンロード
if is_dir_empty(train_path) or is_dir_empty(validation_path) or isUpdate:
    try:
        print(f"'{datasets_name}'をダウンロード中...\n")
        # データセット全体をロード(見た感じtrainしかないっぽい)
        dataset = load_dataset(datasets_name, split="train")
       
        print(f"データセット数: {len(dataset)}")
        print(f"利用可能な列: {dataset.column_names}\n")
       
        # 必要なキーを生成し当てはまる値を設定、不要な列を削除
        dataset = dataset.map(
            lambda record: {
                'text': record['content'],  # レビューテキスト
                'label': record['score'] - 1  # 星評価を0-4に変換(ラベルが1-5のため)
            },
            remove_columns = columns_remove(dataset)
        )
       
        print(f"前処理後のデータセット数: {len(dataset)}")
        print(f"前処理後の利用可能な列: {dataset.column_names}\n")
       
        # ラベル列をClassLabel型に変換（層化分割のため）(今回は検証データがないので、ClassLabelに変換してから分割する)
        # ClassLabel型は各ラベルが独立したカテゴリとして扱われるため、層化分割が可能になる＆モデルの学習が安定する
        try:
            dataset = dataset.cast_column('label', ClassLabel(num_classes=5, names=['0', '1', '2', '3', '4']))
            print("ラベル列をClassLabel型に変換しました。")
           
            # データセットを訓練用と検証用に分割（80:20）層化分割
            dataset_split = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column='label')
            print("層化分割でデータセットを分割しました。\n")
        except Exception as e:
            print(f"層化分割でエラーが発生: {e}")
            # 層化分割なしで分割
            dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
            print("通常分割でデータセットを分割しました。\n")

        train_dataset = dataset_split['train']
        validation_dataset = dataset_split['test']
       
        print(f"分割後 - 訓練データ: {len(train_dataset)}, 検証データ: {len(validation_dataset)}\n")
       
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(validation_path), exist_ok=True)

        # データセットを保存
        print(f"データセットを'{train_path}'と'{validation_path}'に保存中...")
        train_dataset.save_to_disk(train_path)
        validation_dataset.save_to_disk(validation_path)
        print(f"'{datasets_name}'が")
        print(f"'{train_path}'と")
        print(f"'{validation_path}'に保存されました。\n")

    except Exception as e:
        print(f"データセットのダウンロード中にエラーが発生しました: {e}\n")
        exit(1)
else:
    print(f"'{Nickname}'データセットは")
    print(train_path)
    print(validation_path)
    print("に既に存在します。\n")

# データセットの読み込み
train_dataset = load_from_disk(train_path)
validation_dataset = load_from_disk(validation_path)
print("保存したディスクからデータセットを読み込みました。\n")

# ラベルの分布を確認
print("訓練データのラベル分布:")
label_counts_train = {}
for label in train_dataset['label']:
    label_counts_train[label] = label_counts_train.get(label, 0) + 1
for label in sorted(label_counts_train.keys()):
    print(f"ラベル {label} ({label+1}つ星): {label_counts_train[label]}件")

print("\n検証データのラベル分布:")
label_counts_val = {}
for label in validation_dataset['label']:
    label_counts_val[label] = label_counts_val.get(label, 0) + 1
for label in sorted(label_counts_val.keys()):
    print(f"ラベル {label} ({label+1}つ星): {label_counts_val[label]}件")

# labelの0-4でそれぞれ均等にサンプリング（オプション）
sampled_train_indices = []
sampled_validation_indices = []

target_train_size = 4000  # 各ラベルあたりの目標サンプル数
target_validation_size = 100

print(f"\n各ラベルから {target_train_size}件（訓練用）、{target_validation_size}件（検証用）をサンプリング...\n")

# 各ラベルから均等にデータをサンプリング
for label_id in range(5):
    print(f"ラベル {label_id} のデータをサンプリング中...")

    # ラベルに一致するインデックスを取得
    train_indices = [i for i, label in enumerate(train_dataset['label']) if label == label_id]
    validation_indices = [i for i, label in enumerate(validation_dataset['label']) if label == label_id]

    # インデックスをシャッフル
    random.shuffle(train_indices)
    random.shuffle(validation_indices)

    # 各ラベルから指定件数分を選択
    sampled_train_indices.extend(train_indices[:min(target_train_size, len(train_indices))])
    sampled_validation_indices.extend(validation_indices[:min(target_validation_size, len(validation_indices))])

# 抽出したインデックスを使ってデータセットを作成
train_dataset = train_dataset.select(sampled_train_indices)
validation_dataset = validation_dataset.select(sampled_validation_indices)
print("サンプリングが完了しました。\n")
print(f"サンプリング後 - 訓練データ： {len(train_dataset)}, 検証データ: {len(validation_dataset)}\n")

# トークナイザーのロード
print("トークナイザーをロード中...")
model_name = "cl-tohoku/bert-base-japanese-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = tokenizer.model_max_length
if max_length > 1024:
    max_length = 512
print(f"トークナイザーの最大長: {max_length}\n")

# データセットをトークン化する関数を定義
def tokenize_function(data):
    return tokenizer(data['text'], padding="max_length", truncation=True, max_length=max_length)

# データセットをトークン化
print("データセットをトークン化中...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
print("データセットのトークン化が完了しました。\n")

# 不要な列を削除
print("不要なtext列を削除中...")
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(['text'])
print("不要なtext列を削除\n")

# 前処理が完了したデータセットを保存
print(f"前処理が完了したデータセットを'{processed_dataset_path}'に保存中...")
if not os.path.exists(processed_dataset_path):
    os.makedirs(processed_dataset_path, exist_ok=True)
tokenized_train_dataset.save_to_disk(tokenized_train_path)
tokenized_validation_dataset.save_to_disk(tokenized_validation_path)

print(f"\n前処理が完了したデータセットが")
print(tokenized_train_path)
print(tokenized_validation_path)
print("に保存されました。\n")

print("最終的なデータセット情報:")
print(f"訓練データの数: {len(tokenized_train_dataset)}")
print(f"検証データの数: {len(tokenized_validation_dataset)}")

# 最終的なラベル分布を確認
print("\n最終的な訓練データのラベル分布:")
final_label_counts_train = {}
for label in tokenized_train_dataset['label']:
    final_label_counts_train[label] = final_label_counts_train.get(label, 0) + 1
for label in sorted(final_label_counts_train.keys()):
    print(f"ラベル {label} ({label+1}つ星): {final_label_counts_train[label]}件")

print("\n最終的な検証データのラベル分布:")
final_label_counts_val = {}
for label in tokenized_validation_dataset['label']:
    final_label_counts_val[label] = final_label_counts_val.get(label, 0) + 1
for label in sorted(final_label_counts_val.keys()):
    print(f"ラベル {label} ({label+1}つ星): {final_label_counts_val[label]}件")