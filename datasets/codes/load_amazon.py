import os
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer

isUpdate = False  # データセットを再ダウンロードする場合はTrueに設定

# データセット情報を定義
datasets_name = "SetFit/amazon_reviews_multi_ja"
local_dataset_path = "../store/raw"
processed_dataset_path = "../store/processed"
Nickname = "amazon"

# データセットの保存先パスを定義
train_path = os.path.join(local_dataset_path, "train", Nickname)
validation_path = os.path.join(local_dataset_path, "validation", Nickname)

# データセットをダウンロード
if not os.path.exists(train_path) or isUpdate:
    try:
        print(f"'{datasets_name}'をダウンロード中...")
        # train, validationのデータセットをtext, labelに限定してロード
        train_dataset = load_dataset(datasets_name, split="train")
        validation_dataset = load_dataset(datasets_name, split="validation")

        # 不要な列を削除
        train_dataset = train_dataset.remove_columns(["id", "label_text"])
        validation_dataset = validation_dataset.remove_columns(["id", "label_text"])

        # デモ用にデータ数を100件に制限
        # train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        # validation_dataset = validation_dataset.select(range(min(100, len(validation_dataset))))

        # ローカルに保存
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(validation_path), exist_ok=True)

        train_dataset.save_to_disk(train_path)
        validation_dataset.save_to_disk(validation_path)
        print(f"'{datasets_name}'が'{local_dataset_path}'に保存されました。\n")
    except Exception as e:
        print(f"データセットのダウンロード中にエラーが発生しました: {e}\n")
        exit(1)
else:
    print(f"'{Nickname}'データセットは'{local_dataset_path}'に既に存在します。\n")

# データセットの読み込み
train_dataset = load_from_disk(train_path)
validation_dataset = load_from_disk(validation_path)

print(f"データセットの情報を表示")
print(f"訓練データの数: {len(train_dataset)}")
print(f"検証データの数: {len(validation_dataset)}")
print(f"訓練データの例: {train_dataset[0]}\n")

# トークナイザーのロード
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
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(['text'])

# 前処理が完了したデータセットを保存
print(f"前処理が完了したデータセットを'{processed_dataset_path}'に保存中...")
if not os.path.exists(processed_dataset_path):
    os.makedirs(processed_dataset_path, exist_ok=True)
tokenized_train_dataset.save_to_disk(os.path.join(processed_dataset_path, "train", Nickname))
tokenized_validation_dataset.save_to_disk(os.path.join(processed_dataset_path, "validation", Nickname))
print(f"前処理が完了したデータセットが'{processed_dataset_path}'に保存されました。\n")

