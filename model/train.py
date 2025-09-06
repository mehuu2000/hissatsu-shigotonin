import os 
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# データセットのパスを定義
datasets_path = "../datasets/store/processed"
amazon_train_path = os.path.join(datasets_path, "train", "amazon")
amazon_validation_path = os.path.join(datasets_path, "validation", "amazon")
google_play_train_path = os.path.join(datasets_path, "train", "google_play")
google_play_validation_path = os.path.join(datasets_path, "validation", "google_play")

# データセット読み込み
print("データセットを読み込み中...")
amazon_train_dataset = load_from_disk(amazon_train_path)
amazon_validation_dataset = load_from_disk(amazon_validation_path)
google_play_train_dataset = load_from_disk(google_play_train_path)
google_play_validation_dataset = load_from_disk(google_play_validation_path)
print("データセットの読み込みが完了しました。\n")

# データセットを結合
print("データセットを結合中...")
combined_train_dataset = concatenate_datasets([amazon_train_dataset, google_play_train_dataset])
combined_validation_dataset = concatenate_datasets([amazon_validation_dataset, google_play_validation_dataset])
print("データセットの結合が完了しました。\n")

print(f"訓練データの数: {len(combined_train_dataset)}")
print(f"検証データの数: {len(combined_validation_dataset)}")

# モデルのロード
print("モデルをロード中...")
model_name = "cl-tohoku/bert-base-japanese-v2"
num_labels = 5 # 星評価1〜5

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#　モデルがGPUかCPUのどちらで動作するか確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"モデルは {device} で動作します。\n")

# 学習引数の設定
args = TrainingArguments(
    output_dir="./results",           # 学習結果（モデルの重みなど）の保存先
    num_train_epochs=1,               # 訓練エポック数
    per_device_train_batch_size=32,   # 訓練バッチサイズ(一度に処理するデータ数)
    per_device_eval_batch_size=32,    # 評価バッチサイズ(一度に処理するデータ数)
    warmup_steps=500,                 # 学習率のウォームアップステップ数
    weight_decay=0.01,                # 正則化
    logging_dir='./logs',             # ログの保存先
    logging_steps=10,                 # 何ステップごとにログを記録するか
    evaluation_strategy="epoch",      # エポックごとに評価を実行
    save_strategy="epoch",            # エポックごとにモデルを保存
    load_best_model_at_end=True,      # 学習終了時に最良モデルをロード
    metric_for_best_model="accuracy", # 最良モデルの基準とする指標
    fp16=torch.cuda.is_available(),   # 混合精度学習の有効化
    dataloader_num_workers=4,         # データローダーのワーカースレッド数
)

# 評価指標の計算関数を定義
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
    }

# Trainerのインスタンス化
trainer = Trainer(
    args = args,
    model = model,
    train_dataset = combined_train_dataset,
    eval_dataset = combined_validation_dataset,
    compute_metrics = compute_metrics,
)

# 学習実行
print("学習を開始します...\n")
trainer.train()
print("学習が完了しました。\n")

# 最終モデルの評価
print("モデルの評価を開始します...\n")
metrics = trainer.evaluate()
print("評価結果:\n, ", metrics)

# 混同行列の作成と表示
print("\n混同行列を生成中...\n")
predictions = trainer.predict(combined_validation_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = combined_validation_dataset['label']

cm = confusion_matrix(true_labels, predicted_labels)
print("混同行列:\n", cm)
print(f"精度 (Accuracy): {metrics['eval_accuracy']:.4f}")
print(f"F1スコア (F1-Score): {metrics['eval_f1']:.4f}")

# 最終モデルとトークナイザーを保存
model_save_path = "./final_model"
print(f"\n最終モデルを '{model_save_path}' に保存中...")

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("最終モデルとトークナイザーが'{model_save_path}'に保存されました。")
