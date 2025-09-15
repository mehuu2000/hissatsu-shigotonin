import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import os

# --- モデルとトークナイザーのロード ---
# train.pyで保存した最終モデルのディレクトリを指定
model_dir = "../model/final_model"

if not os.path.exists(model_dir):
    print(f"エラー: '{model_dir}' ディレクトリが見つかりません。train.py を実行してモデルを保存してください。")
    exit()

# モデルとトークナイザーをロード
# from_pretrained()は、ディレクトリ内のすべての関連ファイルを自動で読み込みます
loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# GPUが利用可能ならモデルをGPUに移動
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)
loaded_model.eval() # 推論モードに設定

print("モデルとトークナイザーのロードが完了しました。\n")

# --- 予測関数の定義 ---
def predict_review_sentiment(review_text):
    # テキストをトークン化し、モデルの入力形式に変換
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # データをGPUに移動
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 勾配計算を無効化し、予測を実行
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        
    # ロジットから最も高いスコアを持つクラスIDを取得
    predicted_class_id = outputs.logits.argmax().item()
    
    # クラスIDは0から始まるため、+1して星評価に変換
    predicted_star_rating = predicted_class_id + 1
    
    return predicted_star_rating

# --- 予測の実行 ---
print("レビューを入力してください ('exit'で終了):")
while True:
    review = input("> ")
    if review.lower() == 'exit':
        break
    
    predicted_rating = predict_review_sentiment(review)
    
    print("\n\n")
    print(f"予測された星評価: {predicted_rating}つ星\n")