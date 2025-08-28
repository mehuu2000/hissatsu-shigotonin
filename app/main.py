import torch
from transformers import BertForSequenceClassification, AutoTokenizer

# モデルを保存したディレクトリを指定
model_dir = "../model/results/checkpoint-6250"

# モデルとトークナイザーをロード
loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# GPU/CPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)
loaded_model.eval()

# モデルの予測関数を定義
def predict_review_sentiment(review_text):
    # テキストをトークン化
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)

    # デバイスに移動
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # モデルにデータを渡し、予測結果を取得
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        
    # ロジット（生の予測スコア）から最も高いスコアを持つクラスを特定
    predicted_class_id = outputs.logits.argmax().item()
    
    # ラベルは0から始まるため、+1して星評価に変換
    predicted_star_rating = predicted_class_id + 1
    
    return predicted_star_rating

# 新しいレビューで予測を試す
review = input("レビューを入力してください: ")
predicted_rating = predict_review_sentiment(review)

print(f"レビュー: '{review}'")
print(f"予測された星評価: {predicted_rating}つ星")