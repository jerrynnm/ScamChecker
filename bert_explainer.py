import torch
from AI_Model_architecture import BertLSTM_CNN_Classifier, BertPreprocessor
from transformers import BertTokenizer
import re
import requests
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 從 Google Drive 載入 model.pth
def load_model_from_drive():
    model_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"  # 替換為你的檔案 ID
    response = requests.get(model_url)
    if response.status_code == 200:
        with open("model.pth", "wb") as f:
            f.write(response.content)
        return True
    return False

if not os.path.exists("model.pth"):
    if not load_model_from_drive():
        raise FileNotFoundError("無法從 Google Drive 載入 model.pth")

model = BertLSTM_CNN_Classifier()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese")

def predict_single_sentence(model, tokenizer, sentence, max_len=256):
    model.eval()
    with torch.no_grad():
        sentence = re.sub(r"\s+", "", sentence)
        sentence = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？:/.\-]", "", sentence)

        encoded = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded["token_type_ids"].to(device)

        output = model(input_ids, attention_mask, token_type_ids)
        prob = output.item()
        label = int(prob > 0.5)

        if prob > 0.9:
            risk = "🔴 高風險（極可能是詐騙）"
        elif prob > 0.5:
            risk = "🟡 中風險（可疑）"
        else:
            risk = "🟢 低風險（正常）"

        pre_label = "詐騙" if label == 1 else "正常"

        print(f"\n📩 訊息內容：{sentence}")
        print(f"✅ 預測結果：{pre_label}")
        print(f"📊 信心值：{round(prob*100, 2)}")
        print(f"⚠️ 風險等級：{risk}")
        return pre_label, prob, risk

def analyze_text(text):
    label, prob, risk = predict_single_sentence(model, tokenizer, text)
    return {
        "status": label,
        "confidence": round(prob*100, 2),
        "suspicious_keywords": [risk]
    }