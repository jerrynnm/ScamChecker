from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from bert_explainer import analyze_text as bert_analyze_text
from firebase_admin import credentials, firestore
import firebase_admin
import pytz
import os
import json
import requests
import torch

app = FastAPI(
    title="詐騙訊息辨識 API",
    description="使用 BERT 模型分析輸入文字是否為詐騙內容",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextAnalysisRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class TextAnalysisResponse(BaseModel):
    status: str
    confidence: float
    suspicious_keywords: List[str]
    analysis_timestamp: datetime
    text_id: str

# 初始化 Firebase 使用環境變數
try:
    cred_data = os.getenv("FIREBASE_CREDENTIALS")
    if not cred_data:
        raise ValueError("FIREBASE_CREDENTIALS 環境變數未設置")
    cred = credentials.Certificate({"type": "service_account", **json.loads(cred_data)})
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase 初始化錯誤: {e}")

# 從 Google Drive 載入 model.pth
def load_model_from_drive():
    model_url = "https://drive.google.com/uc?export=download&id=1UXkOqMPUiPUIbsy8iENHUqbNFLEHcFFg"  # 替換為你的檔案 ID
    response = requests.get(model_url)
    if response.status_code == 200:
        with open("model.pth", "wb") as f:
            f.write(response.content)
        return True
    return False

if not os.path.exists("model.pth"):
    if not load_model_from_drive():
        raise FileNotFoundError("無法從 Google Drive 載入 model.pth")

from AI_Model_architecture import BertLSTM_CNN_Classifier
model = BertLSTM_CNN_Classifier()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

@app.get("/")
async def root():
    return {"message": "詐騙文字辨識 API 已啟動", "version": "1.0.0", "status": "active", "docs": "/docs"}

@app.post("/predict", response_model=TextAnalysisResponse)
async def analyze_text_api(request: TextAnalysisRequest):
    try:
        tz = pytz.timezone("Asia/Taipei")
        taiwan_now = datetime.now(tz)
        collection_name = taiwan_now.strftime("%Y%m%d")
        document_id = taiwan_now.strftime("%Y%m%dT%H%M%S")
        timestamp_str = taiwan_now.strftime("%Y-%m-%d %H:%M:%S")

        result = bert_analyze_text(request.text)

        record = {
            "text_id": document_id,
            "text": request.text,
            "user_id": request.user_id,
            "analysis_result": {
                "status": result["status"],
                "confidence": result["confidence"],
                "suspicious_keywords": result["suspicious_keywords"],
            },
            "timestamp": timestamp_str,
            "type": "text_analysis"
        }

        db.collection(collection_name).document(document_id).set(record)

        return TextAnalysisResponse(
            status=result["status"],
            confidence=result["confidence"],
            suspicious_keywords=result["suspicious_keywords"],
            analysis_timestamp=taiwan_now,
            text_id=document_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def save_user_feedback(feedback: dict):
    try:
        tz = pytz.timezone("Asia/Taipei")
        taiwan_now = datetime.now(tz)
        timestamp_str = taiwan_now.strftime("%Y-%m-%d %H:%M:%S")

        feedback["used_in_training"] = False
        feedback["timestamp"] = timestamp_str

        db.collection("user_feedback").add(feedback)
        return {"message": "✅ 已記錄使用者回饋"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
