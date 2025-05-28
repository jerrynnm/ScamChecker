import torch
from AI_Model_architecture import BertLSTM_CNN_Classifier

try:
    print("🚀 嘗試載入模型...")
    model = BertLSTM_CNN_Classifier()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    print("✅ 模型成功載入！")
except Exception as e:
    print("❌ 錯誤訊息：", str(e))
