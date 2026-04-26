from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
import uvicorn

app = FastAPI()
model = None

def load_model():
    global model
    try:
        model_path = "model/best_model/data.pkl"
        
        if os.path.exists(model_path):
            # YEH SAHI HAI - torch.load use karo, pickle.load nahi
            model = torch.load(model_path, map_location=torch.device('cpu'))
            print("✅ Model loaded on CPU")
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return False

class InputData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: InputData):
    return {"prediction": "ok", "input": data.text}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/")
async def root():
    return {"message": "API running"}

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
