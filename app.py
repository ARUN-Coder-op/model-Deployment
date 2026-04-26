from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import pickle
import os
import uvicorn

app = FastAPI()

model = None
device = "cpu"

def load_model():
    global model
    try:
        model_path = "model/best_model/data.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f, map_location=torch.device('cpu'))
            print(f"✅ Model load ho gaya on {device}")
            return True
        else:
            print(f"❌ Model file nahi mili: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return True

class InputData(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: InputData):
    if model is None:
        return {
            "prediction": "Dummy prediction - model load nahi hua",
            "input_text": data.text,
            "status": "model_not_loaded"
        }
    
    return {
        "prediction": "Model se output",
        "input_text": data.text,
        "status": "success"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    return {
        "message": "Model API chal raha hai",
        "endpoints": ["/predict (POST)", "/health (GET)"]
    }

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
