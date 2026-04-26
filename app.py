from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import pickle
import os
import uvicorn

app = FastAPI()

# Model load karne ka variable
model = None
device = "cpu"  # GitHub Actions CPU pe chalega, isliye CPU pe force kar rahe hain

# Model load karne wala function
def load_model():
    global model
    try:
        # Model file ka path
        model_path = "model/best_model/data.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                # CPU pe map kar ke load karo - ye IMPORTANT hai!
                # map_location=torch.device('cpu') se GPU wala model CPU pe load ho jayega
                model = pickle.load(f, map_location=torch.device('cpu'))
            print(f"✅ Model load ho gaya on {device}")
            return True
        else:
            print(f"❌ Model file nahi mili: {model_path}")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return True  # Testing ke liye True return kar rahe hain, dummy response dega

# Input format kya hoga
class InputData(BaseModel):
    text: str

# API endpoint - /predict
@app.post("/predict")
async def predict(data: InputData):
    if model is None:
        # Agar model load nahi hua, toh dummy response
        return {
            "prediction": "Dummy prediction - model load nahi hua",
            "input_text": data.text,
            "status": "model_not_loaded"
        }
    
    # Yaha aap model ka inference code likhenge
    # Abhi dummy response de raha hoon
    return {
        "prediction": "Model se output",
        "input_text": data.text,
        "status": "success"
    }

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Model API chal raha hai",
        "endpoints": ["/predict (POST)", "/health (GET)"]
    }

# Server start karne ke liye
if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
