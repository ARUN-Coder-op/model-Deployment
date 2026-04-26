from fastapi import FastAPI
import torch
import pickle
import os
import uvicorn

app = FastAPI()
model = None

def load_model():
    global model
    model_path = "model/best_model/data.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            # YE MOST IMPORTANT LINE HAI
            model = pickle.load(f, map_location=torch.device('cpu'))
        print("✅ Model loaded on CPU")
        return True
    print("❌ Model file not found")
    return False

@app.get("/")
def root():
    return {"status": "running", "model_loaded": model is not None}

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
