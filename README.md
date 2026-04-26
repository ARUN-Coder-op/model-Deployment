# Model Deployment with CI/CD

## 📌 Project Overview
This project deploys a PyTorch model using FastAPI with GitHub Actions CI/CD pipeline.

## 🚀 Deployment Options

### Option 1: Render (Recommended)
1. Sign up at [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
5. Click Deploy

### Option 2: Railway
1. Sign up at [railway.app](https://railway.app)
2. Install Railway CLI: `npm i -g @railway/cli`
3. Run: `railway login`
4. Run: `railway init`
5. Run: `railway up`

### Option 3: Ollama (Local)
```bash
ollama pull llama2
ollama run llama2