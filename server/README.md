# 🚀 Multilingual MuRIL Sarcasm API Server

A production-ready **FastAPI** REST microservice serving the fine-tuned multilingual sarcasm detection model [`Mayurkhalse/muril-sarcasm-detector`](https://huggingface.co/Mayurkhalse/muril-sarcasm-detector) across **English**, **Hindi (Devanagari)**, and **Hinglish (Romanized Hindi)**.

---

## 🛠 Features
- **High-speed PyTorch Inference**: Evaluates single or batch sentences.
- **Auto-generated Interactive Docs**: Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **CORS Enabled**: Ready to connect with React, Vue, Next.js, mobile apps, or any web frontend.
- **Batch Endpoint**: Process multiple sentences in a single HTTP request.

---

## 📦 1. Installation

From this directory (`server/`) or project root:
```bash
pip install -r requirements.txt
```

---

## 🏃 2. Running Locally

### Option A: Direct Python Execution
```bash
python app.py
```

### Option B: Uvicorn with Auto-Reload
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will initialize, download/cache the model from Hugging Face Hub, and listen on `http://localhost:8000`.

- **Root Status**: `http://localhost:8000/`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

---

## 📡 3. API Endpoints & Examples

### Single Prediction (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Oh brilliant! Another meeting that could have been an email.\"}"
```

**Response:**
```json
{
  "text": "Oh brilliant! Another meeting that could have been an email.",
  "is_sarcastic": true,
  "prediction": "Sarcastic",
  "confidence": 0.8679,
  "probabilities": {
    "Not Sarcastic": 0.1321,
    "Sarcastic": 0.8679
  }
}
```

### Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d "{\"texts\": [\"Arey wah! 3 ghante late aakar bolte ho I am on time!\", \"आज मौसम बहुत अच्छा है, चलो पार्क चलते हैं।\"]}"
```

---

## 🌐 4. Exposing Your API Publicly (100% Free, No Credit Card)

Since MuRIL requires ~1.5 GB RAM in memory, free cloud tiers (like Render 512MB) will run out of memory. The best free way to expose your running local server to the internet is with **Cloudflare Tunnels** or **ngrok**:

### Using Cloudflare Tunnel (Free & No Signup Needed):
1. Download `cloudflared` or run via Windows Winget:
   ```powershell
   winget install --id Cloudflare.cloudflared
   ```
2. Start the tunnel while `app.py` is running:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
3. Cloudflare gives you an instant, secure public HTTPS URL (e.g. `https://random-name.trycloudflare.com`). Anyone or any frontend in the world can now make API requests to your model!

---

## 🐳 5. Running with Docker

Build and run the container:
```bash
docker build -t muril-sarcasm-api .
docker run -p 8000:8000 muril-sarcasm-api
```
