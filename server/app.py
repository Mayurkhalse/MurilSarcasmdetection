import os
import sys
from contextlib import asynccontextmanager

# Suppress TensorFlow overhead and warnings on Windows
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Ensure UTF-8 output encoding for multilingual logging
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "Mayurkhalse/muril-sarcasm-detector"
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print(f"\n[+] Loading model from Hugging Face Hub: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("[+] Model loaded successfully and ready for inference!")
    print("[+] API is online at: http://127.0.0.1:8000")
    print("[+] Interactive Swagger Docs: http://127.0.0.1:8000/docs\n")
    yield

app = FastAPI(
    title="Multilingual MuRIL Sarcasm Detection API",
    description="Production REST API for detecting sarcasm across English, Hindi, and Hinglish.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for external web/mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class SarcasmRequest(BaseModel):
    text: str = Field(..., example="Oh brilliant! Another meeting that could have been an email.")

class BatchSarcasmRequest(BaseModel):
    texts: list[str] = Field(..., example=[
        "Oh brilliant! Another meeting that could have been an email.",
        "Arey wah! 3 ghante late aakar bolte ho I am on time!",
        "आज मौसम बहुत अच्छा है, चलो पार्क चलते हैं।"
    ])

class SarcasmResponse(BaseModel):
    text: str
    is_sarcastic: bool
    prediction: str
    confidence: float
    probabilities: dict[str, float]

class BatchSarcasmResponse(BaseModel):
    total: int
    results: list[SarcasmResponse]

def _classify_text(text: str) -> SarcasmResponse:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still initializing. Please wait.")
        
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    not_sarcastic_prob = float(probs[0])
    sarcastic_prob = float(probs[1])
    is_sarcastic = sarcastic_prob > not_sarcastic_prob

    return SarcasmResponse(
        text=cleaned,
        is_sarcastic=is_sarcastic,
        prediction="Sarcastic" if is_sarcastic else "Not Sarcastic",
        confidence=round(sarcastic_prob if is_sarcastic else not_sarcastic_prob, 4),
        probabilities={
            "Not Sarcastic": round(not_sarcastic_prob, 4),
            "Sarcastic": round(sarcastic_prob, 4)
        }
    )

@app.get("/")
def root():
    return {
        "status": "online",
        "model": MODEL_NAME,
        "docs": "/docs",
        "endpoints": {
            "single_predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=SarcasmResponse)
def predict(request: SarcasmRequest):
    """Classify a single sentence for sarcasm in English, Hindi, or Hinglish."""
    return _classify_text(request.text)

@app.post("/predict/batch", response_model=BatchSarcasmResponse)
def predict_batch(request: BatchSarcasmRequest):
    """Classify multiple sentences in a single request."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")
    
    results = [_classify_text(t) for t in request.texts]
    return BatchSarcasmResponse(total=len(results), results=results)

if __name__ == "__main__":
    import uvicorn
    # reload=False ensures the 1GB model only loads once in memory
    uvicorn.run(app, host="127.0.0.1", port=8000)
