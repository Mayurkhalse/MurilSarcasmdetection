# 🎭 Multilingual Sarcasm Detection with MuRIL

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Mayurkhalse%2Fmuril--sarcasm--detector-yellow)](https://huggingface.co/Mayurkhalse/muril-sarcasm-detector)
[![Base Model](https://img.shields.io/badge/Base%20Model-google%2Fmuril--base--cased-blue)](https://huggingface.co/google/muril-base-cased)
[![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Hindi%20%7C%20Hinglish-green)](#languages-supported)
[![Task](https://img.shields.io/badge/Task-Binary%20Sarcasm%20Classification-orange)](#overview)

A production-ready multilingual sarcasm classification system built on Google's **MuRIL (`google/muril-base-cased`)** foundation model, fine-tuned and regularized across **English**, **Hinglish** (Romanized Hindi), and **Hindi** (Devanagari script).

The fine-tuned model is deployed and publicly accessible on the Hugging Face Model Hub at:  
👉 **[`Mayurkhalse/muril-sarcasm-detector`](https://huggingface.co/Mayurkhalse/muril-sarcasm-detector)**

> 📖 **Comprehensive Journey Document**: For an in-depth breakdown of all experiments, evaluations, data audits, and label correction metrics, see [**`docs/EXPERIMENTS_AND_JOURNEY.md`**](docs/EXPERIMENTS_AND_JOURNEY.md).

---

## 📌 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Key Experimental Results & Benchmarks](#key-experimental-results--benchmarks)
  - [1. News Headlines Benchmark (Model 1 vs Model 2)](#1-news-headlines-benchmark-model-1-vs-model-2)
  - [2. Synthetic 1,000 Challenge & Domain Shift](#2-synthetic-1000-challenge--domain-shift)
  - [3. Dataset Audit & Refinement Pipeline (`500k_proper_data.csv`)](#3-dataset-audit--refinement-pipeline-500k_proper_datacsv)
- [Quick Start: Inference & Testing](#quick-start-inference--testing)
  - [1. Fast Inference in Python](#1-fast-inference-in-python)
  - [2. Comparing Models via CLI (`evaluation_scripts/test_models.py`)](#2-comparing-models-via-cli-evaluation_scriptstest_modelspy)
  - [3. Live Hub Inference (`evaluation_scripts/test_sarcasm.py`)](#3-live-hub-inference-evaluation_scriptstest_sarcasmpy)
- [Training Pipelines](#training-pipelines)
  - [1. GPU Scaled Training (`training_scripts/train_gpu_500k.py`)](#1-gpu-scaled-training-training_scriptstrain_gpu_500kpy)
  - [2. Regularized Multilingual Training (`training_scripts/train_regularized.py`)](#2-regularized-multilingual-training-training_scriptstrain_regularizedpy)
- [Data Processing & Generation](#data-processing--generation)
- [REST API Server (`server/`)](#rest-api-server-server)
- [Local Setup & Installation](#local-setup--installation)

---

## 📖 Overview

Detecting sarcasm in Indian social contexts is uniquely challenging because real-world conversations and social media posts heavily feature **code-mixing** (Hinglish: *"Arey wah! 3 ghante late aakar bolte ho 'I am on time'!"*) and native script satire (Hindi: *"वाह भाई! पूरी रात रील्स देखकर कहते हैं कि पढ़ाई का समय नहीं मिला।"*). 

Standard English models fail on transliterated text, while basic multilingual BERT models struggle with colloquial nuances. Google's **MuRIL (Multilingual Representations for Indian Languages)** was specifically pre-trained on 17 Indian languages with transliterated pairs, making it the ideal foundation.

---

## 📂 Repository Structure

The repository is organized into dedicated, modular directories:

```text
MURIL/
├── datasets/                      # Raw, synthetic, and refined CSV/JSON datasets
│   ├── 500k_proper_data.csv       # High-quality refined 500k dataset (493,541 balanced rows)
│   ├── sarcasm_500k.csv           # Raw SARC Reddit 500k dataset
│   ├── sarcasm_headlines_dataset.csv # Converted news headlines dataset (26,709 rows)
│   ├── Sarcasm_Headlines_Dataset.json# Original JSON headlines dataset
│   ├── random_synthetic_sarcasm_1000.csv # 1,000-sample balanced challenge benchmark
│   └── sarcasm_1000_dataset.csv   # 1,000-sample balanced multilingual dataset (EN/HI/Hinglish)
│
├── models/                        # Saved fine-tuned checkpoints
│   ├── 500kdataTrained_muril_sarcasm/ # Model 1: Trained on large-scale SARC 500k
│   └── trained_muril_sarcasm/     # Model 2: Regularized on 1k multilingual pairs
│
├── training_scripts/              # Fine-tuning and optimization scripts
│   ├── train_gpu_500k.py          # High-performance GPU trainer (FP16 AMP, accum steps)
│   ├── train_regularized.py       # Regularized training (layer freezing, label smoothing)
│   └── quick_train.py             # Rapid test trainer
│
├── evaluation_scripts/            # Auditing and benchmarking suites
│   ├── test_models.py             # Dual-model comparative benchmark runner
│   ├── test_model1_on_synthetic.py# Challenge evaluation on synthetic 1,000 dataset
│   ├── test_sarcasm.py            # Live tester for Hugging Face deployed checkpoint
│   ├── simple_example.py          # Standalone inference examples across Indian languages
│   ├── quick_audit.py             # Dataset sanity, length, and duplicate auditor
│   ├── audit.py                   # In-depth linguistic analyzer & TF-IDF baseline
│   └── find_relabeled.py          # Audits relabeled rows (0 -> 1 and 1 -> 0)
│
├── data_processing/               # Data engineering and dataset generators
│   ├── refine_sarcasm_dataset.py  # 5-step automated noise filter and relabeling pipeline
│   ├── convert_json_to_csv.py     # JSON to CSV standardizer
│   ├── generate_random_1000.py    # Synthetic 1,000 challenge benchmark generator
│   └── generate_dataset.py        # Multilingual 1,000 sample generator
│
├── reports/                       # Evaluation CSV summaries and prediction logs
│   ├── models_evaluation_summary.csv # Final comparison table
│   └── synthetic_1000_predictions.csv# 1,000 sample detailed predictions with confidences
│
├── server/                        # FastAPI REST microservice with Dockerfile & docs
├── docs/                          # Detailed research notes & project documentation
│   └── EXPERIMENTS_AND_JOURNEY.md # Full experiment history & data audit details
│
├── sarcasm_detection.py           # Core library wrapper class for inference & training
├── upload_to_huggingface.py      # Hugging Face Hub uploader
├── requirements.txt               # Global project dependencies
└── README.md                      # Project documentation (this file)
```

---

## 📊 Key Experimental Results & Benchmarks

### 1. News Headlines Benchmark (Model 1 vs Model 2)
Evaluated on unseen news headlines from `datasets/sarcasm_headlines_dataset.csv` (The Onion vs HuffPost):

| Model Checkpoint | Accuracy | Macro Precision | Macro Recall | Macro F1 | Sarcasm Class F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`500kdataTrained_muril_sarcasm` (Model 1)** | **96.10%** | **96.09%** | **96.10%** | **0.9609** | **0.9573** |
| **`trained_muril_sarcasm` (Model 2)** | 49.70% | 51.57% | 51.15% | 0.4682 | 0.3082 |

- **Insight**: Model 1 demonstrates state-of-the-art headline comprehension (**96.10%**), while Model 2 shows domain divergence because it was trained primarily on short conversational Indian banter.

### 2. Synthetic 1,000 Challenge & Domain Shift
We generated a challenging benchmark of 1,000 balanced sentences (`datasets/random_synthetic_sarcasm_1000.csv`) featuring multi-clause, formal, and corporate sarcasm.
- **Model 1 Accuracy**: **54.50%**
- **Non-Sarcastic (Class 0) Recall**: **98.8%** (494 / 500 correct)
- **Sarcastic (Class 1) Recall**: **10.2%** (51 / 500 correct; 449 missed!)
- **Domain Shift Takeaway**: Reddit-trained models over-rely on short punchy phrases and explicit `/s` markers. When faced with formal or polite sarcasm (*"What a brilliant move to replace our entire technical support team with a broken chatbot"*), they fail to recognize irony.

### 3. Dataset Audit & Refinement Pipeline (`500k_proper_data.csv`)
Auditing `datasets/sarcasm_500k.csv` revealed substantial label noise caused by Reddit self-annotations:
- **30,338 uninformative junk rows** (1–2 word fragments like *"starbucks"*, pure bot commands like `!RemindMe`, deleted comments).
- **2,616 duplicate pairs** with conflicting labels.
- **7,745 false sarcastic rows ($1 \to 0$)** corrected (factual questions and complaints labeled as sarcasm).
- **139 false non-sarcastic rows ($0 \to 1$)** corrected (classic irony posted without `/s`).
- Result: **`datasets/500k_proper_data.csv`** contains **493,541 high-quality, balanced samples** ready for robust retraining.

---

## 🚀 Quick Start: Inference & Testing

### 1. Fast Inference in Python
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "Mayurkhalse/muril-sarcasm-detector"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

texts = [
    "Oh brilliant! Another meeting that could have been an email.",
    "The engineering team resolved the database latency issue on time.",
    "वाह भाई! पूरी रात रील्स देखकर कहते हैं कि पढ़ाई का समय नहीं मिला।"
]

inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

for text, prob in zip(texts, probs):
    is_sarc = prob[1] > prob[0]
    label = "🎭 Sarcastic" if is_sarc else "✅ Not Sarcastic"
    conf = max(prob[0], prob[1]).item() * 100
    print(f"[{label} ({conf:.1f}%)] {text}")
```

### 2. Comparing Models via CLI (`evaluation_scripts/test_models.py`)
Run automated head-to-head comparison between Model 1 and Model 2:
```bash
# Evaluate both models on news headlines
python evaluation_scripts/test_models.py --csv datasets/sarcasm_headlines_dataset.csv --sample_size 1000

# Evaluate both models on the refined 500k dataset
python evaluation_scripts/test_models.py --csv datasets/500k_proper_data.csv --sample_size 1000
```

### 3. Live Hub Inference (`evaluation_scripts/test_sarcasm.py`)
```bash
# Test sample suite
python evaluation_scripts/test_sarcasm.py

# Test any custom phrase
python evaluation_scripts/test_sarcasm.py "Wah kya mast scheme hai, 21 din me paisa double!"
```

---

## 🏋️ Training Pipelines

### 1. GPU Scaled Training (`training_scripts/train_gpu_500k.py`)
Fine-tune MuRIL on the refined 500k dataset with FP16 Mixed Precision, gradient accumulation, and gradient checkpointing (runs on 4GB VRAM cards like GTX 1650):
```bash
python training_scripts/train_gpu_500k.py --data_path datasets/500k_proper_data.csv --epochs 2 --batch_size 32 --accum_steps 2
```

### 2. Regularized Multilingual Training (`training_scripts/train_regularized.py`)
Fine-tunes with 80/20 train/val split, layer freezing (unfreezing only top layers 10-11), L2 weight decay, and label smoothing:
```bash
python training_scripts/train_regularized.py
```

---

## 🛠️ Data Processing & Generation

```bash
# Re-run automated 5-step dataset refinement pipeline
python data_processing/refine_sarcasm_dataset.py

# Convert Sarcasm_Headlines_Dataset.json to CSV
python data_processing/convert_json_to_csv.py

# Generate synthetic 1,000 challenge benchmark
python data_processing/generate_random_1000.py

# Generate multilingual 1,000 dataset
python data_processing/generate_dataset.py
```

---

## 🌐 REST API Server (`server/`)

We have packaged the model into a standalone, production-ready **FastAPI** service:

```bash
# Install server dependencies
pip install -r server/requirements.txt

# Run the API server
cd server
python app.py
```

- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Predict Endpoint**: `POST http://localhost:8000/predict`
- **Batch Predict Endpoint**: `POST http://localhost:8000/predict/batch`

Refer to [`server/README.md`](server/README.md) for Docker instructions.

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mayurkhalse/MurilSarcasmdetection.git
   cd MurilSarcasmdetection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run model evaluation:**
   ```bash
   python evaluation_scripts/test_models.py --sample_size 100
   ```

---

## 📜 License & Acknowledgments
- **Base Model**: [MuRIL by Google Research](https://huggingface.co/google/muril-base-cased)
- **Deployed Checkpoint**: [Mayurkhalse/muril-sarcasm-detector](https://huggingface.co/Mayurkhalse/muril-sarcasm-detector)
- Built with Hugging Face `transformers`, `torch`, and `datasets`.
