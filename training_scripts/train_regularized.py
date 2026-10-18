"""
Regularized Training Pipeline for MuRIL Sarcasm Detection
Features:
  - 80/20 Train-Validation Split (800 train / 200 unseen validation)
  - Label Smoothing (0.1) to prevent overconfidence (e.g. 99.99%)
  - Weight Decay (0.05) for L2 regularization
  - Unfreezes top encoder layers (10, 11) + pooler + classifier
  - Evaluates true unseen generalization metrics per epoch
"""

import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))

# 1. Load 1,000-sample dataset
csv_candidates = [
    os.path.join(project_root, "datasets", "sarcasm_1000_dataset.csv"),
    os.path.join(project_root, "sarcasm_1000_dataset.csv"),
    os.path.join(base_dir, "sarcasm_1000_dataset.csv"),
    "sarcasm_1000_dataset.csv"
]
csv_path = next((p for p in csv_candidates if os.path.exists(p)), "sarcasm_1000_dataset.csv")

print(f"[*] Loading dataset: {csv_path}...")
df = pd.read_csv(csv_path)

# Handle label column name ('label' or 'sarcasm_label')
label_col = "sarcasm_label" if "sarcasm_label" in df.columns else "label"
df["label"] = df[label_col].astype(int)

# 80/20 Stratified Split (800 train / 200 validation)
train_df, val_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)
print(f"[*] Train set: {len(train_df)} samples, Validation set: {len(val_df)} samples")

# 2. Tokenize
model_name = "google/muril-base-cased"
print(f"[*] Loading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_df(dataframe):
    encodings = tokenizer(
        dataframe["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    labels = torch.tensor(dataframe["label"].tolist(), dtype=torch.long)
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)

print("[*] Tokenizing train and validation sets...")
train_dataset = tokenize_df(train_df)
val_dataset = tokenize_df(val_df)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. Model Architecture & Regularization
print(f"[*] Loading base model '{model_name}'...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "Not Sarcastic", 1: "Sarcastic"},
    label2id={"Not Sarcastic": 0, "Sarcastic": 1}
)

# Unfreeze top 2 encoder layers + pooler + classifier for domain adaptation
for name, param in model.bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Optimizer with L2 Weight Decay (0.05)
optimizer = torch.optim.AdamW([
    {"params": model.classifier.parameters(), "lr": 1e-4, "weight_decay": 0.05},
    {"params": [p for n, p in model.bert.named_parameters() if p.requires_grad], "lr": 2e-5, "weight_decay": 0.05}
])

# CrossEntropyLoss with Label Smoothing (0.1) prevents extreme 99.99% logit saturation
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# 4. Training & Validation Loop
num_epochs = 3
print(f"[*] Starting regularized training for {num_epochs} epochs...")

for epoch in range(1, num_epochs + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    for b_input_ids, b_mask, b_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
        loss = loss_fn(outputs.logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase on unseen 200 samples
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for b_input_ids, b_mask, b_labels in val_loader:
            outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
            loss = loss_fn(outputs.logits, b_labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(b_labels.cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    print(
        f"Epoch {epoch}/{num_epochs} -> "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}% | "
        f"Val F1: {f1:.4f}"
    )

# 5. Save Model
output_dir = os.path.join(project_root, "models", "trained_muril_sarcasm")
os.makedirs(output_dir, exist_ok=True)
print(f"[*] Saving regularized model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[✓] Regularized fine-tuning complete and model saved!")
