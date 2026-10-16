import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load data
df = pd.read_csv("sarcasm_train_dataset.csv")
texts = df["text"].tolist()
labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

# 2. Load model & tokenizer
model_name = "google/muril-base-cased"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "Not Sarcastic", 1: "Sarcastic"},
    label2id={"Not Sarcastic": 0, "Sarcastic": 1}
)

# 3. Freeze base BERT encoder so only the classification head trains
# This makes training 10x faster on CPU and lets the classifier learn immediately
for param in model.bert.parameters():
    param.requires_grad = False

# 4. Tokenize training data
print("Tokenizing data...")
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# 5. Fast Training Loop (Head Optimization)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-2, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

print("Training classification head on CPU...")
model.train()
for epoch in range(1, 26):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = loss_fn(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0 or epoch == 1:
        preds = torch.argmax(outputs.logits, dim=-1)
        acc = (preds == labels).float().mean().item()
        print(f"Epoch {epoch:2d}/25 - Loss: {loss.item():.4f} - Training Accuracy: {acc * 100:.1f}%")

# 6. Save fine-tuned model
output_dir = "./trained_muril_sarcasm"
print(f"Saving fine-tuned model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[✓] Model saved successfully!")
