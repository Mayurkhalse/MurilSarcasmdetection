import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Load data
df = pd.read_csv("sarcasm_train_dataset.csv")
texts = df["text"].tolist()
labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

# 2. Tokenize
model_name = "google/muril-base-cased"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
loader = DataLoader(dataset, batch_size=6, shuffle=True)

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "Not Sarcastic", 1: "Sarcastic"},
    label2id={"Not Sarcastic": 0, "Sarcastic": 1}
)

# Unfreeze the top 2 encoder layers + pooler + classifier
# Freezing the lower 10 layers keeps it very fast on CPU while allowing rich domain adaptation!
for name, param in model.bert.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Higher learning rate for classifier, moderate for top encoder layers
optimizer = torch.optim.AdamW([
    {"params": model.classifier.parameters(), "lr": 5e-4},
    {"params": [p for n, p in model.bert.named_parameters() if p.requires_grad], "lr": 3e-5}
], weight_decay=0.01)

loss_fn = torch.nn.CrossEntropyLoss()

print("Training top layers + classifier on CPU...")
model.train()
for epoch in range(1, 13):
    total_loss = 0.0
    correct = 0
    total = 0
    for b_input_ids, b_mask, b_labels in loader:
        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
        loss = loss_fn(outputs.logits, b_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == b_labels).sum().item()
        total += len(b_labels)
        
    avg_loss = total_loss / len(loader)
    acc = correct / total
    print(f"Epoch {epoch:2d}/12 - Loss: {avg_loss:.4f} - Training Accuracy: {acc * 100:.1f}%")

# Save model
output_dir = "./trained_muril_sarcasm"
print(f"Saving fine-tuned model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("[✓] Model saved successfully!")
