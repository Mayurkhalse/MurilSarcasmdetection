import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

# Windows terminal UTF-8 encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": str(self.texts[idx]),
            "label": int(self.labels[idx])
        }


def collate_fn_builder(tokenizer, max_length=64):
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encodings, labels
    return collate_fn


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    # Model and dataset path resolution
    model_candidates = [
        os.path.join(project_root, "models", "500kdataTrained_muril_sarcasm"),
        os.path.join(project_root, "500kdataTrained_muril_sarcasm"),
        os.path.join(base_dir, "500kdataTrained_muril_sarcasm")
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), model_candidates[0])

    csv_candidates = [
        os.path.join(project_root, "datasets", "random_synthetic_sarcasm_1000.csv"),
        os.path.join(project_root, "random_synthetic_sarcasm_1000.csv"),
        os.path.join(base_dir, "random_synthetic_sarcasm_1000.csv")
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), csv_candidates[0])

    print("=" * 75)
    print("  TESTING MODEL 1: 500kdataTrained_muril_sarcasm ON 1,000 SYNTHETIC DATASET")
    print("=" * 75)

    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found: {csv_path}")
        sys.exit(1)

    # 1. Load Dataset
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df["sarcasm_label"].value_counts())

    # 2. Load Model & Tokenizer
    print(f"\nLoading Model 1 from: {model_path} ...")
    t0_load = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"[✓] Model loaded in {time.time() - t0_load:.2f} s on {device}.")

    # 3. Create DataLoader
    batch_size = 64
    max_length = 64
    dataset = SarcasmDataset(df["text"].tolist(), df["sarcasm_label"].tolist(), tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_builder(tokenizer, max_length=max_length)
    )

    # 4. Run Batched Inference
    all_preds = []
    all_probs = []
    all_labels = []

    t0_eval = time.time()
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dataloader, desc="Evaluating Model 1"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            outputs = model(**batch_inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch_labels.numpy())

    eval_duration = time.time() - t0_eval
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 5. Compute Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    throughput = len(df) / eval_duration if eval_duration > 0 else 0

    print("\n" + "=" * 75)
    print("                         MODEL 1 EVALUATION RESULTS")
    print("=" * 75)
    print(f"  Model Tested:           500kdataTrained_muril_sarcasm")
    print(f"  Dataset:                random_synthetic_sarcasm_1000.csv (1,000 samples)")
    print(f"  Inference Duration:     {eval_duration:.2f} s ({throughput:.1f} samples/s)")
    print(f"  Overall Accuracy:       {acc * 100:.2f}%")
    print(f"  Macro Precision:        {prec_macro * 100:.2f}%")
    print(f"  Macro Recall:           {rec_macro * 100:.2f}%")
    print(f"  Macro F1-Score:         {f1_macro * 100:.2f}%")
    print(f"  Sarcastic Class F1:     {f1_bin * 100:.2f}%")
    print("-" * 75)

    print("\nConfusion Matrix:")
    print(f"  Actual Not-Sarcastic (0): Correct: {cm[0][0]:<5} | Misclassified as Sarcastic: {cm[0][1]:<5}")
    print(f"  Actual Sarcastic (1):     Missed:  {cm[1][0]:<5} | Correct:                  {cm[1][1]:<5}")

    target_names = ["Not Sarcastic (0)", "Sarcastic (1)"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0))

    # 6. Save Full Predictions to CSV
    confidences = [round(float(max(p)) * 100, 2) for p in all_probs]
    sarc_probs = [round(float(p[1]) * 100, 2) for p in all_probs]
    df["predicted_label"] = all_preds
    df["confidence_pct"] = confidences
    df["sarcasm_prob_pct"] = sarc_probs
    df["is_correct"] = (df["sarcasm_label"] == df["predicted_label"])

    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    predictions_csv = os.path.join(reports_dir, "synthetic_1000_predictions.csv")
    df.to_csv(predictions_csv, index=False, encoding="utf-8")
    print(f"[✓] Saved all 1,000 detailed predictions to: {predictions_csv}")

    # 7. Print Sample Predictions
    print("\n" + "=" * 75)
    print("                    SAMPLE PREDICTIONS BREAKDOWN")
    print("=" * 75)
    
    # 5 Sarcastic samples
    print("\n--- Sarcastic Examples (Ground Truth = 1) ---")
    sarc_samples = df[df["sarcasm_label"] == 1].head(5)
    for _, row in sarc_samples.iterrows():
        status = "✅ CORRECT" if row["is_correct"] else "❌ WRONG"
        short_text = (row['text'][:65] + '...') if len(row['text']) > 68 else row['text']
        print(f"  Text:       \"{short_text}\"")
        print(f"  Truth: 1 | Pred: {row['predicted_label']} | Conf: {row['confidence_pct']}% | {status}\n")

    # 5 Non-Sarcastic samples
    print("--- Non-Sarcastic Examples (Ground Truth = 0) ---")
    non_sarc_samples = df[df["sarcasm_label"] == 0].head(5)
    for _, row in non_sarc_samples.iterrows():
        status = "✅ CORRECT" if row["is_correct"] else "❌ WRONG"
        short_text = (row['text'][:65] + '...') if len(row['text']) > 68 else row['text']
        print(f"  Text:       \"{short_text}\"")
        print(f"  Truth: 0 | Pred: {row['predicted_label']} | Conf: {row['confidence_pct']}% | {status}\n")

    # Misclassified examples (if any)
    errors = df[~df["is_correct"]]
    if len(errors) > 0:
        print(f"--- Misclassified Samples ({len(errors)} / {len(df)}) ---")
        for _, row in errors.head(5).iterrows():
            short_text = (row['text'][:65] + '...') if len(row['text']) > 68 else row['text']
            print(f"  Text:       \"{short_text}\"")
            print(f"  Truth: {row['sarcasm_label']} | Pred: {row['predicted_label']} | Conf: {row['confidence_pct']}%\n")
    else:
        print("--- Zero misclassifications! Perfect 100% score! ---")

    print("=" * 75)


if __name__ == "__main__":
    main()
