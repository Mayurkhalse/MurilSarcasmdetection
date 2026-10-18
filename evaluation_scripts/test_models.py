import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

# Windows terminal UTF-8 output fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class SarcasmDataset(Dataset):
    """PyTorch Dataset for batched text inference."""
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


def evaluate_model(model_path: str, df: pd.DataFrame, device: torch.device, batch_size: int = 64, max_length: int = 64):
    """Evaluates a fine-tuned sequence classification model on a given dataset."""
    print("\n" + "=" * 70)
    print(f"  Testing Model: {os.path.basename(model_path)}")
    print(f"  Path: {model_path}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    print("Loading tokenizer and model...")
    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print(f"[✓] Model loaded in {time.time() - t0_load:.2f} seconds onto {device}.")

    # 2. Setup DataLoader
    dataset = SarcasmDataset(df["text"].tolist(), df["sarcasm_label"].tolist(), tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_builder(tokenizer, max_length=max_length),
        num_workers=0
    )

    # 3. Inference Loop
    all_preds = []
    all_labels = []
    t0_eval = time.time()

    desc = f"Evaluating {os.path.basename(model_path)}"
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dataloader, desc=desc, unit="batch"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            outputs = model(**batch_inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())

    eval_duration = time.time() - t0_eval
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Metrics Calculation
    acc = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    throughput = len(df) / eval_duration if eval_duration > 0 else 0

    # Print Model Evaluation Summary
    print("\n" + "-" * 70)
    print(f"Evaluation Summary for: {os.path.basename(model_path)}")
    print("-" * 70)
    print(f"  Total Samples Evaluated: {len(df):,}")
    print(f"  Time Taken:             {eval_duration:.2f} s ({throughput:.1f} samples/sec)")
    print(f"  Accuracy:               {acc * 100:.2f}%")
    print(f"  Macro Precision:        {precision_macro * 100:.2f}%")
    print(f"  Macro Recall:           {recall_macro * 100:.2f}%")
    print(f"  Macro F1-Score:         {f1_macro * 100:.2f}%")
    print(f"  Sarcastic F1-Score:     {f1_bin * 100:.2f}%")
    print("\nConfusion Matrix (Rows: Actual [0: Not Sarcastic, 1: Sarcastic], Cols: Predicted):")
    print(f"  [Not Sarcastic -> Correct: {cm[0][0]:<6} | False Sarcastic: {cm[0][1]:<6}]")
    print(f"  [Sarcastic     -> Missed:  {cm[1][0]:<6} | Correct:         {cm[1][1]:<6}]")
    print("-" * 70)

    target_names = ["Not Sarcastic (0)", "Sarcastic (1)"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0))

    # Free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": os.path.basename(model_path),
        "total_samples": len(df),
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_sarcastic": f1_bin,
        "time_seconds": eval_duration,
        "samples_per_sec": throughput,
        "confusion_matrix": cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare MuRIL sarcasm detection models.")
    parser.add_argument(
        "--csv",
        type=str,
        default="sarcasm_headlines_dataset.csv",
        help="Path to CSV dataset (format: text,sarcasm_label)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of samples to evaluate on (0 evaluates entire dataset)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Evaluation batch size (default: 64)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Maximum token sequence length (default: 64)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling (if sample_size > 0)"
    )
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    def resolve_dataset_path(path_str):
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str
        candidates = [
            path_str,
            os.path.join(project_root, "datasets", path_str),
            os.path.join(project_root, "datasets", os.path.basename(path_str)),
            os.path.join(project_root, path_str),
            os.path.join(base_dir, path_str)
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return path_str

    def resolve_model_path(model_dir_name):
        candidates = [
            os.path.join(project_root, "models", model_dir_name),
            os.path.join(project_root, model_dir_name),
            os.path.join(base_dir, model_dir_name),
            model_dir_name
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return os.path.join(project_root, "models", model_dir_name)

    csv_file = resolve_dataset_path(args.csv)

    if not os.path.exists(csv_file):
        print(f"Error: Dataset not found at: {csv_file}")
        print(f"Searched in: datasets/, root, and {base_dir}")
        sys.exit(1)

    print(f"Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)

    # Validate columns
    if "text" not in df.columns or "sarcasm_label" not in df.columns:
        print(f"Error: CSV must contain 'text' and 'sarcasm_label' columns. Found: {list(df.columns)}")
        sys.exit(1)

    # Drop any nulls
    df = df.dropna(subset=["text", "sarcasm_label"])
    df["sarcasm_label"] = df["sarcasm_label"].astype(int)

    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"Sampling {args.sample_size} records out of {len(df)} total records (seed={args.random_seed})...")
        df = df.sample(n=args.sample_size, random_state=args.random_seed).reset_index(drop=True)

    print(f"Evaluation dataset ready: {len(df)} records.")
    print("Class distribution:")
    print(df["sarcasm_label"].value_counts())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model paths to test in sequence
    model_paths = [
        resolve_model_path("500kdataTrained_muril_sarcasm"),
        resolve_model_path("trained_muril_sarcasm")
    ]

    results = []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model directory not found: {model_path}. Skipping.")
            continue
        res = evaluate_model(
            model_path=model_path,
            df=df,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        results.append(res)

    # Comparative Summary Table
    if results:
        print("\n" + "=" * 80)
        print("                      FINAL COMPARATIVE EVALUATION RESULTS")
        print("=" * 80)
        header = f"{'Model Name':<32} | {'Accuracy':<10} | {'Macro F1':<10} | {'Sarcasm F1':<11} | {'Time (s)':<9}"
        print(header)
        print("-" * 80)
        for r in results:
            print(
                f"{r['model_name']:<32} | "
                f"{r['accuracy'] * 100:>8.2f}% | "
                f"{r['f1_macro'] * 100:>8.2f}% | "
                f"{r['f1_sarcastic'] * 100:>9.2f}% | "
                f"{r['time_seconds']:>8.2f}s"
            )
        print("=" * 80)

        # Highlight winner
        best_model = max(results, key=lambda x: x["accuracy"])
        print(f"\n🏆 Best Performing Model on this Benchmark: {best_model['model_name']} ({best_model['accuracy'] * 100:.2f}% Accuracy)\n")

        # Save results to CSV summary in reports directory
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        summary_csv = os.path.join(reports_dir, "models_evaluation_summary.csv")
        summary_df = pd.DataFrame([{
            "Model Name": r["model_name"],
            "Samples": r["total_samples"],
            "Accuracy (%)": round(r["accuracy"] * 100, 2),
            "Macro Precision (%)": round(r["precision_macro"] * 100, 2),
            "Macro Recall (%)": round(r["recall_macro"] * 100, 2),
            "Macro F1 (%)": round(r["f1_macro"] * 100, 2),
            "Sarcasm Class F1 (%)": round(r["f1_sarcastic"] * 100, 2),
            "Time (s)": round(r["time_seconds"], 2),
            "Throughput (samples/s)": round(r["samples_per_sec"], 1)
        } for r in results])
        summary_df.to_csv(summary_csv, index=False)
        print(f"[✓] Saved comparative summary to: {summary_csv}")


if __name__ == "__main__":
    main()
