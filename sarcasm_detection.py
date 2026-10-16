#!/usr/bin/env python3
"""
MuRIL (Multilingual Representations for Indian Languages) Sarcasm Detector
Model: google/muril-base-cased

MuRIL is a BERT-based multilingual foundation model pre-trained on 17 Indian
languages and English, supporting both native scripts and Latin transliteration
(e.g., Hinglish, Tanglish).

NOTE:
    'google/muril-base-cased' is a base masked language model. To achieve
    accurate sarcasm detection, its classification head must be fine-tuned
    on a labeled sarcasm dataset. This script provides both:
      1. Training / Fine-tuning pipeline (`train`)
      2. Inference & prediction engine (`predict` / interactive CLI)
"""

import argparse
import os
import sys
from typing import Dict, List, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class MuRILSarcasmDetector:
    """Wrapper class for training and inference with MuRIL for sarcasm detection."""

    DEFAULT_MODEL_NAME = "google/muril-base-cased"
    ID2LABEL = {0: "Not Sarcastic", 1: "Sarcastic"}
    LABEL2ID = {"Not Sarcastic": 0, "Sarcastic": 1}

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        device: str = None,
        max_length: int = 128,
    ):
        """
        Initialize MuRIL Sarcasm Detector.

        Args:
            model_name_or_path: Hugging Face model hub id or path to saved checkpoint.
            device: 'cuda', 'cpu', or None (auto-detect).
            max_length: Maximum token length for truncation and padding.
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[*] Loading tokenizer for '{model_name_or_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        print(f"[*] Loading model from '{model_name_or_path}' on device '{self.device}'...")
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=2,
            id2label=self.ID2LABEL,
            label2id=self.LABEL2ID,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )
        self.model.to(self.device)
        self.model.eval()

        if model_name_or_path == self.DEFAULT_MODEL_NAME:
            print(
                "[!] NOTE: You are using the raw base model 'google/muril-base-cased'.\n"
                "    The classification head was randomly initialized. For accurate\n"
                "    predictions, fine-tune the model using the '--mode train' option\n"
                "    or supply a fine-tuned checkpoint path.\n"
            )

    def predict(
        self,
        texts: Union[str, List[str]],
        threshold: float = 0.5,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict whether text is Sarcastic or Not Sarcastic.

        Args:
            texts: Single string or list of strings.
            threshold: Sarcasm classification probability threshold (default: 0.5).

        Returns:
            List of dictionaries containing predictions, confidence, and probabilities.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        if not text_list:
            return []

        # Tokenize input batch
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu()

        results = []
        for i, text in enumerate(text_list):
            not_sarcastic_prob = probabilities[i, 0].item()
            sarcastic_prob = probabilities[i, 1].item()

            is_sarcastic = sarcastic_prob >= threshold
            label = "Sarcastic" if is_sarcastic else "Not Sarcastic"
            confidence = sarcastic_prob if is_sarcastic else not_sarcastic_prob

            results.append(
                {
                    "text": text,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "sarcasm_score": round(sarcastic_prob, 4),
                    "probabilities": {
                        "Not Sarcastic": round(not_sarcastic_prob, 4),
                        "Sarcastic": round(sarcastic_prob, 4),
                    },
                }
            )

        return results

    def train(
        self,
        data_path: str,
        output_dir: str = "./trained_muril_sarcasm",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        val_split: float = 0.15,
        weight_decay: float = 0.01,
    ):
        """
        Fine-tune MuRIL on a custom dataset CSV containing 'text' and 'label' (0 or 1).

        Args:
            data_path: Path to dataset CSV file.
            output_dir: Directory where fine-tuned model and tokenizer will be saved.
            epochs: Number of training epochs.
            batch_size: Batch size for training and evaluation.
            learning_rate: Learning rate for AdamW optimizer.
            val_split: Fraction of dataset for validation split.
            weight_decay: Weight decay for regularization.
        """
        try:
            import pandas as pd
            from datasets import Dataset
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            from sklearn.model_selection import train_test_split
        except ImportError as e:
            print(f"[!] Error: Missing required packages for training: {e}")
            print("    Please run: pip install pandas scikit-learn datasets")
            return

        print(f"[*] Loading training data from: {data_path}")
        df = pd.read_csv(data_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' (0 or 1) columns.")

        # Ensure correct types and drop NaNs
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)

        print(f"[*] Dataset size: {len(df)} samples")
        print(f"    Class distribution:\n{df['label'].value_counts().to_string()}")

        # Train/Validation Split
        if val_split > 0 and len(df) > 5:
            train_df, val_df = train_test_split(
                df, test_size=val_split, random_state=42, stratify=df["label"]
            )
        else:
            train_df = df
            val_df = df

        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        val_ds = Dataset.from_pandas(val_df, preserve_index=False)

        def preprocess_function(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_length,
            )

        print("[*] Tokenizing datasets...")
        tokenized_train = train_ds.map(preprocess_function, batched=True)
        tokenized_val = val_ds.map(preprocess_function, batched=True)

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            predictions = preds.argmax(axis=-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average="binary", zero_division=0
            )
            acc = accuracy_score(labels, predictions)
            return {
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

        os.makedirs(output_dir, exist_ok=True)

        # Unfreeze top encoder layers, pooler, and classifier for effective adaptation
        for name, param in self.model.bert.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if len(val_df) > 5 else "no",
            save_strategy="epoch" if len(val_df) > 5 else "no",
            learning_rate=learning_rate if learning_rate != 2e-5 else 5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            logging_steps=5,
            save_total_limit=1,
            load_best_model_at_end=True if len(val_df) > 5 else False,
            metric_for_best_model="f1" if len(val_df) > 5 else None,
            report_to="none",
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val if len(val_df) > 5 else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("[*] Starting fine-tuning...")
        trainer.train()

        print(f"[*] Saving fine-tuned model and tokenizer to: {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("[✓] Training completed successfully!")


def run_interactive(detector: MuRILSarcasmDetector):
    """Run an interactive CLI session."""
    print("=" * 60)
    print("  MuRIL Sarcasm Detector - Interactive Mode")
    print("  (Works with English, Hindi, Hinglish, & Indian languages)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            text = input("\nEnter text: ").strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                print("Exiting.")
                break

            result = detector.predict(text)[0]
            label_badge = "[SARCASTIC]" if result["label"] == "Sarcastic" else "[NOT SARCASTIC]"

            print(f"Prediction  : {label_badge}")
            print(f"Confidence  : {result['confidence'] * 100:.2f}%")
            print(f"Sarcasm Score: {result['sarcasm_score']:.4f}")
            print(f"Details     : {result['probabilities']}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="MuRIL Sarcasm Detection Tool (Inference & Training)"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "train", "interactive"],
        default="interactive",
        help="Execution mode: predict, train, or interactive (default: interactive)",
    )
    parser.add_argument(
        "--model_path",
        default="google/muril-base-cased",
        help="HuggingFace model ID or path to a saved model directory (default: google/muril-base-cased)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text for prediction (used with --mode predict)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="sample_sarcasm_data.csv",
        help="Path to CSV dataset for training (used with --mode train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_muril_sarcasm",
        help="Output directory to save fine-tuned model (used with --mode train)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )

    args = parser.parse_args()

    detector = MuRILSarcasmDetector(model_name_or_path=args.model_path)

    if args.mode == "predict":
        if not args.text:
            print("[!] Please provide input text using --text \"your text here\"")
            sys.exit(1)
        res = detector.predict(args.text)[0]
        print("\n--- Sarcasm Prediction ---")
        print(f"Text        : {res['text']}")
        print(f"Prediction  : {res['label']}")
        print(f"Confidence  : {res['confidence'] * 100:.2f}%")
        print(f"Sarcasm Score: {res['sarcasm_score']}")
        print(f"Breakdown   : {res['probabilities']}")

    elif args.mode == "train":
        if not os.path.exists(args.data):
            print(f"[!] Dataset file not found: {args.data}")
            sys.exit(1)
        detector.train(
            data_path=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    elif args.mode == "interactive":
        run_interactive(detector)


if __name__ == "__main__":
    main()
