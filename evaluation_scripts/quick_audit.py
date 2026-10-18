import pandas as pd
import numpy as np
import re

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))
csv_candidates = [
    os.path.join(project_root, "datasets", "sarcasm_500k.csv"),
    os.path.join(project_root, "sarcasm_500k.csv"),
    os.path.join(base_dir, "sarcasm_500k.csv"),
    "sarcasm_500k.csv"
]
csv_path = next((p for p in csv_candidates if os.path.exists(p)), "sarcasm_500k.csv")

print(f"Loading {csv_path}...", flush=True)
df = pd.read_csv(csv_path)
print(f"Total rows: {len(df)}", flush=True)

# 1. Nulls
null_count = df.isna().sum().to_dict()
print(f"Null values: {null_count}", flush=True)

df = df.dropna(subset=["text", "sarcasm_label"])
df["text"] = df["text"].astype(str)
df["sarcasm_label"] = df["sarcasm_label"].astype(int)

# 2. Duplicate analysis
dup_exact = df.duplicated(subset=["text"]).sum()
print(f"Exact duplicate texts: {dup_exact}", flush=True)

norm_text = df["text"].str.strip().str.lower()
dup_norm = norm_text.duplicated().sum()
print(f"Normalized duplicate texts: {dup_norm}", flush=True)

# Conflicting labels
conflicts = df.groupby(norm_text)["sarcasm_label"].nunique()
num_conflicts = (conflicts > 1).sum()
print(f"Texts with conflicting labels (both 0 and 1): {num_conflicts}", flush=True)

# 3. Short texts & junk
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df["char_count"] = df["text"].apply(len)

print("\nWord count percentiles:")
print(df["word_count"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]), flush=True)

under_3_words = (df["word_count"] < 3).sum()
under_4_words = (df["word_count"] < 4).sum()
print(f"Texts with < 3 words: {under_3_words} ({under_3_words/len(df)*100:.2f}%)", flush=True)
print(f"Texts with < 4 words: {under_4_words} ({under_4_words/len(df)*100:.2f}%)", flush=True)

# 4. Check typical short texts labeled as sarcastic vs not sarcastic
print("\nSample short texts labeled as sarcastic (1):", flush=True)
print(df[(df["sarcasm_label"] == 1) & (df["word_count"] <= 3)][["text", "sarcasm_label"]].head(15), flush=True)

print("\nSample short texts labeled as non-sarcastic (0):", flush=True)
print(df[(df["sarcasm_label"] == 0) & (df["word_count"] <= 3)][["text", "sarcasm_label"]].head(15), flush=True)

# 5. Check URL-only or bot commands
bot_re = re.compile(r"^(!remindme|remindme!|http|www\.)", re.IGNORECASE)
bot_matches = df["text"].str.strip().str.match(bot_re).sum()
print(f"\nBot commands / pure URL starts: {bot_matches}", flush=True)
