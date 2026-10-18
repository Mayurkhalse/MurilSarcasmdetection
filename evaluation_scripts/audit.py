# audit.py
import re, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

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

df = pd.read_csv(csv_path)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(10).to_string(), "\n")
print("nulls:\n", df.isna().sum(), "\n")

df = df.dropna(subset=["text", "sarcasm_label"]).copy()
df["text"] = df["text"].astype(str)
df["sarcasm_label"] = df["sarcasm_label"].astype(int)
print("label balance:\n", df["sarcasm_label"].value_counts(normalize=True), "\n")

# duplicates / conflicting labels
print("exact duplicate texts:", df.duplicated("text").sum())
norm = df["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
print("duplicates after normalisation:", norm.duplicated().sum())
conf = df.assign(norm=norm).groupby("norm")["sarcasm_label"].nunique()
print("normalised texts with conflicting labels:", (conf > 1).sum(), "\n")

# length per class
df["n_words"] = df["text"].str.split().str.len()
print(df.groupby("sarcasm_label")["n_words"].describe(), "\n")

# script / surface cues per class
def script(t):
    dev = len(re.findall(r"[\u0900-\u097F]", t))
    lat = len(re.findall(r"[A-Za-z]", t))
    if dev and lat: return "mixed"
    if dev: return "devanagari"
    if lat: return "latin"
    return "other"
df["script"] = df["text"].map(script)
df["has_hashtag"] = df["text"].str.contains("#")
df["has_url"] = df["text"].str.contains("http")
df["has_mention"] = df["text"].str.contains("@")
emoji_re = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
df["has_emoji"] = df["text"].map(lambda t: bool(emoji_re.search(t)))
df["ends_punct"] = df["text"].str.strip().str[-1:].isin(list("!?.…"))
for c in ["script", "has_hashtag", "has_url", "has_mention", "has_emoji", "ends_punct"]:
    print(f"--- sarcasm rate by {c}")
    print(df.groupby(c)["sarcasm_label"].agg(["mean", "count"]), "\n")

# TF-IDF baseline on deduplicated data
d = df.assign(norm=norm).drop_duplicates("norm")
tr, va = train_test_split(d, test_size=0.05, random_state=42, stratify=d["sarcasm_label"])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=500_000, sublinear_tf=True)
Xtr, Xva = vec.fit_transform(tr["text"]), vec.transform(va["text"])
clf = LinearSVC(C=0.5).fit(Xtr, tr["sarcasm_label"])
pred = clf.predict(Xva)
print("TF-IDF baseline acc:", accuracy_score(va["sarcasm_label"], pred))
print(classification_report(va["sarcasm_label"], pred, digits=4))

names = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
print("top NOT-sarcastic features:", names[order[:40]].tolist())
print("top SARCASTIC features:   ", names[order[-40:]].tolist())