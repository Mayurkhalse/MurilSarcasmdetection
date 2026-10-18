"""
Refinement Pipeline for Sarcasm 500k Dataset
Cleans text noise, filters uninformative fragments, and corrects mislabeled samples.
Outputs: 500k_proper_data.csv (text,sarcasm_label)
"""

import os
import sys
import re
import html
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

# UTF-8 terminal encoding fix for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def clean_text_surface(text: str) -> str:
    """Performs deep cleaning of HTML entities, whitespace, and Reddit artifacts."""
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
    t = html.unescape(text)
    
    # 2. Remove zero-width spaces, non-breaking spaces, control characters
    t = re.sub(r"[\u200b\u200c\u200d\uFEFF\xa0]", " ", t)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
    
    # 3. Strip leading quote markers from Reddit markdown (> or &gt;)
    t = re.sub(r"^[>\s]+", "", t)
    
    # 4. Clean extra or unbalanced quotes
    t = re.sub(r'^["\']+|["\']+$', "", t)
    t = t.replace('""', '"')
    
    # 5. Collapse repeated whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_junk_text(t: str) -> bool:
    """Filters out bot triggers, pure URLs, short fragments, and uninformative noise."""
    if len(t) < 11:
        return True
    
    words = t.split()
    if len(words) < 3 or len(words) > 75:
        return True
        
    # Check for bot commands or deleted/removed posts
    lower_t = t.lower()
    if lower_t.startswith("!remindme") or lower_t.startswith("remindme!"):
        return True
    if lower_t in ("[deleted]", "[removed]"):
        return True
    if lower_t.startswith("http://") or lower_t.startswith("https://") or lower_t.startswith("www."):
        return True
        
    # Check if string has almost no alphanumeric letters (e.g. pure symbols/emojis)
    alpha_chars = sum(1 for c in t if c.isalnum())
    if alpha_chars < 6:
        return True
        
    # Broken single-word corrections
    if t.startswith("*") and len(words) <= 2:
        return True
        
    return False


def build_linguistic_rules():
    """Builds regex patterns for obvious false positives and false negatives."""
    # Sarcastic cues that should almost never be labeled 0
    sarc_cue_patterns = [
        r"^(oh\s+(yeah|sure|great|brilliant|wonderful|fantastic)|because\s+(obviously|clearly)|yeah\s+(right|because))\b",
        r"\b(totally\s+makes\s+sense|what\s+a\s+(great|brilliant|genius|wonderful)\s+idea)\b",
        r"\b(thanks\s+for\s+nothing|because\s+who\s+needs\s+.*\s+anyway|sooo\s+horrible)\b",
        r"\b(as\s+if\s+that\s+(helps|matters|works)|clearly\s+the\s+best\s+solution)\b",
        r"\b(color\s+me\s+surprised|no\s+shit\s+sherlock|what\s+a\s+shocker)\b"
    ]
    re_sarc_cues = re.compile("|".join(sarc_cue_patterns), re.IGNORECASE)

    # Obvious non-sarcastic false positives (labeled 1 by mistake in SARC)
    non_sarc_patterns = [
        # Questions with no sarcastic cues
        r"^(what|where|when|who|which|whose|why|how|is|are|was|were|do|does|did|can|could|would|should|has|have|had)\b.*\?$",
        # Factual acknowledgments or simple courtesies
        r"^(thanks|thank you)\s+(so much|for sharing|for the link|for the help|for the advice)[\.!]*$",
        r"^(great|awesome|nice|good)\s+(job|work|cover|pic|picture|post|catch|point)[\.!]*$",
        r"^(sounds|looks)\s+(great|good|awesome|nice|interesting|cool)[\.!]*$",
        # Plain complaints / vulgar frustration without irony
        r"^[a-z0-9_\-\s]{1,25}\s+sucks(\s+balls|\s+ass|\s+dick)?[\.!]*$",
        r"^(i\s+hate\s+this|this\s+is\s+awful|this\s+is\s+terrible|i\s+don't\s+like\s+this)[\.!]*$"
    ]
    re_non_sarc_cues = re.compile("|".join(non_sarc_patterns), re.IGNORECASE)

    return re_sarc_cues, re_non_sarc_cues


def refine_dataset(input_csv: str = "sarcasm_500k.csv", output_csv: str = "500k_proper_data.csv"):
    t0_start = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    def resolve_dataset_file(filename):
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        candidates = [
            filename,
            os.path.join(project_root, "datasets", filename),
            os.path.join(project_root, "datasets", os.path.basename(filename)),
            os.path.join(project_root, filename),
            os.path.join(base_dir, filename)
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return os.path.join(project_root, "datasets", os.path.basename(filename))

    input_path = resolve_dataset_file(input_csv)
    print("=" * 75, flush=True)
    print("            STARTING REFINEMENT PIPELINE FOR SARCASM 500K", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Original Dataset
    print(f"\n[1/5] Loading original dataset from '{input_path}'...", flush=True)
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"      Loaded {original_count:,} rows.", flush=True)
    print(f"      Original class distribution:\n{df['sarcasm_label'].value_counts()}", flush=True)

    # 2. Text Surface Cleaning
    print("\n[2/5] Cleaning text surfaces, unescaping HTML, fixing quotes & whitespace...", flush=True)
    df["clean_text"] = df["text"].astype(str).map(clean_text_surface)
    
    # 3. Noise Filtering
    print("      Filtering short fragments, bot triggers, URLs, and uninformative noise...", flush=True)
    is_junk = df["clean_text"].map(is_junk_text)
    junk_count = is_junk.sum()
    df = df[~is_junk].copy()
    print(f"      Removed {junk_count:,} junk/low-context rows ({junk_count/original_count*100:.2f}%).", flush=True)
    print(f"      Remaining rows after noise filter: {len(df):,}", flush=True)

    # Deduplication
    norm_text = df["clean_text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    dup_mask = norm_text.duplicated(keep="first")
    dup_count = dup_mask.sum()
    df = df[~dup_mask].copy()
    print(f"      Removed {dup_count:,} duplicate / near-duplicate rows.", flush=True)
    print(f"      Unique quality candidate rows: {len(df):,}", flush=True)

    # 4. Linguistic Heuristics & Machine Learning Noise Detection
    print("\n[3/5] Calibrating labels using linguistic rules and cross-validated out-of-fold predictions...", flush=True)
    re_sarc_cues, re_non_sarc_cues = build_linguistic_rules()

    # Fast TF-IDF representation
    t_tfidf = time.time()
    print("      Fitting TF-IDF vectorizer...", flush=True)
    vec = TfidfVectorizer(max_features=60000, min_df=5, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(df["clean_text"])
    y = df["sarcasm_label"].values
    print(f"      TF-IDF feature matrix ready: {X.shape} in {time.time() - t_tfidf:.1f}s.", flush=True)

    # 3-Fold Stratified Out-of-fold calibration
    t_cv = time.time()
    print("      Computing out-of-fold probabilities with calibrated linear model...", flush=True)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=25, random_state=42 + fold)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_probs[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    print(f"      Out-of-fold probabilities computed in {time.time() - t_cv:.1f}s.", flush=True)

    # 5. Label Refinement Decisions
    print("\n[4/5] Refining ground truth labels...", flush=True)
    refined_labels = y.copy()
    relabeled_1_to_0 = 0
    relabeled_0_to_1 = 0
    pruned_ambiguous = 0
    keep_mask = np.ones(len(df), dtype=bool)

    texts = df["clean_text"].tolist()
    for idx, (t, orig_label, prob) in enumerate(zip(texts, y, oof_probs)):
        has_sarc_cue = bool(re_sarc_cues.search(t))
        has_non_sarc_cue = bool(re_non_sarc_cues.match(t))

        # Case A: False Sarcastic (Original label 1, but clearly non-sarcastic)
        # Conditions: Rule matches non-sarcastic structure OR (model prob < 0.12 and no sarcastic cues)
        if orig_label == 1:
            if has_non_sarc_cue or (prob < 0.12 and not has_sarc_cue):
                refined_labels[idx] = 0
                relabeled_1_to_0 += 1

        # Case B: False Non-Sarcastic (Original label 0, but clearly sarcastic)
        # Conditions: Obvious sarcasm rule matched OR (model prob > 0.88 and has sarcastic cues)
        elif orig_label == 0:
            if has_sarc_cue or (prob > 0.88 and has_sarc_cue):
                refined_labels[idx] = 1
                relabeled_0_to_1 += 1

    df["sarcasm_label"] = refined_labels
    print(f"      [✓] Corrected False Sarcastic (1 -> 0): {relabeled_1_to_0:,} rows.")
    print(f"      [✓] Corrected False Non-Sarcastic (0 -> 1): {relabeled_0_to_1:,} rows.")
    print(f"      Total labels refined/corrected: {relabeled_1_to_0 + relabeled_0_to_1:,} rows.", flush=True)

    # Integrate verified benchmark headlines (if available) for extra domain diversity
    headlines_csv = resolve_dataset_file("sarcasm_headlines_dataset.csv")
    if os.path.exists(headlines_csv):
        print(f"\n      Augmenting with verified high-quality headlines from '{headlines_csv}'...", flush=True)
        hdf = pd.read_csv(headlines_csv)
        hdf["clean_text"] = hdf["text"].map(clean_text_surface)
        hdf = hdf[~hdf["clean_text"].map(is_junk_text)]
        print(f"      Added {len(hdf):,} clean verified headline records.", flush=True)
        df_combined = pd.concat([
            df[["clean_text", "sarcasm_label"]].rename(columns={"clean_text": "text"}),
            hdf[["clean_text", "sarcasm_label"]].rename(columns={"clean_text": "text"})
        ], ignore_index=True)
    else:
        df_combined = df[["clean_text", "sarcasm_label"]].rename(columns={"clean_text": "text"})

    # Final Deduplication & Clean Up
    df_combined = df_combined.drop_duplicates(subset=["text"]).dropna()
    df_combined["sarcasm_label"] = df_combined["sarcasm_label"].astype(int)

    # Balanced Sampling to exactly 500,000 samples (or maximum balanced if slightly less)
    print("\n[5/5] Re-balancing and finalizing dataset...", flush=True)
    count_0 = (df_combined["sarcasm_label"] == 0).sum()
    count_1 = (df_combined["sarcasm_label"] == 1).sum()
    print(f"      Available cleaned samples -> Not Sarcastic (0): {count_0:,} | Sarcastic (1): {count_1:,}", flush=True)

    target_total = 500000
    target_per_class = target_total // 2

    df_class_0 = df_combined[df_combined["sarcasm_label"] == 0]
    df_class_1 = df_combined[df_combined["sarcasm_label"] == 1]

    # Sample equally up to available or target
    n_sample_0 = min(len(df_class_0), target_per_class)
    n_sample_1 = min(len(df_class_1), target_per_class)
    
    # If one class has more, use remaining to reach 500,000 total cleanly
    rem = target_total - (n_sample_0 + n_sample_1)
    if rem > 0:
        if len(df_class_0) > n_sample_0:
            n_sample_0 += min(rem, len(df_class_0) - n_sample_0)
        elif len(df_class_1) > n_sample_1:
            n_sample_1 += min(rem, len(df_class_1) - n_sample_1)

    df_final = pd.concat([
        df_class_0.sample(n=n_sample_0, random_state=42),
        df_class_1.sample(n=n_sample_1, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save to CSV
    output_path = resolve_dataset_file(output_csv)
    df_final[["text", "sarcasm_label"]].to_csv(output_path, index=False, encoding="utf-8")
    
    total_time = time.time() - t0_start
    print("\n" + "=" * 75, flush=True)
    print("                    REFINEMENT COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 75, flush=True)
    print(f"  Output File:           {output_path}", flush=True)
    print(f"  Total Clean Samples:   {len(df_final):,}", flush=True)
    print(f"  Not Sarcastic (0):     {(df_final['sarcasm_label'] == 0).sum():,} ({((df_final['sarcasm_label'] == 0).sum()/len(df_final)*100):.1f}%)", flush=True)
    print(f"  Sarcastic (1):         {(df_final['sarcasm_label'] == 1).sum():,} ({((df_final['sarcasm_label'] == 1).sum()/len(df_final)*100):.1f}%)", flush=True)
    print(f"  Total Pipeline Time:   {total_time:.1f} s ({total_time/60:.2f} mins)", flush=True)
    print("=" * 75, flush=True)

    print("\nSample Rows from 500k_proper_data.csv (First 10):", flush=True)
    for i, r in df_final.head(10).iterrows():
        tag = "🎭 Sarcastic (1)" if r.sarcasm_label == 1 else "✅ Not Sarcastic (0)"
        print(f"  [{tag}] {r.text}", flush=True)

    return output_path


if __name__ == "__main__":
    refine_dataset()
