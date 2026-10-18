import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))

def resolve_file(name):
    candidates = [
        os.path.join(project_root, "datasets", name),
        os.path.join(project_root, name),
        os.path.join(base_dir, name),
        name
    ]
    return next((p for p in candidates if os.path.exists(p)), name)

orig = pd.read_csv(resolve_file('sarcasm_500k.csv'))
clean = pd.read_csv(resolve_file('500k_proper_data.csv'))

# Normalize to merge
orig['norm'] = orig['text'].astype(str).str.strip().str.lower()
clean['norm'] = clean['text'].astype(str).str.strip().str.lower()

merged = pd.merge(orig, clean, on='norm', suffixes=('_orig', '_new'))
diff_0_to_1 = merged[(merged['sarcasm_label_orig'] == 0) & (merged['sarcasm_label_new'] == 1)]
diff_1_to_0 = merged[(merged['sarcasm_label_orig'] == 1) & (merged['sarcasm_label_new'] == 0)]

print(f"Total matched rows: {len(merged):,}")
print(f"Changed from 0 (Non-Sarcastic) -> 1 (Sarcastic): {len(diff_0_to_1):,}")
print(f"Changed from 1 (Sarcastic) -> 0 (Non-Sarcastic): {len(diff_1_to_0):,}")

print("\n=== SAMPLES CHANGED FROM 0 (Non-Sarcastic) -> 1 (Sarcastic) ===")
for i, r in diff_0_to_1.head(10).iterrows():
    print(f"  Orig: {r.sarcasm_label_orig} -> New: {r.sarcasm_label_new} | \"{r.text_orig}\"")

print("\n=== SAMPLES CHANGED FROM 1 (Sarcastic) -> 0 (Non-Sarcastic) ===")
for i, r in diff_1_to_0.head(10).iterrows():
    print(f"  Orig: {r.sarcasm_label_orig} -> New: {r.sarcasm_label_new} | \"{r.text_orig}\"")
