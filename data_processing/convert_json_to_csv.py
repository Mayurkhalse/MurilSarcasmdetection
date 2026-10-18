import json
import os
import pandas as pd

def convert_json_to_csv(json_path: str, csv_path: str):
    print(f"Reading JSON dataset from: {json_path}")
    records = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                headline = data.get("headline", "")
                label = data.get("is_sarcastic", None)
                if label is not None:
                    records.append({
                        "text": headline,
                        "sarcasm_label": int(label)
                    })
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    df = pd.DataFrame(records)
    print(f"Total valid records parsed: {len(df)}")
    print("Class distribution:")
    print(df["sarcasm_label"].value_counts())
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Successfully saved converted CSV to: {csv_path}")
    print(f"First 5 rows:\n{df.head()}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    datasets_dir = os.path.join(project_root, "datasets")
    
    input_json = os.path.join(datasets_dir, "Sarcasm_Headlines_Dataset.json")
    if not os.path.exists(input_json):
        input_json = os.path.join(base_dir, "Sarcasm_Headlines_Dataset.json")
        
    output_csv = os.path.join(datasets_dir, "sarcasm_headlines_dataset.csv")
    convert_json_to_csv(input_json, output_csv)
