import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fix Windows console UTF-8 encoding for Hindi / special characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "Mayurkhalse/muril-sarcasm-detector"

print("=" * 65)
print(f"Loading model & tokenizer from Hugging Face:")
print(f"  Repo: {MODEL_ID}")
print("=" * 65)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# Check id2label in config or fallback to standard mapping
id2label = getattr(model.config, "id2label", None)
if not id2label or id2label.get(0) in ("LABEL_0", 0, None):
    id2label = {0: "Not Sarcastic", 1: "Sarcastic"}

print("[✓] Model successfully loaded!\n")


def predict(text: str):
    """Predict whether a given text is sarcastic or not."""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    prob_not_sarcastic = float(probs[0])
    prob_sarcastic = float(probs[1])

    is_sarcastic = prob_sarcastic > prob_not_sarcastic
    confidence = max(prob_sarcastic, prob_not_sarcastic) * 100
    label = id2label.get(1 if is_sarcastic else 0, "Sarcastic" if is_sarcastic else "Not Sarcastic")

    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "prob_sarcastic": prob_sarcastic,
        "prob_not_sarcastic": prob_not_sarcastic
    }


def run_benchmark():
    test_cases = [
        # English
        ("Oh brilliant! Another meeting that could have been an email.", "Sarcastic (English)"),
        ("The engineering team resolved the database latency issue on time.", "Non-Sarcastic (English)"),
        
        # Hinglish
        ("Arey wah! 3 ghante late aakar bolte ho 'I am on time'!", "Sarcastic (Hinglish)"),
        ("Kal subah 10 baje office me team meeting start hogi.", "Non-Sarcastic (Hinglish)"),
        
        # Hindi
        ("वाह भाई! पूरी रात रील्स देखकर कहते हैं कि पढ़ाई का समय नहीं मिला।", "Sarcastic (Hindi)"),
        ("आज मौसम बहुत अच्छा है और हल्की ठंडी हवा चल रही है।", "Non-Sarcastic (Hindi)"),
        
        # Additional edge cases
        ("I just love standing in traffic for 2 hours every day.", "Sarcastic (English)"),
        ("Please submit the quarterly performance report by Friday.", "Non-Sarcastic (English)"),
    ]

    print("-" * 75)
    print(f"{'Input Sentence':<50} | {'Expected':<12} | {'Prediction':<14} | {'Conf'}")
    print("-" * 75)

    for text, expected in test_cases:
        res = predict(text)
        tag = "🎭" if "Sarcastic" in res["label"] and "Not" not in res["label"] else "✅"
        short_text = text if len(text) <= 47 else text[:44] + "..."
        print(f"{short_text:<50} | {expected:<12} | {tag} {res['label']:<11} | {res['confidence']:.1f}%")

    print("-" * 75)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User passed a custom sentence via CLI
        custom_text = " ".join(sys.argv[1:])
        res = predict(custom_text)
        print(f"\nText: \"{res['text']}\"")
        print(f"Prediction: {res['label']} ({res['confidence']:.2f}% confidence)")
        print(f"Probabilities -> Sarcastic: {res['prob_sarcastic']:.4f}, Not Sarcastic: {res['prob_not_sarcastic']:.4f}\n")
    else:
        # Run standard benchmark suite
        run_benchmark()
