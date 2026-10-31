"""
predict.py
----------
Standalone inference helper.
Use this to classify text from the command line or import into other scripts.

Usage:
    python src/predict.py "I hate everyone from that group"
"""

# --- UTF-8 console guard (Windows cp1252 would crash on the ✓/→/█ chars) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


import sys
import torch
import numpy as np
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification

sys.path.append(str(Path(__file__).parent))
from preprocess import clean_text

BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "toxic_roberta"
MAX_LENGTH  = 128
LABEL_NAMES = ["clean", "offensive", "hate_speech", "cyberbullying"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model) -> dict:
    cleaned  = clean_text(text)
    encoding = tokenizer(
        cleaned, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        "text":       text,
        "label":      LABEL_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "scores":     {name: round(float(probs[i]), 4)
                       for i, name in enumerate(LABEL_NAMES)},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<text to classify>\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer, model = load_model()
    result = classify(text, tokenizer, model)

    print(f"\nText       : {result['text']}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nAll scores:")
    for label, score in result["scores"].items():
        bar = "█" * int(score * 30)
        print(f"  {label:<16}: {score:.4f}  {bar}")
