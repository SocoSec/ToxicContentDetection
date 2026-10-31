"""
app.py
------
Flask REST API demo for the Toxic Content Detection Platform.

Endpoints:
    POST /predict        — Classify a single text
    POST /predict/batch  — Classify multiple texts
    GET  /health         — Health check

Usage:
    python demo/app.py
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "I hate everyone from that country"}'
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
from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Add src to path for preprocess import
sys.path.append(str(Path(__file__).parent.parent / "src"))
from preprocess import clean_text

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "toxic_roberta"
MAX_LENGTH  = 128
LABEL_NAMES = ["clean", "offensive", "hate_speech", "cyberbullying"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# ── Load Model (once at startup) ───────────────────────────────────────────
print(f"[Demo] Loading model from {MODEL_PATH}...")
try:
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    print(f"[Demo] Model loaded on {DEVICE} ✓")
    MODEL_LOADED = True
except Exception as e:
    print(f"[Demo] WARNING: Could not load model — {e}")
    print("[Demo] Running in mock mode. Train the model first.")
    MODEL_LOADED = False
    tokenizer = None
    model     = None


# ── Inference Helper ───────────────────────────────────────────────────────

def predict_text(text: str) -> dict:
    """
    Run inference on a single cleaned text.
    Returns label probabilities and top prediction.
    """
    if not MODEL_LOADED:
        # Mock response for demo without trained model
        return {
            "text":        text,
            "predictions": {name: round(float(np.random.dirichlet([1]*4)[i]), 4)
                            for i, name in enumerate(LABEL_NAMES)},
            "label":       "mock_mode",
            "confidence":  0.0,
            "note":        "Model not loaded. Run train.py first.",
        }

    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_idx    = int(np.argmax(probs))
    pred_label  = LABEL_NAMES[pred_idx]
    confidence  = round(float(probs[pred_idx]), 4)

    return {
        "text":        text,
        "cleaned":     cleaned,
        "predictions": {name: round(float(probs[i]), 4)
                        for i, name in enumerate(LABEL_NAMES)},
        "label":       pred_label,
        "confidence":  confidence,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL_LOADED,
        "device":       str(DEVICE),
        "labels":       LABEL_NAMES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must include a 'text' field."}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "'text' field cannot be empty."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters."}), 400

    result = predict_text(text)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True, silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Request body must include a 'texts' array."}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty array."}), 400
    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 texts per batch."}), 400

    results = [predict_text(str(t).strip()) for t in texts]
    return jsonify({"results": results, "count": len(results)})


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[Demo] Toxic Content Detection API")
    print("[Demo] Endpoints:")
    print("         GET  /health")
    print("         POST /predict        — { \"text\": \"...\" }")
    print("         POST /predict/batch  — { \"texts\": [\"...\", \"...\"] }")
    print("[Demo] Starting on http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
