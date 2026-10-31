"""
adversarial_test.py
-------------------
Tests the model's robustness against text obfuscation attacks:
  1. Leetspeak substitution   (hate → h4t3)
  2. Emoji injection          (hate → hate 💀)
  3. Character repetition     (hate → haaate)
  4. Deliberate misspellings  (hate → haet)
  5. Unicode homoglyphs       (a → а  Cyrillic lookalike)
  6. Whitespace insertion     (hate → h a t e)
  7. Mixed case               (hate → HaTe)

For each attack, the script reports:
  - How many samples changed prediction (evasion rate)
  - Which labels are most vulnerable
"""

# --- UTF-8 console guard (Windows cp1252 would crash on the ✓/→/█ chars) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


import torch
import random
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
ADV_DIR    = BASE_DIR / "adversarial"
ADV_DIR.mkdir(exist_ok=True)

MODEL_PATH  = MODELS_DIR / "toxic_roberta"
MAX_LENGTH  = 128
LABEL_NAMES = ["clean", "offensive", "hate_speech", "cyberbullying"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
random.seed(42)
np.random.seed(42)


# ── Attack Functions ───────────────────────────────────────────────────────

LEET_MAP = {
    "a": "4", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "g": "9", "b": "8",
}

HOMOGLYPH_MAP = {
    "a": "а",  # Cyrillic а
    "e": "е",  # Cyrillic е
    "o": "о",  # Cyrillic о
    "p": "р",  # Cyrillic р
    "c": "с",  # Cyrillic с
}

TOXICITY_EMOJIS = ["💀", "🖕", "😡", "🤬", "👊", "🔥"]


def attack_leetspeak(text: str) -> str:
    """Replace vowels and common chars with numbers."""
    result = []
    for ch in text:
        result.append(LEET_MAP.get(ch.lower(), ch))
    return "".join(result)


def attack_emoji_injection(text: str) -> str:
    """Inject a random toxic emoji into the text."""
    emoji = random.choice(TOXICITY_EMOJIS)
    words = text.split()
    if not words:
        return text
    pos = random.randint(0, len(words))
    words.insert(pos, emoji)
    return " ".join(words)


def attack_char_repetition(text: str, repeat: int = 3) -> str:
    """Randomly repeat characters in some words."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 2 and random.random() < 0.4:
            idx = random.randint(0, len(word) - 1)
            word = word[:idx] + word[idx] * repeat + word[idx + 1:]
        result.append(word)
    return " ".join(result)


def attack_misspelling(text: str) -> str:
    """Swap two adjacent characters in some words."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < 0.4:
            idx = random.randint(0, len(word) - 2)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
        result.append(word)
    return " ".join(result)


def attack_homoglyphs(text: str) -> str:
    """Replace Latin chars with Cyrillic lookalikes."""
    result = []
    for ch in text:
        result.append(HOMOGLYPH_MAP.get(ch.lower(), ch))
    return "".join(result)


def attack_whitespace(text: str) -> str:
    """Insert spaces between characters of some words."""
    words = text.split()
    result = []
    for word in words:
        if len(word) > 3 and random.random() < 0.3:
            word = " ".join(word)
        result.append(word)
    return " ".join(result)


def attack_mixed_case(text: str) -> str:
    """Randomly alternate character case."""
    return "".join(
        ch.upper() if i % 2 == 0 else ch.lower()
        for i, ch in enumerate(text)
    )


ATTACKS = {
    "leetspeak":       attack_leetspeak,
    "emoji_injection": attack_emoji_injection,
    "char_repetition": attack_char_repetition,
    "misspelling":     attack_misspelling,
    "homoglyphs":      attack_homoglyphs,
    "whitespace":      attack_whitespace,
    "mixed_case":      attack_mixed_case,
}


# ── Inference ──────────────────────────────────────────────────────────────

def predict_batch(texts: list[str], model, tokenizer) -> np.ndarray:
    """Returns predicted class indices for a list of texts."""
    encodings = tokenizer(
        texts, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt"
    )
    input_ids      = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return preds


# ── Adversarial Evaluation ─────────────────────────────────────────────────

def run_adversarial_tests(
    test_texts: list[str],
    original_preds: np.ndarray,
    model,
    tokenizer,
    batch_size: int = 32,
) -> dict:
    """
    For each attack, compute:
      - evasion_rate: % of samples where prediction changed
      - per_label_evasion: evasion rate broken down by original label
    """
    results = {}

    for attack_name, attack_fn in ATTACKS.items():
        print(f"  Running attack: {attack_name}...")

        # Apply attack
        attacked_texts = [attack_fn(t) for t in test_texts]

        # Predict in batches
        adv_preds = []
        for i in range(0, len(attacked_texts), batch_size):
            batch = attacked_texts[i: i + batch_size]
            adv_preds.extend(predict_batch(batch, model, tokenizer))
        adv_preds = np.array(adv_preds)

        # Evasion = prediction changed
        changed      = (adv_preds != original_preds)
        evasion_rate = changed.mean()

        # Per-label breakdown
        per_label = {}
        for label_id, label_name in enumerate(LABEL_NAMES):
            mask = (original_preds == label_id)
            if mask.sum() > 0:
                per_label[label_name] = round(float(changed[mask].mean()), 4)

        results[attack_name] = {
            "evasion_rate":    round(float(evasion_rate), 4),
            "samples_changed": int(changed.sum()),
            "total_samples":   len(test_texts),
            "per_label":       per_label,
        }

        print(
            f"    Evasion rate: {evasion_rate:.2%} "
            f"({changed.sum()}/{len(test_texts)} samples changed)"
        )

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[Device] {DEVICE}")

    # Load a sample of test data
    test_path = BASE_DIR / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("test.csv not found. Run preprocess.py first.")

    test_df = pd.read_csv(test_path).sample(n=min(500, len(pd.read_csv(test_path))), random_state=42)
    print(f"[Data] Running adversarial tests on {len(test_df)} samples")

    # Load model
    print(f"[Model] Loading from {MODEL_PATH}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)

    # Get original predictions
    texts          = test_df["text"].tolist()
    original_preds = []
    for i in range(0, len(texts), 32):
        original_preds.extend(predict_batch(texts[i:i+32], model, tokenizer))
    original_preds = np.array(original_preds)

    print(f"\n[Adversarial Tests]")
    print("-" * 50)
    results = run_adversarial_tests(texts, original_preds, model, tokenizer)

    # Summary
    print("\n" + "=" * 50)
    print("ADVERSARIAL ROBUSTNESS SUMMARY")
    print("=" * 50)
    for attack, data in results.items():
        print(f"  {attack:<20}: evasion rate = {data['evasion_rate']:.2%}")

    overall_evasion = np.mean([v["evasion_rate"] for v in results.values()])
    robustness_score = round((1 - overall_evasion) * 100, 1)
    print(f"\n  Overall robustness score: {robustness_score}/100")
    print(f"  (Higher = more robust against adversarial attacks)")

    results["overall_robustness_score"] = robustness_score

    # Save results
    out_path = ADV_DIR / "adversarial_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved → {out_path}")


if __name__ == "__main__":
    main()
