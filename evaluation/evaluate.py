"""
evaluate.py
-----------
Evaluates the trained model on the test set.
Outputs:
  - F1 (macro, per-class)
  - Precision & recall (per-class)
  - Confusion matrix
  - Full classification report
"""

# --- UTF-8 console guard (Windows cp1252 would crash on the ✓/→/█ chars) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


import torch
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
EVAL_DIR      = BASE_DIR / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

MODEL_PATH  = MODELS_DIR / "toxic_roberta"
BATCH_SIZE  = 32
MAX_LENGTH  = 128
LABEL_NAMES = ["clean", "offensive", "hate_speech", "cyberbullying"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────

class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True,
            padding="max_length", max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Inference ──────────────────────────────────────────────────────────────

def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds   = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ── Reporting ──────────────────────────────────────────────────────────────

def print_report(y_true, y_pred):
    # labels=0..N-1 so the report stays stable even if a class is absent from
    # the test set (e.g. cyberbullying when only Davidson data is loaded).
    all_ids = list(range(len(LABEL_NAMES)))
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=all_ids,
                                target_names=LABEL_NAMES, zero_division=0))

    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":          round(f1_score(y_true, y_pred, labels=all_ids, average="macro",    zero_division=0), 4),
        "f1_weighted":       round(f1_score(y_true, y_pred, labels=all_ids, average="weighted", zero_division=0), 4),
        "precision_macro":   round(precision_score(y_true, y_pred, labels=all_ids, average="macro", zero_division=0), 4),
        "recall_macro":      round(recall_score(y_true, y_pred, labels=all_ids, average="macro",    zero_division=0), 4),
        "per_class_f1":      {
            name: round(score, 4)
            for name, score in zip(
                LABEL_NAMES,
                f1_score(y_true, y_pred, labels=all_ids, average=None, zero_division=0)
            )
        },
    }

    print("\nSummary Metrics:")
    for k, v in metrics.items():
        if k != "per_class_f1":
            print(f"  {k:<22}: {v}")
    print("\nPer-class F1:")
    for label, score in metrics["per_class_f1"].items():
        print(f"  {label:<22}: {score}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Confusion matrix saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[Device] {DEVICE}")

    # Load test data
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"[Data] Test samples: {len(test_df)}")

    # Load model
    print(f"[Model] Loading from {MODEL_PATH}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model     = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)

    # DataLoader
    dataset = ToxicDataset(test_df, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict
    print("[Evaluating]...")
    y_pred, y_true, y_probs = get_predictions(model, loader)

    # Report
    metrics = print_report(y_true, y_pred)

    # Save metrics
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved → {metrics_path}")

    # Confusion matrix plot
    plot_confusion_matrix(y_true, y_pred, EVAL_DIR / "confusion_matrix.png")

    # Save predictions
    test_df["predicted_label_id"] = y_pred
    test_df["predicted_label"]    = [LABEL_NAMES[p] for p in y_pred]
    for i, name in enumerate(LABEL_NAMES):
        test_df[f"prob_{name}"] = y_probs[:, i]
    test_df.to_csv(EVAL_DIR / "test_predictions.csv", index=False)
    print(f"[✓] Predictions saved → {EVAL_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
