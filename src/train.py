"""
train.py
--------
Fine-tunes roberta-base for 4-class toxic content classification.
Labels: 0=clean, 1=offensive, 2=hate_speech, 3=cyberbullying

Usage:
    python src/train.py
"""

# --- UTF-8 console guard (Windows cp1252 would crash on the ✓/→/█ chars) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
import json

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR    = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "roberta-base"
NUM_LABELS  = 4
# These four accept env-var overrides so a quick CPU demo run is possible without
# editing the committed defaults. e.g. TOXIC_EPOCHS=1 TOXIC_MAX_LENGTH=64 python src/train.py
MAX_LENGTH  = int(os.getenv("TOXIC_MAX_LENGTH", "128"))
BATCH_SIZE  = int(os.getenv("TOXIC_BATCH_SIZE", "16"))
EPOCHS      = int(os.getenv("TOXIC_EPOCHS", "3"))
MAX_TRAIN   = int(os.getenv("TOXIC_MAX_TRAIN", "0"))   # 0 = use all training rows
LR          = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SAVE_PATH   = MODELS_DIR / "toxic_roberta"

LABEL_NAMES = ["clean", "offensive", "hate_speech", "cyberbullying"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────

class ToxicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: RobertaTokenizer, max_length: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training ───────────────────────────────────────────────────────────────

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse frequency weighting to handle class imbalance.

    Reindexed over the full 0..NUM_LABELS-1 range so a dataset missing a class
    (e.g. Davidson has no cyberbullying) still yields a length-NUM_LABELS weight
    vector; absent classes get count 1 to avoid division by zero."""
    counts = (
        df["label_id"].value_counts()
        .reindex(range(NUM_LABELS), fill_value=0)
        .sort_index()
        .values
    )
    counts = np.where(counts == 0, 1, counts)   # no inf for absent classes
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_LABELS
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"[Device] Using: {DEVICE}")

    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")

    # Optional stratified subsample for a fast CPU demo (TOXIC_MAX_TRAIN).
    if MAX_TRAIN and len(train_df) > MAX_TRAIN:
        frac = MAX_TRAIN / len(train_df)
        # GroupBy.sample keeps all columns (unlike groupby().apply() in pandas 3,
        # which drops the grouping key). Stratified: same frac from each class.
        train_df = (
            train_df.groupby("label_id", group_keys=False)
            .sample(frac=frac, random_state=42)
            .reset_index(drop=True)
        )
        print(f"[Data] TOXIC_MAX_TRAIN={MAX_TRAIN} -> subsampled train to {len(train_df)} rows")

    print(f"[Data] Train: {len(train_df)} | Val: {len(val_df)}")

    # Tokenizer & model
    print(f"[Model] Loading {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model     = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)

    # Datasets & loaders
    train_dataset = ToxicDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = ToxicDataset(val_df,   tokenizer, MAX_LENGTH)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # Class-weighted loss
    class_weights = compute_class_weights(train_df)
    criterion     = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0.0
    history = []

    print(f"\n[Training] {EPOCHS} epochs | Batch size: {BATCH_SIZE} | LR: {LR}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion
        )
        val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_f1":   round(train_f1,   4),
            "val_loss":   round(val_loss,   4),
            "val_f1":     round(val_f1,     4),
            "val_acc":    round(val_acc,    4),
        })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(SAVE_PATH)
            tokenizer.save_pretrained(SAVE_PATH)
            print(f"  → Best model saved (Val F1: {best_val_f1:.4f})")

    # Save training history
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[✓] Training complete. Best Val F1: {best_val_f1:.4f}")
    print(f"[✓] Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
