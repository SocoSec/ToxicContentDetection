"""
preprocess.py
-------------
Text cleaning and tokenization pipeline.
Outputs train/val/test splits ready for model training.
"""

# --- UTF-8 console guard (Windows cp1252 would crash on the ✓/→/█ chars) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


import re
import html
import unicodedata
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Tokenizer ──────────────────────────────────────────────────────────────
TOKENIZER_NAME = "roberta-base"
MAX_LENGTH     = 128


# ── Text Cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw social media text while preserving semantic meaning.
    Steps:
        1. Decode HTML entities (&amp; → &)
        2. Normalize unicode (smart quotes, accented chars)
        3. Remove URLs
        4. Remove @mentions (keep context, remove handle)
        5. Remove # from hashtags (keep the word)
        6. Collapse repeated characters (haaaate → haate, max 2)
        7. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML entities
    text = html.unescape(text)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 4. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 5. Hashtags — keep word, remove #
    text = re.sub(r"#(\w+)", r"\1", text)

    # 6. Collapse repeated characters (keeps up to 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def batch_clean(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply clean_text to a DataFrame column."""
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    # Drop empty texts after cleaning
    df = df[df[text_col].str.len() > 3].reset_index(drop=True)
    return df


# ── Train/Val/Test Split ───────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    label_col:   str   = "label_id",
    random_state: int  = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / val / test sets.
    Default: 70% / 15% / 15%
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df[label_col], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=temp_df[label_col], random_state=random_state
    )

    print(f"[Split] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── Tokenization ───────────────────────────────────────────────────────────

def tokenize_df(
    df: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    text_col: str = "text",
    max_length: int = MAX_LENGTH,
) -> dict:
    """
    Tokenize a DataFrame using the RoBERTa tokenizer.
    Returns a dict with input_ids, attention_mask, labels.
    """
    encodings = tokenizer(
        df[text_col].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encodings["labels"] = list(df["label_id"])
    return encodings


# ── Main ───────────────────────────────────────────────────────────────────

def run_preprocessing():
    dataset_path = PROCESSED_DIR / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "dataset.csv not found. Run src/dataset_loader.py first."
        )

    print("[1/4] Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"      {len(df)} samples loaded")

    print("[2/4] Cleaning text...")
    df = batch_clean(df)
    print(f"      {len(df)} samples after cleaning")

    print("[3/4] Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df)

    print("[4/4] Saving splits...")
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR  / "val.csv",   index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv",  index=False)
    print(f"[✓] Splits saved to {PROCESSED_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_preprocessing()
