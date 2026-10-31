"""
dataset_loader.py
-----------------
Loads, merges, and normalises multiple toxic content datasets into a
unified format with consistent labels:
    0 = clean
    1 = offensive
    2 = hate_speech
    3 = cyberbullying
"""

# --- UTF-8 console guard (Windows cp1252 would crash on the ✓/→/█ chars) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


import os
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "clean":        0,
    "offensive":    1,
    "hate_speech":  2,
    "cyberbullying": 3,
}


# ── Loaders ────────────────────────────────────────────────────────────────

def load_davidson(path: Path) -> pd.DataFrame:
    """
    Davidson et al. (2017) — Hate Speech and Offensive Language dataset.
    Expected columns: tweet, class
        class 0 = hate speech
        class 1 = offensive language
        class 2 = neither
    Download: https://github.com/t-davidson/hate-speech-and-offensive-language
    """
    df = pd.read_csv(path)
    df = df[["tweet", "class"]].rename(columns={"tweet": "text"})

    label_map = {0: "hate_speech", 1: "offensive", 2: "clean"}
    df["label"] = df["class"].map(label_map)
    df = df[["text", "label"]].dropna()
    print(f"[Davidson] Loaded {len(df)} samples")
    return df


def load_kaggle_cyberbullying(path: Path) -> pd.DataFrame:
    """
    Kaggle Cyberbullying Classification dataset.
    Expected columns: tweet_text, cyberbullying_type
    Download: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
    """
    df = pd.read_csv(path)
    df = df[["tweet_text", "cyberbullying_type"]].rename(
        columns={"tweet_text": "text", "cyberbullying_type": "label"}
    )

    # Map Kaggle labels to unified labels
    label_map = {
        "not_cyberbullying": "clean",
        "age":               "cyberbullying",
        "ethnicity":         "hate_speech",
        "gender":            "hate_speech",
        "religion":          "hate_speech",
        "other_cyberbullying": "cyberbullying",
    }
    df["label"] = df["label"].map(label_map)
    df = df[["text", "label"]].dropna()
    print(f"[Kaggle Cyberbullying] Loaded {len(df)} samples")
    return df


def load_hateval(path: Path) -> pd.DataFrame:
    """
    HatEval SemEval-2019 Task 5 dataset.
    Expected columns: text, HS (hate speech binary flag)
    Download: https://github.com/cicl2018/HEval
    """
    df = pd.read_csv(path)
    df = df[["text", "HS"]].dropna()
    df["label"] = df["HS"].apply(lambda x: "hate_speech" if int(x) == 1 else "clean")
    df = df[["text", "label"]]
    print(f"[HatEval] Loaded {len(df)} samples")
    return df


# ── Merge & Save ───────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """
    Looks for known dataset files in data/raw/ and merges them.
    Falls back to a small synthetic sample if no files are found,
    so the rest of the pipeline can still be tested.
    """
    frames = []

    davidson_path = RAW_DIR / "labeled_data.csv"
    if davidson_path.exists():
        frames.append(load_davidson(davidson_path))

    kaggle_path = RAW_DIR / "cyberbullying_tweets.csv"
    if kaggle_path.exists():
        frames.append(load_kaggle_cyberbullying(kaggle_path))

    hateval_path = RAW_DIR / "hateval2019_en_train.csv"
    if hateval_path.exists():
        frames.append(load_hateval(hateval_path))

    using_real_data = bool(frames)
    if not frames:
        print("[WARNING] No dataset files found in data/raw/. Using synthetic sample.")
        frames.append(_synthetic_sample())

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL_MAP.keys())]

    # Top-up for missing classes: Davidson alone has no cyberbullying examples,
    # so the 4th label would be empty. Fill any absent class from the synthetic
    # sample so the pipeline stays 4-class as designed. This is clearly marked as
    # supplementary — swap in the real Kaggle cyberbullying set for true results.
    if using_real_data:
        present = set(df["label"].unique())
        missing = set(LABEL_MAP.keys()) - present
        if missing:
            synth = _synthetic_sample()
            synth = synth[synth["label"].isin(missing)]
            if len(synth):
                print(f"[top-up] classes absent from real data: {sorted(missing)} "
                      f"-> added {len(synth)} synthetic placeholder rows")
                df = pd.concat([df, synth], ignore_index=True)

    df["label_id"] = df["label"].map(LABEL_MAP)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = PROCESSED_DIR / "dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[✓] Combined dataset saved → {out_path}")
    print(f"    Total samples : {len(df)}")
    print(f"    Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def _synthetic_sample() -> pd.DataFrame:
    """Small synthetic dataset for pipeline testing without real data."""
    samples = [
        ("I love this community, everyone is so supportive!", "clean"),
        ("This is absolutely terrible work.", "offensive"),
        ("I hate people from that religion, they are all evil.", "hate_speech"),
        ("You are such a worthless loser, go kill yourself.", "cyberbullying"),
        ("Great game today, looking forward to the next one.", "clean"),
        ("What an idiot, can't believe you said that.", "offensive"),
        ("All immigrants should be deported immediately.", "hate_speech"),
        ("Nobody likes you, you should just disappear.", "cyberbullying"),
        ("The weather is nice today.", "clean"),
        ("This movie was complete garbage.", "offensive"),
        ("People of that race are inferior.", "hate_speech"),
        ("I will make your life miserable every single day.", "cyberbullying"),
    ] * 50  # Repeat for a slightly larger sample
    return pd.DataFrame(samples, columns=["text", "label"])


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_dataset()
    print("\nSample rows:")
    print(df.sample(5).to_string(index=False))
