'''
🧩 FILE — normalization.py
(Feature Normalization Pipeline)

Responsibility
- Compute feature-wise normalization statistics using TRAIN data only
- Apply normalization consistently to train / val / test splits
- Write normalized samples into data/processed/
- Save normalization parameters for reproducibility and inference

Conceptually
This file is:
🧪 The data standardization step that makes ML training stable and valid
'''

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import List


RAW_CLEAN_DIR = Path("data/raw/clean")
RAW_ATTACKED_DIR = Path("data/raw/attacked")

PROCESSED_DIR = Path("data/processed")
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

STATS_DIR = PROCESSED_DIR / "stats"
SPLIT_DIR = PROCESSED_DIR / "splits"


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def ensure_dirs():
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, STATS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_ids(split: str) -> List[int]:
    ids_file = SPLIT_DIR / f"{split}_ids.txt"
    with open(ids_file, "r") as f:
        return [int(line.strip()) for line in f]


def load_raw_sample(sample_id: int) -> np.ndarray:
    """
    Load raw .npy sample by ID from clean or attacked folder.
    """
    clean_path = RAW_CLEAN_DIR / f"x_{sample_id}.npy"
    attacked_path = RAW_ATTACKED_DIR / f"x_{sample_id}.npy"

    if clean_path.exists():
        return np.load(clean_path)
    elif attacked_path.exists():
        return np.load(attacked_path)
    else:
        raise FileNotFoundError(f"Sample x_{sample_id}.npy not found.")


# ---------------------------------------------------------
# Normalization logic
# ---------------------------------------------------------

def compute_train_statistics(train_ids: List[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute feature-wise mean and std using TRAIN samples only.
    """
    features = []

    for sid in train_ids:
        x = load_raw_sample(sid)
        features.append(x)

    X = np.stack(features, axis=0)

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Numerical stability
    std[std < 1e-8] = 1.0

    return mean, std


def normalize_and_save(
    split: str,
    ids: List[int],
    mean: np.ndarray,
    std: np.ndarray
):
    """
    Normalize samples and write them to processed/{split}/
    """
    out_dir = PROCESSED_DIR / split

    for sid in ids:
        x = load_raw_sample(sid)
        x_norm = (x - mean) / std

        np.save(out_dir / f"x_{sid}.npy", x_norm)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ensure_dirs()

    print("Loading dataset splits...")
    train_ids = load_ids("train")
    val_ids = load_ids("val")
    test_ids = load_ids("test")

    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples:   {len(val_ids)}")
    print(f"Test samples:  {len(test_ids)}")

    print("\nComputing normalization statistics (TRAIN only)...")
    mean, std = compute_train_statistics(train_ids)

    np.save(STATS_DIR / "mean.npy", mean)
    np.save(STATS_DIR / "std.npy", std)

    print("Saved normalization statistics.")

    print("\nNormalizing TRAIN split...")
    normalize_and_save("train", train_ids, mean, std)

    print("Normalizing VAL split...")
    normalize_and_save("val", val_ids, mean, std)

    print("Normalizing TEST split...")
    normalize_and_save("test", test_ids, mean, std)

    print("\nNormalization completed successfully.")
    print("Processed data written to data/processed/")


if __name__ == "__main__":
    main()
