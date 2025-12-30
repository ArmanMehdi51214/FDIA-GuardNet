'''
🧩 FILE — windowing.py
(Temporal Windowing / Sequence Builder)

Responsibility
- Convert normalized flat samples into temporal sequences
- Use sliding windows over sample indices
- Assign labels correctly at sequence level
- Write sequence tensors for train / val / test

Conceptually
This file is:
⏱️ The temporal context builder required for LSTM & Transformer models
'''

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List
import json


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROCESSED_DIR = Path("data/processed")

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

META_DIR = Path("data/metadata")

SEQ_DIR = PROCESSED_DIR / "sequences"
TRAIN_SEQ_DIR = SEQ_DIR / "train"
VAL_SEQ_DIR = SEQ_DIR / "val"
TEST_SEQ_DIR = SEQ_DIR / "test"


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

WINDOW_SIZE = 10     # timesteps per sequence
STRIDE = 1           # sliding step


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def ensure_dirs():
    for d in [TRAIN_SEQ_DIR, VAL_SEQ_DIR, TEST_SEQ_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_metadata_label(sample_id: int) -> int:
    meta_file = META_DIR / f"meta_{sample_id}.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    return meta["label"]


def load_sorted_samples(split_dir: Path) -> List[int]:
    """
    Return sorted sample IDs present in a processed split directory.
    """
    ids = []
    for f in split_dir.glob("x_*.npy"):
        sid = int(f.stem.split("_")[1])
        ids.append(sid)
    return sorted(ids)


def load_sample(split_dir: Path, sample_id: int) -> np.ndarray:
    return np.load(split_dir / f"x_{sample_id}.npy")


# ---------------------------------------------------------
# Windowing logic
# ---------------------------------------------------------

def build_sequences(
    split_name: str,
    split_dir: Path,
    out_dir: Path
):
    """
    Build temporal sequences using sliding windows.
    """
    ids = load_sorted_samples(split_dir)
    n = len(ids)

    print(f"\nBuilding sequences for {split_name}: {n} samples")

    seq_count = 0

    for i in range(0, n - WINDOW_SIZE + 1, STRIDE):
        window_ids = ids[i:i + WINDOW_SIZE]

        # Load window data
        seq = np.stack(
            [load_sample(split_dir, sid) for sid in window_ids],
            axis=0
        )

        # Label = label of last timestep
        label = load_metadata_label(window_ids[-1])

        out = {
            "x": seq,
            "y": label
        }

        np.save(out_dir / f"seq_{seq_count}.npy", out)
        seq_count += 1

    print(f"Generated {seq_count} sequences for {split_name}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ensure_dirs()

    build_sequences("train", TRAIN_DIR, TRAIN_SEQ_DIR)
    build_sequences("val", VAL_DIR, VAL_SEQ_DIR)
    build_sequences("test", TEST_DIR, TEST_SEQ_DIR)

    print("\nTemporal windowing completed successfully.")
    print("Sequence data written to data/processed/sequences/")


if __name__ == "__main__":
    main()
