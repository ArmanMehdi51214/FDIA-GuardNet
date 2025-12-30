'''
🧩 FILE — dataset_builder.py
(Train / Validation / Test Split Builder)

Responsibility
- Index raw dataset (.npy + .json)
- Build train / val / test splits
- Preserve class imbalance (stratified split)
- Save split indices for downstream preprocessing

Conceptually
This file is:
🗂️ The dataset indexer that separates learning from evaluation
'''

from __future__ import annotations

import os
import json
import random
from typing import List, Dict
from pathlib import Path


RAW_CLEAN_DIR = Path("data/raw/clean")
RAW_ATTACKED_DIR = Path("data/raw/attacked")
META_DIR = Path("data/metadata")

PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = PROCESSED_DIR / "splits"


def ensure_dirs():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata() -> List[Dict]:
    """
    Load all metadata JSON files and return a list of records.
    Each record contains: sample_id, label, attack type.
    """
    records = []

    for meta_file in META_DIR.glob("meta_*.json"):
        with open(meta_file, "r") as f:
            meta = json.load(f)

        records.append({
            "sample_id": meta["sample_id"],
            "label": meta["label"],
            "attack": meta.get("attack", "none")
        })

    return records


def stratified_split(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Perform stratified split by label (normal vs attacked).
    """
    random.seed(seed)

    normal = [r for r in records if r["label"] == 0]
    attacked = [r for r in records if r["label"] == 1]

    def split_group(group):
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = group[:n_train]
        val = group[n_train:n_train + n_val]
        test = group[n_train + n_val:]

        return train, val, test

    n_train, n_val, n_test = split_group(normal)
    a_train, a_val, a_test = split_group(attacked)

    train = n_train + a_train
    val = n_val + a_val
    test = n_test + a_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_split(name: str, records: List[Dict]):
    """
    Save split sample IDs to disk.
    """
    out_file = SPLIT_DIR / f"{name}_ids.txt"
    with open(out_file, "w") as f:
        for r in records:
            f.write(f"{r['sample_id']}\n")


def main():
    ensure_dirs()

    records = load_metadata()
    print(f"Total samples indexed: {len(records)}")

    train, val, test = stratified_split(records)

    save_split("train", train)
    save_split("val", val)
    save_split("test", test)

    print("Dataset split completed:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")

    # Sanity check
    total = len(train) + len(val) + len(test)
    assert total == len(records), "Split size mismatch!"


if __name__ == "__main__":
    main()
