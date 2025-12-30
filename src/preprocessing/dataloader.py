'''
🧩 FILE — dataloader.py
(PyTorch DataLoader Factory)

Responsibility
- Create DataLoader objects for train / val / test
- Centralize batching, shuffling, and performance flags

Conceptually
This file is:
🚚 The data pipeline feeding the neural network
'''

from __future__ import annotations

from torch.utils.data import DataLoader
from pathlib import Path

from .sequence_dataset import FDIASequenceDataset


def create_dataloader(
    sequence_dir: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    dataset = FDIASequenceDataset(sequence_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
