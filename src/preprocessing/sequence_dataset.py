'''
🧩 FILE — sequence_dataset.py
(PyTorch Dataset for Temporal FDIA Detection)

Responsibility
- Load precomputed temporal sequences
- Return tensors suitable for CNN / Transformer / BiLSTM
- Support train / val / test splits

Conceptually
This file is:
📦 The bridge between preprocessing and model training
'''

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List


class FDIASequenceDataset(Dataset):
    """
    PyTorch Dataset for FDIA temporal sequences.
    """

    def __init__(self, sequence_dir: str | Path):
        self.sequence_dir = Path(sequence_dir)
        self.sequence_files = sorted(self.sequence_dir.glob("seq_*.npy"))

        if len(self.sequence_files) == 0:
            raise RuntimeError(f"No sequences found in {self.sequence_dir}")

    def __len__(self) -> int:
        return len(self.sequence_files)

    def __getitem__(self, idx: int):
        data = np.load(self.sequence_files[idx], allow_pickle=True).item()

        x = torch.tensor(data["x"], dtype=torch.float32)  # (T, F)
        y = torch.tensor(data["y"], dtype=torch.long)     # scalar label

        return x, y
