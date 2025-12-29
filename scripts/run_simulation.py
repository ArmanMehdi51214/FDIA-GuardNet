'''
🧩 FILE — run_simulation.py
(Dataset Orchestration & Generation)

Responsibility
- Orchestrate end-to-end simulation:
  IEEE grid → power flow → measurements → noise → FDIA (optional)
- Generate labeled datasets (clean vs attacked)
- Support safe resume after interruption
- Write outputs to data/raw and data/metadata

Conceptually
This file is:
🏭 The dataset factory for Milestone 1 & large-scale data generation in Milestone 2
'''

from __future__ import annotations

import os
import json
import glob
import re
from typing import Optional
import numpy as np

from src.power_system.ieee30 import load_ieee30
from src.power_system.power_flow import run_power_flow
from src.power_system.measurements import extract_measurements, flatten_measurements
from src.data_generation.noise import GaussianNoiseModel
from src.attacks.random_attack import RandomFDIAttack
from src.attacks.load_shift_attack import LoadShiftFDIAttack
from src.attacks.stealthy_attack import StealthyFDIAttack
from src.attacks.multi_bus_attack import MultiBusFDIAttack


DATA_RAW_CLEAN = "data/raw/clean"
DATA_RAW_ATTACKED = "data/raw/attacked"
DATA_META = "data/metadata"


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def ensure_dirs():
    for d in [DATA_RAW_CLEAN, DATA_RAW_ATTACKED, DATA_META]:
        os.makedirs(d, exist_ok=True)


def get_last_sample_id() -> int:
    """
    Scan existing .npy files and return the highest sample_id found.
    Enables safe resume after interruption.
    """
    pattern = re.compile(r"x_(\d+)\.npy")
    ids = []

    for folder in [DATA_RAW_CLEAN, DATA_RAW_ATTACKED]:
        for path in glob.glob(os.path.join(folder, "x_*.npy")):
            match = pattern.search(path)
            if match:
                ids.append(int(match.group(1)))

    return max(ids) if ids else -1


def get_attack(name: Optional[str]):
    if name is None:
        return None
    if name == "random":
        return RandomFDIAttack()
    if name == "load_shift":
        return LoadShiftFDIAttack()
    if name == "stealthy":
        return StealthyFDIAttack()
    if name == "multi_bus":
        return MultiBusFDIAttack()
    raise ValueError(f"Unknown attack type: {name}")


# -------------------------------------------------------------------
# Core simulation logic
# -------------------------------------------------------------------

def run_single_sample(
    sample_id: int,
    attack_name: Optional[str],
    rng: np.random.Generator
):
    """
    Generate one dataset sample:
    grid → power flow → measurements → noise → optional FDIA
    """
    net = load_ieee30()
    pf = run_power_flow(net)

    meas = extract_measurements(net, pf)
    noise_model = GaussianNoiseModel()
    noisy = noise_model.apply(meas)

    metadata = {
        "sample_id": sample_id,
        "attack": attack_name if attack_name else "none",
        "label": 0
    }

    if attack_name:
        attack = get_attack(attack_name)
        noisy, attack_meta = attack.apply(noisy, rng=rng)
        metadata["label"] = 1
        metadata["attack_meta"] = attack_meta.__dict__

    x = flatten_measurements(noisy)
    return x, metadata


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main(
    total_samples: int = 30000,
    seed: int = 42
):
    """
    Large-scale dataset generation with controlled class ratios
    and resume support.
    """
    ensure_dirs()
    rng = np.random.default_rng(seed)

    # -------------------------------
    # Dataset composition (CONFIG)
    # -------------------------------
    normal_ratio = 0.65

    attack_distribution = {
        "random": 0.20,
        "load_shift": 0.30,
        "stealthy": 0.30,
        "multi_bus": 0.20
    }

    n_normal = int(total_samples * normal_ratio)
    n_attack = total_samples - n_normal

    attack_counts = {
        k: int(v * n_attack)
        for k, v in attack_distribution.items()
    }

    # Fix rounding issues
    while sum(attack_counts.values()) < n_attack:
        attack_counts["stealthy"] += 1

    print("Dataset composition:")
    print(f"  Normal samples: {n_normal}")
    for k, v in attack_counts.items():
        print(f"  {k} attack samples: {v}")

    # -------------------------------
    # Resume logic
    # -------------------------------
    last_id = get_last_sample_id()
    sample_id = last_id + 1
    print(f"\nResuming dataset generation from sample_id = {sample_id}")

    # -------------------------------
    # Generate NORMAL samples
    # -------------------------------
    while sample_id < n_normal:
        x, meta = run_single_sample(
            sample_id=sample_id,
            attack_name=None,
            rng=rng
        )

        np.save(os.path.join(DATA_RAW_CLEAN, f"x_{sample_id}.npy"), x)
        with open(os.path.join(DATA_META, f"meta_{sample_id}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        sample_id += 1

    # -------------------------------
    # Generate ATTACKED samples
    # -------------------------------
    attack_start_id = n_normal
    for attack_name, count in attack_counts.items():
        for _ in range(count):
            if sample_id < attack_start_id:
                sample_id = attack_start_id

            if sample_id >= total_samples:
                break

            x, meta = run_single_sample(
                sample_id=sample_id,
                attack_name=attack_name,
                rng=rng
            )

            np.save(os.path.join(DATA_RAW_ATTACKED, f"x_{sample_id}.npy"), x)
            with open(os.path.join(DATA_META, f"meta_{sample_id}.json"), "w") as f:
                json.dump(meta, f, indent=2)

            sample_id += 1

    print("\nDataset generation complete.")
    print(f"Total samples generated: {sample_id}")


if __name__ == "__main__":
    main()
