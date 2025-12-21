'''
🧩 FILE — run_simulation.py
(Dataset Orchestration & Generation)

Responsibility
- Orchestrate end-to-end simulation:
  IEEE grid → power flow → measurements → noise → FDIA (optional)
- Generate labeled datasets (clean vs attacked)
- Write outputs to data/raw and data/metadata

Conceptually
This file is:
🏭 The dataset factory that completes Milestone 1
'''

from __future__ import annotations

import os
import json
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


def ensure_dirs():
    for d in [DATA_RAW_CLEAN, DATA_RAW_ATTACKED, DATA_META]:
        os.makedirs(d, exist_ok=True)


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


def run_single_sample(sample_id: int, attack_name: Optional[str], rng: np.random.Generator):
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


def main(
    n_samples: int = 200,
    attack_ratio: float = 0.5,
    seed: int = 42
):
    ensure_dirs()
    rng = np.random.default_rng(seed)

    attacks = [None, "random", "load_shift", "stealthy", "multi_bus"]

    for i in range(n_samples):
        attack_name = None
        if rng.random() < attack_ratio:
            attack_name = rng.choice(attacks[1:])

        x, meta = run_single_sample(i, attack_name, rng)

        if meta["label"] == 0:
            out_x = os.path.join(DATA_RAW_CLEAN, f"x_{i}.npy")
        else:
            out_x = os.path.join(DATA_RAW_ATTACKED, f"x_{i}.npy")

        np.save(out_x, x)

        with open(os.path.join(DATA_META, f"meta_{i}.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Simulation complete. Generated {n_samples} samples.")


if __name__ == "__main__":
    main()
