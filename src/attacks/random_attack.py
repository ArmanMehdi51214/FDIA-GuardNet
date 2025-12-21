'''
🧩 FILE — random_attack.py
(Random FDIA Attack)

Responsibility
- Apply simple random false data injection
- Corrupt a random subset of measurements
- Serve as a baseline (non-stealthy) FDIA

Conceptually
This file is:
🎲 A naive attacker that randomly tampers with sensor data
'''

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from .base_attack import BaseAttack, AttackMetadata


class RandomFDIAttack(BaseAttack):
    """
    Random False Data Injection Attack.

    This attacker:
    - Randomly selects a subset of measurements
    - Adds random perturbations
    - Does NOT preserve power-system physics
    """

    def __init__(
        self,
        attack_strength: float = 0.05,
        attack_ratio: float = 0.1
    ):
        """
        Parameters
        ----------
        attack_strength : float
            Maximum relative perturbation (e.g., 0.05 = ±5%)
        attack_ratio : float
            Fraction of measurement entries to attack
        """
        super().__init__(name="random_fdia")
        self.attack_strength = attack_strength
        self.attack_ratio = attack_ratio

    def apply(
        self,
        measurements: Dict[str, np.ndarray],
        rng: Optional[np.random.Generator] = None
    ):
        if rng is None:
            rng = np.random.default_rng()

        attacked = self._copy_measurements(measurements)

        # Flatten keys and indices for random selection
        all_keys = list(attacked.keys())
        total_entries = sum(attacked[k].size for k in all_keys)

        n_attack = int(self.attack_ratio * total_entries)

        # Create a flat index mapping
        flat_map = []
        for key in all_keys:
            for idx in range(attacked[key].size):
                flat_map.append((key, idx))

        attack_indices = rng.choice(
            len(flat_map),
            size=n_attack,
            replace=False
        )

        for i in attack_indices:
            key, idx = flat_map[i]
            value = attacked[key].flat[idx]

            perturbation = value * rng.uniform(
                -self.attack_strength,
                self.attack_strength
            )
            attacked[key].flat[idx] += perturbation

        metadata = AttackMetadata(
            attack_name=self.name,
            attacked=True,
            strength=self.attack_strength,
            extra={
                "attack_ratio": self.attack_ratio,
                "num_corrupted_entries": n_attack
            }
        )

        return attacked, metadata


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    dummy = {
        "v_mag": rng.uniform(0.95, 1.05, size=30),
        "v_ang": rng.uniform(-10, 10, size=30),
        "p_inj": rng.normal(0, 1, size=30),
        "q_inj": rng.normal(0, 1, size=30),
        "line_p": rng.normal(0, 1, size=41),
        "line_q": rng.normal(0, 1, size=41),
    }

    attack = RandomFDIAttack()
    attacked, meta = attack.apply(dummy)

    print("Random FDIA applied.")
    print(meta)
