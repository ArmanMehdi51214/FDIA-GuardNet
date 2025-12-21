'''
🧩 FILE — stealthy_attack.py
(Stealthy False Data Injection Attack)

Responsibility
- Inject carefully crafted false data
- Preserve internal measurement consistency
- Evade traditional bad-data detection mechanisms

Conceptually
This file is:
🕵️ A smart attacker who understands the power system
'''

from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np

from .base_attack import BaseAttack, AttackMetadata


class StealthyFDIAttack(BaseAttack):
    """
    Stealthy False Data Injection Attack.

    This attacker:
    - Targets a subset of buses
    - Injects correlated changes into P and Q
    - Maintains relative structure so attacks look plausible
    """

    def __init__(
        self,
        target_buses: Optional[List[int]] = None,
        attack_strength: float = 0.05
    ):
        """
        Parameters
        ----------
        target_buses : list[int], optional
            Buses to attack. Randomly chosen if None.
        attack_strength : float
            Relative perturbation magnitude (e.g., 0.05 = 5%)
        """
        super().__init__(name="stealthy_fdia")
        self.target_buses = target_buses
        self.attack_strength = attack_strength

    def apply(
        self,
        measurements: Dict[str, np.ndarray],
        rng: Optional[np.random.Generator] = None
    ):
        if rng is None:
            rng = np.random.default_rng()

        attacked = self._copy_measurements(measurements)

        p = attacked["p_inj"]
        q = attacked["q_inj"]
        v = attacked["v_mag"]

        n_bus = p.size

        # Choose target buses
        if self.target_buses is None:
            n_targets = max(1, n_bus // 8)
            buses = rng.choice(n_bus, size=n_targets, replace=False)
        else:
            buses = np.array(self.target_buses, dtype=int)

        # Apply correlated stealthy perturbations
        for b in buses:
            scale = 1.0 + rng.uniform(
                -self.attack_strength,
                self.attack_strength
            )

            # Correlated changes (key idea)
            p[b] *= scale
            q[b] *= scale
            v[b] *= (1.0 - 0.5 * (scale - 1.0))

        metadata = AttackMetadata(
            attack_name=self.name,
            attacked=True,
            target_buses=buses.tolist(),
            strength=self.attack_strength,
            extra={
                "num_target_buses": len(buses),
                "correlated": True,
                "stealthy": True
            }
        )

        return attacked, metadata


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(2)
    dummy = {
        "v_mag": rng.uniform(0.95, 1.05, size=30),
        "v_ang": rng.uniform(-10, 10, size=30),
        "p_inj": rng.normal(1.0, 0.2, size=30),
        "q_inj": rng.normal(0.5, 0.1, size=30),
        "line_p": rng.normal(0, 1, size=41),
        "line_q": rng.normal(0, 1, size=41),
    }

    attack = StealthyFDIAttack(attack_strength=0.05)
    attacked, meta = attack.apply(dummy)

    print("Stealthy FDIA applied.")
    print(meta)
