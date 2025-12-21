'''
🧩 FILE — multi_bus_attack.py
(Coordinated Multi-Bus FDIA Attack)

Responsibility
- Perform coordinated false data injection across multiple buses
- Apply structured, correlated perturbations
- Simulate advanced attackers targeting grid-wide consistency

Conceptually
This file is:
🕸️ A coordinated attacker manipulating multiple grid locations together
'''

from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np

from .base_attack import BaseAttack, AttackMetadata


class MultiBusFDIAttack(BaseAttack):
    """
    Coordinated Multi-Bus False Data Injection Attack.

    This attacker:
    - Selects multiple buses across the grid
    - Applies correlated perturbations to P, Q, and V
    - Mimics coordinated cyber-physical attack behavior
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
            Relative perturbation magnitude.
        """
        super().__init__(name="multi_bus_fdia")
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

        # Choose target buses (distributed across grid)
        if self.target_buses is None:
            n_targets = max(3, n_bus // 6)
            buses = rng.choice(n_bus, size=n_targets, replace=False)
        else:
            buses = np.array(self.target_buses, dtype=int)

        # Global coordination factor
        global_scale = 1.0 + rng.uniform(
            -self.attack_strength,
            self.attack_strength
        )

        for b in buses:
            local_scale = global_scale * rng.uniform(0.95, 1.05)

            p[b] *= local_scale
            q[b] *= local_scale
            v[b] *= (1.0 - 0.4 * (local_scale - 1.0))

        metadata = AttackMetadata(
            attack_name=self.name,
            attacked=True,
            target_buses=buses.tolist(),
            strength=self.attack_strength,
            extra={
                "num_target_buses": len(buses),
                "coordinated": True
            }
        )

        return attacked, metadata


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(3)
    dummy = {
        "v_mag": rng.uniform(0.95, 1.05, size=30),
        "v_ang": rng.uniform(-10, 10, size=30),
        "p_inj": rng.normal(1.0, 0.2, size=30),
        "q_inj": rng.normal(0.5, 0.1, size=30),
        "line_p": rng.normal(0, 1, size=41),
        "line_q": rng.normal(0, 1, size=41),
    }

    attack = MultiBusFDIAttack(attack_strength=0.05)
    attacked, meta = attack.apply(dummy)

    print("Multi-bus FDIA applied.")
    print(meta)
