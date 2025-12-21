'''
🧩 FILE — load_shift_attack.py
(Load Shift FDIA Attack)

Responsibility
- Manipulate load-related measurements (P, Q)
- Preserve relative consistency to appear realistic
- Simulate attacker shifting demand between buses

Conceptually
This file is:
⚖️ A realistic FDIA that alters load patterns without random corruption
'''

from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np

from .base_attack import BaseAttack, AttackMetadata


class LoadShiftFDIAttack(BaseAttack):
    """
    Load Shift False Data Injection Attack.

    This attacker:
    - Selects a subset of buses
    - Increases load at some buses
    - Decreases load at others
    - Net power change ≈ 0 (appears realistic)
    """

    def __init__(
        self,
        target_buses: Optional[List[int]] = None,
        shift_fraction: float = 0.1
    ):
        """
        Parameters
        ----------
        target_buses : list[int], optional
            Buses whose load measurements will be attacked.
            If None, buses are chosen randomly.
        shift_fraction : float
            Fraction of load to shift (e.g., 0.1 = 10%)
        """
        super().__init__(name="load_shift_fdia")
        self.target_buses = target_buses
        self.shift_fraction = shift_fraction

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

        n_bus = p.size

        # Choose target buses
        if self.target_buses is None:
            n_targets = max(2, n_bus // 10)
            buses = rng.choice(n_bus, size=n_targets, replace=False)
        else:
            buses = np.array(self.target_buses, dtype=int)

        # Split buses into increase / decrease groups
        half = len(buses) // 2
        inc_buses = buses[:half]
        dec_buses = buses[half:]

        # Compute total shift magnitude
        total_p_shift = 0.0
        total_q_shift = 0.0

        # Increase loads
        for b in inc_buses:
            delta_p = abs(p[b]) * self.shift_fraction
            delta_q = abs(q[b]) * self.shift_fraction
            p[b] += delta_p
            q[b] += delta_q
            total_p_shift += delta_p
            total_q_shift += delta_q

        # Decrease loads to balance
        for b in dec_buses:
            delta_p = total_p_shift / len(dec_buses)
            delta_q = total_q_shift / len(dec_buses)
            p[b] -= delta_p
            q[b] -= delta_q

        metadata = AttackMetadata(
            attack_name=self.name,
            attacked=True,
            target_buses=buses.tolist(),
            strength=self.shift_fraction,
            extra={
                "num_target_buses": len(buses),
                "balanced": True
            }
        )

        return attacked, metadata


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(1)
    dummy = {
        "v_mag": rng.uniform(0.95, 1.05, size=30),
        "v_ang": rng.uniform(-10, 10, size=30),
        "p_inj": rng.normal(1.0, 0.2, size=30),
        "q_inj": rng.normal(0.5, 0.1, size=30),
        "line_p": rng.normal(0, 1, size=41),
        "line_q": rng.normal(0, 1, size=41),
    }

    attack = LoadShiftFDIAttack(shift_fraction=0.1)
    attacked, meta = attack.apply(dummy)

    print("Load-shift FDIA applied.")
    print(meta)
