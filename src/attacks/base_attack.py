'''
🧩 FILE — base_attack.py
(FDIA Attack Base Class)

Responsibility
- Define a common interface for all FDIA attacks
- Ensure each attack can be applied consistently to measurement dictionaries
- Store attack metadata for dataset labeling and analysis

Conceptually
This file is:
🧠 The attacker "API" that all attack types must follow
'''

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class AttackMetadata:
    """
    Metadata returned alongside attacked measurements for traceability.
    """
    attack_name: str
    attacked: bool
    target_buses: Optional[list[int]] = None
    target_lines: Optional[list[int]] = None
    strength: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


class BaseAttack(ABC):
    """
    Abstract base class for FDIA attacks.

    All attacks must implement:
    - apply(measurements) -> (attacked_measurements, metadata)

    Where 'measurements' is a dict of numpy arrays:
      {
        "v_mag":  (n_bus,),
        "v_ang":  (n_bus,),
        "p_inj":  (n_bus,),
        "q_inj":  (n_bus,),
        "line_p": (n_line,),
        "line_q": (n_line,)
      }
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(
        self,
        measurements: Dict[str, np.ndarray],
        rng: Optional[np.random.Generator] = None
    ) -> tuple[Dict[str, np.ndarray], AttackMetadata]:
        """
        Apply the attack to measurements.

        Parameters
        ----------
        measurements : dict
            Noisy (or clean) measurements dictionary.
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        attacked_measurements : dict
            New measurements dict after attack injection.
        metadata : AttackMetadata
            Metadata describing what was attacked.
        """
        raise NotImplementedError

    def _copy_measurements(self, measurements: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Safe deep-ish copy for numpy arrays to avoid in-place modification.
        """
        return {k: v.copy() for k, v in measurements.items()}


if __name__ == "__main__":
    print("BaseAttack interface loaded successfully.")
