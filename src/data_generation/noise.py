'''
🧩 FILE 4 — noise.py
(Measurement Noise Modeling)

Responsibility
- Add realistic sensor noise to clean measurements
- Simulate imperfect SCADA / PMU readings
- Ensure realism without breaking physics

Conceptually
This file is:
🎚️ The realism layer between sensors and attackers
'''

from __future__ import annotations

from typing import Dict
import numpy as np


class GaussianNoiseModel:
    """
    Gaussian noise model for power system measurements.

    Notes
    -----
    - Noise is zero-mean Gaussian
    - Standard deviation is configurable per measurement type
    - Noise is applied independently to each measurement
    """

    def __init__(
        self,
        v_mag_std: float = 0.005,     # p.u.
        v_ang_std: float = 0.2,       # degrees
        p_inj_std: float = 0.01,      # MW
        q_inj_std: float = 0.01,      # MVAr
        line_p_std: float = 0.01,     # MW
        line_q_std: float = 0.01      # MVAr
    ):
        self.std_map = {
            "v_mag": v_mag_std,
            "v_ang": v_ang_std,
            "p_inj": p_inj_std,
            "q_inj": q_inj_std,
            "line_p": line_p_std,
            "line_q": line_q_std,
        }

    def apply(self, measurements: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply Gaussian noise to clean measurements.

        Parameters
        ----------
        measurements : dict
            Clean measurement dictionary from measurements.py

        Returns
        -------
        noisy_measurements : dict
            Measurements after noise injection
        """
        noisy_measurements = {}

        for key, values in measurements.items():
            std = self.std_map.get(key, 0.0)
            noise = np.random.normal(loc=0.0, scale=std, size=values.shape)
            noisy_measurements[key] = values + noise

        return noisy_measurements


def sanity_check_noise(
    clean: Dict[str, np.ndarray],
    noisy: Dict[str, np.ndarray]
) -> None:
    """
    Sanity checks to ensure noise behaves correctly.
    """
    for key in clean.keys():
        assert clean[key].shape == noisy[key].shape, f"Shape mismatch in {key}"
        assert not np.isnan(noisy[key]).any(), f"NaN detected after noise in {key}"


if __name__ == "__main__":
    # Smoke test: apply noise to dummy measurements
    rng = np.random.default_rng(42)

    dummy_measurements = {
        "v_mag": rng.uniform(0.95, 1.05, size=30),
        "v_ang": rng.uniform(-10, 10, size=30),
        "p_inj": rng.normal(0, 1, size=30),
        "q_inj": rng.normal(0, 1, size=30),
        "line_p": rng.normal(0, 1, size=41),
        "line_q": rng.normal(0, 1, size=41),
    }

    noise_model = GaussianNoiseModel()
    noisy = noise_model.apply(dummy_measurements)
    sanity_check_noise(dummy_measurements, noisy)

    print("Noise model applied successfully.")
