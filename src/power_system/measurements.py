'''
🧩 FILE 3 — measurements.py
(Measurement Extraction / Sensor Layer)

Responsibility
- Convert true power-flow results into sensor measurements
- Extract:
  - Bus voltage magnitudes
  - Bus voltage angles
  - Active & reactive power injections
  - Line power flows

Conceptually
This file is:
📡 What the control center "sees" before noise and attacks
'''

from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandapower as pp


def extract_measurements(
    net: pp.pandapowerNet,
    power_flow_results: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Extract clean (noise-free, attack-free) measurements from power flow results.

    Parameters
    ----------
    net : pandapowerNet
        Network after successful AC power flow.
    power_flow_results : dict
        Output dictionary from run_power_flow().

    Returns
    -------
    measurements : dict
        Clean sensor measurements:
        - v_mag : bus voltage magnitudes (p.u.)
        - v_ang : bus voltage angles (deg)
        - p_inj : active power injections (MW)
        - q_inj : reactive power injections (MVAr)
        - line_p : line active power flows (MW)
        - line_q : line reactive power flows (MVAr)

    Notes
    -----
    - These are IDEAL measurements (no noise, no FDIA)
    - Noise and attacks are applied in later phases
    """

    measurements = {
        "v_mag": power_flow_results["vm_pu"].copy(),
        "v_ang": power_flow_results["va_deg"].copy(),
        "p_inj": power_flow_results["p_inj_mw"].copy(),
        "q_inj": power_flow_results["q_inj_mvar"].copy(),
        "line_p": power_flow_results["line_p_mw"].copy(),
        "line_q": power_flow_results["line_q_mvar"].copy(),
    }

    return measurements


def flatten_measurements(measurements: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten all measurement arrays into a single 1D feature vector.

    This is the representation used later for ML models.

    Order is FIXED and must never change after this point.

    Returns
    -------
    feature_vector : np.ndarray
        Shape: (num_features,)
    """
    feature_vector = np.concatenate([
        measurements["v_mag"],
        measurements["v_ang"],
        measurements["p_inj"],
        measurements["q_inj"],
        measurements["line_p"],
        measurements["line_q"],
    ])

    return feature_vector


def sanity_check_measurements(measurements: Dict[str, np.ndarray]) -> None:
    """
    Basic validation to ensure extracted measurements are sane.
    """
    for key, arr in measurements.items():
        assert not np.isnan(arr).any(), f"NaN detected in {key}"
        assert np.isfinite(arr).all(), f"Infinite values detected in {key}"


if __name__ == "__main__":
    # Smoke test: IEEE 30-bus → power flow → measurements
    from .ieee30 import load_ieee30
    from .power_flow import run_power_flow

    net = load_ieee30()
    pf_results = run_power_flow(net)
    meas = extract_measurements(net, pf_results)

    sanity_check_measurements(meas)

    vec = flatten_measurements(meas)

    print("Measurement extraction successful.")
    print(f"Total measurement dimension: {vec.shape[0]}")
