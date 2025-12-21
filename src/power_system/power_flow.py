'''
🧩 FILE 2 — power_flow.py
(AC Power Flow Solver)

Responsibility
- Run AC power flow on IEEE 30-bus system
- Compute true electrical state:
  - Voltage magnitudes
  - Voltage angles
  - Active power injections
  - Reactive power injections
  - Line power flows

Conceptually
This file is:
⚡ The physics engine of the grid
'''

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandapower as pp


def run_power_flow(net: pp.pandapowerNet, enforce_convergence: bool = True) -> Dict[str, Any]:
    """
    Run AC power flow on a given pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        IEEE 30-bus network (already loaded and configured).
    enforce_convergence : bool
        If True, raise an error when power flow does not converge.

    Returns
    -------
    results : dict
        Dictionary containing true system state:
        - vm_pu : Voltage magnitudes at buses (p.u.)
        - va_deg : Voltage angles at buses (degrees)
        - p_inj_mw : Active power injections at buses (MW)
        - q_inj_mvar : Reactive power injections at buses (MVAr)
        - line_p_mw : Active power flow on lines (MW)
        - line_q_mvar : Reactive power flow on lines (MVAr)

    Notes
    -----
    - This function represents physical ground truth.
    - No noise, no attacks, no measurements here.
    """
    # Run AC power flow
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)

    # Check convergence
    if enforce_convergence and not net.converged:
        raise RuntimeError("AC power flow did not converge for the given network.")

    # -------------------------
    # Bus-level true states
    # -------------------------
    vm_pu = net.res_bus["vm_pu"].values.copy()
    va_deg = net.res_bus["va_degree"].values.copy()

    # Power injections at buses
    # pandapower convention: positive = injected into grid
    p_inj_mw = net.res_bus["p_mw"].values.copy()
    q_inj_mvar = net.res_bus["q_mvar"].values.copy()

    # -------------------------
    # Line-level true states
    # -------------------------
    # From-bus perspective
    line_p_mw = net.res_line["p_from_mw"].values.copy()
    line_q_mvar = net.res_line["q_from_mvar"].values.copy()

    results = {
        "vm_pu": vm_pu,
        "va_deg": va_deg,
        "p_inj_mw": p_inj_mw,
        "q_inj_mvar": q_inj_mvar,
        "line_p_mw": line_p_mw,
        "line_q_mvar": line_q_mvar,
    }

    return results


def sanity_check_results(results: Dict[str, Any]) -> None:
    """
    Perform basic sanity checks on power flow results.

    Raises
    ------
    AssertionError if any check fails.
    """
    vm = results["vm_pu"]
    va = results["va_deg"]

    assert not np.isnan(vm).any(), "NaN detected in voltage magnitudes."
    assert not np.isnan(va).any(), "NaN detected in voltage angles."

    # Typical operating voltage bounds (loose, for sanity only)
    assert vm.min() > 0.8, "Voltage magnitude too low — possible divergence."
    assert vm.max() < 1.2, "Voltage magnitude too high — possible divergence."


if __name__ == "__main__":
    # Minimal smoke test using IEEE 30-bus system
    from src.power_system.ieee30 import load_ieee30

    net = load_ieee30()
    results = run_power_flow(net)

    sanity_check_results(results)

    print("AC power flow successful.")
    print(f"Voltage magnitude range: {results['vm_pu'].min():.3f} – {results['vm_pu'].max():.3f} p.u.")
