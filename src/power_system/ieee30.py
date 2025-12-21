'''
🧩 FILE 1 — ieee30.py
(IEEE 30-bus system definition)

Responsibility
- Load IEEE 30-bus data
- Define:
  - Buses
  - Generators
  - Lines
  - Loads

Conceptually
This file is:
📐 The blueprint of the grid
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandapower as pp
import pandapower.networks as pn


@dataclass(frozen=True)
class IEEE30Config:
    """
    Configuration for loading and optionally modifying the IEEE 30-bus test system.

    Notes
    -----
    - In Milestone 1 Phase 1, we keep this minimal and deterministic.
    - Later milestones can extend this config (e.g., scenario scaling, randomization seeds).
    """
    f_hz: float = 50.0  # power system frequency (informational; pandapower uses it for some elements)
    name: str = "IEEE 30-bus"
    # Optional: scale all loads by a constant factor (useful for scenario generation later).
    load_scaling: float = 1.0
    # Optional: scale all generator setpoints by a constant factor (rarely needed initially).
    gen_scaling: float = 1.0


def load_ieee30(config: Optional[IEEE30Config] = None) -> pp.pandapowerNet:
    """
    Load the IEEE 30-bus test system as a pandapower network.

    Parameters
    ----------
    config : IEEE30Config, optional
        Controls name/frequency and optional global scaling.

    Returns
    -------
    net : pandapowerNet
        A ready-to-simulate pandapower network (AC power flow can be run on it).

    What this function guarantees (Phase 1)
    ---------------------------------------
    - Returns a valid IEEE 30-bus network
    - Applies optional deterministic scaling (if requested)
    - Does NOT run power flow (that's handled in power_flow.py)
    """
    cfg = config or IEEE30Config()

    # Load canonical IEEE 30-bus network provided by pandapower.
    # This includes buses, lines, transformers (if any), loads, generators, ext_grid, etc.
    net = pn.case_ieee30()

    # Basic metadata
    net.name = cfg.name
    net.f_hz = cfg.f_hz

    # Optional deterministic scaling (kept here so scenario generation can reuse it later)
    if cfg.load_scaling != 1.0 and len(net.load) > 0:
        net.load["p_mw"] *= float(cfg.load_scaling)
        if "q_mvar" in net.load.columns:
            net.load["q_mvar"] *= float(cfg.load_scaling)

    # Scale controllable generators (pandapower uses gen table for PV generators)
    if cfg.gen_scaling != 1.0 and len(net.gen) > 0:
        net.gen["p_mw"] *= float(cfg.gen_scaling)

    # NOTE:
    # - ext_grid represents the slack/source; we usually do not scale ext_grid here.
    # - In later scenario generation we can vary setpoints more intelligently.

    return net


def summarize_ieee30(net: pp.pandapowerNet) -> Dict[str, Any]:
    """
    Quick structural summary for sanity checks and logging.

    This is helpful in early development to ensure the network loads correctly.

    Returns a dict with counts of key elements.
    """
    summary = {
        "name": getattr(net, "name", None),
        "f_hz": getattr(net, "f_hz", None),
        "n_bus": len(net.bus),
        "n_line": len(net.line),
        "n_trafo": len(net.trafo) if hasattr(net, "trafo") else 0,
        "n_load": len(net.load),
        "n_gen": len(net.gen),
        "n_sgen": len(net.sgen),
        "n_ext_grid": len(net.ext_grid),
        "n_shunt": len(net.shunt),
        "n_switch": len(net.switch),
    }
    return summary


if __name__ == "__main__":
    # Minimal smoke test: load the network and print element counts.
    net_30 = load_ieee30()
    info = summarize_ieee30(net_30)
    print("IEEE 30-bus loaded successfully.")
    for k, v in info.items():
        print(f"{k}: {v}")
