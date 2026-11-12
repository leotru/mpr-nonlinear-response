"""Loading of TOML configuration files."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from .theory import MPRTheory

__all__ = ["load_config", "theory_from_config", "network_grid"]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def theory_from_config(cfg: dict) -> MPRTheory:
    m = cfg["model"]
    return MPRTheory(m["eta_bar"], m["J"], m["Delta"])


def network_grid(cfg: dict) -> tuple[float, int]:
    """Time step dt and number of steps per measurement window.

    The window length is chosen so that all stimulus frequencies are integer
    multiples of 2 pi / T, which makes the Fourier estimates leakage-free.
    """
    n = cfg["network"]
    T = n["window_over_pi"] * np.pi
    steps = int(n["steps_per_window"])
    return T / steps, steps
