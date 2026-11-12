"""Spectral estimators for simulated firing-rate time series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["coefficient", "noise_floor", "window_mean_error"]


def coefficient(x: np.ndarray, t: np.ndarray, W: float) -> complex:
    """Complex amplitude c(W) = <x(t) exp(i W t)> on a uniform grid.

    Exact (no leakage) when W is a multiple of 2 pi / T for the window length T.
    """
    return complex(np.mean(x * np.exp(1j * W * t)))


def noise_floor(x: np.ndarray, t: np.ndarray, W: float, dW: float,
                exclude: Sequence[float], kmin: int = 3, kmax: int = 12) -> float:
    """RMS amplitude in neighbouring frequency bins W + k dW, kmin <= |k| <= kmax.

    Bins closer than 2.5 dW to any frequency in ``exclude`` are skipped.
    """
    Ws = [W + k * dW for k in range(-kmax, kmax + 1) if abs(k) >= kmin]
    Ws = [w for w in Ws if w > 0 and all(abs(w - e) > 2.5 * dW for e in exclude)]
    return float(np.sqrt(np.mean([abs(coefficient(x, t, w)) ** 2 for w in Ws])))


def window_mean_error(x: np.ndarray, n_window: int) -> float:
    """Standard error of the mean of x estimated from non-overlapping windows."""
    nw = x.size // n_window
    means = x[: nw * n_window].reshape(nw, n_window).mean(axis=1)
    return float(means.std() / np.sqrt(nw))
