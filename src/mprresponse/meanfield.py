"""Numerical integration of the full (nonlinear) MPR equations under periodic input."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .theory import MPRTheory

__all__ = ["integrate_mpr", "fourier_coefficients", "select_amplitude"]


def integrate_mpr(theory: MPRTheory, amps: Sequence[float], freqs: Sequence[float],
                  t_transient: float, t_measure: float, n_samples: int,
                  rtol: float = 1e-10, atol: float = 1e-12):
    """Integrate Eqs. (MPR) with I(t) = sum_k amps[k] cos(freqs[k] t).

    Starts at the stationary state and returns the sampling times in the
    measurement window [t_transient, t_transient + t_measure] and the rate r(t).
    """
    amps = np.asarray(amps, float)
    freqs = np.asarray(freqs, float)
    m = theory

    def rhs(t, y):
        r, v = y
        inp = float(np.sum(amps * np.cos(freqs * t)))
        return [m.Delta / np.pi + 2 * r * v,
                v * v + m.eta_bar + m.J * r - np.pi**2 * r * r + inp]

    t_eval = np.linspace(t_transient, t_transient + t_measure, n_samples)
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), [m.r, m.v], t_eval=t_eval,
                    rtol=rtol, atol=atol, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    return t_eval, sol.y[0]


def fourier_coefficients(t, x, freqs) -> dict[float, complex]:
    """Complex amplitudes c(W) = (1/T) int x(t) exp(i W t) dt (trapezoidal rule)."""
    T = t[-1] - t[0]
    return {float(W): complex(np.trapezoid(x * np.exp(1j * W * t), t) / T) for W in freqs}


def select_amplitude(theory: MPRTheory, w: float, candidates: Sequence[float],
                     t_transient: float, t_measure: float, n_samples: int,
                     tol: float = 0.03, max_linear_fraction: float = 0.35,
                     check_harmonic: bool = True):
    """Largest stimulus amplitude for which second order is accurate.

    An amplitude eps is accepted when (i) the linear modulation |chi1| eps does
    not exceed ``max_linear_fraction`` r*, and (ii) the full MPR equations give
    second-order estimates (mean-rate shift and, optionally, second harmonic)
    within a relative tolerance ``tol`` of the perturbative prediction.

    Returns the selected amplitude and the full-MPR estimates at that amplitude.
    """
    m = theory
    eps, est = None, None
    for eps in candidates:
        if abs(m.chi1_r(w)) * eps > max_linear_fraction * m.r:
            continue
        t, r = integrate_mpr(m, [eps], [w], t_transient, t_measure, n_samples)
        c = fourier_coefficients(t, r, [2 * w, 0.0])
        est = dict(chi2_harmonic=c[2 * w] / (eps**2 / 4),
                   chi2_rect=(c[0.0].real - m.r) / (eps**2 / 2))
        ok = abs(est["chi2_rect"] / m.rectification(w) - 1) < tol
        if check_harmonic:
            ok &= abs(est["chi2_harmonic"] / m.chi2_r(w, w) - 1) < tol
        if ok:
            break
    return eps, est
