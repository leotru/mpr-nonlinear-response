"""Closed-form linear and second-order response of the MPR firing-rate equations.

The Montbrio-Pazo-Roxin (MPR) equations, in units where the membrane time
constant is tau = 1, read

    dr/dt = Delta/pi + 2 r v
    dv/dt = v^2 + eta_bar + J r - pi^2 r^2 + I(t)

Fourier convention: x(t) = int x(omega) exp(-i omega t) d omega / 2 pi, so that
causal susceptibilities are analytic in the upper half of the complex plane.
Equation numbers in the docstrings refer to the accompanying manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

__all__ = ["fixed_points", "MPRTheory"]


def _fp_residual(r, eta_bar, J, Delta, I0=0.0):
    """Stationary-state condition F(r, I0) = 0 (Eq. fp)."""
    return Delta**2 / (4 * np.pi**2 * r**2) + eta_bar + I0 + J * r - np.pi**2 * r**2


def fixed_points(eta_bar: float, J: float, Delta: float, I0: float = 0.0,
                 r_min: float = 1e-4, r_max: float = 1e2, n_grid: int = 20000) -> list[float]:
    """Return all stationary firing rates r* > 0, sorted in increasing order."""
    rs = np.logspace(np.log10(r_min), np.log10(r_max), n_grid)
    f = _fp_residual(rs, eta_bar, J, Delta, I0)
    roots = []
    for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:])):
        roots.append(brentq(_fp_residual, rs[i], rs[i + 1], args=(eta_bar, J, Delta, I0),
                            xtol=1e-15, rtol=1e-14))
    return roots


@dataclass
class MPRTheory:
    """Linear and second-order susceptibilities around a stable stationary state.

    Parameters
    ----------
    eta_bar, J, Delta : float
        Center of the excitability distribution, coupling, and heterogeneity.
    branch : int
        Index of the stable fixed point to use (sorted by rate; -1 = highest).
    """

    eta_bar: float
    J: float
    Delta: float
    branch: int = -1
    r: float = field(init=False)
    v: float = field(init=False)

    def __post_init__(self) -> None:
        stable = [r for r in fixed_points(self.eta_bar, self.J, self.Delta)
                  if self._w0sq(r) > 0]
        if not stable:
            raise ValueError("no stable fixed point for these parameters")
        self.r = stable[self.branch]
        self.v = -self.Delta / (2 * np.pi * self.r)

    # --- linear response (Eqs. oscillator, params, chi1, D) -------------------
    def _w0sq(self, r: float) -> float:
        return self.Delta**2 / (np.pi**2 * r**2) - 2 * self.J * r + 4 * np.pi**2 * r**2

    @property
    def gamma(self) -> float:
        """Damping rate, gamma = -4 v* = 2 Delta / (pi r*)."""
        return 2 * self.Delta / (np.pi * self.r)

    @property
    def w0sq(self) -> float:
        """Stiffness omega_0^2 (determinant of the Jacobian)."""
        return self._w0sq(self.r)

    @property
    def w0(self) -> float:
        return float(np.sqrt(self.w0sq))

    @property
    def omega_c_sq(self) -> float:
        """Square of the sign-reversal frequency of the mean-rate shift (Eq. rect)."""
        return 4 * np.pi**2 * self.r**2 - 0.75 * self.gamma**2

    @property
    def omega_c(self) -> float:
        """Sign-reversal frequency; NaN when omega_c^2 < 0 (no reversal)."""
        return float(np.sqrt(self.omega_c_sq)) if self.omega_c_sq > 0 else float("nan")

    @property
    def jacobian(self) -> np.ndarray:
        return np.array([[2 * self.v, 2 * self.r],
                         [self.J - 2 * np.pi**2 * self.r, 2 * self.v]])

    def D(self, w):
        return self.w0sq - w**2 - 1j * self.gamma * w

    def p(self, w):
        return self.gamma / 2 - 1j * w

    def chi1_r(self, w):
        """Linear rate susceptibility, 2 r* / D(omega)."""
        return 2 * self.r / self.D(w)

    def chi1_v(self, w):
        """Linear voltage susceptibility, p(omega) / D(omega)."""
        return self.p(w) / self.D(w)

    def green(self, w) -> np.ndarray:
        """Matrix Green's function (-i omega 1 - Jacobian)^(-1) at a scalar omega."""
        return np.linalg.inv(-1j * w * np.eye(2) - self.jacobian)

    # --- second-order response (Eqs. chi2, N, rect) ---------------------------
    def numerator(self, w1, w2):
        W = w1 + w2
        return (self.p(W) * (self.p(w1) + self.p(w2)) + self.p(w1) * self.p(w2)
                - 4 * np.pi**2 * self.r**2)

    def chi2_r(self, w1, w2):
        """Symmetric second-order rate susceptibility chi^(2)_r(omega_1, omega_2)."""
        return 2 * self.r * self.numerator(w1, w2) / (self.D(w1 + w2) * self.D(w1) * self.D(w2))

    def chi2_v(self, w1, w2):
        """Symmetric second-order voltage susceptibility (Appendix A)."""
        W = w1 + w2
        d12 = self.D(w1) * self.D(w2)
        s_r = 2 * self.r * (self.p(w1) + self.p(w2)) / d12
        s_v = (self.p(w1) * self.p(w2) - 4 * np.pi**2 * self.r**2) / d12
        return ((self.J - 2 * np.pi**2 * self.r) * s_r + self.p(W) * s_v) / self.D(W)

    def rectification(self, w):
        """Mean-rate shift susceptibility chi^(2)_r(omega, -omega), closed form (real)."""
        return 2 * self.r * (w**2 - self.omega_c_sq) / (self.w0sq * np.abs(self.D(w))**2)

    # --- static checks and asymptotics ---------------------------------------
    def static_gain(self) -> float:
        """d r*/d I0 = 2 r*/omega_0^2 (static sum rule)."""
        return 2 * self.r / self.w0sq

    def harmonic_asymptote(self) -> float:
        """Coefficient c in chi^(2)_r(omega, omega) ~ c / omega^4 for omega -> infinity."""
        return 2.5 * self.r

    def summary(self) -> dict:
        return dict(eta_bar=self.eta_bar, J=self.J, Delta=self.Delta, r=self.r, v=self.v,
                    gamma=self.gamma, w0=self.w0, omega_c=self.omega_c,
                    quality_factor=self.w0 / self.gamma)
