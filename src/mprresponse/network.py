"""Network of quadratic integrate-and-fire neurons with Lorentzian heterogeneity.

Each neuron obeys dV/dt = V^2 + eta_j + J r(t) + I(t) (tau = 1) with
V_peak = -V_reset = infinity. Between time steps the Riccati equation is solved
exactly for a piecewise-constant drive by a Moebius map, implemented in
projective coordinates V = x1/x2 with (x1, x2) a unit vector and x2 >= 0.
A spike is registered when x2 changes sign. Recurrent input is delivered as
instantaneous pulses V -> V + J n / N at the end of each step (n spikes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

__all__ = ["NetworkState", "make_network", "simulate", "lorentzian_quantiles"]


@njit(cache=True)
def _simulate(eta, x1, x2, J, dt, nsteps, n0, amps, freqs):  # pragma: no cover (jit)
    N = eta.size
    pop = np.zeros(nsteps, dtype=np.int32)
    nspk = np.zeros(N, dtype=np.int64)
    th0 = np.zeros(N)
    for n in range(nsteps):
        if n == n0:
            for j in range(N):
                th0[j] = 2.0 * np.arctan2(x1[j], x2[j])
        t = (n + 0.5) * dt                       # input evaluated at mid-step
        inp = 0.0
        for k in range(amps.size):
            inp += amps[k] * np.cos(freqs[k] * t)
        nsp = 0
        for j in range(N):
            a = eta[j] + inp
            if a > 0.0:
                s = np.sqrt(a)
                C = np.cos(s * dt)
                Sd = np.sin(s * dt) / s
            elif a < 0.0:
                s = np.sqrt(-a)
                C = np.cosh(s * dt)
                Sd = np.sinh(s * dt) / s
            else:
                C = 1.0
                Sd = dt
            y1 = C * x1[j] + a * Sd * x2[j]
            y2 = -Sd * x1[j] + C * x2[j]
            if y2 < 0.0:                         # V passed through +infinity
                nsp += 1
                if n >= n0:
                    nspk[j] += 1
                y1 = -y1
                y2 = -y2
            nrm = np.sqrt(y1 * y1 + y2 * y2)
            x1[j] = y1 / nrm
            x2[j] = y2 / nrm
        pop[n] = nsp
        if nsp > 0:                              # recurrent pulse V -> V + J n / N
            d = J * nsp / N
            for j in range(N):
                y1 = x1[j] + d * x2[j]
                y2 = x2[j]
                nrm = np.sqrt(y1 * y1 + y2 * y2)
                x1[j] = y1 / nrm
                x2[j] = y2 / nrm
    cycles = np.zeros(N)
    if n0 < nsteps:
        for j in range(N):
            cycles[j] = nspk[j] + (2.0 * np.arctan2(x1[j], x2[j]) - th0[j]) / (2 * np.pi)
    return pop, cycles


def lorentzian_quantiles(N: int) -> np.ndarray:
    """Deterministic quantiles of the standard Cauchy distribution."""
    return np.tan(0.5 * np.pi * (2 * np.arange(1, N + 1) - N - 1) / (N + 1))


@dataclass
class NetworkState:
    """Excitabilities and projective coordinates of all neurons."""

    eta: np.ndarray
    x1: np.ndarray
    x2: np.ndarray

    def copy(self) -> NetworkState:
        return NetworkState(self.eta.copy(), self.x1.copy(), self.x2.copy())


def make_network(N: int, eta_bar: float, Delta: float, r0: float, v0: float,
                 seed: int = 0) -> NetworkState:
    """Network with deterministic Lorentzian excitabilities.

    Initial voltages are placed on the Ott-Antonsen manifold, i.e. Lorentzian
    with center v0 and half-width pi r0, randomly permuted with ``seed``.
    """
    q = lorentzian_quantiles(N)
    eta = eta_bar + Delta * q
    rng = np.random.default_rng(seed)
    half_theta = np.arctan(v0 + np.pi * r0 * rng.permutation(q))
    return NetworkState(eta, np.sin(half_theta), np.cos(half_theta))


def simulate(state: NetworkState, J: float, dt: float, nsteps: int, *,
             amps=(), freqs=(), n_measure_start: int | None = None):
    """Advance ``state`` in place for ``nsteps`` steps of size ``dt``.

    Returns
    -------
    rate : ndarray
        Population rate in each step, spikes / (N dt).
    cycles : ndarray
        Accumulated phase of each neuron, in cycles, from step
        ``n_measure_start`` to the end (zeros if not requested).
    """
    n0 = nsteps if n_measure_start is None else int(n_measure_start)
    pop, cycles = _simulate(state.eta, state.x1, state.x2, float(J), float(dt), int(nsteps), n0,
                            np.asarray(amps, float), np.asarray(freqs, float))
    return pop / (state.eta.size * dt), cycles
