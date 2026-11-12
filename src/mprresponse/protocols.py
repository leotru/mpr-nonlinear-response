"""Simulation protocols used in the manuscript.

Both protocols start from the same stored network state (after an initial
unperturbed transient), so that every stimulated run can be compared with an
unstimulated baseline run from an identical initial condition.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .analysis import coefficient, noise_floor, window_mean_error
from .config import network_grid
from .meanfield import fourier_coefficients, integrate_mpr, select_amplitude
from .network import NetworkState, make_network, simulate
from .theory import MPRTheory

__all__ = ["initial_state", "population_sweep", "neuron_map"]

Log = Callable[[str], None]


def _cplx(z) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def initial_state(cfg: dict, theory: MPRTheory) -> NetworkState:
    """Network state after the initial unperturbed transient."""
    n = cfg["network"]
    dt, _ = network_grid(cfg)
    state = make_network(n["N"], theory.eta_bar, theory.Delta, theory.r, theory.v, n["seed"])
    simulate(state, theory.J, dt, int(round(n["initial_transient"] / dt)))
    return state


def population_sweep(cfg: dict, theory: MPRTheory, log: Log = print,
                     partial: dict | None = None,
                     checkpoint: Callable[[dict], None] | None = None) -> dict:
    """Linear, second-harmonic, mean-rate-shift and intermodulation responses.

    ``partial`` holds results of an interrupted run (items already present are
    skipped) and ``checkpoint`` is called with the results after every item.
    """
    m, sel, ps = theory, cfg["amplitude_selection"], cfg["population_sweep"]
    dt, n_meas = network_grid(cfg)
    T = n_meas * dt
    dW = 2 * np.pi / T
    n0 = int(round(cfg["network"]["transient_over_pi"] * np.pi / dt))
    nsteps = n0 + n_meas
    t = (np.arange(nsteps) + 0.5) * dt
    S0 = initial_state(cfg, m)

    def run(amps, freqs):
        rate, _ = simulate(S0.copy(), m.J, dt, nsteps, amps=amps, freqs=freqs)
        return rate

    def coef(x, W):
        return coefficient(x[n0:], t[n0:], W)

    def noise(x, W, excl):
        return noise_floor(x[n0:], t[n0:], W, dW, excl, ps["noise_kmin"], ps["noise_kmax"])

    base = run([], [])
    dc_err = np.sqrt(2) * window_mean_error(base[n0:], int(ps["dc_window"] / dt))
    out = partial if partial is not None else dict(single=[], two=[])
    out.setdefault("single", [])
    out.setdefault("two", [])
    done_single = {d["w"] for d in out["single"]}
    done_two = {tuple(d["w12"]) for d in out["two"]}
    out["baseline_rate"] = float(base[n0:].mean())
    for w in ps["frequencies"]:
        if w in done_single:
            continue
        t0 = time.time()
        eps, ode = select_amplitude(m, w, sel["candidates"], sel["ode_transient"], T,
                                    sel["ode_samples"], sel["tol"], sel["max_linear_fraction"],
                                    check_harmonic=True)
        rate = run([eps], [w])
        x = rate - rate[n0:].mean()
        excl = [w, 2 * w, 3 * w]
        out["single"].append(dict(
            w=w, eps=eps,
            chi1=_cplx(coef(x, w) / (eps / 2)), chi1_err=noise(x, w, excl) / (eps / 2),
            chi2h=_cplx(coef(x, 2 * w) / (eps**2 / 4)),
            chi2h_err=noise(x, 2 * w, excl) / (eps**2 / 4),
            chi2d=float((rate[n0:].mean() - base[n0:].mean()) / (eps**2 / 2)),
            chi2d_err=float(dc_err / (eps**2 / 2)),
            ode_chi2h=_cplx(ode["chi2_harmonic"]), ode_chi2d=float(ode["chi2_rect"]),
            th_chi1=_cplx(m.chi1_r(w)), th_chi2h=_cplx(m.chi2_r(w, w)),
            th_chi2d=float(m.rectification(w))))
        if checkpoint:
            checkpoint(out)
        log(f"single tone w={w:g} eps={eps:g} ({time.time() - t0:.0f} s)")
    for w1, w2, eps in ps["two_tone"]:
        if (w1, w2) in done_two:
            continue
        t0 = time.time()
        tt, r = integrate_mpr(m, [eps, eps], [w1, w2], sel["ode_transient"], T, sel["ode_samples"])
        c = fourier_coefficients(tt, r, [w1 + w2, w2 - w1])
        rate = run([eps, eps], [w1, w2])
        x = rate - rate[n0:].mean()
        excl = [w1, w2, w1 + w2, w2 - w1, 2 * w1, 2 * w2]
        norm = eps * eps / 2
        out["two"].append(dict(
            w12=[w1, w2], eps=eps,
            sum=_cplx(coef(x, w1 + w2) / norm), sum_err=noise(x, w1 + w2, excl) / norm,
            dif=_cplx(coef(x, w2 - w1) / norm), dif_err=noise(x, w2 - w1, excl) / norm,
            ode_sum=_cplx(c[w1 + w2] / norm), ode_dif=_cplx(c[w2 - w1] / norm),
            th_sum=_cplx(m.chi2_r(w1, w2)), th_dif=_cplx(m.chi2_r(w2, -w1))))
        if checkpoint:
            checkpoint(out)
        log(f"two tones ({w1:g}, {w2:g}) eps={eps:g} ({time.time() - t0:.0f} s)")
    out["single"].sort(key=lambda d: d["w"])
    return out


def neuron_map(cfg: dict, theory: MPRTheory, log: Log = print,
               partial: dict | None = None,
               checkpoint: Callable[[dict], None] | None = None) -> dict:
    """Time-averaged rate of every neuron with and without periodic stimulation.

    ``partial``/``checkpoint`` allow resuming an interrupted run (see
    :func:`population_sweep`); rows with ``done == False`` are recomputed.
    """
    m, sel, nm = theory, cfg["amplitude_selection"], cfg["neuron_map"]
    dt, _ = network_grid(cfg)
    n0 = int(round(nm["transient_over_pi"] * np.pi / dt))
    n_meas = int(round(nm["window_over_pi"] * np.pi / dt))
    T = n_meas * dt
    S0 = initial_state(cfg, m)

    def rates(amps, freqs):
        _, cyc = simulate(S0.copy(), m.J, dt, n0 + n_meas, amps=amps, freqs=freqs,
                          n_measure_start=n0)
        return cyc / T

    freqs = np.asarray(nm["frequencies"], float)
    if partial is not None and np.array_equal(partial.get("frequencies"), freqs):
        res = partial
    else:
        base = rates([], [])
        res = dict(frequencies=freqs, eps=np.zeros(freqs.size), base_rate=base,
                   stim_rate=np.zeros((freqs.size, base.size)), window=T,
                   done=np.zeros(freqs.size, dtype=bool))
        log("baseline done")
    for i, w in enumerate(freqs):
        if res["done"][i]:
            continue
        t0 = time.time()
        eps, _ = select_amplitude(m, w, sel["candidates"], sel["ode_transient"], T,
                                  nm["ode_samples"], sel["tol"], sel["max_linear_fraction"],
                                  check_harmonic=False)
        res["eps"][i] = eps
        res["stim_rate"][i] = rates([eps], [w])
        res["done"][i] = True
        if checkpoint:
            checkpoint(res)
        log(f"w={w:g} eps={eps:g} ({time.time() - t0:.0f} s)")
    return res
