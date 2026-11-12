#!/usr/bin/env python3
"""Check that the data reproduce every number quoted in the manuscript.

Usage: python scripts/verify_results.py [--data data]
Exits with status 1 if any quoted value is not reproduced to its quoted precision.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mprresponse.theory import MPRTheory

# (description, quoted value, tolerance = half a unit of the last quoted digit)
QUOTED_THEORY = [
    ("r*", 1.2844, 5e-5), ("gamma", 0.496, 5e-4), ("omega_0", 5.163, 5e-4),
    ("omega_c", 8.06, 5e-3), ("2 pi r*", 8.07, 5e-3),
]
# Table I: omega -> (eps, network, error, unit of last digit, full MPR, unit of last digit)
TABLE_I = {
    0.5: (0.70, -9.19e-3, 0.35e-3, 1e-5, -9.09e-3, 1e-5),
    3.0: (0.50, -1.76e-2, 0.07e-2, 1e-4, -1.72e-2, 1e-4),
    5.0: (0.05, -0.41, 0.07, 1e-2, -0.437, 1e-3),
    7.0: (1.50, -2.81e-3, 0.08e-3, 1e-5, -2.92e-3, 1e-5),
    8.0: (2.00, -0.4e-4, 0.4e-4, 1e-5, -0.62e-4, 1e-6),
    9.0: (3.00, 5.23e-4, 0.19e-4, 1e-6, 5.16e-4, 1e-6),
    10.0: (3.00, 6.24e-4, 0.19e-4, 1e-6, 6.22e-4, 1e-6),
    12.0: (3.00, 5.46e-4, 0.19e-4, 1e-6, 5.51e-4, 1e-6),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    m = MPRTheory(-3.0, 15.0, 1.0)
    failures = []

    def check(name, value, quoted, tol):
        ok = abs(value - quoted) <= tol
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value:.6g} (quoted {quoted:g})")
        if not ok:
            failures.append(name)

    th = dict(zip(["r*", "gamma", "omega_0", "omega_c", "2 pi r*"],
                  [m.r, m.gamma, m.w0, m.omega_c, 2 * np.pi * m.r], strict=True))
    for name, q, tol in QUOTED_THEORY:
        check(name, th[name], q, tol)

    R = json.load(open(data / "population_sweep.json"))
    S = {d["w"]: d for d in R["single"]}
    check("baseline network rate", R["baseline_rate"], 1.2800, 5e-5)
    for w, (eps, net, err, unit, ode, unit_ode) in TABLE_I.items():
        d = S[w]
        check(f"Table I eps(w={w})", d["eps"], eps, 1e-12)
        check(f"Table I network(w={w})", d["chi2d"], net, 0.5 * unit)
        check(f"Table I error(w={w})", d["chi2d_err"], err, 0.5 * unit)
        check(f"Table I MPR(w={w})", d["ode_chi2d"], ode, 0.5 * unit_ode)
    c = S[5.0]["chi1"]
    check("chi1(5) Re", c[0], 0.395, 5e-4)
    check("chi1(5) Im", c[1], 0.739, 5e-4)
    c = S[3.5]["chi2h"]
    check("chi2(3.5,3.5) Re", c[0], 0.0665, 5e-5)
    check("chi2(3.5,3.5) Im", c[1], 0.0111, 5e-5)
    two = {tuple(d["w12"]): d for d in R["two"]}
    c = two[(1.0, 4.0)]["dif"]
    check("chi2(4,-1) Re", c[0], -0.0350, 5e-5)
    check("chi2(4,-1) Im", c[1], -0.0120, 5e-5)

    M = np.load(data / "neuron_map.npz")
    i = int(np.argmin(abs(M["frequencies"] - 10.0)))
    shift = (M["stim_rate"][i] - M["base_rate"]) / (M["eps"][i] ** 2 / 2)
    check("neuron map, population average at w=10", shift.mean(), 6.36e-4, 5e-6)
    Om = 2 * np.pi * M["base_rate"]
    check("fraction of neurons with 6 < Omega < 10", np.mean((Om > 6) & (Om < 10)), 0.92, 5e-3)
    i = int(np.argmin(abs(M["frequencies"] - 7.0)))
    shift = (M["stim_rate"][i] - M["base_rate"]) / (M["eps"][i] ** 2 / 2)
    # neurons with Omega_j < w - 0.5 (silent neurons included), as in Sec. V B
    check("neuron map, Omega < w at w=7", shift[Om < 6.5].mean(), -7.6e-4, 5e-6)

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
