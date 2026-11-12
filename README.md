# mpr-response

[![tests](https://github.com/leotru/mpr-nonlinear-response/actions/workflows/tests.yml/badge.svg)](https://github.com/leotru/mpr-nonlinear-response/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code and data accompanying

> H. Berry and L. Trujillo, *Periodic inputs can lower or raise the firing rate of a
> neural population: exact nonlinear response and Kramers–Kronig relations*
> (submitted to Physical Review E, 2026).

The package computes, in closed form, the linear and second-order (Volterra)
susceptibilities of the Montbrió–Pazó–Roxin firing-rate equations for heterogeneous
populations of quadratic integrate-and-fire (QIF) neurons, verifies their
Kramers–Kronig relations and sum rules, and compares all predictions with
simulations of large spiking networks.

## Installation

Python ≥ 3.11 is required.

```bash
git clone https://github.com/leotru/mpr-nonlinear-response.git
cd mpr-nonlinear-response
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test        # about 30 s
```

## Quick start

```python
from mprresponse import MPRTheory

m = MPRTheory(eta_bar=-3.0, J=15.0, Delta=1.0)
m.summary()                  # r*, gamma, omega_0, omega_c, ...
m.chi1_r(2.0)                # linear rate susceptibility
m.chi2_r(2.0, 3.0)           # second-order susceptibility chi^(2)(omega_1, omega_2)
m.rectification(10.0)        # mean-rate shift chi^(2)(omega, -omega)
```

## Reproducing the manuscript

All figures are regenerated from the data in `data/` in about one minute:

```bash
make figures     # writes figures/*.pdf
make verify      # checks every number quoted in the manuscript against data/
```

To regenerate the data from scratch (about 70 min on one CPU core):

```bash
make data
```

| Manuscript | File | Script | Data |
|---|---|---|---|
| Fig. 1 | `fig1_linear.pdf` | `make_figures.py` | `population_sweep.json` |
| Fig. 2 | `fig_bispectrum.pdf` | `make_figures.py` | theory (+ probed frequencies) |
| Fig. 3 | `fig2_harmonic_IM.pdf` | `make_figures.py` | `population_sweep.json` |
| Fig. 4, Table I | `fig3_rectification.pdf` | `make_figures.py` | `population_sweep.json` |
| Fig. 5 | `fig4_KK_sumrules.pdf` | `make_figures.py` | theory |
| Fig. 6 | `fig5_pulling.pdf` | `make_figures.py` | `neuron_map.npz` |

Data are produced by `scripts/run_population_sweep.py` (≈ 40 min) and
`scripts/run_neuron_map.py` (≈ 30 min). All parameters are in
[`configs/paper.toml`](configs/paper.toml); [`configs/quick.toml`](configs/quick.toml)
runs a reduced version in a few minutes (`make quick`) for testing purposes only.

## Data

| File | Content |
|---|---|
| `data/population_sweep.json` | For each stimulus frequency: amplitude, network estimates of the linear, second-harmonic and mean-rate-shift susceptibilities with statistical errors, full mean-field estimates at the same amplitude, and theory. Same for two-tone intermodulation. |
| `data/neuron_map.npz` | Time-averaged rate of each of the 20 000 neurons without (`base_rate`) and with (`stim_rate`, one row per frequency) periodic stimulation. |

Each file stores its provenance: configuration, package version, git commit,
library versions, platform, and creation time.

## Numerical methods

* **Mean field**: the full nonlinear equations are integrated with an
  eighth-order Runge–Kutta method (`DOP853`, relative tolerance 1e-10).
* **Spiking network**: each QIF neuron is advanced exactly between time steps for a
  piecewise-constant drive (Möbius map of the Riccati equation in projective
  coordinates); spikes are detected without a finite threshold
  (V_peak = −V_reset = ∞). Excitabilities are deterministic Lorentzian quantiles.
  The simulator is compiled with Numba.
* **Determinism**: given the configuration, all results are deterministic. They may
  differ in the last digits across platforms or library versions because of
  floating-point rounding.

## Repository layout

```
src/mprresponse/   theory, mean-field integration, network simulator, protocols, KK tools
scripts/           data generation, figures, verification
configs/           parameter files (paper and quick)
data/              data used in the manuscript
tests/             unit and integration tests (pytest)
```

## Citation

If you use this code, please cite the article above and this repository
(see [`CITATION.cff`](CITATION.cff)).

## License

MIT, see [LICENSE](LICENSE).
