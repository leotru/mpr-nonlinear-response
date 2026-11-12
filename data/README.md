# Data

Generated with the configuration in `configs/paper.toml`; each file stores its
provenance (git commit, library versions, platform, creation time).

* `population_sweep.json` — produced by `scripts/run_population_sweep.py`.
  Fields per single-tone entry: `w` (frequency, units of 1/tau), `eps` (amplitude),
  network estimates `chi1`, `chi2h` (second harmonic), `chi2d` (mean-rate shift) with
  errors `*_err`, full mean-field estimates at the same amplitude `ode_*`, and theory
  `th_*`. Complex numbers are stored as `[real, imag]`. Two-tone entries give the
  intermodulation responses at `w1 + w2` (`sum`) and `w2 - w1` (`dif`).
* `neuron_map.npz` — produced by `scripts/run_neuron_map.py`. Arrays: `frequencies`,
  `eps`, `base_rate` (time-averaged rate of each neuron without input), `stim_rate`
  (one row per frequency), `window` (measurement window), `metadata` (JSON string).

Note on two-tone pairs: for (w1, w2) = (2, 3) and (1, 4), the difference frequency
coincides with a third-order product (2 w1 - w2 and 3 w1, respectively), so the
second-order estimates at w2 - w1 carry O(eps) corrections. The comparison with the
full mean-field equations at the same amplitude (`ode_dif`) is unaffected.
