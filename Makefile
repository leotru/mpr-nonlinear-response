PY ?= python
CONFIG ?= configs/paper.toml

.PHONY: install test test-fast lint figures verify data quick clean

install:
	$(PY) -m pip install -e ".[dev]"

test:            ## full test suite (~30 s)
	$(PY) -m pytest -q

test-fast:       ## skip the spiking-network integration test
	$(PY) -m pytest -q -m "not slow"

lint:
	ruff check src scripts tests

figures:         ## all manuscript figures from data/ (~1 min)
	$(PY) scripts/make_figures.py --data data --out figures --config $(CONFIG)

verify:          ## check the numbers quoted in the manuscript
	$(PY) scripts/verify_results.py --data data

data:            ## regenerate data/ from scratch (~70 min on one core)
	$(PY) scripts/run_population_sweep.py --config $(CONFIG) --out data/population_sweep.json
	$(PY) scripts/run_neuron_map.py --config $(CONFIG) --out data/neuron_map.npz

quick:           ## reduced pipeline for testing (a few minutes); writes to quick/
	$(PY) scripts/run_population_sweep.py --config configs/quick.toml --out quick/population_sweep.json
	$(PY) scripts/run_neuron_map.py --config configs/quick.toml --out quick/neuron_map.npz
	$(PY) scripts/make_figures.py --data quick --out quick/figures --config configs/quick.toml

clean:
	rm -rf quick figures/*.png .pytest_cache .ruff_cache
