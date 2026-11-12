#!/usr/bin/env python3
"""Neuron-resolved mean-rate shift: data for the frequency-pulling figure.

Usage: python scripts/run_neuron_map.py [--config configs/paper.toml]
                                        [--out data/neuron_map.npz]
Runtime with the paper configuration: about 30 min on one CPU core.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mprresponse.config import load_config, theory_from_config
from mprresponse.protocols import neuron_map
from mprresponse.provenance import metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/paper.toml")
    ap.add_argument("--out", default="data/neuron_map.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)
    theory = theory_from_config(cfg)
    res = neuron_map(cfg, theory, log=lambda s: print(s, flush=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **res,
                        metadata=json.dumps(dict(theory=theory.summary(), **metadata(cfg))))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
