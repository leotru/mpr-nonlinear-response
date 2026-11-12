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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.stem + ".partial.npz")

    def save(res, path):
        meta = json.dumps(dict(theory=theory.summary(), **metadata(cfg)))
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **res, metadata=meta)
        tmp.replace(path)

    partial = None
    if part.exists():                     # resume an interrupted run
        with np.load(part) as z:
            partial = {k: z[k] for k in z.files if k != "metadata"}
            if json.loads(str(z["metadata"]))["config"] != cfg:
                partial = None
    res = neuron_map(cfg, theory, log=lambda s: print(s, flush=True),
                     partial=partial, checkpoint=lambda r: save(r, part))
    save(res, out)
    part.unlink(missing_ok=True)
    print(f"written {out}")


if __name__ == "__main__":
    main()
