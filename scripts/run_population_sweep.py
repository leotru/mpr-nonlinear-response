#!/usr/bin/env python3
"""Population response of the QIF network: data for Figs. 1, 3, 4 and Table I.

Usage: python scripts/run_population_sweep.py [--config configs/paper.toml]
                                              [--out data/population_sweep.json]
Runtime with the paper configuration: about 40 min on one CPU core.
"""

import argparse
import json
from pathlib import Path

from mprresponse.config import load_config, theory_from_config
from mprresponse.protocols import population_sweep
from mprresponse.provenance import metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/paper.toml")
    ap.add_argument("--out", default="data/population_sweep.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    theory = theory_from_config(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.json")

    def save(res, path):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(dict(res, theory=theory.summary(), metadata=metadata(cfg)), f, indent=1)
        tmp.replace(path)

    partial = None
    if part.exists():                     # resume an interrupted run
        with open(part) as f:
            partial = json.load(f)
        if partial.get("metadata", {}).get("config") != cfg:
            partial = None
    results = population_sweep(cfg, theory, log=lambda s: print(s, flush=True),
                               partial=partial, checkpoint=lambda r: save(r, part))
    save(results, out)
    part.unlink(missing_ok=True)
    print(f"written {out}")


if __name__ == "__main__":
    main()
