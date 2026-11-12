"""Metadata recorded alongside every generated data file."""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from pathlib import Path

__all__ = ["metadata"]


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001  (not a git checkout, git missing, ...)
        return "unknown"


def metadata(config: dict) -> dict:
    """Configuration, code version, library versions, and timestamp."""
    import matplotlib
    import numba
    import numpy
    import scipy

    from . import __version__
    return dict(
        package_version=__version__,
        git_commit=_git_commit(),
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        python=platform.python_version(),
        platform=platform.platform(),
        versions=dict(numpy=numpy.__version__, scipy=scipy.__version__,
                      numba=numba.__version__, matplotlib=matplotlib.__version__),
        config=config,
    )
