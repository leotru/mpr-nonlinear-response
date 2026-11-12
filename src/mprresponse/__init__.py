"""Exact nonlinear response of the Montbrio-Pazo-Roxin firing-rate equations."""

__version__ = "1.0.0"

from .theory import MPRTheory, fixed_points  # noqa: E402

__all__ = ["MPRTheory", "fixed_points", "__version__"]
