"""Kramers-Kronig relations and sum rules for susceptibilities chi(omega).

Valid for functions analytic in the upper half plane with chi(-w) = chi(w)*.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

__all__ = ["kk_real_from_imag", "kk_imag_from_real", "moment"]

Chi = Callable[[float], complex]


def kk_real_from_imag(chi: Chi, w: float, cutoff: float = 400.0) -> float:
    """Re chi(w) = (2/pi) P int_0^inf w' Im chi(w') / (w'^2 - w^2) dw'."""
    a = quad(lambda x: x * chi(x).imag / (x + w), 0, cutoff, weight="cauchy", wvar=w,
             limit=2000)[0]
    b = quad(lambda x: x * chi(x).imag / (x**2 - w**2), cutoff, np.inf, limit=500)[0]
    return 2 / np.pi * (a + b)


def kk_imag_from_real(chi: Chi, w: float, cutoff: float = 400.0) -> float:
    """Im chi(w) = -(2 w/pi) P int_0^inf Re chi(w') / (w'^2 - w^2) dw'."""
    a = quad(lambda x: chi(x).real / (x + w), 0, cutoff, weight="cauchy", wvar=w,
             limit=2000)[0]
    b = quad(lambda x: chi(x).real / (x**2 - w**2), cutoff, np.inf, limit=500)[0]
    return -2 * w / np.pi * (a + b)


def moment(chi: Chi, k: int) -> float:
    """Odd moment int_0^inf w^k Im chi(w) dw (k odd) entering the sum rules."""
    return quad(lambda x: x**k * chi(x).imag, 0, np.inf, limit=2000)[0]
