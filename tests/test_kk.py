"""Kramers-Kronig relations and sum rules (Sec. IV)."""

import numpy as np
import pytest

from mprresponse.kk import kk_imag_from_real, kk_real_from_imag, moment
from mprresponse.theory import MPRTheory


@pytest.fixture(scope="module")
def m():
    return MPRTheory(-3.0, 15.0, 1.0)


@pytest.mark.parametrize("w", [0.7, 2.6, 5.1, 9.0])
def test_kk_linear(m, w):
    assert kk_real_from_imag(m.chi1_r, w) == pytest.approx(m.chi1_r(w).real, abs=1e-8)
    assert kk_imag_from_real(m.chi1_r, w) == pytest.approx(m.chi1_r(w).imag, abs=1e-8)


@pytest.mark.parametrize("w", [0.7, 2.6, 5.1, 9.0])
def test_kk_second_harmonic(m, w):
    def h(x):
        return m.chi2_r(x, x)
    assert kk_real_from_imag(h, w) == pytest.approx(h(w).real, abs=1e-8)
    assert kk_imag_from_real(h, w) == pytest.approx(h(w).imag, abs=1e-8)


def test_rectification_violates_kk(m):
    w = 3.0
    rect = m.chi2_r(w, -w)
    assert abs(rect.imag) < 1e-14 and abs(rect.real) > 1e-3   # KK from Im = 0 would give 0


def test_sum_rules(m):
    assert moment(m.chi1_r, 1) == pytest.approx(np.pi * m.r, rel=1e-8)

    def h(x):
        return m.chi2_r(x, x)
    assert moment(h, 1) == pytest.approx(0.0, abs=1e-9)
    assert 2 / np.pi * moment(h, 3) == pytest.approx(-m.harmonic_asymptote(), rel=1e-7)
