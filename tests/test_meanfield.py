"""Perturbation theory versus integration of the full nonlinear MPR equations."""

import pytest

from mprresponse.meanfield import fourier_coefficients, integrate_mpr
from mprresponse.theory import MPRTheory

T = 80 * 3.141592653589793      # all test frequencies are multiples of 2 pi / T


@pytest.fixture(scope="module")
def m():
    return MPRTheory(-3.0, 15.0, 1.0)


def test_single_tone(m):
    w, eps = 3.0, 0.05
    t, r = integrate_mpr(m, [eps], [w], 150.0, T, 60001)
    c = fourier_coefficients(t, r, [w, 2 * w, 0.0])
    assert c[w] / (eps / 2) == pytest.approx(m.chi1_r(w), rel=2e-3)
    assert c[2 * w] / (eps**2 / 4) == pytest.approx(m.chi2_r(w, w), rel=1e-2)
    assert (c[0.0].real - m.r) / (eps**2 / 2) == pytest.approx(m.rectification(w), rel=1e-2)


def test_two_tones(m):
    # (w1, w2) chosen so that no third-order product (2 w1 - w2, 3 w1, ...) falls on
    # w1 + w2 or w2 - w1; otherwise the second-order estimate has O(eps) corrections.
    w1, w2, eps = 2.0, 3.3, 0.05
    t, r = integrate_mpr(m, [eps, eps], [w1, w2], 150.0, T, 60001)
    c = fourier_coefficients(t, r, [w1 + w2, w2 - w1])
    assert c[w1 + w2] / (eps**2 / 2) == pytest.approx(m.chi2_r(w1, w2), rel=1e-2)
    assert c[w2 - w1] / (eps**2 / 2) == pytest.approx(m.chi2_r(w2, -w1), rel=1e-2)
