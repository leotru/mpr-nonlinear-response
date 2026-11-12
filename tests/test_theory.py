"""Analytical properties of the closed-form susceptibilities."""

import numpy as np
import pytest

from mprresponse.theory import MPRTheory, _fp_residual, fixed_points

PAPER = dict(eta_bar=-3.0, J=15.0, Delta=1.0)


@pytest.fixture(scope="module")
def m():
    return MPRTheory(**PAPER)


def test_paper_parameters(m):
    assert m.r == pytest.approx(1.2844, abs=1e-4)
    assert m.gamma == pytest.approx(0.4957, abs=1e-4)
    assert m.w0 == pytest.approx(5.1627, abs=1e-4)
    assert m.omega_c == pytest.approx(8.0585, abs=1e-4)


def test_fixed_point(m):
    assert _fp_residual(m.r, m.eta_bar, m.J, m.Delta) == pytest.approx(0, abs=1e-10)
    assert len(fixed_points(-5.0, 15.0, 1.0)) == 3        # bistable regime of MPR


def test_linear_susceptibility_is_green_function_column(m):
    for w in [0.3, 2.0, 5.0, 11.0]:
        G = m.green(w)
        assert G[0, 1] == pytest.approx(m.chi1_r(w), rel=1e-12)
        assert G[1, 1] == pytest.approx(m.chi1_v(w), rel=1e-12)


def test_static_gain_equals_slope_of_transfer_function(m):
    h = 1e-5
    rp = MPRTheory(m.eta_bar + h, m.J, m.Delta).r
    rm = MPRTheory(m.eta_bar - h, m.J, m.Delta).r
    assert (rp - rm) / (2 * h) == pytest.approx(m.chi1_r(0.0).real, rel=1e-6)


def test_static_chi2_equals_half_curvature(m):
    h = 1e-3
    r = [MPRTheory(m.eta_bar + k * h, m.J, m.Delta).r for k in (-1, 0, 1)]
    curvature = (r[2] - 2 * r[1] + r[0]) / h**2
    assert m.chi2_r(0.0, 0.0).real == pytest.approx(0.5 * curvature, rel=1e-4)


def test_chi2_symmetry_and_reality(m):
    w1, w2 = 1.7, -3.2
    assert m.chi2_r(w1, w2) == pytest.approx(m.chi2_r(w2, w1), rel=1e-12)
    assert m.chi2_r(-w1, -w2) == pytest.approx(np.conj(m.chi2_r(w1, w2)), rel=1e-12)


def test_rectification_closed_form(m):
    w = np.linspace(0.1, 15, 50)
    assert np.allclose(m.chi2_r(w, -w).real, m.rectification(w), rtol=1e-10)
    assert np.allclose(m.chi2_r(w, -w).imag, 0, atol=1e-14)
    assert m.rectification(m.omega_c) == pytest.approx(0, abs=1e-15)
    assert m.rectification(0.5 * m.omega_c) < 0 < m.rectification(1.5 * m.omega_c)


def test_no_reversal_for_strong_heterogeneity():
    # omega_c^2 < 0  <=>  Delta > (2/sqrt 3) pi^2 r*^2: shift positive at all frequencies
    m = MPRTheory(-3.0, 5.0, 4.0)
    assert m.Delta > 2 / np.sqrt(3) * np.pi**2 * m.r**2
    assert np.isnan(m.omega_c)
    assert np.all(m.rectification(np.linspace(0, 20, 200)) > 0)


def test_harmonic_asymptote(m):
    w = 2000.0
    assert (w**4 * m.chi2_r(w, w)).real == pytest.approx(m.harmonic_asymptote(), rel=1e-3)
