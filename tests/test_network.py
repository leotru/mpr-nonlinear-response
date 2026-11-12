"""Spiking-network integrator: exactness for a single neuron and mean-field agreement."""

import numpy as np
import pytest

from mprresponse.network import NetworkState, make_network, simulate
from mprresponse.theory import MPRTheory


def test_single_neuron_period_is_exact():
    # uncoupled QIF neuron with constant drive a > 0 fires with period pi / sqrt(a)
    a = 2.3
    period = np.pi / np.sqrt(a)
    dt = period / 37                           # deliberately coarse: 37 steps per period
    state = NetworkState(np.array([a]), np.array([0.0]), np.array([1.0]))
    _, cycles = simulate(state, 0.0, dt, 37 * 500, n_measure_start=0)
    assert cycles[0] == pytest.approx(500.0, abs=1e-8)


def test_lorentzian_quantiles_are_symmetric():
    s = make_network(1001, 0.0, 1.0, 1.0, 0.0)
    assert np.median(s.eta) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.sort(s.eta), -np.sort(s.eta)[::-1])


@pytest.mark.slow
def test_network_rate_matches_mean_field():
    m = MPRTheory(-3.0, 15.0, 1.0)
    state = make_network(4000, m.eta_bar, m.Delta, m.r, m.v, seed=0)
    dt = 2e-3
    rate, _ = simulate(state, m.J, dt, int(80 / dt))
    assert rate[int(40 / dt):].mean() == pytest.approx(m.r, rel=2e-2)
