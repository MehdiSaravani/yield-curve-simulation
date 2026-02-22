# tests/test_rsim.py

import sys
import os
import re
# Add parent directory to path to find rsim module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from rsim import YieldCurveSimulator


@pytest.fixture
def yield_curve_simulator():
    return YieldCurveSimulator(n_factors=3)


@pytest.fixture
def example_yield_data():
    dates = pd.date_range(start="2000-01-01", periods=10, freq="M")
    yield_data = pd.DataFrame(
        data=np.random.rand(10, 5),  # Random 5 maturities over 10 periods
        index=dates,
        columns=[1, 3, 5, 7, 10],  # Maturities in years
    )
    return yield_data


def test_simulate_before_fit_raises_error(yield_curve_simulator):
    # Use re.escape() to properly escape the regex special characters
    with pytest.raises(ValueError, match=re.escape("Model must be fitted before simulation. Call fit() first.")):
        yield_curve_simulator.simulate()


def test_simulate_returns_list_of_arrays(yield_curve_simulator, example_yield_data):
    yield_curve_simulator.fit(example_yield_data)
    results = yield_curve_simulator.simulate(n_paths=100, n_steps=12, seed=123)
    assert len(results) == 2, "Simulation must return two arrays: X_sim and Y_sim"
    assert isinstance(results[0], np.ndarray), "First element of results must be a numpy array (X_sim)"
    assert isinstance(results[1], np.ndarray), "Second element of results must be a numpy array (Y_sim)"


def test_simulate_path_dimensions(yield_curve_simulator, example_yield_data):
    yield_curve_simulator.fit(example_yield_data)
    n_paths = 50
    n_steps = 24
    results = yield_curve_simulator.simulate(n_paths=n_paths, n_steps=n_steps, seed=111)
    X_sim, Y_sim = results
    assert X_sim.shape == (n_paths, n_steps,
                           3), "X_sim shape does not match expected dimensions (n_paths, n_steps, n_factors)"
    assert Y_sim.shape[0] == n_paths, "Y_sim paths dimension does not match n_paths"
    assert Y_sim.shape[1] == n_steps, "Y_sim time dimension does not match n_steps"
    assert Y_sim.shape[2] == example_yield_data.shape[
        1], "Y_sim maturities dimension does not match the number of maturities"


def test_simulation_reproducibility(yield_curve_simulator, example_yield_data):
    yield_curve_simulator.fit(example_yield_data)
    seed = 42
    sim1 = yield_curve_simulator.simulate(n_paths=10, n_steps=5, seed=seed)
    sim2 = yield_curve_simulator.simulate(n_paths=10, n_steps=5, seed=seed)
    assert np.array_equal(sim1[0], sim2[0]), "X_sim results should be reproducible when using the same seed"
    assert np.array_equal(sim1[1], sim2[1]), "Y_sim results should be reproducible when using the same seed"


def test_simulate_with_multiple_start_indices(yield_curve_simulator, example_yield_data):
    yield_curve_simulator.fit(example_yield_data)
    sim_start_indices = [0, 1, 2]
    n_paths = 10
    n_steps = 5
    results = yield_curve_simulator.simulate(
        n_paths=n_paths, n_steps=n_steps, sim_start_index=sim_start_indices, seed=456
    )
    X_sim, Y_sim = results
    expected_total_paths = n_paths * len(sim_start_indices)
    assert X_sim.shape[0] == expected_total_paths, "X_sim paths dimension does not account for multiple start indices"
    assert Y_sim.shape[0] == expected_total_paths, "Y_sim paths dimension does not account for multiple start indices"
