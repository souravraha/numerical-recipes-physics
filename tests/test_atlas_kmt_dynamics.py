import numpy as np

from numerical_recipes.atlas import kmt_dynamics as kmt


def test_run_produces_finite_trajectory():
    # Conservation-like: the sleeve and kMT trajectories stay finite.
    res = kmt.run(n_steps=60, num=15, seed=42)
    assert res["Kr"].shape[0] == 61
    assert res["kmtr"].shape == (61, 15)
    assert np.all(np.isfinite(res["kmtr"]))
    assert np.all(np.isfinite(res["kmtl"]))
    assert np.all(np.isfinite(res["Kr"]))
    assert np.all(np.isfinite(res["Kl"]))


def test_run_solves_velocity_system():
    # The force balance A V = B is solved to a finite velocity vector of the
    # expected size (2 sleeves + 2*num microtubules).
    res = kmt.run(n_steps=40, num=15, seed=3)
    assert res["last_v"] is not None
    assert res["last_v"].shape == (2 * (15 + 1),)
    assert np.all(np.isfinite(res["last_v"]))


def test_run_is_deterministic_given_seed():
    # Reproducibility: the same seed gives bitwise-identical trajectories.
    a = kmt.run(n_steps=40, num=15, seed=1)
    b = kmt.run(n_steps=40, num=15, seed=1)
    assert np.array_equal(a["kmtr"], b["kmtr"])
    assert np.array_equal(a["kmtl"], b["kmtl"])
    assert np.array_equal(a["last_v"], b["last_v"])


def test_different_seeds_diverge():
    # The stochastic rule actually uses the RNG: different seeds differ.
    a = kmt.run(n_steps=40, num=15, seed=1)
    b = kmt.run(n_steps=40, num=15, seed=2)
    assert not np.array_equal(a["kmtr"], b["kmtr"])
