import numpy as np

from numerical_recipes.atlas import lattice_ft as lt


def test_alignment_increases_with_beta():
    # Weaker coupling (larger beta) -> more aligned links -> larger <Re P>.
    p_lo = lt.plaquette_expectation(L=8, beta=0.5, n_sweeps=1500, equilibration=300, seed=3)[0]
    p_mid = lt.plaquette_expectation(L=8, beta=2.0, n_sweeps=1500, equilibration=300, seed=3)[0]
    p_hi = lt.plaquette_expectation(L=8, beta=8.0, n_sweeps=1500, equilibration=300, seed=3)[0]
    assert p_hi > p_mid > p_lo


def test_weak_coupling_aligns_to_one():
    # Deep in the Coulomb phase the links align and <Re P> -> 1.
    p = lt.plaquette_expectation(L=8, beta=20.0, n_sweeps=2500, equilibration=500, seed=4)[0]
    assert p > 0.85


def test_strong_coupling_nearly_random():
    # Strong coupling (beta -> 0): random links, <Re P> -> 0.
    p = lt.plaquette_expectation(L=8, beta=0.3, n_sweeps=1500, equilibration=300, seed=5)[0]
    assert p < 0.2


def test_strong_coupling_upper_bound():
    # The 1-loop strong-coupling result 1 - 1/(2 beta) is an upper bound on
    # <Re P> (higher-order terms are negative).
    beta = 6.0
    p = lt.plaquette_expectation(L=8, beta=beta, n_sweeps=2500, equilibration=500, seed=6)[0]
    assert p < lt.strong_coupling_expectation(beta)
    assert 0.0 < p < 1.0
