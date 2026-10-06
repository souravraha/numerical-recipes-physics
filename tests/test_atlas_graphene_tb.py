import numpy as np

from numerical_recipes.atlas import graphene_tb as gt


def test_dirac_cone_gapless():
    # The honeycomb band touches at K: E(K) ~ 0 (gapless Dirac point).
    kK, EK = gt.dirac_point(t=1.0, nk=301)
    assert EK < 0.05


def test_linear_dispersion():
    # Near K the dispersion is linear, E ~ v_F |k - K|, so E(2r)/E(r) ~ 2.
    v_F, ratio = gt.linear_dispersion(t=1.0)
    assert v_F > 0.0
    assert abs(ratio - 2.0) < 0.35


def test_ssh_topological_edge_states():
    # t1 > t2: topological phase -> a pair of near-zero edge states in the gap.
    w = gt.ssh_spectrum(t1=1.0, t2=0.5, n_cells=60)
    aw = np.sort(np.abs(w))
    assert aw[1] < 0.1        # two edge states at E ~ 0
    assert aw[2] > 0.4        # ... separated from the bulk (gap ~ 2|t1 - t2|)


def test_ssh_trivial_no_edge_states():
    # t1 < t2: trivial phase -> fully gapped, no edge states.
    w = gt.ssh_spectrum(t1=0.5, t2=1.0, n_cells=60)
    aw = np.sort(np.abs(w))
    assert aw[0] > 0.4        # smallest |E| is a bulk state


def test_bipartite_spectrum_symmetric():
    # A bipartite tight-binding lattice has a spectrum symmetric about E = 0.
    w = gt.ssh_spectrum(t1=0.9, t2=0.6, n_cells=30)
    assert np.max(np.abs(w - (-w[::-1]))) < 1e-8
