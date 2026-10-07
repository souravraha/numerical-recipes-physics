import numpy as np

from numerical_recipes.atlas import nfe_bands as nb


def test_1d_gap_opens_at_bz_edge():
    # At k = K/2 the k=0 and k=K plane waves are degenerate; the weak periodic
    # potential lifts the degeneracy.  The converged gap is ~0.462 U for
    # U = 0.5, approached monotonically from above as the cutoff grows.
    U = 0.5
    gap_small = nb.bands_1d(0.5, 5, U)[1] - nb.bands_1d(0.5, 5, U)[0]
    gap_large = nb.bands_1d(0.5, 40, U)[1] - nb.bands_1d(0.5, 40, U)[0]
    assert gap_large > 0.4
    assert gap_small > gap_large


def test_1d_band_structure_shape_and_symmetric():
    # Conservation/symmetry: the bands are even in k (symmetric about k = 0).
    Nmax = 5
    k, e = nb.band_structure_1d(Nmax=Nmax, U=0.5, n_k=200)
    assert e.shape == (200, 2 * Nmax + 1)
    assert np.all(np.isfinite(e))
    assert np.allclose(e[:, 0], e[::-1, 0])


def test_2d_bands_finite_and_ascending():
    # Independent reference: a Hermitian matrix has real, ascending eigenvalues.
    e = nb.bands_2d(0.1, 0.1, Nmax=2, U=0.1)
    assert e.shape == (25,)
    assert np.all(np.isfinite(e))
    assert np.all(np.diff(e) >= -1e-12)


def test_2d_bands_have_center_degeneracy():
    # At the BZ centre k=0 the bands are the (G)^2 eigenvalues shifted by U;
    # the lowest is U (the k=0, G=0 plane wave minus the U_0 shift).
    e = nb.bands_2d(0.0, 0.0, Nmax=2, U=0.3)
    assert e[0] < 0.0           # the -U (U_0) shift pulls the lowest band below 0
    assert np.all(np.isfinite(e))
