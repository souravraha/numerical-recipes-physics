import numpy as np
from scipy.linalg import eigh as scipy_eigh

from numerical_recipes import linalg_eig as le


def _sym(rng, n):
    A = rng.normal(size=(n, n))
    return 0.5 * (A + A.T)


def test_householder_tridiagonal_recovers_A():
    rng = np.random.default_rng(0)
    n = 20
    A = _sym(rng, n)
    T, Q = le.householder_tridiagonalize(A)
    # A == Q T Q^T
    assert np.max(np.abs(Q @ T @ Q.T - A)) < 1e-10
    # T is tridiagonal
    assert np.max(np.abs(np.diag(T, 2))) < 1e-12
    assert np.max(np.abs(np.diag(T, -2))) < 1e-12
    # Q orthogonal
    assert np.max(np.abs(Q @ Q.T - np.eye(n))) < 1e-12


def test_sturm_count_brackets():
    d = np.array([3.0, 2.0, 1.0])
    e = np.array([1.0, 1.0])
    # eigenvalues of this tridiagonal (via scipy)
    T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    w = np.linalg.eigvalsh(T)
    w = np.sort(w)
    # sturm_count(x) = number of eigenvalues below x
    for x in np.linspace(w[0] - 0.1, w[-1] + 0.1, 11):
        assert le.sturm_count(d, e, x) == np.sum(w < x)


def test_eigh_matches_scipy_random():
    rng = np.random.default_rng(1)
    n = 30
    A = _sym(rng, n)
    w, V = le.eigh_symmetric(A)
    w_ref, V_ref = scipy_eigh(A)
    # Eigenvalues (both ascending) agree.
    assert np.max(np.abs(w - w_ref)) < 1e-7
    # Each computed eigenvector is parallel to a reference one.
    for j in range(n):
        proj = np.max(np.abs(V_ref.T @ V[:, j]))
        assert proj > 1.0 - 1e-6


def test_eigh_eigenvectors_orthonormal():
    rng = np.random.default_rng(2)
    A = _sym(rng, 25)
    _, V = le.eigh_symmetric(A)
    assert np.max(np.abs(V.T @ V - np.eye(25))) < 1e-8


def test_particle_in_box_spectrum():
    n = 60
    H, exact = le.particle_in_box(n=n)
    w, V = le.eigh_symmetric(H)
    # The solver must reproduce the TRUE (discrete) spectrum of the
    # finite-difference matrix, E_k = 4 sin^2(k pi / 2(n+1)) / dx^2,
    # to solver accuracy.
    dx = 1.0 / (n + 1)
    k = np.arange(1, n + 1)
    exact_fd = 4.0 * np.sin(k * np.pi / (2 * (n + 1))) ** 2 / dx ** 2
    assert np.max(np.abs(w - exact_fd)) / np.max(exact_fd) < 1e-7
    # And the low-lying modes approach the continuum (k pi)^2 spectrum
    # (the physics), within the O(dx^4) discretization error.
    assert np.max(np.abs(w[:8] - exact[:8])) / np.max(exact[:8]) < 5e-2
    # Ground state is the first (lowest) eigenvalue and positive.
    assert w[0] > 0
    assert np.all(np.diff(w) > 0)


def test_phonon_chain_has_zero_mode():
    n = 40
    D, exact = le.phonon_chain(n=n)
    w, V = le.eigh_symmetric(D)
    # Free ends -> one zero (translation) mode, and it should be the lowest.
    assert abs(w[0]) < 1e-8
    # All others non-negative.
    assert np.all(w >= -1e-9)
    # Compare the non-zero spectrum to the analytic dispersion.
    nz = np.sort(w[1:])
    ref = np.sort(exact[1:])
    assert np.max(np.abs(nz - ref)) / np.max(ref) < 1e-4


def test_lanczos_extremes_of_large_sparse():
    n = 400
    # Random SPD tridiagonal-ish operator; Lanczos should nail the extremes.
    rng = np.random.default_rng(3)
    d = np.abs(rng.normal(size=n)) + 1.0
    e = 0.5 * rng.normal(size=n - 1)
    A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    A = A + A.T
    w_exact = np.linalg.eigvalsh(A)
    k = 5
    w_l, V_l = le.lanczos(A, k=k, n_iter=200)
    # Largest-magnitude eigenvalues captured.
    largest = np.sort(w_exact)[-k:][::-1]
    smallest = np.sort(w_exact)[:k]
    captured = np.sort(np.abs(w_l))
    extremes = np.sort(np.abs(np.concatenate([largest, smallest])))[-k:]
    assert np.max(np.abs(captured - extremes)) < 1e-3
