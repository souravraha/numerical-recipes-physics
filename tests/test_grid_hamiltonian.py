import numpy as np

from numerical_recipes import grid_hamiltonian as gh


def test_ke_matrix_is_real_symmetric():
    # KE is a real symmetric circulant: KE == KE.T and purely real.
    x = np.linspace(-2.0, 2.0, 257)
    KE = gh.kinetic_energy_matrix(x, algorithm="fft")
    assert np.max(np.abs(KE - KE.T)) < 1e-10
    assert np.max(np.abs(KE)) > 0.0


def test_fft_and_cosine_agree():
    # Independent code paths (one inverse FFT vs the direct cosine series)
    # must give the same kinetic matrix to roundoff.
    x = np.linspace(-3.0, 3.0, 513)
    a = gh.kinetic_energy_matrix(x, algorithm="fft")
    b = gh.kinetic_energy_matrix(x, algorithm="cosine")
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-9


def test_harmonic_spectrum_matches_exact():
    # V = k x^2/2: E_n = (n + 1/2) hbar sqrt(k/m), the analytic reference.
    N, L, k = 513, 16.0, 4.0
    x = np.linspace(-L, L, N)
    w, _ = gh.solve_bound_states(lambda xx: k * xx**2 / 2.0, x, m=1.0)
    exact = (np.arange(6) + 0.5) * np.sqrt(k)
    assert np.max(np.abs(w[:6] - exact)) / np.max(exact) < 0.05


def test_square_well_below_continuum():
    # A finite well: every bound state sits below the continuum (E = 0) and
    # the first is the deepest (most negative).
    x = np.linspace(-8.0, 8.0, 513)
    V = np.where(np.abs(x) < 1.0, -25.0, 0.0)
    w, _ = gh.solve_bound_states(lambda xx: V, x)
    bound = w[w < 0.0]
    assert bound.size >= 3
    assert np.all(bound < 0.0)
    assert np.all(np.diff(bound) > 0.0)


def test_normalize_gives_unit_integral():
    # Normalising the columns gives sum |psi|^2 dx = 1 exactly.
    x = np.linspace(-3.0, 3.0, 513)
    _, vecs = gh.solve_bound_states(lambda xx: xx**2 / 2.0, x)
    vecs = gh.normalize(vecs, x)
    dx = x[1] - x[0]
    norms = np.sum(np.abs(vecs[:, :8]) ** 2, axis=0) * dx
    assert np.max(np.abs(norms - 1.0)) < 1e-9
