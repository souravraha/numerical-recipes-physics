"""Fourier Grid Hamiltonian (FGH): a grid-based bound-state eigenvalue solver.

The FGH method is the standard tool for turning a one-dimensional (or
even-extended radial) Schrodinger problem into a dense eigenvalue problem on
a uniform grid.  The Hamiltonian is

    H = KE + diag(V(x)),

where the kinetic-energy operator is diagonalised in reciprocal space.  On a
grid of ``N`` points over a length ``L`` with spacing ``dx = L/N`` and
``delta_k = 2 pi / L``, the kinetic matrix elements are

    KE[i, j] = (hbar^2 / 2 m) <x_i|k^2|x_j>
             = (hbar^2 / 2 m) (2/N) sum_l T_l cos(2 pi l (i-j)/N),
    T_l = (l * delta_k)^2,

and depend only on the separation ``(i - j) mod N``.  That makes ``KE`` a real
symmetric *circulant* matrix, so its first row is a single inverse FFT of the
multiplier ``T`` and the whole matrix is a gather of that row -- O(N^2), and
identical to the classical per-row FFT construction to roundoff.

Two independent constructions are provided so each can cross-check the other:

  * ``"fft"``    -- the O(N^2) circulant build above (the FGH method);
  * ``"cosine"`` -- the direct cosine series, a separate code path.

The FGH method is spectrally (exponentially) accurate for smooth potentials
and handles hard-wall and even-extended radial problems on a periodic grid.
Combine it with any potential ``V(x)`` and diagonalise with the shared
dense symmetric eigensolver.


References (see REFERENCES.md):
  - C. J. Umrigar, J. P. Duan, X. Wang, S. J. Kim, M. J. Hoffmann,
    "The Fourier grid Hamiltonian method for bound state eigenvalues and
    eigenfunctions," J. Chem. Phys. 91, 3571 (1989). [FGH]
  - M. D. Feit, F. A. Fleischer, A. Steyer, J. Chem. Phys. 92, 3955 (1990).
    [grid-based Schrodinger propagation]
  - A. D. McRae, J. D. Dugger, "A comparison of the Fourier grid Hamiltonian
    method and the Lanczos method for quantum mechanical bound states,"
    Chem. Phys. Lett. 157, 157 (1989).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "kinetic_energy_matrix",
    "solve_bound_states",
    "normalize",
]


def _potential(V, x):
    """Accept a callable ``V(x)`` or a pre-evaluated array, return an array."""
    V = V(x) if callable(V) else V
    return np.asarray(V, dtype=float)


def kinetic_energy_matrix(x, hbar=1.0, m=1.0, algorithm="fft"):
    """The FGH kinetic-energy matrix ``KE[i,j] = (hbar^2/2m)<i|k^2|j>`` on grid ``x``.

    Parameters
    ----------
    x : array_like
        Uniform spatial grid (any count; the FFT path needs the grid spacing
        implied by ``L = x[-1] - x[0]``).
    hbar, m : float
        Reduced Planck constant and particle mass (atomic units by default).
    algorithm : {"fft", "cosine"}
        ``"fft"`` builds the real symmetric circulant from one inverse FFT of
        the multiplier (the FGH method, O(N^2)).  ``"cosine"`` evaluates the
        direct cosine series as an independent code path.

    Returns
    -------
    (N, N) real symmetric array.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    L = x[-1] - x[0]
    delta_k = 2.0 * np.pi / L
    pref = hbar**2 / (2.0 * m)

    if algorithm == "fft":
        # T[l] = (l * delta_k)^2 with l folded to the |k| magnitude, so the
        # circulant first row is exactly ifft(T).  Real & symmetric because
        # T is even, hence KE is a real symmetric circulant.
        k = np.arange(N)
        lmag = np.minimum(k, N - k)
        T = pref * (lmag * delta_k) ** 2
        first_row = np.fft.ifft(T).real
        i = np.arange(N)
        return first_row[(i[None, :] - i[:, None]) % N]

    if algorithm == "cosine":
        # Direct FGH sum: KE[i,j] = (2/N) sum_{l=1}^{n} T_l cos(2 pi l d/N)
        # with d = (i-j) mod N and T_0 = 0.  Depends only on d, so build the
        # generator once and gather.
        n = (N - 1) // 2
        l = np.arange(1, n + 1)[:, None]
        d = np.arange(N)[None, :]
        T = pref * (l * delta_k) ** 2
        first_row = (2.0 / N) * np.sum(np.cos(2.0 * np.pi * l * d / N) * T, axis=0)
        i = np.arange(N)
        return first_row[(i[None, :] - i[:, None]) % N]

    raise ValueError(f"unknown algorithm {algorithm!r}; use 'fft' or 'cosine'")


def solve_bound_states(V, x, hbar=1.0, m=1.0, algorithm="fft"):
    """Diagonalise ``H = KE + diag(V(x))`` on the grid ``x``.

    ``V`` may be a callable ``V(x)`` or a pre-evaluated potential array.
    Returns ``(eigenvalues, eigenvectors)`` ascending, eigenvectors as
    columns.  Columns are l2-normalised by the dense eigensolver; use
    :func:`normalize` to normalise each to ``sum |psi|^2 dx = 1``.
    """
    x = np.asarray(x, dtype=float)
    KE = kinetic_energy_matrix(x, hbar=hbar, m=m, algorithm=algorithm)
    H = KE + np.diag(_potential(V, x))
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def normalize(eigenvectors, x):
    """Normalise eigenvector columns to ``sum |psi|^2 dx = 1``."""
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0] if x.size > 1 else 1.0
    norms = np.sqrt(np.sum(np.abs(eigenvectors) ** 2, axis=0) * dx)
    return eigenvectors / norms
