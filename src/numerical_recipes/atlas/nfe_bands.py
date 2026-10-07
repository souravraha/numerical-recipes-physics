"""Nearly-free-electron band structure (1D and 2D) from a weak periodic potential.

An electron in a weak periodic potential is expanded in plane waves over the
reciprocal-lattice vectors kept within a cutoff.  With the potential Fourier
coefficient ``U`` (the structure factor ``U_0`` set to zero), the Hamiltonian
at wavevector ``k`` has diagonal ``(k - K)^2`` and constant off-diagonal
coupling ``U``.  Its eigenvalues are the bands.  As the plane-wave cutoff
grows the band gap at the Brillouin-zone edge (``k = K/2``) converges
monotonically to the nearly-free-electron value (~0.462 ``U`` for ``U = 0.5``
in 1D), and the bands are symmetric about ``k = 0``.

Energies are scaled by ``hbar^2 K^2 / (2 m)`` (``K`` the reciprocal-lattice
vector).  This is the classic textbook "weak potential" model behind
Kittel's band picture (``onedimension.m`` / ``twodimensions.m``).


References (see REFERENCES.md):
  - C. Kittel, Introduction to Solid State Physics, 8th ed., Wiley (2005).
    [nearly-free-electron band model]
  - J. M. Ziman, Principles of the Theory of Solids, Cambridge University
    Press (1964). [plane-wave expansion in a weak potential]
  - N. W. Ashcroft, N. D. Mermin, Solid State Physics, Cengage (1976).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "bands_1d",
    "band_structure_1d",
    "bands_2d",
    "band_surface_2d",
]


def bands_1d(k, Nmax=5, U=0.5, K=1.0):
    """Ascending band energies of the 1D weak periodic potential at ``k``.

    The plane-wave basis is ``K, 2K, ..., +/- Nmax*K`` (``2 Nmax + 1`` states);
    diagonal ``-U`` (the ``U_0`` shift) plus constant off-diagonal ``U``.
    """
    kvalues = np.arange(-Nmax, Nmax + 1) * K
    h = np.diag((k - kvalues) ** 2 - U) + U * np.ones((len(kvalues), len(kvalues)))
    return np.linalg.eigvalsh(h)


def band_structure_1d(Nmax=5, U=0.5, K=1.0, n_k=1000):
    """``(k_values, energies)`` with ``energies`` of shape ``(n_k, 2*Nmax+1)``.

    Sweeps ``k`` across the first Brillouin zone ``[-K/2, K/2]``.
    """
    k_values = np.linspace(-K / 2, K / 2, n_k)
    energies = np.vstack([bands_1d(k, Nmax, U, K) for k in k_values])
    return k_values, energies


def bands_2d(kx, ky, Nmax=2, U=0.1, K=1.0):
    """Ascending band energies of the 2D weak periodic potential at ``(kx, ky)``.

    Square reciprocal lattice; the ``(2 Nmax + 1)^2`` plane waves inside the
    cutoff give a symmetric matrix of diagonal ``(kx-Gx)^2 + (ky-Gy)^2 - U``
    and constant off-diagonal ``U``.
    """
    g = np.arange(-Nmax, Nmax + 1) * K
    gx, gy = np.meshgrid(g, g)
    eps0 = (kx - gx) ** 2 + (ky - gy) ** 2
    n = eps0.size
    h = np.diag(eps0.ravel() - U) + U * np.ones((n, n))
    return np.linalg.eigvalsh(h)


def band_surface_2d(Nmax=2, U=0.1, K=1.0, n_k=50, n_bands=3):
    """The lowest ``n_bands`` bands as a surface over the first BZ.

    Returns ``(kx_grid, ky_grid, bands)`` with ``bands`` of shape
    ``(n_bands, n_k, n_k)``.
    """
    k_values = np.linspace(-K / 2, K / 2, n_k)
    kx_grid, ky_grid = np.meshgrid(k_values, k_values)
    bands = np.empty((n_bands,) + kx_grid.shape)
    for b in range(n_bands):
        bands[b] = np.vectorize(
            lambda x, y: bands_2d(x, y, Nmax, U, K)[b]
        )(kx_grid, ky_grid)
    return kx_grid, ky_grid, bands
