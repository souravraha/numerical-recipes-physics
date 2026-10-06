"""Graphene tight-binding model and edge/topological states.

Graphene's low-energy physics is the nearest-neighbour (NN) tight-binding
model on the honeycomb lattice with sublattices A and B.  The single-particle
Hamiltonian is
    H(k) = t |f(k)|,   f(k) = sum_{delta} exp(i k . delta),
giving the spectrum ``E(k) = +/- t |f(k)|`` -- the "Dirac cones" that touch at
the K and K' points, where ``f(K) = 0``.  This produces gapless, *linear*
dispersion ``E ~ v_F |k - K|`` and the massless-relativistic (Dirac)
quasiparticles behind graphene's anomalous optical conductivity and integer
quantum Hall effect (F. R. Wallace, 1947).

We also build the 1D SSH (Su-Schrieffer-Heeger) chain -- the canonical 1D
topological insulator with alternating strong/weak bonds.  In the
topological phase (t1 > t2) it hosts a pair of zero-energy *edge states*; in
the trivial phase (t1 < t2) it is fully gapped.  The spectrum of a bipartite
tight-binding lattice is symmetric about E = 0.

All eigenvalues are obtained with the shared ``linalg_eig.eigh_symmetric``.


References (see REFERENCES.md):
  - P. R. Wallace, Phys. Rev. 71, 396 (1947); 77, 39 (1950) [graphene, QHE].
  - A. H. Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009) [graphene review].
  - H. S. S. Su, J. R. Schrieffer, A. J. Heeger, Phys. Rev. Lett. 42, 714 (1979) [SSH chain].
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import linalg_eig as le

__all__ = [
    "honeycomb_f",
    "band_structure",
    "dirac_point",
    "linear_dispersion",
    "ssh_hamiltonian",
    "ssh_spectrum",
]


def _nn_vectors(a=1.0):
    """Nearest-neighbour vectors of the honeycomb lattice (a = 1 by default)."""
    return np.array([
        [0.0, a],
        [np.sqrt(3.0) * a / 2.0, -a / 2.0],
        [-np.sqrt(3.0) * a / 2.0, -a / 2.0],
    ])


def honeycomb_f(k, a=1.0):
    """Structure factor ``f(k) = sum_delta exp(i k.delta)``."""
    k = np.asarray(k, dtype=float)
    out = 0.0j
    for d in _nn_vectors(a):
        out += np.exp(1j * (k[0] * d[0] + k[1] * d[1]))
    return out


def band_structure(t=1.0, a=1.0, nk=201):
    """E(k) = t |f(k)| on a square k-grid spanning the BZ.  Returns KX, KY, E."""
    k = np.linspace(-np.pi / a, np.pi / a, nk)
    KX, KY = np.meshgrid(k, k)
    f = np.zeros(KX.shape, dtype=complex)
    for vec in _nn_vectors(a):
        f += np.exp(1j * (KX * vec[0] + KY * vec[1]))
    return KX, KY, t * np.abs(f)


def dirac_point(t=1.0, a=1.0, nk=401):
    """Locate K (min |f|) and return (k_K, E_K); the Dirac cone gives E_K ~ 0."""
    KX, KY, E = band_structure(t=t, a=a, nk=nk)
    i = int(np.argmin(E))
    return np.array([KX.flat[i], KY.flat[i]]), float(E.flat[i])


def linear_dispersion(t=1.0, a=1.0):
    """Check the linear E ~ v_F |k - K| near K.

    Returns ``(v_F, ratio)`` where ``ratio = E(2r)/E(r)`` (linearity => 2).
    """
    kK, _ = dirac_point(t=t, a=a, nk=501)
    def e_at(r, ang=0.35):
        k = kK + r * np.array([np.cos(ang), np.sin(ang)])
        return t * abs(honeycomb_f(k, a))
    r1, r2 = 0.02, 0.04
    e1, e2 = e_at(r1), e_at(r2)
    return float(e1 / r1), float(e2 / e1)


def ssh_hamiltonian(t1, t2, n_cells):
    """1D SSH chain of ``2 n_cells`` sites.

    Bonds (in a 0-indexed chain A_1-B_1-...-A_N-B_N):
        A_i --t2-- B_i --t1-- A_{i+1}
    so the strong intra-cell bond is ``t1`` and the weak inter-cell bond is
    ``t2``.  With open boundaries the two end atoms are dangling, so the
    topological phase ``t1 > t2`` hosts a pair of zero-energy edge states;
    the trivial phase ``t1 < t2`` is fully gapped (gap ``2|t1 - t2|``).
    """
    n = 2 * n_cells
    H = np.zeros((n, n))
    for i in range(n_cells):
        a, b = 2 * i, 2 * i + 1
        H[a, b] = H[b, a] = t2            # intra-cell (weak) bond A_i-B_i
        if i < n_cells - 1:
            H[b, a + 2] = H[a + 2, b] = t1  # inter-cell (strong) bond B_i-A_{i+1}
    return H


def ssh_spectrum(t1, t2, n_cells):
    """Sorted eigenvalues of the SSH chain (via shared eigh_symmetric)."""
    w, _ = le.eigh_symmetric(ssh_hamiltonian(t1, t2, n_cells))
    return np.asarray(w, dtype=float)
