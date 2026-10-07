"""Hartree and restricted Hartree-Fock ground states: helium and beryllium.

Two closed-shell atoms, the standard first-principles benchmark for
mean-field electronic-structure methods (atomic units, hbar = m_e = e = 1):

  * **helium** (``1s^2``, ``z = 2``) -- self-consistent *Hartree* (direct
    field only, no exchange).  The radial 1s orbital is the even-extension
    bound state of the effective potential ``-z/r + V_Hartree(r)``, solved on
    the shared ``grid_hamiltonian`` grid; the total energy is
    ``2 e_1s - self_energy``.  It sits between the exact nonrelativistic
    value (-2.9037 Ha) and Hartree-Fock (-2.8617 Ha).

  * **beryllium** (``1s^2 2s^2``, ``z = 4``) -- *restricted Hartree-Fock*
    (Roothaan equations) over six Slater-type radial basis functions with the
    Clementi exponents.  The radial kinetic operator uses a spline second
    derivative, the two-electron integrals use the ``1/max(r, r')`` kernel,
    and the self-consistent Fock matrix is diagonalised as a generalised
    eigenvalue problem against the overlap matrix.  The total energy is about
    -14.667 Ha.

Both are deterministic given their grid and tolerance.


References (see REFERENCES.md):
  - R. P. Feynman, Phys. Rev. 56, 340 (1939). doi:10.1103/PhysRev.56.340
    -- the helium total energy.
  - S. Fermi, "Sulla struttura degli atomi," (1930); E. C. Stoner, Proc. R.
    Soc. A 105, 582 (1924). -- Hartree self-consistent fields.
  - B. Roothaan, Rev. Mod. Phys. 23, 69 (1951). doi:10.1103/RevModPhys.23.69
    -- the Roothaan (matrix) Hartree-Fock equations.
  - E. Clementi, C. Raimondi, J. Chem. Phys. 68, 44 (1973).
    -- Slater-type orbitals and Clementi exponents used for Be.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as spl
import scipy.interpolate as fit
from scipy.integrate import simpson

from numerical_recipes import grid_hamiltonian as gh

__all__ = [
    "he_hartree",
    "be_roothaan",
]


# --------------------------------------------------------------------------- #
# Helium -- self-consistent Hartree (direct field only)
# --------------------------------------------------------------------------- #
def _build_half_grid(n, r_max):
    """``n`` points in ``(0, r_max]`` (the positive radial half-line)."""
    return np.linspace(-r_max, r_max, 2 * n)[n:]


def _radial_norm(r, p):
    """Normalisation constant ``sqrt(int |p|^2 dr)`` on the half-line."""
    return np.sqrt(simpson(np.abs(p) ** 2, dx=r[1] - r[0]))


def _hartree_potential(r, p):
    """Electron-electron potential ``V(r) = int |p(r')|^2 / max(r, r') dr'``.

    ``p`` is normalised to unit radial norm.  Evaluated in O(n) by splitting
    the integral at ``r' = r`` and using cumulative trapezoid sums:
    ``V(r) = F(r)/r + (G_total - G(r))``.
    """
    p = p / _radial_norm(r, p)
    f = np.abs(p) ** 2
    dr = r[1] - r[0]
    f_trap = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dr)))
    fr_trap = np.concatenate(([0.0], np.cumsum(0.5 * ((f / r)[1:] + (f / r)[:-1]) * dr)))
    return f_trap / r + (fr_trap[-1] - fr_trap)


def he_hartree(n=256, r_max=10.0, tol=1e-5, max_iter=200, z=2.0):
    """Self-consistent Hartree ground state of a ``1s^2`` atom (helium, ``z=2``).

    Returns a dict with ``r`` (grid), ``p`` (normalised 1s radial
    wavefunction), ``e_1s`` (orbital energy) and ``e_total`` (total energy).
    """
    r = _build_half_grid(n, r_max)
    r_ext = np.concatenate((-r[::-1], r))
    ke = gh.kinetic_energy_matrix(r_ext)                 # reuse the FGH primitive
    core = -z / r

    def solve(v_half):
        v_ext = np.concatenate((v_half[::-1], v_half))
        eig, vecs = gh.solve_bound_states(lambda xx: v_ext, r_ext)
        return eig[1], vecs[len(r):, 1]                  # 1s = index 1

    e, p = solve(core)
    for _ in range(max_iter):
        v = core + _hartree_potential(r, p)
        e_new, p_new = solve(v)
        err = np.max(np.abs(np.abs(p_new) - np.abs(p)))
        e, p = e_new, p_new
        if err < tol:
            break

    p = p / _radial_norm(r, p)
    self_energy = simpson(np.abs(p) ** 2 * _hartree_potential(r, p), dx=r[1] - r[0])
    e_total = 2.0 * e - self_energy
    return {"r": r, "p": p, "e_1s": e, "e_total": e_total}


# --------------------------------------------------------------------------- #
# Beryllium -- restricted Hartree-Fock (Roothaan equations)
# --------------------------------------------------------------------------- #
_Z = 4.0                    # Be nuclear charge
_N_OCC = 2                  # two doubly-occupied (closed) s orbitals
_BASIS = [                  # six Slater-type radial basis: r^power * exp(-zeta r)
    (1, 3.47116),
    (1, 6.36861),
    (2, 0.77820),
    (2, 0.94067),
    (2, 1.48725),
    (2, 2.71830),
]


def _be_basis_functions(r):
    """Six normalised Slater-type radial basis functions on ``r``."""
    out = []
    for power, zeta in _BASIS:
        phi = (r ** power) * np.exp(-zeta * r)
        out.append(phi / _radial_norm(r, phi))
    return out


def _second_derivative(r, p):
    """Second derivative of ``p`` on the uniform grid ``r`` (spline)."""
    tck = fit.splrep(r, p, k=5)
    return fit.splev(r, tck, der=2)


def _be_core_matrix(r, basis):
    """One-electron matrix ``<p_i | h1 | p_j>`` with ``h1 = -0.5 d^2/dr^2 - Z/r``."""
    nb = len(basis)
    h1 = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            d2 = _second_derivative(r, basis[j])
            h1[i, j] = simpson(basis[i] * (-0.5 * d2 - (_Z / r) * basis[j]), dx=r[1] - r[0])
    return h1


def _be_two_electron(r, basis):
    """Four-index integrals ``h2[a,b,c,d] = (a c | b d)`` with the 1/max(r,r') kernel.

    ``(alpha beta | gamma delta) = int int phi_a(r1) phi_b(r1)
    phi_c(r2) phi_d(r2) / max(r1, r2) dr1 dr2``.
    """
    nb = len(basis)
    n = len(r)
    dr2 = (r[1] - r[0]) ** 2
    inv = 1.0 / r
    rgt = np.where(np.arange(n)[:, None] >= np.arange(n)[None, :], inv[:, None], inv[None, :])
    g = np.empty((nb, nb, n))
    for b in range(nb):
        for d in range(nb):
            g[b, d] = rgt @ (basis[b] * basis[d])
    h2 = np.zeros((nb, nb, nb, nb))
    for a in range(nb):
        for c in range(nb):
            fa = basis[a] * basis[c]
            for b in range(nb):
                for d in range(nb):
                    h2[a, b, c, d] = np.dot(fa, g[b, d]) * dr2
    return h2


def _be_total_energy(c, h1, h2):
    """RHF total energy ``2 sum h_ii + sum_{ij} (2 J_ij - K_ij)``."""
    nb = h1.shape[0]

    def two_elec(ci, cj, kind):
        acc = 0.0
        for mu in range(nb):
            for nu in range(nb):
                for lam in range(nb):
                    for sig in range(nb):
                        w = h2[mu, lam, nu, sig]           # (mu nu | lam sig)
                        if kind == "J":
                            acc += ci[mu] * ci[nu] * cj[lam] * cj[sig] * w
                        else:
                            acc += ci[mu] * cj[nu] * cj[lam] * ci[sig] * w
        return acc

    e1 = 2.0 * sum(c[i] @ h1 @ c[i] for i in range(_N_OCC))
    ee = 0.0
    for i in range(_N_OCC):
        for j in range(_N_OCC):
            ee += 2.0 * two_elec(c[i], c[j], "J") - two_elec(c[i], c[j], "K")
    return e1 + ee


def be_roothaan(n=256, r_min=1e-10, r_max=20.0, tol=1e-3, max_iter=100, verbose=False):
    """Restricted Hartree-Fock ground state of beryllium.

    Returns a dict with ``r``, ``basis``, ``energies`` (2 occupied orbital
    energies, ascending), ``coefficients`` (2 x nb) and ``e_total``.
    """
    r = np.linspace(r_min, r_max, n)
    basis = _be_basis_functions(r)
    nb = len(basis)

    h1 = _be_core_matrix(r, basis)
    s = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            s[i, j] = simpson(basis[i] * basis[j], dx=r[1] - r[0])
    h2 = _be_two_electron(r, basis)

    def fock(c):
        ee = np.zeros((nb, nb))
        for p in range(nb):
            for m in range(nb):
                acc = 0.0
                for i in range(_N_OCC):
                    for n_idx in range(nb):
                        for q_idx in range(nb):
                            acc += c[i, n_idx] * c[i, q_idx] * (
                                2.0 * h2[p, q_idx, m, n_idx] - h2[q_idx, p, m, n_idx])
                ee[p, m] = acc
        return h1 + ee

    def solve(f):
        eigvals, eigvecs = spl.eigh(f, s)
        return eigvals[:_N_OCC], eigvecs[:, :_N_OCC].T

    e, c = solve(fock(np.tile(np.ones((_N_OCC, nb)), 1)))
    for it in range(max_iter):
        e_new, c_new = solve(fock(c))
        c_new = np.array([c_new[i] / _radial_norm(r, c_new[i] @ basis) for i in range(_N_OCC)])
        err = np.max(np.abs(e_new - e))
        e, c = e_new, c_new
        if verbose:
            print(f"iter {it:2d}: Err = {err:.3e}  E = {list(np.round(e, 6))}")
        if err < tol:
            break

    e_total = _be_total_energy(c, h1, h2)
    return {
        "r": r,
        "basis": basis,
        "energies": e,
        "coefficients": c,
        "e_total": e_total,
    }
