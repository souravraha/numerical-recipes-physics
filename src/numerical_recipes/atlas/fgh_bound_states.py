"""FGH bound-state case studies: harmonic well, square well, hydrogen, central.

Four canonical bound-state problems solved with the shared Fourier Grid
Hamiltonian primitive (``grid_hamiltonian``), each checked against an
independent analytic reference:

  * 1D harmonic oscillator   -- E_n = (n + 1/2) hbar omega (NumPy reference);
  * finite square well       -- the even/odd transcendental equations, solved
    with ``scipy.optimize.brentq`` as the independent reference;
  * hydrogen 1s              -- the even-extension trick maps the radial
    s-wave problem to the full line; the 1s energy is ``-Z^2/2`` (a.u.);
  * 3D truncated harmonic    -- a central potential with angular momentum
    ``l``; the s-wave (``l = 0``) ground state sits at ``E = 1.5 hbar w``.

The FGH method is spectrally accurate for the smooth potentials here, so a
modest grid already reproduces the analytic spectra to well below 1 %.


References (see REFERENCES.md):
  - C. J. Umrigar et al., J. Chem. Phys. 91, 3571 (1989). [FGH method]
  - C. Cohen-Tannoudji, B. Diu, F. Laloe, Quantum Mechanics, Wiley (1977).
    [harmonic oscillator, square well, hydrogen atom]
  - M. D. Feit, F. A. Fleischer, A. Steyer, J. Chem. Phys. 92, 3955 (1990).
    [grid-based radial problems]
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from numerical_recipes import grid_hamiltonian as gh

__all__ = [
    "harmonic_energies",
    "square_well_states",
    "solve_harmonic",
    "solve_square_well",
    "hydrogen_radial",
    "central_potential",
]


# --------------------------------------------------------------------------- #
# Analytic references (independent code paths)
# --------------------------------------------------------------------------- #
def harmonic_energies(n_states, k=1.0, m=1.0, hbar=1.0):
    """Analytic energies ``(n + 1/2) hbar omega`` of ``V = k x^2 / 2``."""
    omega = np.sqrt(k / m)
    return (np.arange(n_states) + 0.5) * hbar * omega


def _square_well_roots(V0, a):
    """Roots (E' measured from the well bottom) of the even/odd equations."""
    roots = []
    top = np.sqrt(2.0 * V0) * a

    def branch(lo, hi, fun):
        if hi <= lo:
            return
        try:
            alpha = brentq(fun, lo, hi)
        except ValueError:
            return
        E = alpha**2 / 2.0
        if E < V0:
            roots.append(E)

    # Even parity: alpha*a in (p*pi, (p+1/2)*pi);  alpha*tan(alpha a) = sqrt(2(V0-E')).
    p = 0
    while p * np.pi < top - 1e-9:
        lo = p * np.pi + 1e-9
        hi = min((p + 0.5) * np.pi - 1e-9, top - 1e-9)
        branch(lo, hi, lambda al: al * np.tan(al * a) - np.sqrt(max(2.0 * (V0 - al**2 / 2.0), 0.0)))
        p += 1
    # Odd parity: alpha*a in ((p-1/2)*pi, p*pi);  alpha/tan(alpha a) = -sqrt(2(V0-E')).
    p = 1
    while (p - 0.5) * np.pi < top - 1e-9:
        lo = (p - 0.5) * np.pi + 1e-9
        hi = min(p * np.pi - 1e-9, top - 1e-9)
        branch(lo, hi, lambda al: al / np.tan(al * a) + np.sqrt(max(2.0 * (V0 - al**2 / 2.0), 0.0)))
        p += 1
    return roots


def square_well_states(V0, a, n_states=None):
    """Analytic bound states of a finite square well ``V = -V0`` for ``|x| < a``.

    Convention: the well bottom is at ``-V0`` and ``V -> 0`` outside, so every
    bound energy is negative.  The even/odd transcendental equations are solved
    in terms of the kinetic energy measured from the well bottom (``E' = E+V0``)
    and shifted back at the end -- ``scipy.optimize.brentq`` on each single-
    root branch.
    """
    energies = sorted(e - V0 for e in _square_well_roots(V0, a))
    return energies[: (n_states or 100)]


# --------------------------------------------------------------------------- #
# FGH solves (built on the shared grid_hamiltonian primitive)
# --------------------------------------------------------------------------- #
def solve_harmonic(n_states=6, k=4.0, m=1.0, hbar=1.0, N=513, L=16.0):
    """FGH spectrum of ``V = k x^2 / 2`` on ``[-L, L]``.

    Returns a dict with ``x``, ``energies`` (ascending), unit-normalised
    ``states`` (columns), and the analytic ``exact`` reference.
    """
    x = np.linspace(-L, L, N)
    V = k * x**2 / 2.0
    eig, vecs = gh.solve_bound_states(lambda xx: V, x, hbar=hbar, m=m)
    vecs = gh.normalize(vecs, x)
    return {
        "x": x,
        "energies": eig,
        "states": vecs,
        "exact": harmonic_energies(n_states, k=k, m=m, hbar=hbar),
    }


def solve_square_well(V0, a, n_states=4, N=513, L=16.0):
    """FGH spectrum of a finite square well ``V = -V0`` for ``|x| < a``.

    Returns a dict with ``x``, ``energies`` (ascending, negative), unit-
    normalised ``states``, and the analytic ``exact`` reference.
    """
    x = np.linspace(-L, L, N)
    V = np.where(np.abs(x) < a, -V0, 0.0)
    eig, vecs = gh.solve_bound_states(lambda xx: V, x)
    vecs = gh.normalize(vecs, x)
    return {
        "x": x,
        "energies": eig,
        "states": vecs,
        "exact": square_well_states(V0, a, n_states),
    }


def hydrogen_radial(N=256, r_max=10.0, hbar=1.0, m=1.0, charge=1.0):
    """Hydrogen 1s radial state via the even-extension FGH trick.

    The radial s-wave equation is mapped to the full line by mirroring
    ``r > 0`` to ``x < 0``; the Coulomb ``-Z/r`` becomes the even ``-Z/|x|``.
    The 1s state is the first *physical* (even-parity) solution, at index 1
    (index 0 is the spurious odd-parity state).  Returns a dict with ``r``,
    ``E_1s``, the normalised radial amplitude ``R_1s``, the analytic
    ``theory_E = -Z^2/2``, and the analytic ``theory_density``.
    """
    r = np.linspace(-r_max, r_max, 2 * N)[N:]           # r > 0 half grid
    r_ext = np.concatenate((-r[::-1], r))
    V_half = -charge / r
    V_ext = np.concatenate((V_half[::-1], V_half))
    eig, vecs = gh.solve_bound_states(lambda xx: V_ext, r_ext, hbar=hbar, m=m)
    E_1s = float(eig[1])
    R_1s = vecs[len(r):, 1]
    dr = r[1] - r[0]
    R_1s = R_1s / np.sqrt(np.sum(np.abs(R_1s) ** 2) * dr)
    theory_E = -0.5 * charge**2
    theory_density = (charge**3 / np.pi) * np.exp(-2.0 * charge * r)
    return {
        "r": r,
        "E_1s": E_1s,
        "R_1s": R_1s,
        "theory_E": theory_E,
        "theory_density": theory_density,
    }


def central_potential(N=512, w=10.0, a=4.0, l=0, mu=1.0, hbar=1.0,
                      r0=-2.0, r1=2.0, n_states=3):
    """Bound states of a truncated 3D harmonic central potential.

    ``V(r) = mu (w min(|r|, a))^2 / 2`` plus the centrifugal term
    ``l(l+1) hbar^2 / (2 mu r^2)``, even-extended to the full line.  Parity
    selects the physical radial states (``(-1)^l``): the s-wave (``l = 0``)
    ground state sits at the analytic ``E = 1.5 hbar w`` of the 3D harmonic.
    Returns a dict with ``r``, ``E`` (parity-filtered, ascending), unit-
    normalised radial amplitudes ``R`` (on ``r > 0``), the well peak ``V0``,
    and the analytic ``theory_E0``.
    """
    r = np.linspace(r0, r1, N)
    centrifugal = l * (l + 1) * hbar**2 / (2.0 * mu) * np.where(
        np.abs(r) > 1e-12, 1.0 / r**2, 1e12)
    well = mu * (w * np.minimum(np.abs(r), a)) ** 2 / 2.0
    V = centrifugal + well
    V0 = l * (l + 1) * hbar**2 / (2.0 * mu * a**2) + mu * (w * a) ** 2 / 2.0

    eig, vecs = gh.solve_bound_states(lambda xx: V, r, hbar=hbar, m=mu)
    mask = eig < V0
    E_all = eig[mask]
    R_all = [vecs[N // 2:, i] for i in np.where(mask)[0]]
    if l % 2 == 0:
        E, R = E_all[1::2], R_all[1::2]
    else:
        E, R = E_all[0::2], R_all[0::2]
    E = E[:n_states]
    R = R[:n_states]
    dr = r[1] - r[0]
    R = [Ri / np.sqrt(np.sum(np.abs(Ri) ** 2) * dr) for Ri in R]
    return {
        "r": r,
        "E": E,
        "R": R,
        "V0": V0,
        "theory_E0": 1.5 * hbar * w if l == 0 else None,
    }
