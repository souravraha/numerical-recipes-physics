"""2D Ising model: finite-size scaling and critical exponents.

At the critical point ``T = Tc`` the 2D Ising model is scale-invariant, so
thermodynamic observables are power laws in the linear size ``L``:

    susceptibility   chi(L) ~ L^{gamma/nu}
    specific heat    C(L)   ~ L^{alpha/nu}

with the exact 2D Ising exponents ``nu = 1, gamma = 7/4, alpha = 0`` (the
specific heat has only a logarithmic divergence).  The correlation length
``xi ~ L`` diverges at ``Tc``, so a finite lattice "sees" a correlation length
of order ``L`` and the exponents are read off as the log-log slope of
``chi(L)`` across a sequence of lattice sizes.

We sample each size with the shared cluster Monte Carlo (``mcmc_ising``)
cluster algorithm (which removes critical slowing down at ``Tc``), estimate
``chi`` and ``C``, and verify: (a) ``chi`` grows with ``L`` and peaks at
``Tc``; (b) the finite-size slope of ``chi(L)`` is the non-trivial Ising
exponent ``gamma/nu ~ 7/4``; (c) the specific heat stays sub-power (``alpha
~ 0``).


References (see REFERENCES.md):
  - L. Onsager, Phys. Rev. 65, 117 (1944) [exact Tc].
  - J. Cardy, J. Phys. A 17, L455 (1984) [2D Ising exponents].
  - H. E. Stanley, Introduction to Phase Transitions and Critical Phenomena (1971).
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import mcmc_ising as mi

__all__ = ["onsager_tc", "scaling_data", "loglog_slope", "fit_exponents"]


def onsager_tc(J=1.0):
    """Exact critical temperature ``kTc/J = 2/ln(1 + sqrt(2))``."""
    return float(mi.onsager_tc(J=J, k_B=1.0))


def scaling_data(T, sizes, seed=0, n_sweeps=900, equilibration=150):
    """Susceptibility and specific heat vs lattice size ``L`` at temperature T.

    Uses Wolff cluster MC (fast at ``Tc``).  Returns ``(L, chi, C)``.
    """
    sizes = list(sizes)
    chi, C = [], []
    for L in sizes:
        r = mi.ising_thermo(L, T, n_sweeps=n_sweeps, equilibration=equilibration,
                            seed=seed, method="wolff")
        chi.append(r["susceptibility_per_spin"])
        C.append(r["specific_heat_per_spin"])
    return np.asarray(sizes, dtype=float), np.asarray(chi), np.asarray(C)


def loglog_slope(x, y):
    """Slope of log y vs log x (the finite-size scaling exponent)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def fit_exponents(T=None, sizes=(8, 16, 32, 64), seed=0, **kw):
    """Estimate ``gamma/nu`` (from chi ~ L^{gamma/nu}) and ``alpha/nu``.

    Returns a dict with the two exponent estimates and the raw data.
    """
    T = onsager_tc() if T is None else T
    L, chi, C = scaling_data(T, sizes, seed=seed, **kw)
    return {
        "gamma_over_nu": loglog_slope(L, chi),
        "alpha_over_nu": loglog_slope(L, C),
        "L": L, "chi": chi, "C": C,
    }
