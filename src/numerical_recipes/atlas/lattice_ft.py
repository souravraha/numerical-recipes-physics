"""Lattice field theory: 2D compact U(1) gauge theory (lattice QED).

Compact U(1) lattice gauge theory places a unit link variable
``U_mu(x) = exp(i theta_mu(x))`` on every oriented bond of a 2D lattice.  The
Wilson action is
    S = beta * sum_plaquettes (1 - Re P_p),     P_p = product of the 4 links,
and the plaquette expectation ``<Re P_p>`` is the key observable: at weak
coupling (large beta) the links align and ``<Re P_p> -> 1`` (Coulomb phase),
while at strong coupling (small beta) the plaquettes are disordered and
``<Re P_p>`` drops.  The strong-coupling series expansion gives the classic
result ``<Re P_p> = 1 - 1/(2 beta) + O(beta^-2)``.

We sample the link phases with a Metropolis single-link update and verify the
strong-coupling expansion and the monotonic alignment ``<Re P_p>(beta)``.


References (see REFERENCES.md):
  - K. G. Wilson, Phys. Rev. D 10, 2445 (1974) [lattice gauge theory].
  - M. Creutz, Quarks and Gluons (1983) [lattice QCD / compact QED].
  - A. M. Polyakov, Nucl. Phys. B 30, 282 (1970) [duality / confinement].
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "init_links",
    "plaquette_sum",
    "plaquette_expectation",
    "strong_coupling_expectation",
]


def init_links(L, seed=0):
    """Random link phases on an L x L 2D lattice: shape (2, L, L) in [0, 2pi)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=(2, L, L))


def plaquette_sum(theta, L):
    """Sum of Re P over all (0,1)-plane plaquettes (periodic BCs).

    P(x,y) = U_0(x,y) U_1(x+1,y) U_0(x,y+1)^{-1} U_1(x,y)^{-1}.
    """
    two_pi = 2.0 * np.pi
    s = 0.0
    for x in range(L):
        xp = (x + 1) % L
        for y in range(L):
            yp = (y + 1) % L
            ang = theta[0, x, y] + theta[1, xp, y] - theta[0, x, yp] - theta[1, x, y]
            s += np.cos(ang % two_pi)
    return s


def plaquette_expectation(L=8, beta=2.0, n_sweeps=2000, equilibration=300, seed=0):
    """Monte Carlo estimate of ``<Re P_p>`` (Metropolis single-link updates).

    Returns ``(mean_plaquette, acceptance_rate)``.
    """
    rng = np.random.default_rng(seed)
    two_pi = 2.0 * np.pi
    theta = init_links(L, seed=seed)
    base = plaquette_sum(theta, L)
    acc = 0
    samples = []
    for step in range(n_sweeps):
        mu = int(rng.integers(0, 2))
        x = int(rng.integers(0, L))
        y = int(rng.integers(0, L))
        old = theta[mu, x, y]
        new = (old + rng.normal(0.0, 0.6)) % two_pi   # wide-enough step for the stiff high-beta phase
        theta[mu, x, y] = new
        cand = plaquette_sum(theta, L)
        theta[mu, x, y] = old
        if np.log(rng.random()) < beta * (cand - base):
            theta[mu, x, y] = new
            base = cand
            acc += 1
        if step >= equilibration:
            samples.append(base / (L * L))
    return float(np.mean(samples)), acc / n_sweeps


def strong_coupling_expectation(beta):
    """Strong-coupling 1-loop: ``<Re P_p> ~= 1 - 1/(2 beta)`` (large beta)."""
    return 1.0 - 1.0 / (2.0 * beta)
