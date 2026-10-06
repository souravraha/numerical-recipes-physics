"""Monte Carlo for the 2D Ising model -- the canonical phase transition.

The 2D Ising model is *the* worked example of statistical mechanics: N spins
on a square lattice, energy ``E = -J sum <ij> s_i s_j``, and the Boltzmann
weight ``exp(-E / kT)``.  It has a sharp (in the thermodynamic limit) second-
order phase transition at ``kT_c / J = 2 / ln(1 + sqrt(2)) ~ 2.269`` (Onsager's
exact result), where the magnetisation vanishes and the specific heat peaks.

We sample the equilibrium distribution with two Markov-chain Monte Carlo
schemes:

  * **Metropolis** -- single-spin flips accepted with probability
    ``min(1, exp(-dE / kT))``.  Simple, detailed-balance, but slow near Tc
    because spins flip one at a time (critical slowing down).
  * **Wolff cluster** -- flip whole *clusters* of aligned spins at once.
  *Much* more efficient near Tc; the clustering *is* the physics.

Both obey detailed balance and converge to the same Boltzmann distribution --
a great lesson that the *path* through phase space can change the speed by
orders of magnitude without changing the answer.


References (see REFERENCES.md):
  - N. Metropolis et al., J. Chem. Phys. 21, 1087 (1953) [Metropolis].
  - U. Wolff, Phys. Rev. Lett. 56, 116 (1986) [Wolff cluster].
  - R. H. Swendsen, J.-S. Wang, Phys. Rev. Lett. 58, 86 (1987) [Swendsen-Wang].
  - L. Onsager, Phys. Rev. 65, 117 (1944) [exact critical temperature]."""

from __future__ import annotations

import numpy as np

__all__ = [
    "ising_energy",
    "ising_magnetization",
    "metropolis_sweep",
    "wolff_sweep",
    "sw_sweep",
    "ising_thermo",
    "onsager_tc",
]


def onsager_tc(J=1.0, k_B=1.0):
    """Onsager's exact critical temperature: kTc/J = 2 / ln(1 + sqrt(2))."""
    return J / (k_B * 0.5 * np.log(1.0 + np.sqrt(2.0)))


def _neighbors(n, periodic=True):
    """Integer neighbour offsets for an n x n lattice."""
    if periodic:
        return [
            lambda i, j: ((i + 1) % n, j),
            lambda i, j: ((i - 1) % n, j),
            lambda i, j: (i, (j + 1) % n),
            lambda i, j: (i, (j - 1) % n),
        ]
    return [
        lambda i, j: (i + 1, j),
        lambda i, j: (i - 1, j),
        lambda i, j: (i, j + 1),
        lambda i, j: (i, j - 1),
    ]


def ising_energy(s, J=1.0, periodic=True):
    """Total energy E = -J sum <ij> s_i s_j of a spin configuration."""
    s = np.asarray(s)
    n = s.shape[0]
    E = 0.0
    for i in range(n):
        for j in range(n):
            si = s[i, j]
            if periodic:
                E += -J * si * (s[(i + 1) % n, j] + s[(i - 1) % n, j]
                               + s[i, (j + 1) % n] + s[i, (j - 1) % n])
            else:
                if i + 1 < n:
                    E += -J * si * s[i + 1, j]
                if i - 1 >= 0:
                    E += -J * si * s[i - 1, j]
                if j + 1 < n:
                    E += -J * si * s[i, j + 1]
                if j - 1 >= 0:
                    E += -J * si * s[i, j - 1]
    return E / 2.0  # each bond counted twice


def ising_magnetization(s):
    """Magnetisation per spin, M = (1/N) sum s_i."""
    s = np.asarray(s)
    return float(np.sum(s) / s.size)


def metropolis_sweep(s, kT, J=1.0, rng=None, periodic=True):
    """One Metropolis sweep (one attempted flip per spin).

    Returns the updated spin array (modified in place) and the number of
    accepted flips.
    """
    rng = rng if rng is not None else np.random.default_rng()
    s = np.asarray(s)
    n = s.shape[0]
    accepted = 0
    for i in range(n):
        for j in range(n):
            # dE from flipping s[i, j]:  dE = 2 J s[i,j] * (sum of neighbours)
            if periodic:
                nbr = s[(i + 1) % n, j] + s[(i - 1) % n, j] + s[i, (j + 1) % n] + s[i, (j - 1) % n]
            else:
                nbr = 0.0
                if i + 1 < n:
                    nbr += s[i + 1, j]
                if i - 1 >= 0:
                    nbr += s[i - 1, j]
                if j + 1 < n:
                    nbr += s[i, j + 1]
                if j - 1 >= 0:
                    nbr += s[i, j - 1]
            dE = 2.0 * J * s[i, j] * nbr
            if dE <= 0.0 or np.log(rng.random()) < -dE / kT:
                s[i, j] *= -1.0
                accepted += 1
    return s, accepted


def wolff_sweep(s, kT, J=1.0, rng=None, periodic=True):
    """One Wolff cluster sweep: grow a cluster from a random seed, flip it.

    A spin joins the cluster with probability ``1 - exp(-J s_i s_c / kT)``.
    The whole cluster is flipped at once (detailed balance holds).
    """
    rng = rng if rng is not None else np.random.default_rng()
    s = np.asarray(s)
    n = s.shape[0]
    i0, j0 = rng.integers(0, n), rng.integers(0, n)
    cluster = [(i0, j0)]
    in_cluster = np.zeros((n, n), dtype=bool)
    in_cluster[i0, j0] = True
    # Wolff cluster-growth probability.  With our energy E = -J sum s_i s_j,
    # flipping an aligned boundary bond changes E by 2J, so detailed balance
    # gives p = 1 - exp(-2J/kT).
    p = 1.0 - np.exp(-2.0 * J / kT) if kT > 0 else 1.0

    stack = [(i0, j0)]
    while stack:
        i, j = stack.pop()
        si = s[i, j]
        nbrs = []
        if periodic:
            nbrs = [((i + 1) % n, j), ((i - 1) % n, j), (i, (j + 1) % n), (i, (j - 1) % n)]
        else:
            if i + 1 < n:
                nbrs.append((i + 1, j))
            if i - 1 >= 0:
                nbrs.append((i - 1, j))
            if j + 1 < n:
                nbrs.append((i, j + 1))
            if j - 1 >= 0:
                nbrs.append((i, j - 1))
        for (ni, nj) in nbrs:
            if not in_cluster[ni, nj] and s[ni, nj] == si:
                if kT <= 0 or rng.random() < p:
                    in_cluster[ni, nj] = True
                    cluster.append((ni, nj))
                    stack.append((ni, nj))
    for (i, j) in cluster:
        s[i, j] *= -1.0
    return s, len(cluster)


def ising_thermo(n=16, T=2.2, n_sweeps=2000, equilibration=500, seed=0, J=1.0, method="metropolis", periodic=True):
    """Equilibrate + sample an n x n Ising lattice; return thermodynamic data.

    Returns a dict with mean energy/spin, mean |magnetisation|/spin, specific
    heat per spin C = <E^2> - <E>^2 / (kT^2 N), and susceptibility per spin.
    """
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, size=(n, n)) * 2 - 1  # random +-1
    sweep = metropolis_sweep if method == "metropolis" else wolff_sweep
    N = n * n
    energies = []
    magnets = []
    for sw in range(n_sweeps):
        s, _ = sweep(s, T, J=J, rng=rng, periodic=periodic)
        if sw < equilibration:
            continue
        E = ising_energy(s, J=J, periodic=periodic)
        M = abs(ising_magnetization(s))
        energies.append(E)
        magnets.append(M)
    E = np.asarray(energies)
    M = np.asarray(magnets)
    mean_E = E.mean()
    mean_M = M.mean()
    C = (np.var(E) / (T**2 * N))
    chi = (np.mean(M**2) - mean_M**2) * N
    return {
        "energy_per_spin": float(mean_E / N),
        "magnetization_per_spin": float(mean_M),
        "specific_heat_per_spin": float(C),
        "susceptibility_per_spin": float(chi),
        "n_samples": len(energies),
    }


def sw_sweep(s, kT, J=1.0, rng=None, periodic=True):
    """One Swendsen-Wang cluster sweep.

    Build *all* clusters at once by percolating same-spin bonds with
    probability ``p = 1 - exp(-2 J / kT)``, then flip every cluster with
    probability 1/2.  This generalises the Wolff sweep (which grows a single
    cluster from one seed) to the whole configuration: every cluster is
    identified simultaneously and each is independently flipped.  It obeys
    detailed balance, converges to the same Boltzmann distribution, and is the
    standard cluster algorithm for the Ising model.

    Returns the updated spin array (modified in place) and the number of
    clusters.
    """
    rng = rng if rng is not None else np.random.default_rng()
    s = np.asarray(s)
    n = s.shape[0]
    p = 1.0 - np.exp(-2.0 * J / kT) if kT > 0 else 1.0

    parent = {(i, j): (i, j) for i in range(n) for j in range(n)}
    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:          # path compression
            parent[a], a = root, parent[a]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(n):
            nbrs = [((i + 1) % n, j), (i, (j + 1) % n)] if periodic else \
                   ([( (i + 1, j)) if i + 1 < n else None,
                     ((i, j + 1)) if j + 1 < n else None])
            for nb in nbrs:
                if nb is not None and s[nb] == s[i, j] and rng.random() < p:
                    union((i, j), nb)

    clusters = {}
    for i in range(n):
        for j in range(n):
            clusters.setdefault(find((i, j)), []).append((i, j))

    n_clusters = 0
    for members in clusters.values():
        n_clusters += 1
        if rng.random() < 0.5:
            for (i, j) in members:
                s[i, j] *= -1.0
    return s, n_clusters
