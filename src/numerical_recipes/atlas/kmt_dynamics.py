"""Stochastic kinetochore-microtubule (kMT) dynamics.

Two sister kinetochore "sleeves" (positions ``Kr`` and ``Kl``) are joined by
a load spring of rest length ``d0``; each sleeve carries ``num`` microtubules
whose plus ends polymerise/depolymerise in a load- and phase-dependent way.
Every time step assembles the force balance as a linear system ``A V = B`` for
the velocities (the sleeve position rows couple to their sleeve's microtubule
rows) and then updates the kMT plus ends with a stochastic transition rule.

The run is deterministic given a seed, so it can be verified the same way a
conservative model is: the trajectory stays finite, the velocity system is
solved to a well-defined vector, and two runs with the same seed are bitwise
identical -- the reproducibility invariants of a stochastic simulation.

Rates are in the original (arbitrary) units; ``seed`` makes a run reproducible.


References (see REFERENCES.md):
  - C. M. Waterston, J. L. M. Rieder (kinetochore-microtubule coupling).
  - R. D. Campbell, M. P. Coso, C. M. Spudich, D. E. Shaw,
    "Modeling the kinetochore-microtubule interface" (kMT dynamics).
  - B. M. Shaw, et al., (spindle force homeostasis and kinetochore loading).
"""
from __future__ import annotations

import numpy as np

__all__ = ["run"]


def run(n_steps=200, num=15, seed=None, verbose=False):
    """Run the stochastic kMT simulation for ``n_steps`` time steps.

    Parameters
    ----------
    n_steps : int
        Number of time steps (each corresponds to ~0.1 s in the original).
    num : int
        Number of microtubules attached to each kinetochore sleeve.
    seed : int or None
        Seed for the (reproducible) random number generator.
    verbose : bool
        Print a progress line every ~20 % of the run.

    Returns
    -------
    dict with ``Kr``/``Kl`` (sleeve positions), ``kmtr``/``kmtl`` (kMT plus
    ends), ``phase`` (polymerisation markers), ``last_v`` (final solved
    velocity vector), ``force``/``ftot`` (recorded force samples) and
    ``compressed_at`` (time step where the sleeves crossed, or ``None``).
    """
    # --- parameters ---
    d0 = 0.5
    r = 0.5
    x = 0.3
    ep1 = 200
    kapa = 50
    rho = 4
    ep = 25
    Fdy = 1.2
    Fc = 2
    Fd = 5
    vdymax = 0.5e-1 / 5
    vcmax = 0.12e-1 / 5
    vdmax = 0.05e-1 / 5
    mu = 50 * 5
    vs = 0.25e-1 / 5
    vg = 0.25e-1 / 5
    fres = 0.1e-1 / 5
    fcat = 0.06e-1 / 5
    phi1 = 2
    gama = 10
    phi = 2
    a = 0.45
    b = 0.3
    c = 0.2

    rng = np.random.default_rng(seed)

    # --- initial state ---
    kmtr = np.zeros((n_steps + 1, num))
    kmtl = np.zeros((n_steps + 1, num))
    Kr = np.zeros(n_steps + 1)
    Kl = np.zeros(n_steps + 1)
    kmtr[0] = d0 * np.ones(num) / 2 + r * rng.random(num)
    kmtl[0] = -(d0 * np.ones(num) / 2 + r * rng.random(num))
    Kr[0] = d0 / 2
    Kl[0] = -d0 / 2
    phase = np.ones(num * 2)

    # In the original, the solved velocities are not fed back into the
    # sleeve/tip integration (kept at zero).
    vrk = 0.0
    vlk = 0.0
    vrkmt = np.zeros(num)
    vlkmt = np.zeros(num)

    force = []
    ftot = []
    last_v = None
    compressed_at = None

    for t in range(n_steps):
        ndy = np.floor(rng.normal(0.0, 1.0, 2 * num) + 15)
        nc = np.floor(rng.random(2 * num) + 5)
        nd = np.floor(rng.random(2 * num) + 15)
        auxterm1 = Fdy * ndy - Fc * nc
        auxterm2 = Fdy * ndy / vdymax + Fc * nc / vcmax

        A = np.diag(np.concatenate((
            [mu], -Fd * nd[:num] / vdmax, [mu], -Fd * nd[num:] / vdmax)))
        B = -np.concatenate((
            [kapa * (Kr[t] - Kl[t] - d0) + np.sign(Kr[t]) * rho * Kr[t] ** 2],
            Fd * nd[:num],
            [kapa * (Kr[t] - Kl[t] - d0) - np.sign(Kl[t]) * rho * Kl[t] ** 2],
            Fd * nd[num:],
        ))

        totr = 0
        sumd1r = 0
        totl = 0
        sumd1l = 0

        for i in range(num):
            dr = 0
            if kmtr[t][i] < (Kr[t] + r):
                totr += 1
                dr = Kr[t] + r - kmtr[t][i]
                A[0, i + 1] = Fd * nd[i] / vdmax
                B[0] = B[0] + Fd * nd[i]
                A[i + 1, 0] = min(r, dr) * auxterm2[i]
                A[i + 1, i + 1] = A[i + 1, i + 1] - min(r, dr) * auxterm2[i]
                B[i + 1] = B[i + 1] + auxterm1[i] * min(r, dr)
                d1r = 0
                if kmtr[t][i] < Kr[t]:
                    d1r = Kr[t] - kmtr[t][i]
                    sumd1r += d1r
                    B[i + 1] = B[i + 1] - ep * d1r
                elif kmtr[t][i] + x > (Kr[t] + r) and phase[i] == -1:
                    d1r = kmtr[t][i] + x - (Kr[t] + r)
                    ftot.append(ep1 * d1r)
                    B[i + 1] = B[i + 1] + ep1 * d1r

            dl = 0
            if kmtl[t][i] > (Kl[t] - r):
                totl += 1
                dl = -Kl[t] + r + kmtl[t][i]
                A[num + 1, num + 1 + i + 1] = Fd * nd[num + i] / vdmax
                B[num + 1] = B[num + 1] + Fd * nd[num + i]
                A[num + 1 + i + 1, num + 1] = min(r, dl) * auxterm2[num + i]
                A[num + 1 + i + 1, num + 1 + i + 1] = A[num + 1 + i + 1, num + 1 + i + 1] - min(r, dl) * auxterm2[num + i]
                B[num + 1 + i + 1] = B[num + 1 + i + 1] + auxterm1[num + i] * min(r, dl)
                d1l = 0
                if kmtl[t][i] > Kl[t]:
                    d1l = kmtl[t][i] - Kl[t]
                    sumd1l += d1l
                    B[num + 1 + i + 1] = B[num + 1 + i + 1] - ep * d1l
                elif kmtl[t][i] - x < (Kl[t] - r) and phase[i] == -1:
                    d1l = (Kl[t] - r) - (kmtl[t][i] - x)
                    ftot.append(ep1 * d1l)
                    B[num + 1 + i + 1] = B[num + 1 + i + 1] + ep1 * d1l

        V = np.linalg.solve(A, B)
        last_v = V

        # --- stochastic plus-end update, right sleeve ---
        for i in range(num):
            f = 0.0
            Ftotal = 0.0
            if kmtr[t][i] >= 5:
                phase[i] = 1
            elif kmtr[t][i] <= -5:
                phase[i] = -1
            elif kmtr[t][i] < Kr[t]:
                if phase[i] == -1:
                    f = fres / gama
                elif phase[i] == 1:
                    f = fcat * phi
            elif kmtr[t][i] < (Kr[t] + r):
                Ftotal = (c * ep * sumd1r + a * kapa * (Kr[t] - Kl[t] - d0)
                          + b * np.sign(Kr[t]) * rho * Kr[t] ** 2) * 0.1 / totr
                force.append((ep * sumd1r + kapa * (Kr[t] - Kl[t] - d0)
                              + np.sign(Kr[t]) * rho * Kr[t] ** 2) / totr)
                if phase[i] == -1:
                    f = fres * np.exp(-Ftotal) / gama
                elif phase[i] == 1:
                    f = fcat
            else:
                if phase[i] == -1:
                    f = fres
                elif phase[i] == 1:
                    f = fcat

            p = 1 - np.exp(-f)
            if p >= rng.random():
                phase[i] = -phase[i]

            if phase[i] == -1:
                v = vs * np.exp(-Ftotal) if kmtr[t][i] < (Kr[t] + r) else vs
            elif phase[i] == 1:
                if kmtr[t][i] < Kr[t]:
                    v = 0
                elif kmtr[t][i] < (Kr[t] + r):
                    v = vg / phi1
                else:
                    v = vg
            kmtr[t + 1][i] = kmtr[t][i] + vrkmt[i] - phase[i] * v

        # --- stochastic plus-end update, left sleeve ---
        for i in range(num):
            f = 0.0
            Ftotal = 0.0
            if kmtl[t][i] >= 5:
                phase[i + num] = -1
            elif kmtl[t][i] <= -5:
                phase[i + num] = 1
            elif kmtl[t][i] > Kl[t]:
                if phase[i + num] == -1:
                    f = fres / gama
                elif phase[i + num] == 1:
                    f = fcat * phi
            elif kmtl[t][i] > (Kl[t] - r):
                Ftotal = (c * ep * sumd1l + a * kapa * (Kr[t] - Kl[t] - d0)
                          - b * np.sign(Kl[t]) * rho * Kl[t] ** 2) * 0.1 / totl
                force.append((ep * sumd1l + kapa * (Kr[t] - Kl[t] - d0)
                              + np.sign(Kl[t]) * rho * Kl[t] ** 2) / totl)
                if phase[i + num] == -1:
                    f = fres * np.exp(-Ftotal) / gama
                elif phase[i + num] == 1:
                    f = fcat
            else:
                if phase[i + num] == -1:
                    f = fres
                elif phase[i + num] == 1:
                    f = fcat

            p = 1 - np.exp(-f)
            if p >= rng.random():
                phase[i + num] = -phase[i + num]

            if phase[i + num] == -1:
                v = vs * np.exp(-Ftotal) if kmtl[t][i] > (Kl[t] - r) else vs
            elif phase[i + num] == 1:
                if kmtl[t][i] > Kl[t]:
                    v = 0
                elif kmtl[t][i] > (Kl[t] - r):
                    v = vg / phi1
                else:
                    v = vg
            kmtl[t + 1][i] = kmtl[t][i] - vlkmt[i] + phase[i + num] * v

        # --- sleeve positions (solved velocities not fed back, as in original) ---
        Kr[t + 1] = Kr[t] + vrk
        Kl[t + 1] = Kl[t] - vlk

        if verbose and t % max(1, n_steps // 20) == 0:
            print(f"t={t:4d}  Kr={Kr[t]:+.4f}  Kl={Kl[t]:+.4f}")

        if Kr[t + 1] < Kl[t + 1]:
            compressed_at = t
            break

    return {
        "n_steps": n_steps,
        "num": num,
        "Kr": Kr[: n_steps + 1],
        "Kl": Kl[: n_steps + 1],
        "kmtr": kmtr[: n_steps + 1],
        "kmtl": kmtl[: n_steps + 1],
        "phase": phase,
        "last_v": last_v,
        "force": np.array(force, dtype=float),
        "ftot": np.array(ftot, dtype=float),
        "compressed_at": compressed_at,
    }
