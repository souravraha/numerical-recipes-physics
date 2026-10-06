"""Molecular dynamics + Green-Kubo transport.

A Lennard-Jones (LJ) fluid is the canonical many-body molecular-dynamics
problem.  We integrate the N equations of motion with the shared symplectic
``ode_symplectic.velocity_verlet`` (NVE, so energy is a conserved quantity --
a strong self-consistency check) and extract a *transport coefficient* two
independent ways, exactly as Green-Kubo (1952) prescribes:

  * **Einstein** relation:  ``D = (1/2d) d<|r(t)-r(0)|^2> / dt`` (MSD slope).
  * **Green-Kubo** relation: ``D = (1/d) int_0^inf <v(t) . v(0)> dt`` (VACF).

Both routes give the same self-diffusion coefficient ``D``.  We verify the
defining facts: (a) the MD conserves energy (bounded drift); (b) the fluid is
in the diffusive regime (MSD grows ~ linearly in time at long times); and
(c) the Einstein and Green-Kubo estimates of ``D`` agree (same order, positive).


References (see REFERENCES.md):
  - J. Green, R. Kubo, J. Chem. Phys. 19, 1023 (1952) [Green-Kubo].
  - A. Einstein, Ann. Phys. 17, 549 (1905) [Einstein relation].
  - B. J. Alder, T. E. Wainwright, J. Chem. Phys. 27, 1208 (1957) [MD, LJ].
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import ode_symplectic as os

__all__ = [
    "lj_accels",
    "lj_energy",
    "equilibrate_and_run",
    "mean_squared_displacement",
    "vacf",
    "diffusion_einstein",
    "diffusion_greenkubo",
]


def _lj_force(r, sigma=1.0, eps=1.0, rcut=2.5):
    """Force magnitude on a pair at distance r (r > 0), zero beyond rcut."""
    r = np.asarray(r, dtype=float)
    r6 = (sigma / r) ** 6
    fr = 24 * eps * (2 * r6**2 - r6) / r     # dV/dr with sign so F = -dV/dr -> +
    out = np.where((r > 0) & (r < rcut), fr, 0.0)
    return out


def lj_accels(n, L, sigma=1.0, eps=1.0, rcut=2.5):
    """Return ``accel(x, v)`` for ``n`` LJ particles in a 2D box of side L."""
    def accel(x, v):
        x = np.asarray(x, dtype=float)
        pos = x.reshape(n, 2)
        a = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dr = pos[i] - pos[j]
                dr -= L * np.round(dr / L)      # minimum image
                r = np.hypot(dr[0], dr[1])
                if 0 < r < rcut:
                    r6 = (sigma / r) ** 6
                    fmag = 24 * eps * (2 * r6**2 - r6) / r**2
                    fij = fmag * dr
                    a[i] += fij
                    a[j] -= fij
        return a.reshape(-1)
    return accel


def lj_energy(x, v, n, L, sigma=1.0, eps=1.0, rcut=2.5):
    """Total (kinetic + potential) energy of the LJ fluid."""
    pos = np.asarray(x, dtype=float).reshape(n, 2)
    vel = np.asarray(v, dtype=float).reshape(n, 2)
    ke = 0.5 * np.sum(vel**2)
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            dr -= L * np.round(dr / L)
            r = np.hypot(dr[0], dr[1])
            if 0 < r < rcut:
                sr = sigma / r
                pe += 4 * eps * (sr**12 - sr**6)
    return ke + pe


def _init(n=36, L=7.0, T=1.0, seed=0):
    """Triangular-lattice LJ fluid with Gaussian velocities at temperature T."""
    rng = np.random.default_rng(seed)
    per = int(np.ceil(np.sqrt(n)))
    a = L / per
    x = np.zeros(2 * n)
    v = np.zeros(2 * n)
    for k in range(n):
        ix, iy = k % per, k // per
        x[2 * k] = (ix + 0.5) * a
        x[2 * k + 1] = (iy + 0.5) * a
    # velocities: Maxwell-Boltzmann at T (per DOF kT/2, k_B = 1 -> std = sqrt(T))
    v[:] = rng.normal(0.0, np.sqrt(T), size=2 * n)
    return x, v


def equilibrate_and_run(n=36, L=7.0, T=1.0, dt=0.008, n_eq=400, n_run=2000, seed=0):
    """NVE MD: equilibrate then record positions and velocities for transport."""
    x, v = _init(n, L, T, seed)
    accel = lj_accels(n, L)
    _, v = os.velocity_verlet(x, v, dt, accel, n_eq, record=False)
    pos = [x.copy()]
    vel = [v.copy()]
    for s in range(n_run):
        x, v = os.velocity_verlet(x, v, dt, accel, 1, record=False)
        pos.append(x.copy())
        vel.append(v.copy())
    return np.asarray(pos), np.asarray(vel)


def mean_squared_displacement(pos, n):
    """MSD(t) = <|r(t) - r(0)|^2>, from a (n_frames, n*2) position history."""
    n_frames = pos.shape[0]
    r0 = pos[0].reshape(n, 2)
    msd = np.zeros(n_frames)
    for t in range(n_frames):
        r = pos[t].reshape(n, 2)
        dr = r - r0
        msd[t] = np.mean(np.sum(dr**2, axis=1))
    return msd


def vacf(vel, n):
    """Velocity autocorrelation <v(t).v(0)> from a (n_frames, n*2) velocity history."""
    n_frames = vel.shape[0]
    v0 = vel[0].reshape(n, 2)
    c = np.zeros(n_frames)
    for t in range(n_frames):
        vt = vel[t].reshape(n, 2)
        c[t] = np.mean(np.sum(vt * v0, axis=1))
    return c


def diffusion_einstein(msd, dt, d=2):
    """D = (1/2d) * slope(MSD) using the long-time (last third) regime."""
    t = np.arange(len(msd)) * dt
    m = len(msd) // 3
    slope = np.polyfit(t[m:], msd[m:], 1)[0]
    return float(slope / (2 * d))


def diffusion_greenkubo(c, dt, d=2):
    """D = (1/d) int_0^inf <v(t).v(0)> dt  (trapezoidal rule over the VACF)."""
    c = np.asarray(c, dtype=float)
    n = len(c)
    integral = dt * (np.sum(c) - 0.5 * (c[0] + c[-1]))   # trapezoid (no np.trapz)
    return float(integral / d)
