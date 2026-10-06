"""Particle-in-cell plasma simulation (1D Langmuir oscillations).

A 1D electron plasma (mobile electrons + a fixed neutralising ion background)
is the classic PIC system.  The loop is:
  1. *Deposit* particle charges onto a grid (CIC cloud), adding the uniform
     background so the net charge has zero mean.
  2. *Solve* the Poisson equation ``nabla^2 phi = -rho/eps0`` with the shared
     ``spectral_fft.poisson_periodic_1d`` (spectrally exact).
  3. *Interpolate* the field back to the particles.
  4. *Push* the particles (``m dv/dt = -e E``, ``dx/dt = v``).

A coherent density perturbation then oscillates at the plasma frequency
``omega_p = sqrt(n0 e^2 / (m eps0))`` (Langmuir oscillation).  We verify:
(a) *charge neutrality* -- a uniform plasma has a zero field; (b) *field-solve
correctness* -- the Poisson residual and the antisymmetric field about a charge;
and (c) the *Langmuir* mode oscillates at ``omega_p``.  Geometric units
(e = m = eps0 = 1).


References (see REFERENCES.md):
  - L. Spitzer, Physics of Fully Ionized Gases (1956) [plasma frequency].
  - C. K. Birdsall, A. B. Langdon, Plasma Physics via Computer Simulation (1985).
  - O. Buneman, Phys. Rev. 115, 529 (1959) [plasma instability/oscillations].
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import spectral_fft as sf

__all__ = [
    "plasma_frequency",
    "deposit_charge",
    "field_from_density",
    "pic_step",
    "langmuir",
]


def plasma_frequency(n0, e=1.0, m=1.0, eps0=1.0):
    """``omega_p = sqrt(n0 e^2 / (m eps0))``."""
    return float(np.sqrt(n0 * e**2 / (m * eps0)))


def deposit_charge(x_particles, L, n, e=1.0):
    """Deposit electron charges ``-e`` onto ``n`` grid points via a CIC cloud."""
    rho = np.zeros(n)
    for xp in x_particles:
        f = (xp % L) * n / L
        i = int(f) % n
        frac = f - int(f)
        rho[i] += -e * (1.0 - frac)
        rho[(i + 1) % n] += -e * frac
    return rho


def _interp(field, x_particles, L, n):
    out = np.zeros_like(x_particles)
    for j, xp in enumerate(x_particles):
        f = (xp % L) * n / L
        i = int(f) % n
        frac = f - int(f)
        out[j] = (1.0 - frac) * field[i] + frac * field[(i + 1) % n]
    return out


def field_from_density(rho, L, eps0=1.0):
    """Solve ``nabla^2 phi = -rho/eps0`` -> ``(phi, E)`` with ``E = -phi'``."""
    n = len(rho)
    phi = sf.poisson_periodic_1d(rho / eps0, L=L)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    E = np.fft.ifft(-1j * k * np.fft.fft(phi)).real
    return phi, E


def pic_step(x, v, n0, L, n, dt, e=1.0, m=1.0, eps0=1.0):
    """One PIC push: deposit, solve, interpolate, push.  Returns (x, v)."""
    rho = deposit_charge(x, L, n, e=e) + n0
    _, E = field_from_density(rho, L, eps0=eps0)
    Ep = _interp(E, x, L, n)
    v = v + dt * (-e * Ep / m)
    x = (x + dt * v) % L
    return x, v


def langmuir(x0, L, n0, n=256, dt=0.02, n_steps=500, e=1.0, m=1.0, eps0=1.0):
    """Evolve a 1D PIC plasma and return the k=1 density-wave amplitude over time."""
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    amp = np.zeros(n_steps + 1)
    for s in range(n_steps + 1):
        rho = deposit_charge(x, L, n, e=e) + n0
        rho1 = rho - rho.mean()
        amp[s] = abs(np.fft.fft(rho1)[1])
        if s < n_steps:
            x, v = pic_step(x, v, n0, L, n, dt, e=e, m=m, eps0=eps0)
    return amp
