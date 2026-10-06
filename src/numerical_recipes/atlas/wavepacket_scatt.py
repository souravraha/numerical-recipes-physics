"""Quantum wave-packet scattering.

A Gaussian wave packet (the standard model of a quantum projectile) is sent
toward a one-dimensional potential.  Because the time evolution is unitary,
the total probability is conserved, and the packet splits into transmitted and
reflected parts.  The transmission + reflection coefficients therefore satisfy
``T + R = 1`` (probability conservation).  Physically: a tall/wide barrier
reflects (``R -> 1``), a low barrier mostly transmits (``T -> 1``), and a free
packet drifts in the direction of its mean momentum.

We use the shared split-operator propagator
(``spectral_fft.tdschrodinger_split_operator``), which is unitary to machine
precision, so ``T + R = 1`` and probability conservation are exact checks.


References (see REFERENCES.md):
  - R. P. Feynman, R. B. Leighton, M. Sands, The Feynman Lectures on Physics III (1965).
  - A. J. Leggett, Rev. Mod. Phys. 53, 657 (1981) [quantum tunneling].
  - A. Feit, F. A. Fleischer, J. Comput. Phys. 14, 235 (1973) [split-operator].
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import spectral_fft as sf

__all__ = [
    "make_packet",
    "barrier_potential",
    "scatter",
    "total_probability",
    "transmission_reflection",
]


def make_packet(n, L, x0, k0, sigma):
    """Normalised right-moving Gaussian packet.  Returns (x, psi)."""
    return sf.gaussian_wavepacket(n=n, L=L, x0=x0, k0=k0, sigma=sigma)


def barrier_potential(x, height, width):
    """Rectangular barrier of ``height`` centred at x=0 with half-width width."""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < width, height, 0.0)


def scatter(psi, V, dx, dt, n_steps, record=False):
    """Unitary time evolution of the packet in the potential."""
    return sf.tdschrodinger_split_operator(psi, V, dx, dt, n_steps, record=record)


def total_probability(psi, dx):
    """Integrated |psi|^2 (the conserved quantity for unitary evolution)."""
    return float(np.sum(np.abs(psi) ** 2) * dx)


def transmission_reflection(psi, x, x_split=0.0, dx=1.0):
    """Split probability at x_split into (right=T, left=R)."""
    x = np.asarray(x, dtype=float)
    right = float(np.sum(np.abs(psi[x >= x_split]) ** 2) * dx)
    left = float(np.sum(np.abs(psi[x < x_split]) ** 2) * dx)
    return right, left
