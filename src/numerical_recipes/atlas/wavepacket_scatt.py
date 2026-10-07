"""Quantum wave-packet scattering.

A Gaussian wave packet (the standard model of a quantum projectile) is sent
toward a one-dimensional potential.  Because the time evolution is unitary,
the total probability is conserved, and the packet splits into transmitted and
reflected parts.  The transmission + reflection coefficients therefore satisfy
``T + R = 1`` (probability conservation).  Physically: a tall/wide barrier
reflects (``R -> 1``), a low barrier mostly transmits (``T -> 1``), and a free
packet drifts in the direction of its mean momentum.

The module also carries the *energy-resolved* band picture of a lattice of
identical barriers: the analytic (Kronig-Penney / Chebyshev) transmission
``|T(E)|^2`` oscillates with ``E`` and develops *gaps* (near-zero transmission)
between the resonances -- the same physics the real-time packet sees as the
packet transmits or reflects from the periodic structure.

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
    "barrier_lattice_potential",
    "scatter",
    "total_probability",
    "transmission_reflection",
    "single_barrier",
    "barrier_lattice",
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


# --------------------------------------------------------------------------- #
# Lattice of identical barriers -- potential + analytic (Kronig-Penney) reference
# --------------------------------------------------------------------------- #
def barrier_lattice_potential(x, n=10, a=0.1, b=0.2, x_start=10.0, V0=400.25):
    """``n`` identical square barriers of width ``a``, height ``V0``.

    Barriers are centred at ``x_start + i(a + b)`` for ``i = 0 .. n-1`` (``b``
    is the gap between barriers).  Feed this to :func:`scatter` with the
    shared split-operator propagator for the real-time packet dynamics.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(n):
        centre = x_start + i * (a + b)
        out = np.where((x > centre - a / 2) & (x < centre + a / 2), V0, out)
    return out


def _chebyshev_u(n, z):
    """Chebyshev polynomial of the second kind ``U_n(z)`` (stable recurrence)."""
    z = np.asarray(z)
    if n == 0:
        return np.ones_like(z)
    if n == 1:
        return 2.0 * z
    u0 = np.ones_like(z)
    u1 = 2.0 * z
    for _ in range(2, n + 1):
        u0, u1 = u1, 2.0 * z * u1 - u0
    return u1


def _wave_numbers(E, V0, m, hbar):
    """Incident ``k1 = sqrt(2 m E)/hbar`` and in-barrier ``k2`` wave numbers."""
    k1 = np.sqrt(2.0 * m * E) / hbar
    k2 = np.sqrt(2.0 * m * (E - V0)) / hbar
    return k1, k2


def single_barrier(E, V0, a, m=1.0, hbar=1.0):
    """Transmission ``t`` and reflection ``r`` amplitudes for one barrier.

    Requires ``E > V0`` (above-barrier scattering); satisfies
    ``|t|^2 + |r|^2 = 1`` exactly.
    """
    k1, k2 = _wave_numbers(E, V0, m, hbar)
    den = 2.0 * k1 * k2 * np.cos(k2 * a) - 1j * (k1**2 + k2**2) * np.sin(k2 * a)
    t = np.exp(-1j * k1 * a) * 2.0 * k1 * k2 / den
    r = 1j * (k2**2 - k1**2) * np.sin(k2 * a) / den
    return t, r


def barrier_lattice(E, V0, a, s, n, m=1.0, hbar=1.0):
    """Transmission ``T`` and reflection ``R`` for ``n`` identical barriers.

    The Kronig-Penney result via the Chebyshev polynomials of the second
    kind: for separation ``s`` between barrier centres,

        z   = (e^{-i k1 s}/t + e^{i k1 s}/t*) / 2,
        Den = U_n(z) - e^{i k1 s} U_{n-1}(z) / t*,
        T   = e^{-i k1 s n} / Den,
        R   = r U_{n-1}(z) e^{i k1 s} / (t Den),

    with ``t, r`` the single-barrier amplitudes.  This is the independent,
    energy-resolved reference against which the real-time packet scattering
    (``scatter`` + ``barrier_lattice_potential``) is compared; ``|T|^2``
    develops band gaps as ``n`` grows.
    """
    k1, _ = _wave_numbers(E, V0, m, hbar)
    t, r = single_barrier(E, V0, a, m=m, hbar=hbar)
    z = (np.exp(-1j * k1 * s) / t + np.exp(1j * k1 * s) / np.conj(t)) / 2.0
    U_n = _chebyshev_u(n, z)
    U_nm1 = _chebyshev_u(n - 1, z)
    den = U_n - np.exp(1j * k1 * s) * U_nm1 / np.conj(t)
    T = np.exp(-1j * k1 * s * n) / den
    R = r * U_nm1 * np.exp(1j * k1 * s) / (t * den)
    return T, R
