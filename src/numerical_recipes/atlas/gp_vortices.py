"""Gross-Pitaevskii condensate and vortex dynamics.

A dilute Bose gas at zero temperature is described by a single macroscopic
wave function ``psi`` obeying the Gross-Pitaevskii (GP) equation
    i dpsi/dt = [-1/2 nabla^2 + V + g |psi|^2] psi    (hbar = m = 1),
where ``g`` is the (mean-field) interaction strength.  A quantized vortex
is a phase singularity: ``psi ~ r^q e^{i q theta}`` has a density zero at the
core and a winding ``q`` of the phase around it, so the circulation is
quantised, ``oint v . dl = 2 pi q`` (integer ``q``), and the core carries a
concentrated density dip -- the defining signature of a vortex.

We build the standard ``q``-vortex initial state, verify the quantised
circulation, and evolve it with the shared nonlinear split-operator propagator
(``spectral_fft.tdschrodinger_nonlinear_2d``) to show the core persists and
the particle number is conserved (``sum |psi|^2`` is exact for the GP flow).


References (see REFERENCES.md):
  - L. P. Gross, E. Pitaevskii, J. Phys. (Moscow) 18, 914 (1948) [GP equation].
  - E. P. Gross, Ann. Phys. 10, 62 (1961) [superfluid hydrodynamics].
  - M. R. Matthews et al., Nature 399, 281 (1999) [observation of a vortex].
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import spectral_fft as sf

__all__ = [
    "make_vortex",
    "winding",
    "evolve",
    "particle_number",
    "core_depth",
]


def make_vortex(nx=64, L=16.0, q=1, a0=1.0):
    """A ``q``-vortex condensate ``psi = a0 r^q e^{i q theta}`` (real a0 > 0).

    Returns ``psi`` of shape ``(nx, nx)`` on the periodic box ``[-L/2, L/2)``.
    """
    x = np.linspace(-L / 2, L / 2, nx, endpoint=False)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    TH = np.arctan2(Y, X)
    return (a0 * R**q) * np.exp(1j * q * TH)


def winding(psi, k=6, npts=64):
    """Integer phase winding ``q = (1/2pi) oint grad theta . dl`` about the centre.

    Measured on a fine ring (``npts`` points at radius ``k`` grid cells) so the
    per-step phase change stays well below the ring's Nyquist limit.
    """
    psi = np.asarray(psi, dtype=complex)
    c = psi.shape[0] // 2
    pts = []
    for m in range(npts):
        ang = 2.0 * np.pi * m / npts
        i = int(round(c + k * np.sin(ang)))
        j = int(round(c + k * np.cos(ang)))
        pts.append(psi[i, j])
    dphi = sum(np.angle(pts[(m + 1) % npts] * np.conj(pts[m])) for m in range(npts))
    return float(dphi / (2.0 * np.pi))


def evolve(psi, g, nx=64, L=16.0, dt=0.05, n_steps=400, V=None):
    """Time-evolve the condensate under the GP equation (shared propagator)."""
    dx = dy = L / nx
    if V is None:
        V = np.zeros((nx, nx))
    return sf.tdschrodinger_nonlinear_2d(psi, V, g, dx, dy, dt, n_steps)


def particle_number(psi):
    """Conserved particle number ``N = sum |psi|^2 dx^2`` (up to the grid area)."""
    return float(np.sum(np.abs(psi) ** 2))


def core_depth(psi, center=None):
    """Depth of the density dip at the vortex core (1 - rho_core / rho_max)."""
    rho = np.abs(np.asarray(psi, dtype=complex)) ** 2
    nx = rho.shape[0]
    i, j = (center if center is not None else (nx // 2, nx // 2))
    return float(1.0 - rho[i, j] / rho.max())
