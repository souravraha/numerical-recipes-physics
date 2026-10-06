"""2D incompressible Navier-Stokes turbulence (spectral vorticity method).

The incompressible 2D Navier-Stokes equations can be written for the scalar
vorticity ``w = curl u``:
    dw/dt + (u . nabla) w = nu nabla^2 w,        u = curl^{-1}(w),
where the velocity is recovered from the vorticity by inverting the Laplacian
with periodic boundary conditions -- exactly what the shared
``spectral_fft.poisson_periodic_2d`` solves.  This pseudo-spectral vorticity
method is the standard tool for studying 2D turbulence and the inverse energy
cascade (Kolmogorov ``k^{-5/3}`` in the energy spectrum).

We implement the vorticity step and verify the two invariants that make it
correct: (a) *incompressibility* -- the Poisson projection leaves the flow
divergence-free to machine precision, and (b) *energy dissipation* -- viscosity
drives the kinetic energy down monotonically.


References (see REFERENCES.md):
  - U. Frisch, Turbulence (1995) [2D turbulence, inverse cascade].
  - A. J. Chorin, J. Comput. Phys. 2, 12 (1967) [vorticity/NS].
  - O. M. Phillips, The Dynamics of the Upper Atmosphere (1982) [2D NS].
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import spectral_fft as sf

__all__ = [
    "streamfunction",
    "velocities",
    "curl",
    "divergence",
    "laplacian",
    "step",
    "kinetic_energy",
]


def _k(n, L):
    return 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)


def _mesh(n, L):
    k = _k(n, L)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return KX, KY


def streamfunction(w, L):
    """Solve ``laplacian(phi) = -w`` (the velocity potential) via the FFT."""
    return sf.poisson_periodic_2d(w, L=L)


def velocities(w, L):
    """``u = (dphi/dy, -dphi/dx)`` from the vorticity ``w`` (divergence-free)."""
    n = w.shape[0]
    KX, KY = _mesh(n, L)
    phi_hat = np.fft.fft2(streamfunction(w, L))
    u = np.fft.ifft2(1j * KY * phi_hat).real      # u = dphi/dy
    v = np.fft.ifft2(-1j * KX * phi_hat).real     # v = -dphi/dx
    return u, v


def curl(u, v, L):
    """``w = curl u = dv/dx - du/dy`` (spectral)."""
    n = u.shape[0]
    KX, KY = _mesh(n, L)
    return np.fft.ifft2(1j * KX * np.fft.fft2(v) - 1j * KY * np.fft.fft2(u)).real


def divergence(u, v, L):
    """``div u = du/dx + dv/dy`` (spectral)."""
    n = u.shape[0]
    KX, KY = _mesh(n, L)
    return np.fft.ifft2(1j * KX * np.fft.fft2(u) + 1j * KY * np.fft.fft2(v)).real


def laplacian(f, L):
    """Spectral Laplacian with periodic BCs."""
    n = f.shape[0]
    KX, KY = _mesh(n, L)
    return np.fft.ifft2(-(KX**2 + KY**2) * np.fft.fft2(f)).real


def _advect_and_diffuse(w, nu, L):
    n = w.shape[0]
    u, v = velocities(w, L)
    KX, KY = _mesh(n, L)
    wh = np.fft.fft2(w)
    dw_dx = np.fft.ifft2(1j * KX * wh).real
    dw_dy = np.fft.ifft2(1j * KY * wh).real
    adv = u * dw_dx + v * dw_dy
    return -adv + nu * laplacian(w, L)


def step(w, nu, L, dt):
    """One RK2 (midpoint) vorticity step:  ``dw/dt + (u.nabla)w = nu lap w``."""
    w1 = w + dt * _advect_and_diffuse(w, nu, L)
    return w + dt * _advect_and_diffuse(w1, nu, L)


def kinetic_energy(u, v, L):
    n = u.shape[0]
    dx = L / n
    return float(0.5 * np.sum(u**2 + v**2) * dx**2)
