"""Spectral methods and the FFT, written from scratch where it teaches.

The FFT is the engine behind spectral methods: represent a function by its
Fourier coefficients, and differential operators (derivatives, the Laplacian)
become *multiplication by wavenumber* in Fourier space.  This gives two
enormous wins:

  * **Spectral (exponential) accuracy** for smooth problems -- the error
    decays faster than any power of the grid spacing, unlike the 2nd-order
    finite differences in ``pde_fdm``.
  * **Fast solves.**  The 2D Poisson equation with periodic BCs is solved in
    one shot: FFT, divide by ``-|k|^2``, inverse FFT.

This module implements:

  * ``dft`` -- the naive O(n^2) discrete Fourier transform, used to *verify*
    ``numpy.fft`` (and to make the definition concrete).
  * ``spectrum`` -- amplitude/phase spectrum of a signal (the "look inside
    the signal" routine).
  * ``poisson_periodic_2d`` -- FFT-based Poisson solver (periodic BCs).
  * ``tdschrodinger_split_operator`` -- the split-operator method for the
    time-dependent Schrodinger equation: the gold-standard quantum
    wavepacket propagator (exact for the free particle to O(dt^2), unitary).
  * ``spectral_wave_1d`` -- spectral time-stepping of the 1D wave equation.

Everything is built on ``numpy.fft`` (the FFT itself is a compiled O(n log n)
routine we take as given); the physics routines are written by hand.


References (see REFERENCES.md):
  - J. W. Cooley, J. W. T. Tukey, Math. Comp. 19, 297 (1965) [FFT].
  - A. Feit, F. A. Fleischer, J. Comput. Phys. 14, 235 (1973) [split-operator].
  - A. Kosloff, Phys. Rep. 194, 195 (1990) [time-dependent Schrodinger methods]."""

from __future__ import annotations

import numpy as np

__all__ = [
    "dft",
    "spectrum",
    "poisson_periodic_2d",
    "poisson_periodic_1d",
    "tdschrodinger_nonlinear_2d",
    "tdschrodinger_split_operator",
    "spectral_wave_1d",
    "gaussian_wavepacket",
]


# --------------------------------------------------------------------------- #
# Discrete Fourier transform + spectra
# --------------------------------------------------------------------------- #
def dft(x):
    """Naive O(n^2) discrete Fourier transform (definition, not performance).

    ``X[k] = sum_j x[j] exp(-2 pi i j k / n)``.  Used to verify ``np.fft``
    and to make the definition concrete.
    """
    x = np.asarray(x, dtype=complex)
    n = x.size
    j = np.arange(n)
    W = np.exp(-2j * np.pi * np.outer(j, j) / n)
    return W @ x


def spectrum(x, dt):
    """One-sided amplitude and phase spectrum of a real signal ``x``.

    Returns ``(freq, amplitude, phase)`` for the positive frequencies.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.rfft(x)
    amp = 2.0 * np.abs(X) / n
    amp[0] = np.abs(X[0]) / n  # DC term is not doubled
    phase = np.angle(X)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, amp, phase


# --------------------------------------------------------------------------- #
# FFT-based 2D Poisson solver (periodic boundary conditions)
# --------------------------------------------------------------------------- #
def poisson_periodic_2d(rho, L=1.0):
    """Solve ``laplacian(u) = -rho`` on a periodic [0, L]^2 box via the FFT.

    ``rho`` is an (n x n) array.  In Fourier space the Laplacian is the
    multiplier ``-(kx^2 + ky^2)``, so ``u = rho / (kx^2 + ky^2)`` with the
    zero mode (mean) set to zero.  Exact up to roundoff -- the spectral win.
    Returns the solution ``u`` (with zero mean).
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[0]
    dx = L / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    k2 = KX**2 + KY**2
    rho_hat = np.fft.fft2(rho)
    u_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = k2 > 0
    u_hat[mask] = rho_hat[mask] / k2[mask]
    u_hat[~mask] = 0.0  # enforce zero mean
    u = np.fft.ifft2(u_hat).real
    return u


# --------------------------------------------------------------------------- #
# Split-operator time-dependent Schrodinger equation
# --------------------------------------------------------------------------- #
def gaussian_wavepacket(n=4096, L=40.0, x0=0.0, k0=5.0, sigma=1.0):
    """Normalised Gaussian wavepacket centred at x0 with mean momentum k0.

    ``psi = (1/(2 pi sigma^2))^{1/4} exp(-(x-x0)^2/(4 sigma^2) + i k0 x)``.
    Returns ``(x, psi)``.
    """
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = x[1] - x[0]
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2) + 1j * k0 * x)
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return x, psi


def tdschrodinger_split_operator(psi, V, dx, dt, n_steps, record=False):
    """Split-operator (Trotter) propagator for i dpsi/dt = H psi.

    H = -1/2 d^2/dx^2 + V(x)  (hbar = m = 1).  The kinetic step is exact in
    Fourier space (multiply by exp(-i k^2 dt/2)), the potential step is exact
    in real space (multiply by exp(-i V dt)).  Unitary and 2nd-order in dt --
    the standard quantum wavepacket propagator.

    Returns the final ``psi`` (or ``(psi, history)``).
    """
    psi = np.asarray(psi, dtype=complex).copy()
    n = psi.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kinetic = np.exp(-0.5j * k**2 * dt)
    potential = np.exp(-1j * V * dt)
    half_kin = np.exp(-0.5j * k**2 * (dt / 2))

    hist = [psi.copy()] if record else None
    for _ in range(n_steps):
        # Kinetic half-step (Fourier), potential full-step (real), kinetic half.
        psi = np.fft.ifft(half_kin * np.fft.fft(psi))
        psi = psi * potential
        psi = np.fft.ifft(half_kin * np.fft.fft(psi))
        if record:
            hist.append(psi.copy())
    if record:
        return psi, np.asarray(hist)
    return psi


# --------------------------------------------------------------------------- #
# Spectral 1D wave equation
# --------------------------------------------------------------------------- #
def spectral_wave_1d(u0, v0, dx, dt, n_steps, c=1.0, record=False):
    """Spectral (FFT) time-stepping of u_tt = c^2 u_xx with periodic BCs.

    The second spatial derivative is computed by multiplying the FFT by
    ``-(k^2)`` -- spectrally accurate for smooth data.  Returns ``(u, v)``.
    """
    u = np.asarray(u0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    n = u.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    u_hat0 = np.fft.fft(u0)
    hist = [u.copy()] if record else None
    # Leapfrog in time, spectral in space.
    for step in range(n_steps):
        u_hat = np.fft.fft(u)
        acc_hat = -c**2 * k**2 * u_hat
        acc = np.fft.ifft(acc_hat).real
        v = v + dt * acc
        u = u + dt * v
        if record:
            hist.append(u.copy())
    if record:
        return u, v, np.asarray(hist)
    return u, v


def poisson_periodic_1d(rho, L=1.0):
    """Solve ``laplacian(u) = -rho`` on a periodic [0, L] line via the FFT.

    ``rho`` is a 1-D array (length n).  In Fourier space the Laplacian is the
    multiplier ``-k^2``, so ``u = rho_hat / k^2`` with the zero mode (mean) set
    to zero.  Exact up to roundoff.  Returns the solution ``u`` (zero mean).
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.size
    dx = L / n
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    k2 = k ** 2
    rho_hat = np.fft.fft(rho)
    u_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = k2 > 0
    u_hat[mask] = rho_hat[mask] / k2[mask]
    return np.fft.ifft(u_hat).real


def tdschrodinger_nonlinear_2d(psi, V, g, dx, dy, dt, n_steps, record=False):
    """Split-operator (Strang) propagator for the 2D Gross-Pitaevskii equation.

    ``i dpsi/dt = [-1/2 (d^2/dx^2 + d^2/dy^2) + V(x, y) + g |psi|^2] psi``
    (hbar = m = 1).  The kinetic step is exact in Fourier space (multiply by
    ``exp(-i (kx^2 + ky^2) dt/2)``); the external + interaction potential step
    is exact in real space (multiply by ``exp(-i (V + g|psi|^2) dt)``).  Second
    order in dt.  Unlike the linear propagator this is *not* unitary in |psi|:
    it conserves the integrated particle number ``sum |psi|^2 dx dy`` but the
    density profile evolves self-consistently through the nonlinearity -- which
    is precisely how vortices and the condensate profile form.

    ``psi`` and ``V`` are ``(ny, nx)``; ``g`` is the (real) interaction
    strength.  Returns the final ``psi`` (or ``(psi, history)``).
    """
    psi = np.asarray(psi, dtype=complex).copy()
    ny, nx = psi.shape
    V = np.asarray(V, dtype=float)
    fx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    fy = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    K2 = fx[None, :] ** 2 + fy[:, None] ** 2
    half_kin = np.exp(-0.5j * K2 * (dt / 2))
    hist = [psi.copy()] if record else None
    for _ in range(n_steps):
        psi = np.fft.ifft2(half_kin * np.fft.fft2(psi))
        psi = psi * np.exp(-1j * (V + g * np.abs(psi) ** 2) * dt)
        psi = np.fft.ifft2(half_kin * np.fft.fft2(psi))
        if record:
            hist.append(psi.copy())
    if record:
        return psi, np.asarray(hist)
    return psi
