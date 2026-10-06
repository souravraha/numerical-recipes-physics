"""Finite-difference PDE solvers, written from scratch.

Finite differences are the most direct way to turn a PDE into algebra: replace
derivatives by differences on a grid.  The ideas this module drills are the
ones every applied physicist must internalise:

  * **Stability (CFL).**  The explicit (FTCS) heat solver is stable only for
    ``r = D dt / dx^2 <= 0.5``; beyond that the shortest grid modes blow up.
    The wave solver needs ``c dt / dx <= 1``.  The implicit (Crank-Nicolson)
    heat solver trades a tridiagonal solve per step for unconditional
    stability.
  * **The tridiagonal (Thomas) solve.**  1D second-order FD always produces a
    tridiagonal matrix; the Thomas algorithm solves it in O(n).
  * **Iteration and over-relaxation.**  The 2D Poisson equation is solved by
    Jacobi (slow, reference) and red-black SOR (fast) -- showing why
    over-relaxation is the workhorse of grid PDE solvers.

Reference systems span the disciplines: heat conduction (thermal), the
vibrating string (acoustics/mechanics), linear advection (fluids), and the
2D Poisson equation (electrostatics/gravitation).  Second-order accuracy is
the honest starting point; spectral accuracy lives in ``spectral_fft``.


References (see REFERENCES.md):
  - L. Courant, K. Friedrichs, R. Lewy, Math. Ann. 100, 32 (1928) [CFL].
  - J. Crank, P. Nicolson, Math. Proc. Camb. Phil. Soc. 42, 50 (1946) [Crank-Nicolson].
  - L. H. Thomas, J. Opt. Soc. Am. 24, 595 (1934) [tridiagonal solve].
  - R. S. Varga, Matrix Iterative Analysis (1962) [SOR]."""

from __future__ import annotations

import numpy as np

__all__ = [
    "thomas",
    "heat_1d_explicit",
    "heat_1d_cn",
    "wave_1d",
    "advection_1d_upwind",
    "poisson_2d_jacobi",
    "poisson_2d_sor",
]


# --------------------------------------------------------------------------- #
# Thomas algorithm for tridiagonal systems
# --------------------------------------------------------------------------- #
def thomas(diag, sub, sup, rhs):
    """Solve a tridiagonal system by the Thomas algorithm (O(n)).

    ``diag`` (length n), ``sub`` and ``sup`` (length n-1) are the main,
    sub- and super-diagonal; ``rhs`` the right-hand side.
    """
    n = len(diag)
    c = np.empty(n - 1)
    d = np.empty(n)
    c[0] = sup[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - sub[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = sup[i] / denom
        d[i] = (rhs[i] - sub[i - 1] * d[i - 1]) / denom
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


# --------------------------------------------------------------------------- #
# 1D heat equation   u_t = D u_xx
# --------------------------------------------------------------------------- #
def heat_1d_explicit(u0, dt, dx, n_steps, D=1.0, u_left=0.0, u_right=0.0, record=False):
    """Explicit FTCS heat solver with Dirichlet BCs.

    Stable only for ``r = D dt / dx^2 <= 0.5``.  Returns the final profile
    (or ``(u, history)`` if ``record``).
    """
    u = np.array(u0, dtype=float, copy=True)
    r = D * dt / dx**2
    if r > 0.5 + 1e-12:
        raise ValueError(f"FTCS heat scheme unstable: r={r:.4f} > 0.5")
    hist = [u.copy()] if record else None
    for _ in range(n_steps):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[:-2] - 2 * u[1:-1] + u[2:])
        u_new[0] = u_left
        u_new[-1] = u_right
        u = u_new
        if record:
            hist.append(u.copy())
    if record:
        return u, np.asarray(hist)
    return u


def heat_1d_cn(u0, dt, dx, n_steps, D=1.0, u_left=0.0, u_right=0.0, record=False):
    """Crank-Nicolson heat solver (unconditionally stable, 2nd order).

    Solves a tridiagonal system per step via the Thomas algorithm.
    """
    u = np.array(u0, dtype=float, copy=True)
    n = u.size
    r = D * dt / dx**2
    hist = [u.copy()] if record else None
    for _ in range(n_steps):
        # Interior rows of the Crank-Nicolson system.
        diag = np.full(n, 1.0 + r)
        sub = np.full(n - 1, -r / 2)
        sup = np.full(n - 1, -r / 2)
        rhs = np.empty(n)
        rhs[1:-1] = (1 - r) * u[1:-1] + (r / 2) * (u[:-2] + u[2:])
        # Apply Dirichlet BCs (absorbing the off-diagonal into the RHS).
        diag[0] = 1.0
        sub[0] = 0.0
        rhs[0] = u_left + (r / 2) * u[1]
        diag[-1] = 1.0
        sup[-1] = 0.0
        rhs[-1] = u_right + (r / 2) * u[-2]
        u = thomas(diag, sub, sup, rhs)
        u[0] = u_left
        u[-1] = u_right
        if record:
            hist.append(u.copy())
    if record:
        return u, np.asarray(hist)
    return u


# --------------------------------------------------------------------------- #
# 1D wave equation   u_tt = c^2 u_xx
# --------------------------------------------------------------------------- #
def wave_1d(u0, v0, dt, dx, n_steps, c=1.0, record=False):
    """Leapfrog (2nd-order) wave solver with fixed ends.

    Stable for ``c dt / dx <= 1`` (CFL).  ``v0`` is the initial velocity.
    Returns ``(u, v)`` or ``(u, v, history)``.
    """
    u = np.array(u0, dtype=float, copy=True)
    v = np.array(v0, dtype=float, copy=True)
    if c * dt / dx > 1.0 + 1e-12:
        raise ValueError(f"wave CFL violated: c*dt/dx={c * dt / dx:.4f} > 1")
    hist = [u.copy()] if record else None
    for _ in range(n_steps):
        a = np.zeros_like(u)
        a[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
        v[1:-1] += dt * c**2 * a[1:-1]
        u = u + dt * v
        u[0] = u[-1] = 0.0
        v[0] = v[-1] = 0.0
        if record:
            hist.append(u.copy())
    if record:
        return u, v, np.asarray(hist)
    return u, v


# --------------------------------------------------------------------------- #
# 1D advection   u_t + a u_x = 0   (upwind, stable for a > 0)
# --------------------------------------------------------------------------- #
def advection_1d_upwind(u0, dt, dx, n_steps, a=1.0, record=False):
    """First-order upwind advection solver (stable for ``a dt / dx <= 1``).

    The canonical transport problem; upwinding adds numerical diffusion
    (2nd-order side effect) -- the whole teaching point.
    """
    u = np.array(u0, dtype=float, copy=True)
    nu = a * dt / dx
    if nu > 1.0 + 1e-12:
        raise ValueError(f"upwind CFL violated: nu={nu:.4f} > 1")
    hist = [u.copy()] if record else None
    for _ in range(n_steps):
        u_new = u.copy()
        u_new[1:] = u[1:] - nu * (u[1:] - u[:-1])
        u = u_new
        if record:
            hist.append(u.copy())
    if record:
        return u, np.asarray(hist)
    return u


# --------------------------------------------------------------------------- #
# 2D Poisson   laplacian(u) = -rho   (Dirichlet u=0 on the boundary)
# --------------------------------------------------------------------------- #
def poisson_2d_jacobi(rho, dx=1.0, tol=1e-10, max_iter=200000):
    """Solve the 2D Poisson equation with (plain) Jacobi iteration.

    ``rho`` is the source on the full (n x n) grid; the boundary is held at
    zero.  Returns ``(u, iters)``.  Jacobi is the reference (slow) method --
    it converges but slowly, which motivates SOR.
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[0]
    dx2 = dx * dx
    u = np.zeros_like(rho)
    for it in range(1, max_iter + 1):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] + dx2 * rho[1:-1, 1:-1]
        )
        if np.max(np.abs(u_new - u)) < tol:
            return u_new, it
        u = u_new
    return u, max_iter


def poisson_2d_sor(rho, dx=1.0, omega=None, tol=1e-10, max_iter=200000):
    """Solve the 2D Poisson equation with red-black SOR.

    Red-black ordering lets each sweep be applied point-by-point in a stable
    order, and over-relaxation with ``omega`` near the optimal value
    ``2 / (1 + sin(pi dx))`` converges far faster than Jacobi -- the key
    practical PDE lesson.  Returns ``(u, iters)``.
    """
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[0]
    dx2 = dx * dx
    if omega is None:
        omega = min(2.0 / (1.0 + np.sin(np.pi * dx)), 1.9999)
    i, j = np.indices((n, n))
    interior = (i >= 1) & (i <= n - 2) & (j >= 1) & (j <= n - 2)
    even = interior & (((i + j) % 2) == 0)
    odd = interior & (((i + j) % 2) == 1)
    even_idx = np.argwhere(even)
    odd_idx = np.argwhere(odd)
    u = np.zeros_like(rho)
    for it in range(1, max_iter + 1):
        u_old = u.copy()
        for idx in (even_idx, odd_idx):
            for ii, jj in idx:
                u[ii, jj] = (1.0 - omega) * u[ii, jj] + (omega / 4.0) * (
                    u[ii - 1, jj] + u[ii + 1, jj] + u[ii, jj - 1] + u[ii, jj + 1] + dx2 * rho[ii, jj]
                )
        if np.max(np.abs(u - u_old)) < tol:
            return u, it
    return u, max_iter
