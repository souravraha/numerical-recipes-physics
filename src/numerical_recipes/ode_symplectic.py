"""Symplectic integrators for Hamiltonian systems, plus an RK4 reference.

The point of this module is the *contrast*.  A generic 4th-order integrator
(RK4) is accurate over a short time window, but on a conservative (Hamiltonian)
system it slowly leaks or gains energy, so orbits precess and long
trajectories drift off the true energy surface.  A *symplectic* integrator
(leapfrog / velocity-Verlet) preserves the phase-space area element, so on a
Hamiltonian system with a position-dependent force the energy error stays
*bounded* (it oscillates) instead of drifting -- which is what makes them the
default for celestial mechanics, molecular dynamics, and lattice simulations.

Every routine here takes the acceleration as a callable ``a(x, v)`` and
integrates the *entire* trajectory in one call (``n_steps`` is the total
number of steps), starting from position ``x`` and velocity ``v``.  NumPy
only.


References (see REFERENCES.md):
  - L. Verlet, Phys. Rev. 159, 98 (1967) [leapfrog / velocity-Verlet].
  - C. Runge (1895); W. Kutta, Z. Math. Phys. 46, 43 (1901) [RK4].
  - E. Hairer, C. Lubich, L. Wanner, Geometric Numerical Integration, Springer (2006)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "leapfrog",
    "velocity_verlet",
    "langevin_verlet",
    "rk4",
    "kepler",
    "chain_accel",
    "chain_energy",
    "double_pendulum",
    "double_pendulum_energy",
]


# --------------------------------------------------------------------------- #
# Generic integrators
# --------------------------------------------------------------------------- #
def leapfrog(x, v, dt, accel, n_steps=1, record=False):
    """Kick-drift-kick (leapfrog / staggered) integrator.

    Integrates ``x'' = accel(x, v)`` from position ``x`` and velocity ``v``
    for ``n_steps`` steps of size ``dt``.  Second order, symplectic, and
    time-reversible (when the force is position-dependent).

    Returns ``(x, v)`` or ``(x, v, x_hist)`` if ``record`` is True, where
    ``x_hist`` has shape ``(n_steps+1, ...)``.
    """
    x = np.array(x, dtype=float, copy=True)
    v = np.array(v, dtype=float, copy=True)
    hist = [x.copy()] if record else None

    v = v + 0.5 * dt * accel(x, v)          # initial half kick
    for _ in range(n_steps):
        x = x + dt * v                       # drift
        if hist is not None:
            hist.append(x.copy())
        v = v + dt * accel(x, v)             # kick
    if record:
        return x, v, np.asarray(hist)
    return x, v


def velocity_verlet(x, v, dt, accel, n_steps=1, record=False):
    """Velocity-Verlet integrator (same order and symplecticity as leapfrog).

    Keeps positions and velocities on the *same* time grid (no staggering),
    which is convenient when both are needed at every step.
    """
    x = np.array(x, dtype=float, copy=True)
    v = np.array(v, dtype=float, copy=True)
    a = accel(x, v)
    hist = [x.copy()] if record else None

    for _ in range(n_steps):
        x = x + dt * v + 0.5 * dt * dt * a
        if hist is not None:
            hist.append(x.copy())
        a_new = accel(x, v)
        v = v + 0.5 * dt * (a + a_new)
        a = a_new
    if record:
        return x, v, np.asarray(hist)
    return x, v


def rk4(x, v, dt, accel, n_steps=1, record=False):
    """Classic 4th-order Runge-Kutta for ``x'' = accel(x, v)``.

    Provided as a *reference* to show the energy drift that motivates
    symplectic integrators on conservative systems.  Not symplectic.
    """
    x = np.array(x, dtype=float, copy=True)
    v = np.array(v, dtype=float, copy=True)
    nx = x.shape[0]
    hist = [x.copy()] if record else None

    def rhs(yy):
        xx = yy[:nx]
        vv = yy[nx:]
        return np.concatenate([vv, accel(xx, vv)])

    y = np.concatenate([x, v])
    for _ in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x, v = y[:nx], y[nx:]
        if hist is not None:
            hist.append(x.copy())
    if record:
        return x, v, np.asarray(hist)
    return x, v


# --------------------------------------------------------------------------- #
# Kepler two-body problem (celestial mechanics) -- the flagship demo
# --------------------------------------------------------------------------- #
def kepler(x, v=None, GM=1.0):
    """Gravitational acceleration of a test particle about a mass GM at origin.

    ``a = -GM r / |r|^3``.  Position-dependent, so ``v`` is accepted only for
    signature compatibility with the ``accel(x, v)`` convention and is ignored.
    With GM=1 a particle at unit distance on a circular orbit has period 2*pi.
    """
    r = np.linalg.norm(x)
    return -GM * x / (r ** 3)


def kepler_energy(x, v, GM=1.0):
    """Specific orbital energy 0.5|v|^2 - GM/|r| (constant on a true orbit)."""
    r = np.linalg.norm(x)
    return float(0.5 * np.sum(v ** 2) - GM / r)


# --------------------------------------------------------------------------- #
# Harmonic oscillator chain (phonons / condensed matter)
# --------------------------------------------------------------------------- #
def chain_accel(u, v, k=1.0, m=1.0):
    """Acceleration of a 1D harmonic chain fixed at both ends.

    ``a_i = (k/m)(u_{i-1} + u_{i+1} - 2 u_i)`` with ghost ends held at zero.
    The normal modes are standing waves -- the phonon dispersion.
    """
    u = np.asarray(u, dtype=float)
    n = u.size
    if n == 1:
        return np.zeros_like(u)
    a = np.empty_like(u)
    a[0] = (k / m) * (u[1] - 2.0 * u[0])
    a[-1] = (k / m) * (u[-2] - 2.0 * u[-1])
    if n > 2:
        a[1:-1] = (k / m) * (u[:-2] + u[2:] - 2.0 * u[1:-1])
    return a


def chain_energy(u, v, k=1.0, m=1.0):
    """Total energy of a 1D harmonic chain fixed at both ends.

    Kinetic energy plus the spring potential, *including* the two wall
    springs to the fixed (zero) ends that ``chain_accel``'s boundary
    conditions imply.  Only with those terms is this the first integral of
    ``chain_accel``.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    kinetic = 0.5 * m * np.sum(v ** 2)
    potential = 0.5 * k * (np.sum(np.diff(u) ** 2) + u[0] ** 2 + u[-1] ** 2)
    return float(kinetic + potential)


# --------------------------------------------------------------------------- #
# Double pendulum (chaos) -- the butterfly-effect demo
# --------------------------------------------------------------------------- #
def double_pendulum(theta, omega, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """Angular accelerations of a double pendulum (angles from vertical)."""
    d = theta[0] - theta[1]
    cd, sd = np.cos(d), np.sin(d)
    den = 2.0 * m1 + m2 - m2 * np.cos(2.0 * d)

    a1 = (
        -g * (2.0 * m1 + m2) * np.sin(theta[0])
        - m2 * g * np.sin(theta[0] - 2.0 * theta[1])
        - 2.0 * sd * m2 * (omega[1] ** 2 * l2 + omega[0] ** 2 * l1 * cd)
    ) / (l1 * den)

    a2 = (
        2.0
        * sd
        * (
            omega[0] ** 2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta[0])
            + omega[1] ** 2 * l2 * m2 * cd
        )
    ) / (l2 * den)
    return np.array([a1, a2])


def double_pendulum_energy(theta, omega, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """Total energy of a double pendulum (kinetic + gravitational potential)."""
    th1, th2 = theta
    w1, w2 = omega
    v1_sq = l1 ** 2 * w1 ** 2
    v2_sq = v1_sq + l2 ** 2 * w2 ** 2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
    kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    potential = -m1 * g * l1 * np.cos(th1) - m2 * g * (l1 * np.cos(th1) + l2 * np.cos(th2))
    return float(kinetic + potential)


def langevin_verlet(x, v, dt, accel, gamma, T, n_steps=1, seed=0, m=1.0, record=False):
    """Langevin (velocity-Verlet) thermostatted integrator for an NVT ensemble.

    Integrates ``m x'' = F(x) - gamma m x' + sqrt(2 gamma T / m) xi(t)`` with
    ``F(x) = m * accel(x, v)`` and ``k_B = 1``.  This is the velocity-Verlet
    stepper with a friction term and a random kick whose amplitude is fixed by
    the fluctuation-dissipation relation, so an *arbitrary* force field (e.g. a
    many-body Lennard-Jones fluid) is driven to the canonical distribution:
    equipartition ``<v^2> = T/m`` per degree of freedom.  It is the generic
    generalisation of ``stochastic.langevin_oscillator`` (the 1-D harmonic
    special case) to many-body dynamics.

    Returns the final ``(x, v)`` (or ``(x, v, history)`` when ``record``).
    """
    rng = np.random.default_rng(seed)
    x = np.array(x, dtype=float, copy=True)
    v = np.array(v, dtype=float, copy=True)
    hist = [x.copy()] if record else None
    sigma_half = np.sqrt(gamma * T * dt / m)   # per half-kick (FDT) amplitude
    a = accel(x, v)
    for _ in range(n_steps):
        v = v + 0.5 * dt * (a - gamma * v) + sigma_half * rng.standard_normal(size=v.shape)
        x = x + dt * v
        if hist is not None:
            hist.append(x.copy())
        a = accel(x, v)
        v = v + 0.5 * dt * (a - gamma * v) + sigma_half * rng.standard_normal(size=v.shape)
    if record:
        return x, v, np.asarray(hist)
    return x, v
