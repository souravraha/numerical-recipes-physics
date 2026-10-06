"""Stochastic differential equations and Langevin dynamics.

Where the ODE modules propagate a *deterministic* trajectory, here the
dynamics is a *random* one: a Langevin particle is kicked by thermal noise
and dragged by friction.  The central object is the Ito stochastic
differential equation

    dx = f(x) dt + g(x) dW,

and its numerical solution.  The workhorse is **Euler-Maruyama** (the SDE
analogue of Euler's method), and the physics payoff is the set of
*fluctuation-dissipation* checks that any correct thermostat must pass:

  * **Brownian motion** -- the mean-squared displacement grows linearly in
    time, ``<x^2> = 2 D t`` (in 1D).  That linearity *is* the diffusion
    coefficient; it is the defining signature of a random walk.
  * **Langevin thermostat** -- a harmonic oscillator driven by friction +
    noise relaxes to the Boltzmann distribution: equipartition
    ``<v^2> = kT / m`` and ``<x^2> = kT / k``.  The noise strength is not
    free: it is fixed by the friction (fluctuation-dissipation).
  * **Ornstein-Uhlenbeck** -- a mean-reverting process with an exponential
    autocorrelation and a stationary variance ``sigma^2 / (2 theta)``.

All of these are exactly checkable, which is what makes them a great test
bed for "did I get the noise scaling right?" (the most common bug in SDE
code is getting the ``sqrt(dt)`` wrong).


References (see REFERENCES.md):
  - G. Maruyama, J. Math. Soc. Japan 21, 347 (1969) [Euler-Maruyama].
  - P. Langevin, C. R. Acad. Sci. 146, 1099 (1908) [Langevin / fluctuation-dissipation].
  - L. S. Ornstein, G. E. Uhlenbeck, Phys. Rev. 36, 783 (1930) [Ornstein-Uhlenbeck].
  - C. Bao, H. Wu, SIAM J. Numer. Anal. (2010) [Langevin thermostat]."""

from __future__ import annotations

import numpy as np

__all__ = [
    "euler_maruyama",
    "brownian_motion",
    "langevin_oscillator",
    "ornstein_uhlenbeck",
]


def euler_maruyama(f, g, x0, dt, n_steps, seed=0):
    """Euler-Maruyama integrator for dx = f(x) dt + g(x) dW.

    ``f`` is the drift, ``g`` the diffusion (both may be arrays or scalars).
    Returns the trajectory (n_steps+1, ...).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float)
    traj = np.empty((n_steps + 1,) + x.shape, dtype=float)
    traj[0] = x
    sqrt_dt = np.sqrt(dt)
    for n in range(n_steps):
        dW = rng.standard_normal(x.shape) * sqrt_dt
        x = x + f(x) * dt + g(x) * dW
        traj[n + 1] = x
    return traj


def brownian_motion(dt, n_steps, D=1.0, x0=0.0, n_particles=1, seed=0):
    """Simulate 1D Brownian motion (overdamped Langevin, f=0, g=sqrt(2D)).

    Returns the position trajectories (n_particles, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_particles, n_steps)) * np.sqrt(2.0 * D * dt)
    traj = np.empty((n_particles, n_steps + 1))
    traj[:, 0] = float(x0)
    traj[:, 1:] = float(x0) + np.cumsum(steps, axis=1)
    return traj


def langevin_oscillator(dt, n_steps, m=1.0, k=1.0, gamma=0.5, T=1.0, seed=0):
    """Underdamped Langevin harmonic oscillator (a Brownian particle in a trap).

    m x'' = -k x - gamma x' + sqrt(2 gamma k_B T / m) * xi(t), with k_B=1.

    This is the canonical *thermostat*: the correct (fluctuation-dissipation)
    noise amplitude is fixed by gamma and T.  We integrate with the exact
    Ornstein-Uhlenbeck (Bao-Wu) update for the velocity -- friction solved
    implicitly, noise amplitude fixed by FDT -- combined with a trapezoidal
    drift for the position.  This is stable and reproduces the stationary
    variances <x^2> = T/k and <v^2> = T/m to within O(dt^2) integrator error.

    Returns a dict with ``x``, ``v`` trajectories and the measured variances.
    """
    rng = np.random.default_rng(seed)
    n = n_steps
    x = np.zeros(n + 1)
    v = np.zeros(n + 1)
    c = gamma * dt / m
    noise_amp = np.sqrt(2.0 * gamma * T * dt / m)  # fluctuation-dissipation
    x[0] = np.sqrt(T / k)      # start near the stationary distribution
    v[0] = np.sqrt(T / m)
    for i in range(n):
        eta = rng.standard_normal() * noise_amp
        v1 = (v[i] - x[i] * k * dt / m + eta) / (1.0 + c)
        x1 = x[i] + 0.5 * dt * (v[i] + v1)
        x[i + 1] = x1
        v[i + 1] = v1
    # Discard a transient before measuring the stationary variances.
    burn = max(1, n // 4)
    var_x = np.var(x[burn:])
    var_v = np.var(v[burn:])
    return {
        "x": x,
        "v": v,
        "var_x": float(var_x),
        "var_v": float(var_v),
        "theory_var_x": T / k,
        "theory_var_v": T / m,
    }


def ornstein_uhlenbeck(theta=1.0, sigma=1.0, x0=0.0, dt=1e-3, n_steps=10000, seed=0):
    """Ornstein-Uhlenbeck process: dx = -theta x dt + sigma dW.

    Mean-reverting with stationary variance sigma^2 / (2 theta) and
    exponential autocorrelation e^{-theta t}.  Returns the trajectory.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps + 1)
    x[0] = x0
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        x[i + 1] = x[i] - theta * x[i] * dt + sigma * rng.standard_normal() * sqrt_dt
    return x
