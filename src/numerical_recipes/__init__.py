"""Foundational numerical routines in physics, taught verification-first.

Each module implements one family of numerical routines from scratch (NumPy
only), paired with tests that check the three things a numerical physicist
always checks: (a) it converges at the expected order, (b) it respects the
relevant conservation laws, and (c) it agrees with an independent reference
(an analytic solution, SciPy, or a symmetry).

  * ``ode_symplectic``  -- leapfrog / velocity-Verlet / RK4 for Hamiltonian
    systems; celestial mechanics, phonons, molecular dynamics.
  * ``ode_adaptive``    -- adaptive (Dormand-Prince) embedded RK45; circuits,
    population dynamics, stiff-free initial-value problems.
  * ``linalg_eig``      -- Householder tridiagonalisation, Sturm bisection,
    inverse iteration, and Lanczos; quantum wells, phonons, stiffness.
  * ``pde_fdm``         -- finite differences: heat, wave, advection, Poisson;
    thermal, acoustics, fluid, electrostatics.
  * ``spectral_fft``    -- FFT-based Poisson, split-operator Schrodinger,
    spectral time-stepping; EM, quantum, acoustics.
  * ``mcmc_ising``      -- Metropolis and Wolff cluster Monte Carlo for the
    2D Ising model; the canonical phase-transition study.
  * ``stochastic``      -- Euler-Maruyama and Langevin dynamics; Brownian
    motion, thermostats, Ornstein-Uhlenbeck.
  * ``minimization``    -- gradient descent, Newton, BFGS, Nelder-Mead;
    molecular and data-fitting energy minimisation.

Units and constants are set per problem inside each routine; nothing here is
tied to a single unit system.
"""

__version__ = "0.1.0"

from . import (
    mcmc,
    linalg_eig,
    mcmc_ising,
    minimization,
    ode_adaptive,
    ode_symplectic,
    pde_fdm,
    spectral_fft,
    stochastic,
    verify,
)

__all__ = [
    "mcmc",
    "linalg_eig",
    "mcmc_ising",
    "minimization",
    "ode_adaptive",
    "ode_symplectic",
    "pde_fdm",
    "spectral_fft",
    "stochastic",
    "verify",
]
