"""Case-study "atlas": fourteen flagship computational-physics applications built
entirely from the foundational ``numerical_recipes`` routines.

Each module is a self-contained, verification-first worked example of a real
physics problem.  They *reuse* the shared library (``ode_adaptive``,
``spectral_fft``, ``mcmc``, ``mcmc_ising``, ``linalg_eig``, ``ode_symplectic``,
``minimization``) rather than re-implementing it -- the point is to show how
the primitives compose into a real simulation, and to check each against a
known analytic result or conservation law.

Modules
-------
  * bh_shadow        -- black-hole ray tracing & shadow (null geodesics).
  * ising_fss        -- 2D Ising finite-size scaling & critical exponents.
  * gp_vortices      -- Gross-Pitaevskii condensate & vortex dynamics.
  * ns2d_turb        -- 2D incompressible Navier-Stokes (spectral vorticity).
  * lattice_ft       -- lattice field theory (2D compact U(1) gauge).
  * pic_plasma       -- particle-in-cell plasma (Langmuir oscillations).
  * graphene_tb      -- graphene tight-binding & edge/topological states.
  * wavepacket_scatt -- quantum wave-packet scattering (T/R, unitarity).
  * md_greenkubo     -- molecular dynamics + Green-Kubo / Einstein transport.
  * bayesian_infer   -- Bayesian MCMC inference on a real physics problem.
  * fgh_bound_states -- FGH bound states: harmonic, square well, H, central.
  * nfe_bands        -- nearly-free-electron band structure (1D/2D).
  * atomic_hf        -- Hartree (He) and Hartree-Fock (Be) ground states.
  * kmt_dynamics     -- stochastic kinetochore-microtubule dynamics.

Import a module directly, e.g.  ``from numerical_recipes.atlas import bh_shadow``.
"""

__all__ = [
    "bh_shadow",
    "ising_fss",
    "gp_vortices",
    "ns2d_turb",
    "lattice_ft",
    "pic_plasma",
    "graphene_tb",
    "wavepacket_scatt",
    "md_greenkubo",
    "bayesian_infer",
    "fgh_bound_states",
    "nfe_bands",
    "atomic_hf",
    "kmt_dynamics",
]
