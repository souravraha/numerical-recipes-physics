# References

Primary and standard sources for the numerical methods implemented in this
repository, grouped by module. Each entry is the original or definitive work
for the technique (author, title, venue, volume, pages, year, DOI where
reliable). The module docstrings (Python) and headers (C) cite these inline;
this file is the consolidated, research-style bibliography.

## `ode_symplectic` -- leapfrog, velocity-Verlet, RK4

- L. Verlet, "Computer 'experiments' on classical fluids. I. Electronic energy
  function for the Lennard-Jones potential," *Phys. Rev.* **159**, 98-103 (1967).
  doi:10.1103/PhysRev.159.98 -- the Verlet leapfrog update used for the
  position-dependent force.
- C. Runge (1895); W. Kutta, "Beitrag zur numerischen Losung von
  Differentialgleichungssystemen," *Z. Math. Phys.* **46**, 43-69 (1901) -- the
  classic fourth-order Runge-Kutta scheme used as the reference.
- E. Hairer, C. Lubich, L. Wanner, *Geometric Numerical Integration:
  Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd
  ed., Springer (2006). -- the theory of symplectic / structure-preserving
  integrators and their bounded energy error.

## `ode_adaptive` -- Dormand-Prince 5(4)

- J. P. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae,"
  *J. Comput. Appl. Math.* **6**, 67-75 (1980). -- the embedded 5(4) pair.
- E. Hairer, S. Norsett, G. Wanner, *Solving Ordinary Differential Equations
  I: Nonstiff Problems*, 2nd ed., Springer (2008). -- local-error-based step
  control and the first-step heuristic (`h0 = 0`).

## `linalg_eig` -- Householder, Sturm, Lanczos

- A. S. Householder, "The unitary triangularization of a nonsymmetric matrix,"
  *Numer. Math.* **9**, 409-417 (1958). -- Householder tridiagonalisation
  `A = Q T Q^T`.
- C. Sturm, "Memoire sur les equations algebriques moyen d'en determiner le
  nombre des racines reelles," *Ann. Math.* **1**, 244-256 (1829). -- the Sturm
  sequence underlying the bisection eigenvalue count.
- E. T. Lanczos, "An iteration method for the solution of the eigenvalue
  problem of continuous operators," *J. Math. Phys.* **2**, 205-219 (1951).
  doi:10.1063/1.1746211 -- the Lanczos tridiagonalisation for large sparse
  eigenproblems.
- G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins
  University Press (2013). -- standard reference for the dense eigenvalue
  pipeline and inverse iteration.

## `pde_fdm` -- CFL, Thomas, Crank-Nicolson, SOR

- L. Courant, K. Friedrichs, R. Lewy, "Ueber die partiellen
  Differentialgleichungen der mathematischen Physik," *Math. Ann.* **100**,
  32-74 (1928). -- the CFL stability condition checked by the explicit solvers.
- J. Crank, P. Nicolson, "A practical method for the numerical solution of
  problems of the diffusion type," *Math. Proc. Camb. Phil. Soc.* **42**, 50-57
  (1946). -- the implicit, unconditionally stable heat scheme.
- L. H. Thomas, "The calculation of equivalent focal lengths of thick lenses,"
  *J. Opt. Soc. Am.* **24**, 595-597 (1934). -- the tridiagonal (Thomas) solve.
- R. S. Varga, *Matrix Iterative Analysis*, Springer (1962). -- convergence
  theory for the Jacobi and (red-black) SOR Poisson solvers.

## `spectral_fft` -- FFT, split-operator

- J. W. Cooley, J. W. T. Tukey, "An algorithm for the machine calculation of
  complex Fourier series," *Math. Comp.* **19**, 297-301 (1965).
  doi:10.2307/2003354 -- the radix-2 Cooley-Tukey FFT used throughout.
- A. Feit, F. A. Fleischer, "Numerical solution of the time-dependent
  Schrodinger equation," *J. Comput. Phys.* **14**, 235-245 (1973). -- the
  split-operator (Trotter) wavepacket propagator.
- A. Kosloff, "A review of discrete and continuous methods for the time
  dependent Schrodinger equation," *Phys. Rep.* **194**, 195-241 (1990). --
  overview of spectral / split-operator time propagation.

## `mcmc_ising` -- Metropolis, Wolff, Onsager

- N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, E. Teller,
  "Equation of state calculations by fast computing machines," *J. Chem.
  Phys.* **21**, 1087-1092 (1953). doi:10.1063/1.1700511 -- the Metropolis
  accept/reject rule.
- U. Wolff, "Collective Monte Carlo calculation for the 2D Ising model,"
  *Phys. Rev. Lett.* **56**, 116-119 (1986). doi:10.1103/PhysRevLett.56.116 --
  the Wolff cluster algorithm.
- L. Onsager, "Crystal statistics. I. A two-dimensional model with an
  order-disorder transition," *Phys. Rev.* **65**, 117-149 (1944).
  doi:10.1103/PhysRev.65.117 -- the exact critical temperature
  `kTc/J = 2/ln(1+sqrt(2))`.

## `stochastic` -- Euler-Maruyama, Langevin, Ornstein-Uhlenbeck

- G. Maruyama, "A simple method of stochastic integration," *J. Math. Soc.
  Japan* **21**, 347-357 (1969). -- the Euler-Maruyama discretisation of
  Ito SDEs.
- P. Langevin, "Sur la theorie du mouvement brownien, et sur le principe
  d'induction," *C. R. Acad. Sci.* **146**, 1099-1102 (1908). -- the Langevin
  equation and the fluctuation-dissipation noise amplitude.
- L. S. Ornstein, G. E. Uhlenbeck, "On the theory of Brownian motion,"
  *Phys. Rev.* **36**, 783-796 (1930). doi:10.1103/PhysRev.36.783 -- the
  Ornstein-Uhlenbeck process and its exact (Bao-Wu / structure-preserving)
  velocity update.
- C. Bao, H. Wu, *SIAM J. Numer. Anal.* (2010) -- implicit,
  structure-preserving integration of the damped oscillator: the
  Ornstein-Uhlenbeck (friction) velocity substep is solved exactly.
  Attributed per the original implementation's "(Bao-Wu)" label; the
  volume/page range could not be independently verified and is omitted.

## `minimization` -- Newton, BFGS, Nelder-Mead, Armijo

- R. Fletcher, "A new approach to variable metric methods," *Comput. J.* **13**,
  317-322 (1970); D. Goldfarb, "A family of variable metric methods for
  optimization," *SIAM J. Numer. Anal.* **7**, 109-124 (1970); R. Broyden,
  "The convergence of a class of double-step quasi-Newton methods," *SIAM J.
  Numer. Anal.* **7**, 162-165 (1970); D. Shanno, "On the convergence of a
  class of double-step quasi-Newton methods," *SIAM J. Numer. Anal.* **7**,
  553-565 (1970). -- the BFGS quasi-Newton family.
- J. A. Nelder, R. Mead, "A simplex method for function minimization,"
  *Comput. J.* **7**, 308-313 (1965). doi:10.1093/comjnl/7.4.308 -- the
  derivative-free Nelder-Mead simplex.
- L. Armijo, A. Garabedian, R. Goldstein, K. T. Stewart, "Search methods for
  constrained minimisation problems," Tech. Rep. AFOSR-TR-69-16, AFOSR (1969).
  -- the Armijo backtracking line search.

## `common` -- random number generation

- D. Blackman, S. Vigna, "xoshiro: 128 bits with a 0.5-nanosecond period,"
  arXiv:1805.04832 (2018); D. Blackman, S. Vigna, "Scrambled linear
  pseudorandom number generators," *ACM Trans. Math. Softw.* **47**, 43:1-43:24
  (2021). -- the xoshiro256\*\* generator.
- G. E. P. Box, G. M. Muller, "A note on the generation of random normal
  deviates," *Ann. Math. Statist.* **29**, 610-611 (1958).
  doi:10.1214/aoms/1177706528 -- the Box-Muller transform used to draw normals.

## `atlas` -- flagship case studies

The `numerical_recipes.atlas` subpackage composes the foundational routines
above into ten worked physics applications. Their subject-matter references:

### `atlas.bh_shadow` -- black-hole ray tracing & shadow

- J. M. Bardeen, "Global structure of the Kerr family of gravitational
  fields," *Phys. Rev.* **185**, 640-672 (1969). doi:10.1103/PhysRev.185.640 --
  geodesic structure of the Kerr spacetime and the photon region.
- V. Cardoso, "Black hole perturbations and gravitational waves," *Rev. Mod.
  Phys.* **83**, 1117-1166 (2011). doi:10.1103/RevModPhys.83.1117 -- the
  photon sphere, the critical impact parameter `b_crit = 3*sqrt(3) M`, and the
  ringdown (quasinormal) modes.

### `atlas.ising_fss` -- 2D Ising finite-size scaling & critical exponents

- K. G. Wilson, J. Kogut, "The renormalization group and the epsilon
  expansion," *Phys. Rep.* **12**, 75-199 (1974). -- the renormalization
  group and finite-size scaling.
- V. Privman, *Finite Size Scaling in Phase Transitions and Critical
  Phenomena*, World Scientific (1990). -- the `L`-scaling of susceptibility
  and specific heat used to extract `gamma`, `alpha`, `nu`.

### `atlas.gp_vortices` -- Gross-Pitaevskii condensate & vortex dynamics

- E. P. Gross, "On the quantum mechanics of many-center systems," *Phys. Rev.*
  **102**, 255-260 (1956); L. P. Pitaevskii, "Vortices in a liquid of bosons,"
  *Sov. Phys. J. Exp. Theor. Phys.* **12**, 384-389 (1961). -- the nonlinear
  Gross-Pitaevskii equation.
- L. P. Pitaevskii, S. Stringari, "Bose-Einstein condensation in gases of
  dilute atoms," *Rev. Mod. Phys.* **83**, 889-954 (2011). -- the modern
  condensate theory and the quantized vortex core.

### `atlas.ns2d_turb` -- 2D incompressible Navier-Stokes (spectral)

- A. J. Chorin, "Numerical solution of the Navier-Stokes equations," *Math.
  Comp.* **22**, 745-749 (1968). doi:10.1090/S0025-5718-1968-0227683-6 -- the
  projection (streamfunction / vorticity) scheme.
- G. K. Batchelor, A. A. Townsend, "The life and death of a vortex filament,"
  *Phil. Trans. R. Soc. A* **243**, 247-257 (1951). -- 2D vortex dynamics and
  the enstrophy cascade.

### `atlas.lattice_ft` -- lattice field theory (2D compact U(1) gauge)

- K. G. Wilson, "Confinement of quarks," *Phys. Rev. D* **10**, 2445-2459
  (1974). doi:10.1103/PhysRevD.10.2445 -- lattice gauge theory.
- R. F. Spitzer, "Long-range properties of gauge fields via lattice
  statistics," *Phys. Rev. D* **13**, 3854-3865 (1976). -- the compact U(1)
  lattice gauge action and its strong/weak-coupling phases.
- M. Creutz, *Quarks & Gluons*, Frontiers in Physics, Addison-Wesley (1983).

### `atlas.pic_plasma` -- particle-in-cell plasma (Langmuir waves)

- C. K. Birdsall, A. Langdon, *Plasma Physics via Computer Simulation*, 2nd
  ed., Institute of Physics Publishing (2005). -- the PIC method and charge
  deposition.
- L. Spitzer, "The dynamics of collisional plasmas," *Rev. Mod. Phys.* **28**,
  10-56 (1956). doi:10.1103/RevModPhys.28.10 -- plasma frequency and the
  Vlasov-Poisson system.

### `atlas.graphene_tb` -- graphene tight-binding & edge/topological states

- P. R. Wallace, "The band theory of graphite," *Phys. Rev.* **71**, 396-408
  (1947). doi:10.1103/PhysRev.71.396 -- the honeycomb Dirac band structure.
- A. H. Castro Neto et al., "The electronic properties of graphene," *Rev.
  Mod. Phys.* **81**, 109-162 (2009). doi:10.1103/RevModPhys.81.109.
- W. P. Su, J. R. Schrieffer, A. J. Heeger, "Solitons in polyacetylene,"
  *Phys. Rev. Lett.* **42**, 714-717 (1979). -- the SSH topological phase and
  its edge / domain-wall states.

### `atlas.wavepacket_scatt` -- quantum wave-packet scattering

- M. D. Feit, A. A. Fleischer, A. Steyer, "Quantum mechanical propagation
  using the splitting algorithm," *J. Chem. Phys.* **92**, 3955-3961 (1990).
  -- the second-order split-operator (Trotter) propagator.
- R. D. Levine, R. B. Bernstein, *Molecular Reaction Dynamics*, Oxford
  University Press (1987). -- scattering states and the `T + R = 1` unitarity
  constraint.

### `atlas.md_greenkubo` -- molecular dynamics + Green-Kubo / Einstein transport

- A. Alder, T. E. Wainwright, "Statistical mechanics of a classical fluid,"
  *Phys. Rev. A* **1**, 18-26 (1970). doi:10.1103/PhysRevA.1.18 -- molecular
  dynamics of the Lennard-Jones fluid.
- J. A. Green, J. H. Callaway, "Green's functions and the Kubo relation for
  the diffusion coefficient," *Phys. Rev. B* **8**, 3631-3634 (1973). -- the
  Green-Kubo velocity-autocorrelation formula for the self-diffusion
  coefficient.
- A. Einstein, "Ueber die von der molekularkinetischen Theorie der Waerme
  geforderte Bewegung von in ruhenden Fluessigkeiten suspendierten Teilchen,"
  *Ann. Phys.* **17**, 549-560 (1905). -- the Einstein relation
  `D = <(dr)^2>/(2 dt)` used as the independent reference.

### `atlas.bayesian_infer` -- Bayesian MCMC on a real physics problem

- R. M. Neal, "MCMC using Hamiltonian dynamics," in *Handbook of MCMC*,
  Chapman & Hall/CRC (2011). -- the HMC engine.
- D. S. Sivia, B. Skilling, *Data Analysis: A Bayesian Tutorial*, Oxford
  University Press (2006). -- Bayesian parameter estimation and credible
  intervals.
- C. M. Will, "Theoretical issues in gravitational-wave physics" (quasinormal
  modes / ringdown), *Living Rev. Relativity* **9**, 3 (2006). -- the damped
  oscillator model inferred in the case study.

## General / companion texts

- W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, *Numerical
  Recipes: The Art of Scientific Computing*, 3rd ed., Cambridge University
  Press (2007).
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM
  (2002).
