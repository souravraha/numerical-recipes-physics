# Numerical Recipes in Physics

Foundational numerical routines in physics -- ODEs, eigenvalues, PDEs, FFT,
Monte Carlo, stochastic dynamics, and optimization -- implemented from scratch
(NumPy only) and taught **verification-first**, spanning mechanics, EM,
quantum, stat-mech, and molecular physics.

## The verification-first philosophy

A numerical routine is not trusted until three independent checks pass. Every
module ships tests that verify all three:

1. **Convergence** -- the discretisation error shrinks at the expected order
   when the step size is halved.
2. **Conservation laws** -- the invariants the physics demands (energy,
   probability, equipartition, ...) are respected.
3. **Independent reference** -- the result agrees with an analytic solution,
   SciPy, or a symmetry, computed by a different code path.

The shared helpers live in [`verify.py`](src/numerical_recipes/verify.py) so
each module can spend its lines on the physics rather than the bookkeeping.

## Modules

| Module | Methods | Physics it serves |
|---|---|---|
| `ode_symplectic` | leapfrog, velocity-Verlet, RK4 | celestial mechanics, phonons, molecular dynamics |
| `ode_adaptive` | embedded RK45 (Dormand-Prince) | circuits, population dynamics, non-stiff IVPs |
| `linalg_eig` | Householder tridiagonalisation, Sturm bisection, inverse iteration, Lanczos | quantum wells, phonon chains, stiffness spectra |
| `pde_fdm` | finite differences: heat, wave, advection, Poisson | thermal, acoustics, fluid, electrostatics |
| `spectral_fft` | FFT Poisson, split-operator Schrodinger, spectral stepping | EM, quantum, acoustics |
| `mcmc_ising` | Metropolis and Wolff cluster Monte Carlo | the 2D Ising phase transition |
| `stochastic` | Euler-Maruyama, Langevin (Bao-Wu) | Brownian motion, thermostats, Ornstein-Uhlenbeck |
| `minimization` | gradient descent, Newton, BFGS, Nelder-Mead | molecular ground states, data fitting |

Units and constants are set per problem inside each routine; nothing is tied
to a single unit system.

## The atlas -- flagship case studies

On top of the primitives, [`src/numerical_recipes/atlas/`](src/numerical_recipes/atlas/)
is a subpackage of ten **worked physics applications**. Each one composes the
shared routines above into a real simulation and is checked the same way --
convergence, conservation, and an independent analytic reference. Nothing is
re-implemented; every project imports the foundational modules.

| Project | Physics | Built on |
|---|---|---|
| `bh_shadow` | Schwarzschild/Kerr black-hole ray tracing & the photon shadow | `ode_adaptive` (null geodesics), `minimization` (critical impact parameter) |
| `ising_fss` | 2D Ising finite-size scaling & critical exponents | `mcmc_ising` (Wolff), `verify` |
| `gp_vortices` | Gross-Pitaevskii condensate & quantized vortex dynamics | `spectral_fft` (nonlinear Schrodinger), `stochastic` |
| `ns2d_turb` | 2D incompressible Navier-Stokes turbulence (vorticity) | `spectral_fft` (FFT Poisson), `verify` |
| `lattice_ft` | Lattice gauge theory -- 2D compact U(1), phases & plaquette | `mcmc` (Metropolis), `verify` |
| `pic_plasma` | Particle-in-cell plasma & Langmuir oscillations | `pde_fdm` / `spectral_fft` (Poisson), `ode_symplectic` |
| `graphene_tb` | Graphene Dirac cone + SSH edge/topological states | `linalg_eig` (eigh_symmetric) |
| `wavepacket_scatt` | Quantum wave-packet scattering, `T + R = 1` | `spectral_fft` (split-operator) |
| `md_greenkubo` | Molecular dynamics + Green-Kubo / Einstein diffusion | `ode_symplectic` (velocity-Verlet), `stochastic` |
| `bayesian_infer` | Bayesian MCMC inference on a damped-oscillator fit | `mcmc` (HMC), `minimization` (BFGS/Nelder-Mead) |

Run any single study headlessly (plots go to `output/`):

```bash
python -m examples.bh_shadow        # or: ./examples/run_examples.sh
```

## Get started
## Get started

The environment is managed with [Pixi](https://pixi.sh) (conda-forge
channels); the lockfile is committed for reproducibility.

```bash
# install the environment (numpy, scipy, pytest, Python 3.10)
pixi install

# run the full verification-first test suite
pixi run test

# run the ten atlas case studies (figures -> output/)
pixi run examples
```

Without Pixi, any Python >= 3.10 with `numpy`, `scipy`, and `pytest` works:

```bash
pip install numpy scipy pytest
pytest -q
```

## C port

Every module is also available as a from-scratch **C11** port in [`c/`](c/):
the same verification-first tests and the same hand-rolled algorithms, with no
dependency beyond the C standard library and `-lm` (no NumPy, FFTW, or BLAS).
The radix-2 FFT, the RNG (xoshiro256** + Box-Muller), the Householder
tridiagonaliser, the Lanczos iteration, the split-operator propagator, and the
Metropolis/Wolff sweeps are all written by hand.

```bash
cd c
make test      # build build/libnr.a + every test, run them all, fail on first exit != 0
make clean     # wipe build/
```

Built with `-std=c11 -O2 -Wall -Wextra`. Conventions: row-major `double`
arrays; the caller supplies output buffers (optional history/trajectory buffers
are nullable); callbacks carry a `void *ud` payload; complex data uses C99
`double complex`. All nine physics modules plus `verify` are ported, with
`common` holding the shared matrix helpers and RNG.

## Development

A top-level [`Makefile`](Makefile) gives one entry point for both languages.
From the repository root, after activating the Python environment
(`pixi shell`):

```bash
make test      # run the C and Python test suites together (default)
make c-test    # C port only            (c/)
make py-test   # Python package only    (src/ + tests/)
make examples  # run the ten atlas case studies (figures -> output/)
make clean     # remove C build artifacts (c/build/)
```

Continuous integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs all three checks on every push and pull request: the C suite with warnings
as errors, the C suite under AddressSanitizer + UBSan, and the Python suite.

## Layout

```
src/numerical_recipes/   the Python routines (one family per module) + verify.py
tests/                   one Python test file per module; checks (1)-(3) above
c/                       the from-scratch C11 port (include/nr, src, tests, Makefile)
Makefile                 top-level build/test entry point for both languages
.github/workflows/       CI: C (-Werror, ASan+UBSan) + Python (pytest)
```

## References

The routines are cited to their primary sources, as is standard in research.
See [`REFERENCES.md`](REFERENCES.md) for the full bibliography -- landmark
papers including Verlet (1967), Cooley-Tukey (1965), Metropolis (1953),
Onsager (1944), Wolff (1986), Lanczos (1951), Dormand-Prince (1980),
Nelder-Mead (1965), and more. Each module's docstring (Python) and header (C)
also cites its references inline.

## License

MIT -- see [LICENSE](LICENSE).
