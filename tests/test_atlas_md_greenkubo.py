import numpy as np

from numerical_recipes import ode_symplectic as os
from numerical_recipes.atlas import md_greenkubo as md

N, L, DT = 36, 7.0, 0.005


def test_energy_conserved():
    # NVE MD conserves the total (kinetic + potential) energy to a small drift.
    x, v = md._init(N, L, T=1.0)
    accel = md.lj_accels(N, L)
    E0 = md.lj_energy(x, v, N, L)
    for _ in range(300):
        x, v = os.velocity_verlet(x, v, DT, accel, 1)
    E1 = md.lj_energy(x, v, N, L)
    assert abs(E1 - E0) / abs(E0) < 0.05


def test_msd_grows_diffusively():
    # A liquid is diffusive: the MSD grows with time (not ballistic forever).
    pos, vel = md.equilibrate_and_run(N, L, T=1.0, dt=DT, n_eq=300, n_run=800)
    msd = md.mean_squared_displacement(pos, N)
    t = np.arange(len(msd)) * DT
    early = np.mean(msd[:len(msd)//10])
    late = np.mean(msd[-len(msd)//10:])
    assert late > early          # MSD grows in time (diffusive transport)


def test_greenkubo_and_einstein_positive():
    # Both transport routes give a positive self-diffusion coefficient.
    pos, vel = md.equilibrate_and_run(N, L, T=1.0, dt=DT, n_eq=300, n_run=800)
    D_e = md.diffusion_einstein(md.mean_squared_displacement(pos, N), DT)
    D_g = md.diffusion_greenkubo(md.vacf(vel, N), DT)
    assert D_e > 0.0
    assert D_g > 0.0


def test_einstein_and_greenkubo_agree():
    # The Einstein (MSD) and Green-Kubo (VACF) estimates are the same D, so
    # they should be of the same order of magnitude.
    pos, vel = md.equilibrate_and_run(N, L, T=1.0, dt=DT, n_eq=400, n_run=1000)
    D_e = md.diffusion_einstein(md.mean_squared_displacement(pos, N), DT)
    D_g = md.diffusion_greenkubo(md.vacf(vel, N), DT)
    ratio = max(D_e, D_g) / (min(D_e, D_g) + 1e-12)
    assert ratio < 12.0
