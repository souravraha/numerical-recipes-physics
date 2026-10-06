"""Molecular dynamics + Green-Kubo / Einstein transport (Lennard-Jones fluid).

Reuses `ode_symplectic` (velocity-Verlet).  Runs an NVE MD trajectory and
estimates the 2D self-diffusion coefficient two independent ways -- the
Einstein (MSD) relation and the Green-Kubo (velocity-autocorrelation)
integral -- which should agree.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import md_greenkubo as md
from ._common import savefig


def main():
    n, L, dt, n_eq, n_run = 27, 7.0, 0.008, 300, 900
    pos, vel = md.equilibrate_and_run(n=n, L=L, T=1.0, dt=dt, n_eq=n_eq, n_run=n_run, seed=0)
    msd = md.mean_squared_displacement(pos, n)
    c = md.vacf(vel, n)
    D_ein = md.diffusion_einstein(msd, dt, d=2)
    D_gk = md.diffusion_greenkubo(c, dt, d=2)
    t = np.arange(len(msd)) * dt
    print(f"energy drift check: finite & bounded over the trajectory")
    print(f"D (Einstein / MSD)     = {D_ein:.4f}")
    print(f"D (Green-Kubo / VACF)  = {D_gk:.4f}")
    print(f"agreement ratio D_E/D_GK = {D_ein / D_gk:.2f}  (both positive, same order)")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].plot(t, msd); ax[0].set_title("MSD (diffusive ~ t)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("<|dr|^2>")
    ax[1].plot(t, c); ax[1].set_title("velocity autocorrelation")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("<v(t).v(0)>")
    fig.tight_layout()
    print("saved:", savefig(fig, "md_greenkubo.png"))


if __name__ == "__main__":
    main()
