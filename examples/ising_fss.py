"""2D Ising finite-size scaling: extract gamma/nu and alpha/nu at Tc.

Reuses `mcmc_ising` (Wolff cluster MC).  Fits chi(L) ~ L^{gamma/nu} and
C(L) ~ L^{alpha/nu} from a set of lattice sizes and prints the estimates
(2D Ising: gamma/nu = 22/9, alpha/nu = 0).
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import ising_fss as ifss
from ._common import savefig


def main():
    Tc = ifss.onsager_tc()
    res = ifss.fit_exponents(T=Tc, sizes=(8, 16, 32), seed=0,
                             n_sweeps=600, equilibration=150)
    print(f"Onsager Tc              = {Tc:.5f} kT/J")
    print(f"gamma/nu (from chi(L))  = {res['gamma_over_nu']:.3f}   (theory 22/9 = 2.444)")
    print(f"alpha/nu (from C(L))    = {res['alpha_over_nu']:.3f}   (theory 0)")
    L, chi, C = res["L"], res["chi"], res["C"]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.5))
    ax[0].loglog(L, chi, "-o")
    ax[0].set_title(f"chi ~ L^(gamma/nu)\n slope {res['gamma_over_nu']:.2f}")
    ax[0].set_xlabel("lattice size L"); ax[0].set_ylabel("susceptibility chi")
    ax[1].loglog(L, C, "-s")
    ax[1].set_title(f"C ~ L^(alpha/nu)\n slope {res['alpha_over_nu']:.2f}")
    ax[1].set_xlabel("lattice size L"); ax[1].set_ylabel("specific heat C")
    fig.tight_layout()
    print("saved:", savefig(fig, "ising_fss.png"))


if __name__ == "__main__":
    main()
