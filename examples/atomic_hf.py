"""Helium (Hartree) and beryllium (Hartree-Fock) ground states.

Reuses `grid_hamiltonian` (FGH radial kinetic operator for He).  Prints the
total energies against their physics reference bands, the orbital
normalisation, and the Be orbital ordering, then plots the He 1s radial
density and the Be 1s/2s orbitals.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import atomic_hf as hf
from ._common import savefig


def main():
    he = hf.he_hartree(n=256, r_max=10.0, tol=1e-5, max_iter=200)
    dr = he["r"][1] - he["r"][0]
    nrm = float(np.sum(np.abs(he["p"]) ** 2) * dr)
    print(f"He Hartree    e_total = {he['e_total']:.5f} a.u.   (band -3.05 .. -2.70, HF -2.8617, exact -2.9037)")
    print(f"He 1s orbital normalised to {nrm:.5f}   (should be 1)")

    be = hf.be_roothaan(n=256, r_max=20.0, tol=1e-4, max_iter=100)
    print(f"Be Hartree-Fock e_total = {be['e_total']:.5f} a.u.   (band -15.05 .. -14.50, HF ~ -14.667)")
    print(f"Be orbital energies = {np.round(be['energies'], 4)}   (1s more bound than 2s: {be['energies'][0] < be['energies'][1] < 0})")

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].plot(he["r"], np.abs(he["p"]) ** 2, "b-", lw=2)
    ax[0].set_xlabel("r (a0)"); ax[0].set_ylabel("|p(r)|^2")
    ax[0].set_title(f"He 1s Hartree  (E_tot = {he['e_total']:.4f} a.u.)")

    r = be["r"]
    psi1 = be["coefficients"][0] @ np.array(be["basis"])
    psi2 = be["coefficients"][1] @ np.array(be["basis"])
    ax[1].plot(r, np.abs(psi1) ** 2, "b-", lw=2, label="1s^2")
    ax[1].plot(r, np.abs(psi2) ** 2, "r-", lw=2, label="2s^2")
    ax[1].set_xlabel("r (a0)"); ax[1].set_ylabel("|psi|^2")
    ax[1].set_title("Be 1s / 2s orbitals"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    print("saved:", savefig(fig, "atomic_hf.png"))


if __name__ == "__main__":
    main()
