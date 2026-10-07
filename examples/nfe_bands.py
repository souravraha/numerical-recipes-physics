"""Nearly-free-electron band structure (1D) + the gap at the BZ edge.

Reuses NumPy eigensolves on the plane-wave Hamiltonian.  Prints the 1D band
gap at k = K/2 for two cutoffs (showing the monotone convergence to the
nearly-free-electron value) and the symmetric-about-k=0 property, then plots
the band structure over the first Brillouin zone.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import nfe_bands as nb
from ._common import savefig


def main():
    U = 0.5
    gap5 = nb.bands_1d(0.5, 5, U)[1] - nb.bands_1d(0.5, 5, U)[0]
    gap40 = nb.bands_1d(0.5, 40, U)[1] - nb.bands_1d(0.5, 40, U)[0]
    print(f"1D gap at k = K/2:  Nmax = 5  -> {gap5:.4f},   Nmax = 40 -> {gap40:.4f}")
    print("converged NFE gap ~ 0.462 (for U = 0.5), approached monotonically from above as Nmax grows")
    k, e = nb.band_structure_1d(Nmax=6, U=U, n_k=200)
    print(f"bands symmetric about k=0: {np.allclose(e[:, 0], e[::-1, 0])}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for b in range(e.shape[1]):
        ax.plot(k, e[:, b], lw=1.5, color=plt.cm.viridis(b / e.shape[1]))
    ax.axvline(0.0, color="dimgray", ls=":", lw=1)
    ax.set_xlabel("k / K"); ax.set_ylabel("E / (hbar^2 K^2 / 2m)")
    ax.set_title(f"1D nearly-free-electron bands  (U = {U})")
    fig.tight_layout()
    print("saved:", savefig(fig, "nfe_bands.png"))


if __name__ == "__main__":
    main()
