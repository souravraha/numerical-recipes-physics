"""FGH bound states: hydrogen 1s radial density + harmonic spectrum.

Reuses `grid_hamiltonian` (FGH eigenvalue primitive).  Prints the hydrogen
1s energy against the analytic -Z^2/2, and the harmonic spectrum against
(n + 1/2) hbar omega, then plots the hydrogen 1s radial density (FGH vs the
analytic (e^{-r}/sqrt(pi))^2) and the harmonic spectrum.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import fgh_bound_states as fb
from ._common import savefig


def main():
    h = fb.hydrogen_radial(N=256, r_max=10.0)
    rel_h = abs(h["E_1s"] - h["theory_E"]) / abs(h["theory_E"])
    print(f"hydrogen 1s  E_1s = {h['E_1s']:.5f}   (theory {h['theory_E']:.3f},  rel err {rel_h:.2e})")

    r = fb.solve_harmonic(n_states=6, k=4.0, N=513, L=16.0)
    rel = np.max(np.abs(r["energies"][:6] - r["exact"])) / np.max(r["exact"])
    print(f"harmonic  k=4   first 6 states, rel err {rel:.2e}  (theory (n+1/2) hbar omega)")

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax[0].plot(h["r"], np.abs(h["R_1s"]) ** 2, "b-", lw=2, label="FGH  |R_1s|^2")
    ax[0].plot(h["r"], h["theory_density"], "r--", lw=2, label="theory (e^{-r}/sqrt(pi))^2")
    ax[0].set_xlabel("r (a0)"); ax[0].set_ylabel("|psi|^2")
    ax[0].set_title(f"Hydrogen 1s  (E = {h['E_1s']:.4f} a.u.)"); ax[0].legend(fontsize=8)

    n = np.arange(6)
    ax[1].plot(r["exact"], "o-", color="tab:blue", label="FGH")
    ax[1].plot(r["exact"], "s--", color="tab:red", ms=7, mfc="none", label="theory")
    ax[1].set_xlabel("n"); ax[1].set_ylabel("E / (hbar omega)")
    ax[1].set_title("Harmonic spectrum  (k = 4)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    print("saved:", savefig(fig, "fgh_bound_states.png"))


if __name__ == "__main__":
    main()
