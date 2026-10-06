"""Lattice gauge theory: 2D compact U(1), plaquette <Re P> vs beta.

Reuses `mcmc` (Metropolis single-link updates).  Prints <Re P> against the
1-loop strong-coupling bound 1 - 1/(2 beta) over a beta sweep (Coulomb phase
alignment rises with beta), and plots the sweep.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import lattice_ft as lt
from ._common import savefig


def main():
    betas = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    P = np.array([lt.plaquette_expectation(L=8, beta=float(b), n_sweeps=1200,
                                           equilibration=300, seed=int(round(b * 10)))[0]
                  for b in betas])
    print("beta    <Re P>   1-loop bound (1 - 1/(2 beta))")
    for b, p in zip(betas, P):
        print(f"{b:4.1f}   {p:6.4f}   {lt.strong_coupling_expectation(float(b)):6.4f}")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(betas, P, "-o", color="tab:blue", label="Monte Carlo  <Re P>")
    ax.plot(betas, 1.0 - 1.0 / (2.0 * betas), "--", color="dimgray", label="1-loop strong coupling")
    ax.set_xlabel("beta"); ax.set_ylabel("<Re P>")
    ax.set_title("2D compact U(1) gauge: <Re P> vs beta")
    ax.legend(fontsize=8); ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    print("saved:", savefig(fig, "lattice_ft.png"))


if __name__ == "__main__":
    main()
