"""Gross-Pitaevskii condensate with a quantized vortex: winding, core, N.

Reuses `spectral_fft` (nonlinear split-operator propagator).  Prints the
phase winding q, the conserved particle number, and the vortex-core depth,
and plots |psi|^2 showing the core dip.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import gp_vortices as gp
from ._common import savefig


def main():
    nx, L = 64, 16.0
    psi0 = gp.make_vortex(nx=nx, L=L, q=1)
    q0, N0 = gp.winding(psi0), gp.particle_number(psi0)
    psi = gp.evolve(psi0, g=1.0, nx=nx, L=L, dt=0.02, n_steps=200)
    q1, N1 = gp.winding(psi), gp.particle_number(psi)
    print(f"phase winding q : {q0:.2f} -> {q1:.2f}   (quantized, integer)")
    print(f"particle number N: {N0:.4f} -> {N1:.4f}  (drift {abs(N1 - N0) / N0:.2e})")
    print(f"vortex core depth: {gp.core_depth(psi):.3f}")
    x = np.linspace(-L / 2, L / 2, nx, endpoint=False)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.pcolormesh(X, Y, np.abs(psi) ** 2, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="|psi|^2")
    ax.set_title("GP condensate with a q = 1 quantized vortex")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    print("saved:", savefig(fig, "gp_vortices.png"))


if __name__ == "__main__":
    main()
