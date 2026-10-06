"""2D incompressible Navier-Stokes: decaying turbulence (spectral vorticity).

Reuses `spectral_fft` (FFT Poisson solve).  Prints the incompressibility of
the velocity projection and the kinetic-energy decay under viscosity, and
plots the vorticity field and its streamfunction.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import ns2d_turb as ns
from ._common import savefig


def _smooth_zero_mean_w(n, L, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, n))
    w = np.fft.ifft2(np.fft.fft2(w) / (1.0 + (2 * np.pi * np.abs(np.fft.fftfreq(n, d=L / n))) ** 2)).real
    return w - w.mean()


def main():
    n, L, nu = 64, 2.0 * np.pi, 3e-3
    w = _smooth_zero_mean_w(n, L)
    u0, v0 = ns.velocities(w, L)
    div0 = np.max(np.abs(ns.divergence(u0, v0, L)))
    KE0 = ns.kinetic_energy(u0, v0, L)
    enst0 = float(np.sum(w ** 2))
    for _ in range(300):
        w = ns.step(w, nu, L, dt=0.02)
    u1, v1 = ns.velocities(w, L)
    KE1 = ns.kinetic_energy(u1, v1, L)
    enst1 = float(np.sum(w ** 2))
    print(f"incompressibility  |div u|_max = {div0:.2e}")
    print(f"kinetic energy     {KE0:.5f} -> {KE1:.5f}   (dissipated)")
    print(f"enstrophy          {enst0:.2f} -> {enst1:.2f}   (dissipated)")
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8))
    im = ax[0].pcolormesh(X, Y, w, shading="auto", cmap="bwr"); ax[0].set_title("vorticity  w")
    im = ax[1].pcolormesh(X, Y, ns.streamfunction(w, L), shading="auto", cmap="viridis"); ax[1].set_title("streamfunction  phi")
    fig.tight_layout()
    print("saved:", savefig(fig, "ns2d_turb.png"))


if __name__ == "__main__":
    main()
