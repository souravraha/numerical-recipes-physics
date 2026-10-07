"""Quantum wave-packet scattering: T + R = 1, plus the barrier-lattice gaps.

Reuses `spectral_fft` (split-operator Schrodinger propagator).  Scatters a
right-moving Gaussian packet off a single barrier (conserved probability, T+R),
and shows the energy-resolved transmission |T(E)|^2 of a lattice of identical
barriers (the analytic Kronig-Penney reference with band gaps).
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import wavepacket_scatt as ws
from ._common import savefig


def _run(height):
    n, L, x0, k0, sigma = 2048, 80.0, -15.0, 3.0, 2.0
    dx = L / n
    x, psi0 = ws.make_packet(n, L, x0, k0, sigma)
    V = ws.barrier_potential(x, height, 3.0)
    psi = ws.scatter(psi0, V, dx, 0.05, 240)
    T, R = ws.transmission_reflection(psi, x, 0.0, dx)
    return x, psi, V, dx, T, R


def main():
    xH, pscH, VH, dx, TH, RH = _run(60.0)   # high barrier -> reflection
    xL, psiL, VL, _, TL, RL = _run(2.0)    # low barrier  -> transmission
    prob = ws.total_probability(pscH, dx)
    print(f"total probability (conserved) = {prob:.6f}")
    print(f"high barrier (V0 = 60):  T = {TH:.3f},  R = {RH:.3f},  T+R = {TH + RH:.6f}")
    print(f"low barrier  (V0 = 2):   T = {TL:.3f},  R = {RL:.3f},  T+R = {TL + RL:.6f}")

    # Energy-resolved transmission through a lattice of 6 barriers (gaps).
    E = np.linspace(150.0, 1000.0, 400)
    T2 = np.array([abs(ws.barrier_lattice(e, 100.0, 0.2, 0.6, 6)[0]) ** 2 for e in E])
    print(f"barrier lattice |T(E)|^2: max {T2.max():.3f} (resonance), min {T2.min():.3f} (gap)")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax[0].plot(xH, np.abs(pscH) ** 2, label="|psi|^2  (high barrier)")
    ax[0].plot(xH, np.abs(psiL) ** 2, label="|psi|^2  (low barrier)")
    ax[0].plot(xH, VH * 0.12, color="dimgray", lw=2, label="barrier (scaled)")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("|psi|^2")
    ax[0].set_title(f"Wave-packet scattering   (T+R = {TH + RH:.4f})"); ax[0].legend(fontsize=8)

    ax[1].plot(E, T2, "b-", lw=1.8)
    ax[1].set_xlabel("E"); ax[1].set_ylabel("|T(E)|^2")
    ax[1].set_title("6-barrier lattice: resonances & gaps")
    fig.tight_layout()
    print("saved:", savefig(fig, "wavepacket_scatt.png"))


if __name__ == "__main__":
    main()
