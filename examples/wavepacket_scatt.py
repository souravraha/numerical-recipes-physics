"""Quantum wave-packet scattering: transmission / reflection, T + R = 1.

Reuses `spectral_fft` (split-operator Schrodinger propagator).  Scatters a
right-moving Gaussian packet off a rectangular barrier, prints the conserved
probability and the (T, R) split for a high vs. a low barrier, and plots the
packet profile plus the barrier.
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
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(xH, np.abs(pscH) ** 2, label="|psi|^2  (high barrier)")
    ax.plot(xH, np.abs(psiL) ** 2, label="|psi|^2  (low barrier)")
    ax.plot(xH, VH * 0.12, color="dimgray", lw=2, label="barrier (scaled)")
    ax.set_xlabel("x"); ax.set_ylabel("|psi|^2")
    ax.set_title(f"Wave-packet scattering   (T+R = {TH + RH:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    print("saved:", savefig(fig, "wavepacket_scatt.png"))


if __name__ == "__main__":
    main()
