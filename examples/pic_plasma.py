"""Particle-in-cell plasma: the self-consistent field loop and the Langmuir response.

Reuses `spectral_fft` (Poisson field solve) and the PIC push/deposit routines.
Demonstrates the three pillars of the PIC method:
  (1) charge neutrality   -- a uniform plasma has a vanishing self-field;
  (2) the self-consistent field of a charge bump (deposit -> Poisson -> E = -phi');
  (3) the Langmuir response -- a launched electron-density wave drives a
      self-consistent E-field that accelerates the electrons (the PIC loop).
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import pic_plasma as pic
from ._common import savefig


def main():
    n = 256
    L = 2.0 * np.pi
    xg = np.linspace(0, L, n, endpoint=False)

    # (1) Charge neutrality: a perfectly uniform plasma is net-neutral -> E = 0.
    N = 1024
    xu = np.linspace(0, L, N, endpoint=False) + 1e-3
    rho_u = pic.deposit_charge(xu, L, n) + N / L
    _, E_u = pic.field_from_density(rho_u, L)
    print(f"(1) neutrality:  max |E| for a uniform plasma = {np.max(np.abs(E_u)):.2e}  (~ 0)")

    # (2) Self-consistent field of a localized charge bump.
    xc = 0.4 * L
    rho = np.exp(-((xg - xc) ** 2) / 0.15)
    rho = rho - rho.mean()                      # net charge (positive bump)
    phi, E = pic.field_from_density(rho, L)
    i_c = int(round(xc / L * n)) % n
    # Antisymmetry about the bump centre: E(xc+d) = -E(xc-d).
    d = n // 8
    asym = np.max(np.abs(E[(i_c + d) % n] + E[(i_c - d) % n]))
    print(f"(2) bump field:  |E|_max = {np.max(np.abs(E)):.4f}, "
          f"max |E(xc+d)+E(xc-d)| = {asym:.2e}  (antisymmetric ~ 0)")

    # (3) Langmuir response: a launched electron-density wave drives a
    #     self-consistent E-field; we track the k=1 mode amplitude over time.
    Nw = 4096
    base = np.linspace(0, L, Nw, endpoint=False)
    n0 = Nw / L
    x = base + 0.02 * np.sin(2 * np.pi * base / L)   # small coherent k=1 wave
    v = np.zeros_like(x)
    n_steps = 400
    dt = 0.02
    amp = np.zeros(n_steps + 1)
    for s in range(n_steps + 1):
        rho_s = pic.deposit_charge(x, L, n) + n0
        amp[s] = np.fft.fft(rho_s - rho_s.mean())[1].real
        if s < n_steps:
            x, v = pic.pic_step(x, v, n0, L, n, dt)
    omega_p = pic.plasma_frequency(n0)
    print(f"(3) Langmuir:    omega_p = {omega_p:.4f}; "
          f"k=1 mode responds: amplitude {amp[0]:.3f} -> {amp[-1]:.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(6.8, 6.2), sharex=False)
    ax[0].plot(xg, rho, label="net charge rho")
    ax[0].plot(xg, phi, label="potential phi")
    ax[0].plot(xg, E, label="E = -dphi/dx")
    ax[0].axvline(xc, color="gray", ls=":", lw=1)
    ax[0].set_ylabel("field quantities"); ax[0].set_title("Self-consistent PIC field of a charge bump")
    ax[0].legend(fontsize=8); ax[0].set_xlabel("x")
    t = np.arange(len(amp)) * dt
    ax[1].plot(t, amp, lw=1.2)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("Re[k=1 density mode]")
    ax[1].set_title(f"Langmuir response of the electron fluid (omega_p = {omega_p:.3f})")
    fig.tight_layout()
    print("saved:", savefig(fig, "pic_plasma.png"))


if __name__ == "__main__":
    main()
