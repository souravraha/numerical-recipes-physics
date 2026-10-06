"""Black-hole shadow: ray-trace null geodesics and recover the photon shadow.

Reuses `ode_adaptive` (adaptive RK45 null geodesics) and `minimization`.
Prints the bisection-recovered shadow radius and the analytic sqrt(27) M,
then plots a captured and a scattered ray around the photon sphere (r = 3 M).
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import bh_shadow as bh
from ._common import savefig


def main():
    M = 1.0
    r_bis = bh.shadow_radius(M=M)
    r_analytic = np.sqrt(27.0) * M
    print(f"shadow radius (bisection)  = {r_bis / M:.5f} M")
    print(f"shadow radius (sqrt(27) M) = {r_analytic / M:.5f} M")
    print(f"relative error             = {abs(r_bis - r_analytic) / r_analytic:.2e}")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.add_patch(plt.Circle((0, 0), 2 * M, color="black", zorder=5, label="horizon (r = 2M)"))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(3 * M * np.cos(th), 3 * M * np.sin(th), ":", color="dimgray", lw=2, label="photon sphere (r = 3M)")
    for frac in (0.9, 1.1):
        b = frac * np.sqrt(27.0) * M
        ray = bh.trace_ray(b, M=M)
        x = ray["r"] * np.cos(ray["phi"]); y = ray["r"] * np.sin(ray["phi"])
        label = f"b = {b / M:.2f} M  ({'captured' if ray['captured'] else 'scattered'})"
        ax.plot(x, y, "-o", ms=2.5, lw=1, label=label)
    ax.set_xlim(-9.5, 9.5); ax.set_ylim(-7.5, 7.5)
    ax.set_aspect("equal"); ax.set_xlabel("x / M"); ax.set_ylabel("y / M")
    ax.set_title(f"Schwarzschild shadow  (r_crit = {r_bis / M:.3f} M)")
    ax.legend(fontsize=8, loc="upper left")
    print("saved:", savefig(fig, "bh_shadow.png"))


if __name__ == "__main__":
    main()
