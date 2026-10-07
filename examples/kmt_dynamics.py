"""Stochastic kinetochore-microtubule dynamics.

Reuses `kmt_dynamics` (the force-balance + stochastic kMT model).  Prints the
trajectory invariants (finite, deterministic given a seed) and the solved
velocity system, then plots the sleeve positions and a few kMT plus ends.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import kmt_dynamics as kmt
from ._common import savefig


def main():
    res = kmt.run(n_steps=200, num=15, seed=42)
    a = kmt.run(n_steps=60, num=15, seed=1)
    b = kmt.run(n_steps=60, num=15, seed=1)
    print(f"trajectory finite: Kr {np.all(np.isfinite(res['Kr']))}, "
          f"kmtr {np.all(np.isfinite(res['kmtr']))}")
    print(f"deterministic (same seed): {np.array_equal(a['kmtr'], b['kmtr'])}")
    print(f"solved velocity system: shape {res['last_v'].shape}, finite {np.all(np.isfinite(res['last_v']))}")

    t = np.arange(res["Kr"].shape[0])
    fig, ax = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True)
    ax[0].plot(t, res["Kr"], "b-", lw=2, label="right sleeve Kr")
    ax[0].plot(t, res["Kl"], "r-", lw=2, label="left sleeve Kl")
    ax[0].set_ylabel("position"); ax[0].set_title("kMT sleeve positions"); ax[0].legend(fontsize=8)
    for i in (0, 5, 10):
        ax[1].plot(t, res["kmtr"][:, i], lw=1, label=f"kmtr[{i}]")
        ax[1].plot(t, res["kmtl"][:, i], lw=1, ls="--", label=f"kmtl[{i}]")
    ax[1].set_xlabel("time step"); ax[1].set_ylabel("kMT plus end")
    ax[1].set_title("a few microtubule plus ends"); ax[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    print("saved:", savefig(fig, "kmt_dynamics.png"))


if __name__ == "__main__":
    main()
