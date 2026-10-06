"""Graphene tight-binding (Dirac cone) + SSH edge/topological states.

Reuses `linalg_eig` (eigh_symmetric).  Prints the gapless Dirac point and the
linear (E ~ |k - K|) dispersion ratio, plus the topological-vs-trivial SSH
spectrum (a pair of zero-energy edge states when t1 > t2), and plots all three.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import graphene_tb as gt
from ._common import savefig


def main():
    kK, EK = gt.dirac_point(t=1.0, nk=301)
    vF, ratio = gt.linear_dispersion(t=1.0)
    print(f"Dirac point K = ({kK[0]:.3f}, {kK[1]:.3f}),  E(K) = {EK:.5f}  (gapless ~ 0)")
    print(f"linear dispersion: v_F = {vF:.3f},  E(2r)/E(r) = {ratio:.3f}  (-> 2 if linear)")

    w_topo = gt.ssh_spectrum(t1=1.0, t2=0.5, n_cells=60)
    w_triv = gt.ssh_spectrum(t1=0.5, t2=1.0, n_cells=60)
    print(f"SSH topological (t1>t2): |E|_{'min 2'} = {np.sort(np.abs(w_topo))[:3].round(3)}")
    print(f"SSH trivial      (t1<t2): |E|_min     = {abs(w_triv).min():.4f}  (fully gapped)")

    KX, KY, E = gt.band_structure(t=1.0, nk=121)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    im = ax[0].pcolormesh(KX, KY, E, shading="auto", cmap="viridis")
    ax[0].set_title("Graphene band  E(k) = t|f(k)|"); ax[0].set_xlabel("kx"); ax[0].set_ylabel("ky")
    k = np.linspace(-0.3, 0.3, 200); K = kK + k[:, None] * np.array([np.cos(0.35), np.sin(0.35)])
    ax[1].plot(k, [abs(t * gt.honeycomb_f(kk)) for t in (1.0,) for kk in K])
    ax[1].set_title("Linear dispersion near K"); ax[1].set_xlabel("|k - K|")
    for w, c, lbl in [(w_topo, "tab:blue", "topological t1>t2"), (w_triv, "tab:red", "trivial t1<t2")]:
        ax[2].plot(w, np.zeros_like(w), "|", color=c, ms=6, label=lbl)
    ax[2].set_title("SSH spectrum"); ax[2].set_xlabel("E"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    print("saved:", savefig(fig, "graphene_tb.png"))


if __name__ == "__main__":
    main()
