"""Bayesian MCMC inference on a real physics problem (damped-oscillator fit).

Reuses `mcmc` (Hamiltonian Monte Carlo) and `minimization` (BFGS / Nelder-Mead).
Generates a noisy quasinormal ringdown, infers (A, gamma, omega) with HMC,
cross-checks against the MLE, prints the posterior means + 90% CIs, and plots
the data, the best-fit model, and the posterior 1-D histograms.
"""
import numpy as np
import matplotlib.pyplot as plt

from numerical_recipes.atlas import bayesian_infer as bi
from ._common import savefig


def main():
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 10.0, 60)
    true = np.array([2.5, 0.12, 1.3])
    sigma = 0.25
    y = bi.model(t, *true) + rng.normal(0.0, sigma, t.size)

    mle = bi.maximum_likelihood(theta0=[2.0, 0.1, 1.0], t=t, y=y, sigma=sigma)
    smp = bi.posterior_samples(theta0=[2.0, 0.1, 1.0], t=t, y=y, sigma=sigma,
                               n_steps=2000, seed=0, burn=800)
    mean = smp.mean(axis=0); lo = np.percentile(smp, 5, axis=0); hi = np.percentile(smp, 95, axis=0)
    names = ("A", "gamma", "omega")
    print("parameter   true    MLE      posterior 90% CI")
    for i, nm in enumerate(names):
        print(f"{nm:8s}   {true[i]:6.3f}   {mle[i]:6.3f}   [{lo[i]:6.3f}, {hi[i]:6.3f}]")
    for i, nm in enumerate(names):
        print(f"  {nm}: true inside 90% CI = {bool(lo[i] <= true[i] <= hi[i])}")

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8),
                           gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot(t, y, "o", ms=4, color="tab:gray", label="noisy data")
    ax[0].plot(t, bi.model(t, *mean), "-", color="tab:blue", label=f"fit  (A={mean[0]:.2f}, g={mean[1]:.3f}, w={mean[2]:.3f})")
    ax[0].plot(t, bi.model(t, *true), "--", color="tab:red", label="true model")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8)
    ax[0].set_title("Damped-oscillator (ringdown) fit")
    for i, nm in enumerate(names):
        ax[1].hist(smp[:, i], bins=40, density=True, alpha=0.6, color="tab:blue")
        ax[1].axvline(true[i], color="tab:red", ls="--", lw=1.5)
        ax[1].axvline(mean[i], color="tab:blue", lw=2)
    ax[1].set_title("posterior 1-D marginals (red = true)")
    fig.tight_layout()
    print("saved:", savefig(fig, "bayesian_infer.png"))


if __name__ == "__main__":
    main()
