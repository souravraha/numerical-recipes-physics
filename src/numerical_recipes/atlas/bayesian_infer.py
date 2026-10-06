"""Bayesian inference / MCMC on a real physics problem.

We infer the parameters of an *underdamped harmonic oscillator*
    y(t) = A e^{-gamma t} cos(omega t),
a model that appears in RLC circuits, pendulums, and quasinormal modes of
black holes.  Given noisy measurements ``y(t_i)`` with known noise ``sigma``,
Bayes' theorem gives the posterior over ``(A, gamma, omega)`` proportional to
the (Gaussian) likelihood times a weak prior.  We sample that posterior with
the shared Hamiltonian Monte Carlo sampler (``mcmc.hmc``) and cross-check it
against the maximum-likelihood estimate from the shared ``minimization.bfgs``
-- the posterior mean should sit at the MLE (up to the prior) and its
uncertainty should grow with the data noise.

This is the "real physics problem" end of the inference pipeline: a physical
model, realistic noise, and the two canonical ways to extract parameters
(MCMC posterior + frequentist MLE) agreeing.


References (see REFERENCES.md):
  - R. M. Neal, "MCMC using Hamiltonian dynamics," in *Handbook of MCMC*,
    Chapman & Hall/CRC (2011) [HMC].
  - D. S. Sivia, B. Skilling, *Data Analysis: A Bayesian Tutorial*,
    Oxford University Press (2006) [Bayesian inference].
  - The damped-oscillator / quasinormal-mode ringdown model is standard in
    RLC circuits and black-hole perturbation theory (see, e.g., C. M. Will,
    Living Rev. Relativity 9, 3 (2006) [quasinormal modes / ringdown]).
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import mcmc as M
from numerical_recipes import minimization as MIN

__all__ = [
    "model",
    "neglog_posterior",
    "neglog_posterior_grad",
    "posterior_samples",
    "maximum_likelihood",
]


def model(t, A, gamma, omega):
    """y(t) = A e^{-gamma t} cos(omega t)."""
    t = np.asarray(t, dtype=float)
    return A * np.exp(-gamma * t) * np.cos(omega * t)


def _penalty(theta):
    A, gamma, omega = theta
    p = 0.0
    if A <= 0.0:
        p += 100.0 * A * A
    if gamma < 0.0:
        p += 100.0 * gamma * gamma
    if omega <= 0.0:
        p += 100.0 * omega * omega
    return p


def _penalty_grad(theta):
    A, gamma, omega = theta
    g = np.zeros(3)
    if A <= 0.0:
        g[0] = -200.0 * A
    if gamma < 0.0:
        g[1] = -200.0 * gamma
    if omega <= 0.0:
        g[2] = -200.0 * omega
    return g


def neglog_posterior(theta, t, y, sigma, prior_sd=5.0):
    """Negative log posterior (Gaussian likelihood + weak Gaussian prior)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A, gamma, omega = theta
    with np.errstate(over="ignore", invalid="ignore"):
        r = model(t, A, gamma, omega) - y
        ll = 0.5 * np.sum(np.square(r)) / sigma**2
    if not np.isfinite(ll):
        # A divergent optimizer probe flung the model to non-finite values;
        # report a large-but-finite cost so the search simply moves on.
        return 1e12
    prior = 0.5 * np.sum((theta / prior_sd) ** 2)
    return ll + prior + _penalty(theta)


def neglog_posterior_grad(theta, t, y, sigma, prior_sd=5.0):
    """Gradient of the negative log posterior w.r.t. (A, gamma, omega)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    A, gamma, omega = theta
    with np.errstate(over="ignore", invalid="ignore"):
        c = np.cos(omega * t)
        e = np.exp(-gamma * t)
        r = A * e * c - y
        dA = e * c
        dG = -A * t * e * c
        dW = -A * e * (t * np.sin(omega * t))
        g = np.array([dA, dG, dW])
        grad = np.array([np.sum(r * g[i]) / sigma**2 for i in range(3)])
    if not np.all(np.isfinite(grad)):
        # Divergent probe: fall back to a large finite push back toward the
        # prior centre so a line search can always take a safe step.
        grad = np.where(np.isfinite(grad), grad, -np.sign(theta) * 1e6)
    return grad + theta / prior_sd**2 + _penalty_grad(theta)


def _hessian(theta, t, y, sigma, prior_sd=5.0, h=1e-5):
    """Symmetric Hessian of the negative log posterior (central finite differences)."""
    f = lambda th: neglog_posterior(th, t, y, sigma, prior_sd)
    d = len(theta)
    H = np.zeros((d, d))
    for i in range(d):
        ei = np.zeros(d); ei[i] = h
        for j in range(i, d):
            ej = np.zeros(d); ej[j] = h
            H[i, j] = (f(theta + ei + ej) - f(theta + ei - ej)
                       - f(theta - ei + ej) + f(theta - ei - ej)) / (4.0 * h * h)
    return 0.5 * (H + H.T)


def _adaptive_step(theta0, t, y, sigma, prior_sd=5.0,
                   ceiling=0.05, floor=1e-4):
    """A leapfrog step safe for this posterior's steepest curvature.

    HMC's leapfrog integrator is stable only for ``eps < 2 / w_max`` where
    ``w_max = sqrt(lam_max)`` and ``lam_max`` is the largest eigenvalue of the
    (negative-log) posterior Hessian.  With small data noise the posterior is
    very narrow and a naive step is rejected almost every time, so we estimate
    ``lam_max`` at the starting point and at the MLE (where the posterior is
    usually sharpest) and return a safety fraction of the bound.
    """
    points = [np.asarray(theta0, dtype=float)]
    try:
        points.append(maximum_likelihood(theta0, t, y, sigma, prior_sd))
    except Exception:
        pass
    lam_max = 0.0
    for th in points:
        try:
            eig = np.linalg.eigvalsh(_hessian(th, t, y, sigma, prior_sd))
            lam_max = max(lam_max, float(eig.max()))
        except np.linalg.LinAlgError:
            pass
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return ceiling
    eps = 0.5 * 2.0 / np.sqrt(lam_max)
    return float(min(max(eps, floor), ceiling))


def posterior_samples(theta0, t, y, sigma, n_steps=2000, step_size=None,
                      n_leapfrog=20, seed=0, burn=1000, prior_sd=5.0):
    """HMC samples of the posterior (burn-in discarded).  Shape (n, 3).

    ``step_size=None`` picks the leapfrog step automatically from the
    posterior curvature (see ``_adaptive_step``); pass a value to override.
    """
    log_target = lambda th: -neglog_posterior(th, t, y, sigma, prior_sd)
    grad = lambda th: -neglog_posterior_grad(th, t, y, sigma, prior_sd)
    if step_size is None:
        step_size = _adaptive_step(theta0, t, y, sigma, prior_sd)
    _, hist = M.hmc(log_target, grad, np.asarray(theta0, dtype=float),
                    step_size=step_size, n_leapfrog=n_leapfrog, n_steps=n_steps,
                    seed=seed, record=True)
    return np.asarray(hist, dtype=float)[burn:]


def maximum_likelihood(theta0, t, y, sigma, prior_sd=5.0):
    """MLE (minimisation of the negative log posterior).  Shape (3,).

    We run the gradient-based BFGS *and* the derivative-free Nelder-Mead and
    return whichever lands on the lower objective.  BFGS is fast but can be
    derailed by this ill-scaled, multi-well likelihood (its first step can
    fling one parameter into a spurious valley); Nelder-Mead is robust.
    Returning the better of the two is a cheap, dependable MLE.
    """
    f = lambda th: float(neglog_posterior(th, t, y, sigma, prior_sd))
    g = lambda th: neglog_posterior_grad(th, t, y, sigma, prior_sd)
    x0 = np.asarray(theta0, dtype=float)
    candidates = []
    try:
        bfgs_x, _ = MIN.bfgs(f, g, x0, max_iter=300)
        candidates.append(bfgs_x)
    except Exception:
        pass
    nm_x, _ = MIN.nelder_mead(f, x0)
    candidates.append(nm_x)
    return min(candidates, key=f)
