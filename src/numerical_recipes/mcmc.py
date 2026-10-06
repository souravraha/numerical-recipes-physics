"""General-purpose Markov-chain Monte Carlo: Metropolis-Hastings, HMC, and
chain diagnostics (autocorrelation, effective sample size, Gelman-Rubin R-hat).

Where ``mcmc_ising`` samples a *specific* model (the 2D Ising spins), this
module samples an *arbitrary* target ``pi(x) proportional to exp(log_target(x))``
in R^d, so it is the workhorse for continuous fields (lattice field theory)
and for Bayesian inference.  Two samplers are provided:

  * **Metropolis-Hastings** -- a Gaussian random walk; simple, robust, but
    mixes slowly (O(d) and with strong autocorrelation).
  * **Hamiltonian Monte Carlo** -- uses the *gradient* of the log-target to
    take long, correlated steps that follow the curvature of the distribution;
    far better mixing in moderate/high dimension.  This is the standard
    Bayesian engine (the "NUTS" idea, in its simplest fixed-step form).

The diagnostics answer the question "did my chain actually sample the
target?": the autocorrelation time, the effective sample size (how many
*independent* draws a correlated chain is worth), and the Gelman-Rubin
potential scale reduction across independent chains.


References (see REFERENCES.md):
  - N. Metropolis et al., J. Chem. Phys. 21, 1087 (1953) [Metropolis].
  - W. K. Hastings, Biometrika 56, 97 (1970) [Metropolis-Hastings].
  - C. Leimkuhler, S. E. Reich, "Simulating Hamiltonian Dynamics,"
    Cambridge Univ. Press (2004) [leapfrog / symplectic integrators].
  - R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of MCMC,
    Chapman & Hall/CRC (2011) [HMC].
  - R. E. Gelman, D. B. Rubin, Stat. Sci. 7, 473 (1992) [Gelman-Rubin R-hat].
  - A. D. Gordon, "Measurement of uncertainty in estimation of posterior
    distributions," (autocorrelation / ESS); see also J. S. Hunter,
    "Handbook of Markov Chain Monte Carlo" (2018)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "metropolis_hastings",
    "hmc",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "gelman_rubin",
]


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def metropolis_hastings(
    log_target,
    x0,
    *,
    cov=None,
    n_steps=1000,
    seed=0,
    record=True,
):
    """Gaussian random-walk Metropolis-Hastings for ``pi(x) ~ exp(log_target(x))``.

    Parameters
    ----------
    log_target : callable
        ``x (d,) -> float``; the log of the (unnormalised) target density.
        May return ``-inf`` in regions of zero weight.
    x0 : array_like
        Initial state, shape ``(d,)``.
    cov : array_like, optional
        ``(d, d)`` proposal covariance.  Defaults to the identity (unit
        isotropic Gaussian steps).
    n_steps : int
        Number of proposals.
    seed : int
        RNG seed (reproducible).
    record : bool
        If True also return the full trajectory.

    Returns
    -------
    x : (d,)
        Final state.  With ``record`` also ``(x, history)`` where history has
        shape ``(n_steps + 1, d)`` (the initial state is row 0).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).ravel().copy()
    d = x.size
    if cov is None:
        cov = np.eye(d)
    cov = np.asarray(cov, dtype=float)
    hist = [x.copy()] if record else None
    cur = _safe_log_target(log_target, x)
    for _ in range(n_steps):
        prop = x + rng.multivariate_normal(np.zeros(d), cov)
        lp = _safe_log_target(log_target, prop)
        if _accept(cur, lp, rng):
            x = prop
            cur = lp
        if record:
            hist.append(x.copy())
    if record:
        return x, np.asarray(hist)
    return x


def hmc(
    log_target,
    grad_log_target,
    x0,
    *,
    step_size=0.1,
    n_leapfrog=20,
    n_steps=500,
    seed=0,
    record=True,
):
    """Hamiltonian Monte Carlo (fixed-step leapfrog + Metropolis accept/reject).

    Parameters
    ----------
    log_target : callable
        ``x (d,) -> float``; the log of the (unnormalised) target.
    grad_log_target : callable
        ``x (d,) -> (d,)``; the gradient of ``log_target``.
    x0 : array_like
        Initial state, shape ``(d,)``.
    step_size : float
        Leapfrog step size (the HMC ``epsilon``).
    n_leapfrog : int
        Number of leapfrog steps per trajectory.
    n_steps : int
        Number of HMC iterations.
    seed : int
        RNG seed.
    record : bool
        If True also return the full trajectory.

    Returns
    -------
    x : (d,)
        Final state (or ``(x, history)`` with ``record``).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).ravel().copy()
    d = x.size
    eps = float(step_size)
    hist = [x.copy()] if record else None
    for _ in range(n_steps):
        # Sample momentum p ~ N(0, I).
        p = rng.standard_normal(d)
        # Kinetic + potential define the Hamiltonian H = p^2/2 - log_target(x).
        H_old = 0.5 * np.dot(p, p) - _safe_log_target(log_target, x)
        q, r = _leapfrog(x, p, grad_log_target, eps, n_leapfrog)
        # Reverse momentum so the map is invertible (symplectic).
        r = -r
        H_new = 0.5 * np.dot(r, r) - _safe_log_target(log_target, q)
        if _accept(H_old, H_new, rng, invert=True):
            x = q
        if record:
            hist.append(x.copy())
    if record:
        return x, np.asarray(hist)
    return x


def _leapfrog(x, p, grad_log_target, eps, n_leapfrog):
    """One HMC trajectory by the leapfrog (Stormer-Verlet) integrator."""
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)
    p = p + 0.5 * eps * grad_log_target(x)
    for i in range(n_leapfrog):
        x = x + eps * p
        if i != n_leapfrog - 1:
            p = p + eps * grad_log_target(x)
    p = p + 0.5 * eps * grad_log_target(x)
    return x, p


def _safe_log_target(log_target, x):
    v = log_target(x)
    v = float(v)
    return v if np.isfinite(v) else -np.inf


def _accept(old, new, rng, invert=False):
    """Metropolis accept test on (log) weights.

    ``invert=False`` compares log-densities (accept if log u < new - old).
    ``invert=True``  compares Hamiltonians (accept if log u < old - new, since
    a lower H is better).
    """
    if new == -np.inf:
        return False
    delta = (old - new) if invert else (new - old)
    if delta >= 0.0:
        return True
    return np.log(rng.random()) < delta


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def autocorrelation(x, max_lag=None):
    """Sample autocorrelation function of a (possibly multivariate) chain.

    ``x`` may be shape ``(n,)`` or ``(n, d)``; in the latter case the
    *average* over dimensions is returned.  Returns ``acf[0..max_lag]`` with
    ``acf[0] == 1``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        cols = [x]
    else:
        cols = [x[:, j] for j in range(x.shape[1])]
    n = x.shape[0]
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    acfs = []
    for v in cols:
        v = v - v.mean()
        c0 = np.dot(v, v)
        if c0 <= 0:
            acfs.append(np.ones(max_lag + 1))
            continue
        acf = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf[k] = np.dot(v[: n - k], v[k:]) / c0
        acfs.append(acf)
    return np.mean(np.asarray(acfs), axis=0)


def integrated_autocorrelation_time(x, max_lag=None):
    """Integrated autocorrelation time (truncated at the first negative acf).

    ``tau_int = 1 - 1/n + 2 * sum_{k>=1} rho_k`` (truncated).  For iid draws
    this is ~1; for a strongly correlated chain it is much larger, which is
    exactly what shrinks the effective sample size.
    """
    acf = autocorrelation(x, max_lag=max_lag)
    # Truncate the sum at the first lag where the running total turns negative
    # (a standard, bias-reducing choice).
    n = 1
    s = 0.0
    best = 1.0
    for k in range(1, acf.size):
        s += acf[k]
        tau = 1.0 + 2.0 * s
        if tau < best:
            break
        best = tau
    return float(max(best, 1.0))


def effective_sample_size(x):
    """Effective sample size: N / tau_int (averaged over dimensions)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if x.ndim == 1:
        tau = integrated_autocorrelation_time(x)
        return float(n / tau)
    taus = [integrated_autocorrelation_time(x[:, j]) for j in range(x.shape[1])]
    return float(np.mean([n / t for t in taus]))


def gelman_rubin(chains):
    """Gelman-Rubin potential scale reduction factor R-hat.

    ``chains`` is shape ``(m, n)`` for a scalar target sampled by ``m``
    independent chains of length ``n``, or ``(m, n, d)`` for a ``d``-dimensional
    target (R-hat is the max over dimensions).  R-hat ~ 1 means the chains
    agree (converged); R-hat >> 1 means they do not.
    """
    c = np.asarray(chains, dtype=float)
    if c.ndim == 2:
        c = c[:, :, None]
    m, n, d = c.shape
    if m < 2:
        raise ValueError("gelman_rubin needs at least two independent chains")
    mean_t = c.mean(axis=1)  # (m, d)
    var_t = c.var(axis=1, ddof=1)  # (m, d)
    overall = mean_t.mean(axis=0)  # (d,)
    B = n * np.sum((mean_t - overall) ** 2, axis=0) / (m - 1)  # (d,)
    W = var_t.mean(axis=0)  # (d,)
    var_c = (n - 1) / n * W + B / n  # (d,)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = var_c / W
    r2 = np.where(W > 0, r2, 1.0)
    return float(np.sqrt(np.max(r2)))
