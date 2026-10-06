import numpy as np

from numerical_recipes import mcmc as M

MU = np.array([1.0, -2.0])
SIGMA = np.array([[1.0, 0.3], [0.3, 2.0]])
PREC = np.linalg.inv(SIGMA)


def log_t(x):
    d = x - MU
    return -0.5 * d @ PREC @ d


def grad_t(x):
    return -PREC @ (x - MU)


def test_metropolis_recovers_mean_cov():
    # Independent reference: the target is a known Gaussian, so the chain
    # mean/cov must match it (after burn-in).
    _, hist = M.metropolis_hastings(log_t, np.zeros(2), cov=1.5 * np.eye(2),
                                    n_steps=20000, seed=1)
    s = hist[2000:]
    assert np.max(np.abs(s.mean(0) - MU)) < 0.1
    assert np.max(np.abs(np.cov(s.T) - SIGMA)) < 0.3


def test_hmc_recovers_mean():
    _, hist = M.hmc(log_t, grad_t, np.zeros(2), step_size=0.5, n_leapfrog=15,
                    n_steps=2000, seed=1)
    assert np.max(np.abs(hist[100:].mean(0) - MU)) < 0.1


def test_metropolis_and_hmc_agree():
    _, hm = M.metropolis_hastings(log_t, np.zeros(2), cov=1.5 * np.eye(2),
                                  n_steps=30000, seed=2)
    _, hh = M.hmc(log_t, grad_t, np.zeros(2), step_size=0.5, n_leapfrog=15,
                  n_steps=4000, seed=2)
    assert np.max(np.abs(hm[5000:].mean(0) - hh[500:].mean(0))) < 0.2


def test_hamiltonian_conservation_small_steps():
    # With tiny step size the HMC trajectory is near-Hamiltonian: the
    # accept rate should be high (a broken leapfrog would reject almost all).
    rng = np.random.default_rng(0)
    x = np.zeros(2)
    accepted = 0
    for _ in range(200):
        p = rng.standard_normal(2)
        H0 = 0.5 * np.dot(p, p) - log_t(x)
        q, r = M._leapfrog(x, p, grad_t, 0.02, 10)
        r = -r
        H1 = 0.5 * np.dot(r, r) - log_t(q)
        if np.log(rng.random()) < H0 - H1:
            accepted += 1
            x = q
    assert accepted / 200 > 0.9


def test_autocorrelation_iid_and_ar1():
    rng = np.random.default_rng(0)
    iid = rng.normal(0, 1, 5000)
    acf = M.autocorrelation(iid, max_lag=5)
    assert abs(acf[0] - 1.0) < 1e-12
    assert abs(acf[1]) < 0.05
    # AR(1) with rho=0.9 has acf[1] ~= 0.9.
    N = 20000
    x = np.empty(N); x[0] = 0.0
    for t in range(1, N):
        x[t] = 0.9 * x[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal()
    assert abs(M.autocorrelation(x, max_lag=3)[1] - 0.9) < 0.05


def test_effective_sample_size_matches_ar1_theory():
    # Independent reference: for AR(1) with ratio rho, ESS ~= N (1-rho)/(1+rho).
    rng = np.random.default_rng(0)
    N = 20000
    x = np.empty(N); x[0] = 0.0
    for t in range(1, N):
        x[t] = 0.9 * x[t - 1] + np.sqrt(1 - 0.81) * rng.standard_normal()
    ess = M.effective_sample_size(x)
    theory = N * (1 - 0.9) / (1 + 0.9)
    assert 0.6 * theory < ess < 1.5 * theory


def test_effective_sample_size_iid_is_large():
    rng = np.random.default_rng(1)
    iid = rng.normal(0, 1, 5000)
    assert M.effective_sample_size(iid) > 3000  # nearly all draws are independent


def test_gelman_rubin_near_one_when_converged():
    # Independent chains from the same target: R-hat ~ 1 (the B/W ratio can
    # dip slightly below 1 with finite sampling, so test the band around 1).
    rng = np.random.default_rng(0)
    chains = np.stack([rng.normal(1.0, 1.0, 4000) for _ in range(4)])
    assert 0.9 < M.gelman_rubin(chains) < 1.1


def test_gelman_rubin_grows_when_divergent():
    rng = np.random.default_rng(0)
    bad = np.stack([rng.normal(mu, 1.0, 4000) for mu in (0.0, 2.0, 4.0, 6.0)])
    good = np.stack([rng.normal(1.0, 1.0, 4000) for _ in range(4)])
    assert M.gelman_rubin(bad) > M.gelman_rubin(good) * 2
