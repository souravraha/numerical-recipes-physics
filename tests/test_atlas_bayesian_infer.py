import numpy as np

from numerical_recipes.atlas import bayesian_infer as bi


def _data(sigma, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, 60)
    true = np.array([2.5, 0.12, 1.3])
    y = bi.model(t, *true) + rng.normal(0.0, sigma, t.size)
    return t, y, true


def test_posterior_mean_recovers_true_params():
    t, y, true = _data(sigma=0.25)
    smp = bi.posterior_samples(theta0=[2.0, 0.1, 1.0], t=t, y=y, sigma=0.25,
                               n_steps=2000, seed=0, burn=800)
    mean = smp.mean(axis=0)
    std = smp.std(axis=0)
    for i, name in enumerate(("A", "gamma", "omega")):
        assert abs(mean[i] - true[i]) < 3.0 * std[i], (name, mean[i], true[i], std[i])


def test_credible_intervals_cover_truth():
    t, y, true = _data(sigma=0.25)
    smp = bi.posterior_samples(theta0=[2.0, 0.1, 1.0], t=t, y=y, sigma=0.25,
                               n_steps=2000, seed=1, burn=800)
    lo = np.percentile(smp, 5, axis=0)
    hi = np.percentile(smp, 95, axis=0)
    for i in range(3):
        assert lo[i] <= true[i] <= hi[i], (i, lo[i], hi[i], true[i])


def test_mle_agrees_with_posterior_mean():
    t, y, true = _data(sigma=0.25)
    mle = bi.maximum_likelihood(theta0=[2.0, 0.1, 1.0], t=t, y=y, sigma=0.25)
    # The MLE sits near the true parameters and near the posterior mode.
    for i in range(3):
        assert abs(mle[i] - true[i]) < 0.12, (i, mle[i], true[i])


def test_uncertainty_grows_with_noise():
    t1, y1, _ = _data(sigma=0.25)
    t2, y2, _ = _data(sigma=0.50)
    s1 = bi.posterior_samples(theta0=[2.0, 0.1, 1.0], t=t1, y=y1, sigma=0.25,
                              n_steps=2000, seed=2, burn=800)
    s2 = bi.posterior_samples(theta0=[2.0, 0.1, 1.0], t=t2, y=y2, sigma=0.50,
                              n_steps=2000, seed=2, burn=800)
    w1 = np.percentile(s1, 95, axis=0) - np.percentile(s1, 5, axis=0)
    w2 = np.percentile(s2, 95, axis=0) - np.percentile(s2, 5, axis=0)
    # Doubling the noise roughly doubles the credible-interval width (A drives it).
    assert w2[0] > w1[0]
    assert 0.8 < w2[0] / w1[0] < 2.5
