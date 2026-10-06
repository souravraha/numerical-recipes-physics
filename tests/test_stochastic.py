import numpy as np

from numerical_recipes import stochastic as st


def test_brownian_msd_linear():
    # <(x(t)-x0)^2> = 2 D t  (1D Brownian motion).  Check the MSD slope.
    dt = 1e-4
    n_steps = 20000
    D = 1.0
    n_particles = 2000
    traj = st.brownian_motion(dt=dt, n_steps=n_steps, D=D, n_particles=n_particles, seed=0)
    t = np.arange(n_steps + 1) * dt
    msd = np.mean((traj - traj[:, :1]) ** 2, axis=0)
    mask = (t > 0.3) & (t < 1.7)
    slope = np.polyfit(t[mask], msd[mask], 1)[0]
    assert abs(slope - 2 * D) / (2 * D) < 0.10


def test_brownian_var_grows_linearly():
    dt = 1e-3
    n_steps = 20000
    n_particles = 3000
    traj = st.brownian_motion(dt=dt, n_steps=n_steps, D=1.0, n_particles=n_particles, seed=1)
    t = np.arange(n_steps + 1) * dt
    var = np.mean(traj ** 2, axis=0)  # x0 = 0
    slope = np.polyfit(t, var, 1)[0]
    assert abs(slope - 2.0) / 2.0 < 0.12


def test_langevin_equipartition():
    # <v^2> = T/m and <x^2> = T/k at stationarity (fluctuation-dissipation).
    res = st.langevin_oscillator(dt=1e-3, n_steps=120000, m=1.0, k=1.0, gamma=1.0, T=1.0, seed=2)
    assert abs(res["var_v"] - 1.0) < 0.20   # T/m = 1
    assert abs(res["var_x"] - 1.0) < 0.25   # T/k = 1


def test_langevin_temperature_scaling():
    # Raising T should raise both variances roughly proportionally.
    r1 = st.langevin_oscillator(dt=1e-3, n_steps=60000, T=1.0, seed=3)
    r2 = st.langevin_oscillator(dt=1e-3, n_steps=60000, T=2.0, seed=3)
    assert r2["var_v"] > r1["var_v"]
    assert r2["var_x"] > r1["var_x"]
    # Ratio should be roughly 2 (within noise).
    assert 1.0 < r2["var_v"] / r1["var_v"] < 3.0


def test_ou_stationary_variance():
    # OU stationary variance = sigma^2 / (2 theta).
    theta, sigma = 1.0, 1.0
    x = st.ornstein_uhlenbeck(theta=theta, sigma=sigma, dt=1e-3, n_steps=200000, seed=4)
    x = x[50000:]
    var = np.var(x)
    theory = sigma ** 2 / (2 * theta)
    assert abs(var - theory) / theory < 0.15


def test_ou_autocorrelation_exponential():
    # OU autocorrelation ~ exp(-theta t).
    theta = 1.0
    x = st.ornstein_uhlenbeck(theta=theta, sigma=1.0, dt=1e-3, n_steps=300000, seed=5)
    x = x[100000:]
    x = x - x.mean()

    def acf(x, lag):
        return np.mean(x[lag:] * x[:-lag]) / np.mean(x ** 2)

    lag_steps = int(1.0 / 1e-3)
    measured = acf(x, lag_steps)
    theory = np.exp(-theta * 1.0)
    assert abs(measured - theory) < 0.12


def test_euler_maruyama_ou_matches_stationary():
    # Euler-Maruyama on dx = -x dt + dW (theta=1, sigma=1) -> var -> 1/2.
    x = st.euler_maruyama(f=lambda y: -y, g=lambda y: 1.0, x0=0.0,
                          dt=1e-3, n_steps=200000, seed=6)
    x = x[50000:]
    var = np.var(x)
    assert abs(var - 0.5) / 0.5 < 0.15
