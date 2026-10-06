import numpy as np

from numerical_recipes import ode_symplectic as os
from numerical_recipes import verify


def _ho_accel(x, v, k=1.0, m=1.0):
    return -(k / m) * x


def _ho_energy(x, v):
    return 0.5 * np.sum(v**2) + 0.5 * np.sum(x**2)


def test_leapfrog_harmonic_energy_bounded():
    # Symplectic integrators keep the energy error BOUNDED (oscillatory),
    # not secular, over a long run of the harmonic oscillator.  The
    # oscillation amplitude is O(dt^2) -- for leapfrog the staggered
    # velocity makes it ~ (dt^2)/8 of the energy -- so the tolerance is a
    # loose O(dt^2) bound (5e-3 at dt=0.1), not a tight number.
    x0 = np.array([1.0, 0.0])
    v0 = np.array([0.0, 1.0])
    E0 = _ho_energy(x0, v0)
    dt, n = 0.1, 4000
    for method in (os.leapfrog, os.velocity_verlet):
        x, v = method(x0, v0, dt, _ho_accel, n_steps=n)
        drift = abs(_ho_energy(x, v) - E0) / E0
        assert drift < 5e-3


def test_leapfrog_converges_second_order():
    # At fixed final time T, the global error of a 2nd-order method ~ O(dt^2).
    # T is deliberately NOT an integer multiple of the period: at resonant
    # times the phase error cancels and the measured order is artificially high.
    T = 1.5

    def err(dt):
        n = int(round(T / dt))
        x, v = os.leapfrog(np.array([1.0, 0.0]), np.array([0.0, 0.0]), dt, _ho_accel, n_steps=n)
        return float(abs(x[0] - np.cos(T)))  # exact x(T) = [cos(T), 0]

    p = verify.convergence_order(err, h=T / 8, n_points=4)
    assert 1.8 < p < 2.2


def test_rk4_matches_scipy_on_kepler_arc():
    from scipy.integrate import solve_ivp

    def f(t, y):
        x, v = y[:2], y[2:]
        return np.concatenate([v, os.kepler(x)])

    x0, v0 = np.array([1.0, 0.0]), np.array([0.0, 0.8])
    T, h = 1.0, 0.01
    ref = solve_ivp(f, [0, T], np.concatenate([x0, v0]), method="RK45", rtol=1e-11, atol=1e-12)
    x, v = os.rk4(x0, v0, h, lambda x, v: os.kepler(x), n_steps=int(T / h))
    assert np.max(np.abs(x - ref.y[:2, -1])) < 1e-5
    assert np.max(np.abs(v - ref.y[2:, -1])) < 1e-5


def test_kepler_orbit_energy_drift_leapfrog_vs_rk4():
    # Unit circular Kepler orbit over long runs: the symplectic integrator's
    # energy drift stays BOUNDED at O(dt^2) (same magnitude at 20 and 400
    # periods), while RK4 (non-symplectic) drifts SECULARLY (keeps growing
    # with time).  This is the headline demonstration of why symplectic
    # integrators are used for long orbits.
    x0, v0 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    dt = 0.02
    E0 = os.kepler_energy(x0, v0)

    def drift(integrator, n_periods):
        n = int(round(n_periods * 2 * np.pi / dt))
        x, v = integrator(x0, v0, dt, lambda x, v: os.kepler(x), n_steps=n)
        return abs(os.kepler_energy(x, v) - E0) / abs(E0)

    leap_short = drift(os.leapfrog, 20)
    leap_long = drift(os.leapfrog, 400)
    rk4_short = drift(os.rk4, 20)
    rk4_long = drift(os.rk4, 400)
    assert leap_long < 1e-3            # small, O(dt^2)
    assert leap_long < 2.0 * leap_short  # bounded: does not grow with time
    assert rk4_long > 3.0 * rk4_short   # secular: keeps growing with time


def test_chain_energy_conserved():
    rng = np.random.default_rng(0)
    n = 16
    u0, v0 = rng.normal(size=n), rng.normal(size=n)
    u, v = os.leapfrog(u0, v0, 0.05, lambda u, v: os.chain_accel(u, v), n_steps=4000)
    # Symplectic leapfrog conserves a modified Hamiltonian: the naive-energy
    # error is bounded at O((omega_max dt)^2) ~ 1e-3 here, not vanishingly
    # small -- and crucially it does NOT grow with run length.
    assert abs(os.chain_energy(u, v) - os.chain_energy(u0, v0)) / abs(os.chain_energy(u0, v0)) < 5e-3


def test_double_pendulum_butterfly_effect():
    # Chaos: two initially-nearby double-pendulum states separate
    # exponentially (sensitive dependence on initial conditions).
    def run(th0):
        th, om = os.rk4(
            np.array(th0), np.array([0.0, 0.0]), 0.01,
            lambda th, om: os.double_pendulum(th, om), n_steps=3000,
        )
        return th

    base = [np.pi / 2, np.pi / 4]
    sep = np.linalg.norm(run(base) - run([base[0] + 1e-6, base[1]]))
    assert sep > 1e-3


def test_langevin_verlet_equilibrates_harmonic_oscillator():
    # Strong friction -> short correlation time -> a tight equipartition check.
    # Independent reference: <x^2> = T/k and <v^2> = T/m for the canonical
    # distribution of the harmonic oscillator (m = k = T = 1).
    x, v, hist = os.langevin_verlet(np.zeros(1), np.array([1.0]), 0.005,
                                    _ho_accel, gamma=1.0, T=1.0,
                                    n_steps=100000, seed=7, record=True)
    xs = hist[20000:, 0]
    assert abs(np.mean(xs**2) - 1.0) < 0.15  # <x^2> = T/k
    assert np.all(np.isfinite(hist))


def test_langevin_verlet_matches_bao_wu_oscillator():
    # Cross-check the generic thermostat against the dedicated Bao-Wu
    # harmonic-oscillator integrator (stochastic.langevin_oscillator): both
    # must give <x^2> ~ T/k at the same T.
    from numerical_recipes import stochastic as st
    _, _, hist = os.langevin_verlet(np.zeros(1), np.array([1.0]), 0.01,
                                    _ho_accel, gamma=1.0, T=1.0,
                                    n_steps=60000, seed=3, record=True)
    vx = np.mean(hist[15000:, 0]**2)
    rw = st.langevin_oscillator(dt=0.01, n_steps=60000, m=1.0, k=1.0,
                                gamma=1.0, T=1.0, seed=3)
    assert abs(vx - rw["var_x"]) < 0.2
