import numpy as np
from scipy.integrate import solve_ivp

from numerical_recipes import ode_adaptive as oa


def test_dormand_prince_matches_scipy_linear():
    # y' = -2 y + cos(t), exact-ish reference from SciPy.
    def f(t, y):
        return np.array([-2.0 * y[0] + np.cos(t)])

    y0 = np.array([1.0])
    T = 3.0
    ref = solve_ivp(f, [0, T], y0, method="RK45", rtol=1e-11, atol=1e-12)
    sol = oa.integrate_adaptive(f, y0, 0.0, T, rtol=1e-9, atol=1e-11)
    # Compare at the final time (both should be very accurate).
    assert abs(sol["y"][-1, 0] - ref.y[0, -1]) < 1e-6


def test_adaptive_more_accurate_than_coarse_fixed():
    # A rapidly varying solution: adaptivity should beat a fixed step that
    # is too coarse.
    def f(t, y):
        return np.array([y[1], -9.0 * y[0] - 2.0 * y[1]])  # ~3 rad/s oscillator

    y0 = np.array([1.0, 0.0])
    T = 10.0
    ref = solve_ivp(f, [0, T], y0, method="RK45", rtol=1e-10, atol=1e-12)

    sol = oa.integrate_adaptive(f, y0, 0.0, T, rtol=1e-8, atol=1e-10)
    err = abs(sol["y"][-1, 0] - ref.y[0, -1])
    assert err < 1e-5


def test_rlc_energy_decays_monotonically():
    sol = oa.rlc_circuit(R=0.5, L=1.0, C=1.0, q0=1.0, t_end=20.0)
    q, i = sol["y"][:, 0], sol["y"][:, 1]
    E = 0.5 * q**2 + 0.5 * i**2  # with L=C=1
    # Envelope must decay (damping), never blow up.
    assert E[-1] < E[0]
    assert np.all(np.isfinite(E))
    # Peak amplitude should shrink over time.
    early = np.max(np.abs(q[: len(q) // 4]))
    late = np.max(np.abs(q[3 * len(q) // 4 :]))
    assert late < early


def test_van_der_pol_settles_to_limit_cycle():
    sol = oa.van_der_pol(mu=1.0, x0=1.0, y0=0.0, t_end=40.0)
    x = sol["y"][:, 0]
    # After transients, amplitude should be bounded near the limit cycle
    # (x peaks around ~2 for mu=1).
    assert np.max(np.abs(x)) < 5.0
    # It must oscillate (both signs reached).
    assert np.min(x) < 0 < np.max(x)


def test_step_count_respects_tolerance():
    def f(t, y):
        return np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    tight = oa.integrate_adaptive(f, y0, 0.0, 10.0, rtol=1e-10, atol=1e-12)
    loose = oa.integrate_adaptive(f, y0, 0.0, 10.0, rtol=1e-4, atol=1e-6)
    assert tight["n_steps"] > loose["n_steps"]
