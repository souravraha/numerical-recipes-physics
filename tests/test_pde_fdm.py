import numpy as np

from numerical_recipes import pde_fdm as pf


def test_thomas_matches_numpy():
    n = 50
    d = 4.0 * np.ones(n)
    e = -1.0 * np.ones(n - 1)
    rng = np.random.default_rng(0)
    rhs = rng.normal(size=n)
    A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    x = pf.thomas(d, e, e, rhs)
    assert np.max(np.abs(A @ x - rhs)) < 1e-10


def test_heat_cn_matches_analytic_decay():
    # u_t = u_xx, u(0)=sin(pi x) on [0,1] -> u(t)=exp(-pi^2 t) sin(pi x).
    n = 61
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    t_end = 0.05
    dt = 0.01 * dx**2
    n_steps = int(round(t_end / dt))
    u = pf.heat_1d_cn(u0, dt, dx, n_steps)
    u_exact = np.exp(-np.pi**2 * t_end) * np.sin(np.pi * x)
    assert np.max(np.abs(u - u_exact)) < 2e-3


def test_heat_explicit_unstable_beyond_cfl():
    # r = dt/dx^2 > 0.5 must blow up (or be rejected).
    n = 21
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    dt = 0.8 * dx**2  # r = 0.8 > 0.5
    try:
        u = pf.heat_1d_explicit(u0, dt, dx, 50)
        assert not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e6
    except ValueError:
        pass  # implementation chose to raise


def test_wave_propagates_at_correct_speed():
    # u_tt = c^2 u_xx, u(0)=sin(pi x/L), v(0)=0 -> u(t)=sin(pi x/L) cos(c pi t / L).
    L = 1.0
    n = 81
    x = np.linspace(0, L, n)
    dx = x[1] - x[0]
    c = 1.0
    u0 = np.sin(np.pi * x / L)
    v0 = np.zeros_like(u0)
    t_end = 0.25
    dt = 0.5 * dx / c  # CFL = 0.5
    n_steps = int(round(t_end / dt))
    u, v = pf.wave_1d(u0, v0, dt, dx, n_steps, c=c)
    u_exact = np.sin(np.pi * x / L) * np.cos(c * np.pi * t_end / L)
    assert np.max(np.abs(u - u_exact)) < 3e-2


def test_advection_upwind_moves_bump():
    n = 201
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    a = 1.0
    u0 = np.exp(-((x - 0.3) / 0.05) ** 2)
    dt = 0.5 * dx / a  # nu = 0.5
    n_steps = int(round(0.2 / dt))
    u = pf.advection_1d_upwind(u0, dt, dx, n_steps, a=a)
    # Peak should have moved right by ~ a*t = 0.2
    x_peak = x[np.argmax(u)]
    assert 0.45 < x_peak < 0.65


def test_poisson_jacobi_vs_analytic():
    # lap(u) = -rho on [0,1]^2, u=0 on boundary.
    # Choose rho = 2*pi^2 * sin(pi x) sin(pi y) -> u = sin(pi x) sin(pi y).
    n = 33
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    rho = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    dx = x[1] - x[0]
    u, it = pf.poisson_2d_jacobi(rho, dx=dx, tol=1e-9, max_iter=200000)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    # Compare interior (boundary is fixed at 0 in both).
    assert np.max(np.abs(u[1:-1, 1:-1] - u_exact[1:-1, 1:-1])) < 2e-2


def test_poisson_sor_faster_than_jacobi():
    n = 33
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    rho = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    dx = x[1] - x[0]
    u_j, it_j = pf.poisson_2d_jacobi(rho, dx=dx, tol=1e-8, max_iter=200000)
    u_s, it_s = pf.poisson_2d_sor(rho, dx=dx, tol=1e-8, max_iter=200000)
    # Same answer...
    assert np.max(np.abs(u_j - u_s)) < 1e-4
    # ...but SOR takes fewer iterations.
    assert it_s < it_j
