import numpy as np

from numerical_recipes import minimization as mn


def _lj_grad(x):
    x1 = x[:2]
    x2 = x[2:]
    r = x2 - x1
    d = np.linalg.norm(r)
    dr = 4.0 * (-12 * d ** -13 + 6 * d ** -7)
    grad_r = dr * (r / d)
    return np.concatenate([-grad_r, grad_r])


def _lj_hess(x):
    h = np.zeros((4, 4))
    eps = 1e-5
    f = mn.lennard_jones_dimer
    for i in range(4):
        for j in range(4):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            if i == j:
                h[i, j] = (f(xp) - 2 * f(x) + f(xm)) / eps ** 2
            else:
                xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                h[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return h


def _check_lj_minimum(x):
    # LJ dimer minimum: separation 2^(1/6), potential -1.
    r = np.linalg.norm(x[2:] - x[:2])
    assert abs(r - 2 ** (1 / 6)) < 1e-3
    assert abs(mn.lennard_jones_dimer(x) + 1.0) < 1e-3


# --- a well-conditioned quadratic (GD-friendly) ---
_A = np.array([[4.0, 1.0], [1.0, 3.0]])
_b = np.array([2.0, 1.0])
_xstar = np.linalg.solve(_A, _b)


def _quad(x):
    return 0.5 * x @ _A @ x - _b @ x


def _quad_g(x):
    return _A @ x - _b


def test_all_methods_find_quadratic_minimum():
    x0 = np.array([5.0, -3.0])
    for x in [
        mn.newton(_quad, _quad_g, lambda x: _A, x0)[0],
        mn.bfgs(_quad, _quad_g, x0)[0],
        mn.gradient_descent(_quad, _quad_g, x0, lr=0.3, max_iter=2000)[0],
        mn.nelder_mead(_quad, x0, max_iter=2000)[0],
    ]:
        assert np.linalg.norm(x - _xstar) < 1e-4


def test_newton_quadratic_is_fast():
    x0 = np.array([5.0, -3.0])
    x, it = mn.newton(_quad, _quad_g, lambda x: _A, x0)
    assert np.linalg.norm(_A @ x - _b) < 1e-8
    assert it <= 6  # quadratic -> essentially one Newton step


def test_newton_finds_lj_minimum():
    x0 = np.array([0.0, 0.0, 1.2, 0.3])
    x, it = mn.newton(mn.lennard_jones_dimer, _lj_grad, _lj_hess, x0)
    _check_lj_minimum(x)


def test_bfgs_finds_lj_minimum():
    x0 = np.array([0.0, 0.0, 1.2, 0.3])
    x, it = mn.bfgs(mn.lennard_jones_dimer, _lj_grad, x0)
    _check_lj_minimum(x)


def test_nelder_mead_finds_lj_minimum():
    x0 = np.array([0.0, 0.0, 1.2, 0.3])
    x, it = mn.nelder_mead(mn.lennard_jones_dimer, x0, max_iter=5000)
    _check_lj_minimum(x)
    assert np.linalg.norm(_lj_grad(x)) < 1e-3


def _rosen_g(x):
    return np.array([
        2 * (x[0] - 1) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ])


def _rosen_h(x):
    return np.array([
        [2 + 400 * x[0] - 1200 * x[0] ** 2, -400 * x[0]],
        [-400 * x[0], 200],
    ])


def test_rosenbrock_minimum():
    x0 = np.array([-1.0, 1.5])
    for x in [
        mn.newton(mn.rosenbrock, _rosen_g, _rosen_h, x0)[0],
        mn.bfgs(mn.rosenbrock, _rosen_g, x0)[0],
        mn.nelder_mead(mn.rosenbrock, x0, max_iter=6000)[0],
    ]:
        assert abs(x[0] - 1.0) < 5e-3
        assert abs(x[1] - 1.0) < 5e-3


def test_armijo_reduces_function():
    x = np.array([-1.0, 1.5])
    d = -_rosen_g(x)
    t, f_new = mn.armijo(mn.rosenbrock, x, d, _rosen_g(x))
    assert f_new < mn.rosenbrock(x)
