"""Nonlinear minimisation routines, written from scratch.

Nearly every "find the ground state" problem in physics is a minimisation:
the ground-state geometry of a molecule, the lowest mode of a structure, the
best-fit parameters of a model.  The methods form a ladder of information
used per step:

  * **Gradient descent** -- uses only the gradient.  Simple, robust, but
    slow (linear convergence) and sensitive to the learning rate.
  * **Newton's method** -- uses the gradient *and* the Hessian.  Quadratic
    convergence near the minimum, but needs a Hessian (expensive in high D).
  * **BFGS** -- a *quasi-Newton* method: builds an approximation to the
    Hessian from gradient differences.  Gets Newton-like speed without ever
    computing a Hessian.  The workhorse for medium-D physics problems.
  * **Nelder-Mead** -- derivative-free (only function values).  The go-to
    when the objective is a black box (a simulation) or non-smooth.

All four use an Armijo backtracking line search for step-length control.
Reference problems span the disciplines: a 2D Lennard-Jones dimer (molecular
geometry), the Rosenbrock banana function (a classic convergence benchmark),
and a least-squares fit of noisy data (data analysis).


References (see REFERENCES.md):
  - R. Fletcher (1970); D. Goldfarb (1970); R. Broyden (1970); D. Shanno (1970) [BFGS].
  - J. A. Nelder, R. Mead, Comput. J. 7, 308 (1965) [Nelder-Mead].
  - L. Armijo et al., Tech. Rep. AFOSR-TR-69-16 (1969) [Armijo line search]."""

from __future__ import annotations

import numpy as np

__all__ = [
    "armijo",
    "gradient_descent",
    "newton",
    "bfgs",
    "nelder_mead",
    "lennard_jones_dimer",
    "rosenbrock",
]


# --------------------------------------------------------------------------- #
# Line search
# --------------------------------------------------------------------------- #
def armijo(f, x, d, g, alpha=0.5, c1=1e-4, max_iter=50):
    """Backtracking Armijo line search along direction d from x.

    Finds a step size t satisfying f(x + t d) <= f(x) + c1 t (g . d).
    Returns ``(t, f_new)``.
    """
    f0 = f(x)
    slope = g @ d
    if slope >= 0:
        # d is not a descent direction; fall back to a tiny step.
        return 1e-4, f0
    t = 1.0
    for _ in range(max_iter):
        f_new = f(x + t * d)
        if f_new <= f0 + c1 * t * slope:
            return t, f_new
        t *= alpha
    return t, f_new


# --------------------------------------------------------------------------- #
# Gradient-based methods
# --------------------------------------------------------------------------- #
def gradient_descent(f, g, x0, lr=0.01, tol=1e-8, max_iter=5000, verbose=False):
    """Plain gradient descent with a fixed learning rate."""
    x = np.array(x0, dtype=float)
    for it in range(1, max_iter + 1):
        grad = g(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x - lr * grad
        if verbose:
            print(f"gd   {it:5d}  f={f(x):.8f}  |g|={np.linalg.norm(grad):.2e}")
    return x, it


def newton(f, g, hess, x0, tol=1e-10, max_iter=200, verbose=False):
    """Newton's method with an Armijo line search.

    Uses the true Hessian: the Newton step ``d = -H^{-1} g`` is taken when
    it is a descent direction (``g . d < 0``), which gives quadratic
    convergence near a positive-definite minimum.  Far from the minimum the
    Hessian of a hard problem (e.g. the flat valley of the Rosenbrock
    function) is often *indefinite*, in which case the raw Newton step
    points uphill.  There we fall back to the steepest-descent direction
    ``-g`` (always a descent direction) and let the Armijo line search pick
    a safe step length.  This keeps Newton's speed near the minimum while
    staying globally robust.
    """
    x = np.array(x0, dtype=float)
    for it in range(1, max_iter + 1):
        grad = g(x)
        if np.linalg.norm(grad) < tol:
            break
        d = None
        try:
            d = -np.linalg.solve(hess(x), grad)
        except np.linalg.LinAlgError:
            d = None
        if d is None or float(grad @ d) >= 0.0:
            d = -grad
        t, _ = armijo(f, x, d, grad)
        x = x + t * d
        if verbose:
            print(f"newt {it:5d}  f={f(x):.8f}")
    return x, it


def bfgs(f, g, x0, tol=1e-9, max_iter=500, verbose=False):
    """BFGS quasi-Newton with Armijo line search.

    The standard default for smooth, medium-dimensionality minimisation.
    """
    n = len(x0)
    x = np.array(x0, dtype=float)
    g0 = g(x)
    B = np.eye(n)  # Hessian approximation
    for it in range(1, max_iter + 1):
        if np.linalg.norm(g0) < tol:
            break
        d = -np.linalg.solve(B, g0)
        t, _ = armijo(f, x, d, g0)
        x_new = x + t * d
        g_new = g(x_new)
        s = x_new - x
        y = g_new - g0
        sy = float(s @ y)
        if sy > 1e-12:
            # Direct BFGS update of the Hessian approximation.
            Bs = B @ s
            B = B - np.outer(Bs, Bs) / float(s @ Bs) + np.outer(y, y) / sy
        x = x_new
        g0 = g_new
        if verbose:
            print(f"bfgs {it:5d}  f={f(x):.8f}  |g|={np.linalg.norm(g0):.2e}")
    return x, it


def nelder_mead(f, x0, tol=1e-8, max_iter=2000, alpha=1.0, gamma=2.0, rho=0.5, verbose=False):
    """Nelder-Mead simplex method (derivative-free).

    The standard black-box minimiser: only function values are needed.
    """
    n = len(x0)
    # Build the initial simplex: x0 plus n points offset along each axis.
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        pt = np.array(x0, dtype=float)
        step = 0.05 if abs(x0[i]) < 1e-8 else 0.05 * abs(x0[i])
        pt[i] += step
        simplex.append(pt)
    vals = [f(p) for p in simplex]

    for it in range(1, max_iter + 1):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if np.max(vals) - np.min(vals) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection.
        xr = centroid + alpha * (centroid - worst)
        fr = f(xr)
        if fr < vals[0]:
            # Expansion.
            xe = centroid + gamma * (centroid - worst)
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            # Contraction.
            xc = centroid + rho * (worst - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                # Shrink.
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
        if verbose:
            print(f"nm   {it:5d}  f_best={min(vals):.8f}")
    best = np.argmin(vals)
    return simplex[best], it


# --------------------------------------------------------------------------- #
# Reference problems
# --------------------------------------------------------------------------- #
def lennard_jones_dimer(x):
    """Lennard-Jones potential of a 2-atom dimer (2D positions packed in x).

    x = [x1, y1, x2, y2].  The minimum is at separation 2^(1/6) (the well
    depth).  A clean molecular-geometry minimisation.
    """
    r = np.linalg.norm(x[2:] - x[:2])
    return 4.0 * (r ** -12 - r ** -6)


def rosenbrock(x):
    """Rosenbrock 'banana' function -- a classic convergence benchmark."""
    a = x[0] - 1.0
    b = x[1] - x[0] ** 2
    return a ** 2 + 100.0 * b ** 2
