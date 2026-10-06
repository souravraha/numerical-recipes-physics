"""Adaptive-step Runge-Kutta (Dormand-Prince embedded RK45).

A *fixed*-step integrator must use a step small enough for the *worst* part
of a trajectory, which is wasteful when the solution is easy most of the
time.  An *adaptive* integrator estimates the local error at every step from
an embedded lower-order solution and rescales the step: take bigger steps
where the solution is smooth, smaller steps where it varies rapidly.  The
workhorse pair is the Dormand-Prince 5(4) method: a 5th-order propagator
paired with a 4th-order one whose difference estimates the local error and
drives the step controller.

This module implements the Dormand-Prince coefficients, a local-error-based
step controller (PI control with a safety factor), and two reference systems
that show why adaptivity matters: a series RLC circuit (damped driven
oscillator) and the Van der Pol oscillator (a limit cycle whose sharp, slow
relaxation part stresses a fixed-step code).


References (see REFERENCES.md):
  - J. P. Dormand, P. J. Prince, J. Comput. Appl. Math. 6, 67 (1980) [Dormand-Prince 5(4)].
  - E. Hairer, S. Norsett, G. Wanner, Solving Ordinary Differential Equations I, Springer (2008)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "dormand_prince_step",
    "integrate_adaptive",
    "rlc_circuit",
    "van_der_pol",
]

# Dormand-Prince 5(4) Butcher tableau (6 stages).
# Dormand-Prince 5(4) Butcher tableau (6 stages).
_DP_C = np.array([0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0])
_DP_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40, 9.0 / 40, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45, -56.0 / 15, 32.0 / 9, 0.0, 0.0, 0.0],
        [19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0.0, 0.0],
        [9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656, 0.0],
    ]
)
# Weights for the 5th-order (propagated) and 4th-order (error estimate) solutions.
_DP_B5 = np.array([35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84])
_DP_B4 = np.array([5321.0 / 57600, 0.0, 7429.0 / 16695, 1321.0 / 1920, -126603.0 / 339200, 207.0 / 1400])


def dormand_prince_step(t, y, h, f):
    """One Dormand-Prince step.

    Returns ``(y5, y4, k)`` where ``y5`` is the 5th-order result, ``y4`` the
    4th-order result (their difference estimates the local error), and ``k``
    the six stage vectors.
    """
    k = np.empty((6, y.size), dtype=float)
    k[0] = f(t, y)
    for i in range(1, 6):
        y_stage = y + h * sum(_DP_A[i, j] * k[j] for j in range(i))
        k[i] = f(t + _DP_C[i] * h, y_stage)
    y5 = y + h * (k.T @ _DP_B5)
    y4 = y + h * (k.T @ _DP_B4)
    return y5, y4, k


def integrate_adaptive(
    f,
    y0,
    t0,
    t1,
    *,
    rtol=1e-7,
    atol=1e-9,
    h0=None,
    max_step=None,
    min_step=1e-12,
):
    """Integrate ``y' = f(t, y)`` from ``t0`` to ``t1`` with adaptive RK45.

    Parameters
    ----------
    f : callable
        ``f(t, y) -> array`` right-hand side.
    y0 : array_like
        Initial state.
    t0, t1 : float
        Integration interval (t1 may be < t0 for reverse integration).
    rtol, atol : float
        Relative / absolute tolerances driving the step controller.
    h0 : float, optional
        Initial step; defaults to a heuristic based on the first derivative.
    max_step : float, optional
        Upper bound on the step size (defaults to the whole interval / 10).
    min_step : float
        Lower bound on the step size.

    Returns
    -------
    dict
        ``t``, ``y`` (trajectories), ``n_steps``, and ``rejections``.
    """
    y0 = np.asarray(y0, dtype=float)
    t = [t0]
    y = [y0.copy()]
    n_steps = 0
    rejections = 0

    direction = 1.0 if t1 >= t0 else -1.0
    max_step = max_step if max_step is not None else abs(t1 - t0) / 10

    # Heuristic first step (Hairer-Norsett-Wanner).
    scale = atol + rtol * np.abs(y0)
    d0 = np.linalg.norm(y0 / scale) if np.any(y0 != 0) else 1.0
    d1 = np.linalg.norm(f(t0, y0) / scale)
    h = 0.01 / max(d0, d1) ** 0.2 if max(d0, d1) > 0 else max_step
    h = float(np.clip(h, min_step, max_step))

    cur_t = t0
    cur_y = y0.copy()
    while (direction > 0 and cur_t < t1) or (direction < 0 and cur_t > t1):
        # Clamp the step so we land exactly on t1.
        if direction > 0:
            h = min(h, t1 - cur_t)
        else:
            h = max(h, t1 - cur_t)
        if abs(h) < min_step:
            break

        y5, y4, _ = dormand_prince_step(cur_t, cur_y, h, f)
        err = np.max(np.abs(y5 - y4) / (atol + rtol * np.maximum(np.abs(y5), np.abs(y4))))

        if err <= 1.0 or abs(h) <= min_step:
            cur_t = cur_t + direction * abs(h)
            cur_y = y5
            t.append(cur_t)
            y.append(cur_y.copy())
            n_steps += 1
        else:
            rejections += 1

        factor = 5.0 if err == 0.0 else 0.9 * err ** (-0.2)  # PI control, order 5
        factor = min(5.0, max(0.2, factor))
        h = direction * abs(h) * factor
        h = float(np.clip(h, -max_step, max_step))
        h = direction * max(abs(h), min_step)

    return {
        "t": np.asarray(t),
        "y": np.asarray(y),
        "n_steps": n_steps,
        "rejections": rejections,
    }


# --------------------------------------------------------------------------- #
# Reference systems
# --------------------------------------------------------------------------- #
def rlc_circuit(R=1.0, L=1.0, C=1.0, q0=1.0, t_end=10.0):
    """Series RLC circuit: charge ringing down through a resistor.

    State ``y = [q, i]`` (charge, current).  ``q'' + (R/L) q' + (1/LC) q = 0``.
    Underdamped for small R -- amplitude decays while the phase advances, a
    nice showcase for an adaptive controller.
    """

    def f(t, y):
        q, i = y
        return np.array([i, -(R / L) * i - (1.0 / (L * C)) * q])

    return integrate_adaptive(f, np.array([q0, 0.0]), 0.0, t_end)


def van_der_pol(mu=1.0, x0=1.0, y0=0.0, t_end=20.0):
    """Van der Pol oscillator:  x' = y,  y' = mu(1 - x^2) y - x.

    A self-sustaining (limit-cycle) oscillator.  Small mu is a weakly damped
    harmonic oscillator; large mu becomes relaxation-like with a sharp jump
    and a slow part that stresses any fixed-step code.
    """

    def f(t, y):
        x, v = y
        return np.array([v, mu * (1.0 - x * x) * v - x])

    return integrate_adaptive(f, np.array([x0, y0]), 0.0, t_end)
