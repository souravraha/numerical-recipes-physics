"""Shared verification helpers used across every module.

The pedagogical core of this project: a numerical routine is not trusted
until it (a) converges at the expected order, (b) respects the relevant
conservation laws, and (c) agrees with an independent reference (an analytic
solution, SciPy, or a symmetry).  These small helpers make that check uniform
so each module can spend its lines on the physics rather than the bookkeeping.


References (see REFERENCES.md): G. H. Golub, C. F. Van Loan, Matrix Computations (2013);
  N. J. Higham, Accuracy and Stability of Numerical Algorithms (2002)."""

from __future__ import annotations

import numpy as np


def convergence_order(error_fn, h, n_points=4, halving=2.0):
    """Estimate the order ``p`` of ``error_fn(h) ~ C * h**p``.

    Parameters
    ----------
    error_fn : callable
        Maps a discretisation parameter ``h`` to a positive scalar error.
    h : float
        Starting value of ``h`` (the coarsest).
    n_points : int
        Number of refinements (h, h/halving, h/halving**2, ...).
    halving : float
        Refinement factor (default 2).

    Returns
    -------
    float
        Slope of log(error) vs log(h) from the finest two points (the
        asymptotic regime), which for error ~ C h**p is the order p.
    """
    h_values = np.array([h * halving ** (-i) for i in range(n_points)])
    errors = np.array([float(error_fn(float(hh))) for hh in h_values])
    if np.any(errors <= 0) or np.any(~np.isfinite(errors)):
        raise ValueError(f"non-positive or non-finite errors: {errors}")
    ratio = h_values[1] / h_values[0]
    slopes = np.diff(np.log(errors)) / np.log(ratio)
    return float(slopes[-1])


def max_relative_error(approx, exact):
    """max |approx - exact| scaled by max(1, max |exact|)."""
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    scale = max(1.0, float(np.max(np.abs(exact))))
    return float(np.max(np.abs(approx - exact)) / scale)


def energy_drift(energies):
    """Max relative drift of an energy history: max |E - E0| / |E0|."""
    energies = np.asarray(energies, dtype=float)
    e0 = energies[0]
    if e0 == 0:
        return float(np.max(np.abs(energies - e0)))
    return float(np.max(np.abs(energies - e0)) / abs(e0))
