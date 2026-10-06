"""Black-hole ray tracing and shadow formation.

A distant observer sees a Schwarzschild black hole as a dark disk ("shadow"):
light rays whose impact parameter ``|b| < b_crit`` fall in and never return,
while ``|b| > b_crit`` are deflected back to infinity.  ``b_crit`` is set by
the unstable photon sphere -- the maximum of the photon effective potential
``V_eff(r) = (b^2/r^2)(1 - 2M/r)`` -- and the shadow radius is the classic
result ``b_crit = sqrt(27) M = (3 sqrt(3)/2) r_s`` with ``r_s = 2M`` the
Schwarzschild radius.

We (a) locate the photon sphere and ``b_crit`` from the effective potential,
and (b) *confirm* them by tracing actual null geodesics with the shared
adaptive RK45 integrator (``ode_adaptive.integrate_adaptive``), classifying
each ray as captured or scattered and recovering ``b_crit`` by bisection.
Geometric units (G = c = 1), mass ``M``.


References (see REFERENCES.md):
  - A. Einstein, Ann. Phys. 32, 355 (1911) [light deflection].
  - C. M. Cunningham, J. M. Bardeen, Astrophys. J. 218, 995 (1977) [shadow/deflection].
  - S. W. Hawking, R. S. Penrose, Proc. R. Soc. A 314, 529 (1970) [structure].
  - M. P. Norris (n/a) -> C. W. Misner, K. S. Thorne, J. A. Wheeler, Gravitation (1973).
"""
from __future__ import annotations

import numpy as np

from numerical_recipes import ode_adaptive as oa

__all__ = [
    "effective_potential",
    "photon_sphere",
    "critical_impact_parameter",
    "trace_ray",
    "shadow_radius",
]


def effective_potential(r, b, M=1.0):
    """Photon effective potential ``V_eff = (b^2/r^2)(1 - 2M/r)`` (E = 1)."""
    r = np.asarray(r, dtype=float)
    return (b * b / r**2) * (1.0 - 2.0 * M / r)


def _shape_function(M=1.0, rmax=40.0, n=40001):
    """g(r) = (1/r^2)(1 - 2M/r) = (r - 2M)/r^3 on a fine radial grid."""
    r = np.linspace(2.001 * M, rmax * M, n)
    return r, (r - 2.0 * M) / r**3


def photon_sphere(M=1.0):
    """Radius of the unstable circular photon orbit (max of g) = 3M."""
    r, g = _shape_function(M)
    return float(r[int(np.argmax(g))])


def critical_impact_parameter(M=1.0):
    """b_crit = 1/sqrt(max_r g(r)) = sqrt(27) M."""
    r, g = _shape_function(M)
    return float(1.0 / np.sqrt(g.max()))


def trace_ray(b, M=1.0, r0=80.0, phi_max=30.0):
    """Trace one incoming null ray of impact parameter ``b``.

    Integrates the first-order ray equation in the azimuth ``phi``,
        dr/dphi = -(r^2/b) sqrt(1 - (b^2/r^2)(1 - 2M/r))   (incoming branch),
    with the adaptive RK45 integrator.  The square root is clamped at zero so
    the trajectory stops cleanly at a turning point.  Returns a dict with
    ``r_min`` (closest approach), ``captured`` (r_min at/inside the horizon),
    and the ``phi``/``r`` trajectory.
    """
    def rhs(phi, y):
        r = max(float(y[0]), 1e-6)
        f = 1.0 - (b * b / (r * r)) * (1.0 - 2.0 * M / r)
        s = np.sqrt(max(f, 0.0))
        return np.array([-(r * r / b) * s])

    res = oa.integrate_adaptive(rhs, [r0], 0.0, phi_max, rtol=1e-10, atol=1e-10)
    r = np.asarray(res["y"], dtype=float)[:, 0]
    r_min = float(r.min())
    return {"r_min": r_min, "captured": bool(r_min <= 2.0 * M + 1e-3),
            "phi": np.asarray(res["t"]), "r": r}


def shadow_radius(M=1.0, iters=45):
    """Recover the shadow radius (critical impact parameter) by bisection on
    the captured/scattered transition -- an independent check of sqrt(27) M.
    """
    lo, hi = 0.3 * M, 10.0 * M
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if trace_ray(mid, M=M)["captured"]:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
