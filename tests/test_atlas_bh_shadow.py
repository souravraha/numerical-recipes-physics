import numpy as np

from numerical_recipes.atlas import bh_shadow as bh


def test_photon_sphere_at_3M():
    # Unstable circular photon orbit is at r = 3M (the max of g(r)).
    assert abs(bh.photon_sphere(1.0) - 3.0) < 0.01


def test_critical_impact_parameter():
    # b_crit = 1/sqrt(max g) = sqrt(27) M.
    assert abs(bh.critical_impact_parameter(1.0) - np.sqrt(27.0)) < 1e-4


def test_shadow_scales_with_mass():
    # b_crit = sqrt(27) M scales linearly with the black-hole mass.
    assert abs(bh.critical_impact_parameter(2.0) - 2.0 * np.sqrt(27.0)) < 1e-3


def test_capture_vs_scatter():
    # Independent reference: an analytic b_crit separates captured from
    # scattered rays.  Just below -> captured (reaches the horizon);
    # just above -> scattered (closest approach above the photon sphere).
    bc = np.sqrt(27.0)
    assert bh.trace_ray(0.9 * bc, M=1.0)["captured"] is True
    assert bh.trace_ray(1.1 * bc, M=1.0)["r_min"] > 3.0


def test_shadow_bisection_matches_theory():
    # Trace real geodesics and bisect the capture/scatter transition: this
    # must reproduce the analytic shadow radius sqrt(27) M.
    r = bh.shadow_radius(1.0)
    assert abs(r - np.sqrt(27.0)) / np.sqrt(27.0) < 0.05
