import numpy as np
from scipy.integrate import simpson

from numerical_recipes.atlas import atomic_hf as hf


def test_he_hartree_total_energy_physics_band():
    # Independent reference: Hartree (no exchange) He sits between the exact
    # nonrelativistic value (-2.9037 Ha) and Hartree-Fock (-2.8617 Ha).
    res = hf.he_hartree(n=192, r_max=10.0, tol=1e-5, max_iter=100)
    assert np.isfinite(res["e_total"])
    assert -3.05 < res["e_total"] < -2.7


def test_he_orbital_is_normalised():
    # Conservation: the 1s orbital is normalised to unit radial integral.
    res = hf.he_hartree(n=192, r_max=10.0, tol=1e-5, max_iter=100)
    dr = res["r"][1] - res["r"][0]
    assert abs(simpson(np.abs(res["p"]) ** 2, dx=dr) - 1.0) < 1e-3


def test_he_is_deterministic():
    # Reproducibility: same grid/tol -> same self-consistent solution.
    a = hf.he_hartree(n=192, r_max=10.0, tol=1e-5, max_iter=60)
    b = hf.he_hartree(n=192, r_max=10.0, tol=1e-5, max_iter=60)
    assert np.allclose(a["e_total"], b["e_total"])
    assert np.allclose(a["p"], b["p"], rtol=1e-8, atol=1e-10)


def test_be_two_occupied_orbitals_ordered():
    # Independent reference: a 1s^2 2s^2 closed shell has two occupied orbitals,
    # 1s more bound than 2s (e[0] < e[1] < 0).
    res = hf.be_roothaan(n=256, r_max=20.0, tol=1e-3, max_iter=100)
    e = res["energies"]
    assert e.shape == (2,)
    assert e[0] < e[1] < 0


def test_be_total_energy_physics_band():
    # Independent reference: Be Hartree-Fock total energy is about -14.667 Ha.
    res = hf.be_roothaan(n=256, r_max=20.0, tol=1e-4, max_iter=100)
    assert -15.05 < res["e_total"] < -14.5


def test_be_coefficients_shape():
    # Structure: two occupied orbitals expanded over six Slater-type basis fns.
    res = hf.be_roothaan(n=256, r_max=20.0, tol=1e-3, max_iter=100)
    assert res["coefficients"].shape == (2, 6)
