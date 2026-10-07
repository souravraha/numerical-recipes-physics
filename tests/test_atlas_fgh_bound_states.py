import numpy as np

from numerical_recipes.atlas import fgh_bound_states as f


def test_harmonic_spectrum_matches_analytic():
    # V = k x^2/2: FGH spectrum must match (n + 1/2) hbar sqrt(k/m).
    r = f.solve_harmonic(n_states=6, k=4.0, N=513, L=16.0)
    rel = np.max(np.abs(r["energies"][:6] - r["exact"])) / np.max(r["exact"])
    assert rel < 0.05


def test_square_well_matches_transcendental():
    # Independent reference: the even/odd transcendental equations (brentq).
    r = f.solve_square_well(25.0, 1.0, n_states=4, N=513, L=16.0)
    assert len(r["exact"]) == 4
    rel = np.max(np.abs(r["energies"][:4] - r["exact"])) / np.max(np.abs(r["exact"]))
    assert rel < 0.10


def test_harmonic_states_normalised():
    # Conservation: each state is normalised to unit integral on the grid.
    r = f.solve_harmonic(n_states=4, N=513, L=16.0)
    dx = r["x"][1] - r["x"][0]
    norms = np.sum(np.abs(r["states"][:, :4]) ** 2, axis=0) * dx
    assert np.max(np.abs(norms - 1.0)) < 1e-9


def test_hydrogen_1s_energy():
    # Independent reference: hydrogen 1s energy is -Z^2/2 (a.u.), = -0.5 for Z=1.
    h = f.hydrogen_radial(N=256, r_max=10.0)
    assert abs(h["E_1s"] - h["theory_E"]) < 0.02
    # And the radial amplitude is normalised and finite.
    dr = h["r"][1] - h["r"][0]
    assert abs(np.sum(np.abs(h["R_1s"]) ** 2) * dr - 1.0) < 1e-6
    assert np.all(np.isfinite(h["R_1s"]))


def test_central_3d_s_wave_ground_state():
    # Independent reference: 3D harmonic l=0 ground state is E = 1.5 hbar w.
    c = f.central_potential(N=512, w=10.0, a=4.0, l=0)
    assert c["theory_E0"] is not None
    assert abs(c["E"][0] - c["theory_E0"]) / c["theory_E0"] < 5e-3


def test_central_3d_radial_normalised():
    # Conservation: each radial amplitude is normalised to unit integral on r>0.
    c = f.central_potential(N=512, w=10.0, a=4.0, l=0, n_states=2)
    dr = c["r"][1] - c["r"][0]
    for Ri in c["R"]:
        rpos = c["r"][c["r"] > 0]
        assert abs(np.sum(np.abs(Ri) ** 2) * dr - 1.0) < 1e-6
