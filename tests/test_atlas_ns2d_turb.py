import numpy as np

from numerical_recipes.atlas import ns2d_turb as ns


def _rand_vorticity(n=32, L=4.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n))


def test_incompressible_projection():
    # velocities(w) = curl^{-1} w is divergence-free to machine precision.
    w = _rand_vorticity()
    u, v = ns.velocities(w, L=4.0)
    div = ns.divergence(u, v, L=4.0)
    assert np.max(np.abs(div)) / (np.max(np.abs(u)) + 1e-12) < 1e-10


def test_curl_inverse_roundtrip():
    # curl(velocities(w)) = w for a smooth (zero-mean) vorticity -- the exact
    # Biot-Savart / Poisson inversion (spectral).
    n, L = 32, 4.0
    x = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = np.cos(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L)
    u, v = ns.velocities(w, L)
    assert np.max(np.abs(ns.curl(u, v, L) - w)) / np.max(np.abs(w)) < 1e-10


def test_mean_vorticity_conserved():
    # In 2D the advection of a divergence-free field and periodic diffusion
    # leave the mean vorticity invariant.
    w = _rand_vorticity(n=32, L=4.0, seed=1)
    w0 = w.mean()
    for _ in range(200):
        w = ns.step(w, nu=0.005, L=4.0, dt=0.05)
    assert abs(w.mean() - w0) / abs(w0) < 1e-8


def test_viscosity_dissipates_energy():
    # With nu > 0 and no forcing, both kinetic energy and enstrophy decay.
    n, L = 32, 4.0
    x = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = np.sin(2 * np.pi * X / L) * np.cos(2 * np.pi * Y / L)
    e0 = ns.kinetic_energy(*ns.velocities(w, L), L=L)
    enst0 = float(np.sum(w**2))
    for _ in range(300):
        w = ns.step(w, nu=0.03, L=L, dt=0.02)
    e1 = ns.kinetic_energy(*ns.velocities(w, L), L=L)
    assert e1 < e0                          # kinetic energy dissipated
    assert float(np.sum(w**2)) < enst0      # enstrophy dissipated
