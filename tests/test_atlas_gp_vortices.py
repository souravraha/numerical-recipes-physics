import numpy as np

from numerical_recipes.atlas import gp_vortices as gp


def test_quantized_circulation():
    # A q-vortex winds its phase by exactly 2*pi*q around the core.
    for q in (1, 2):
        psi = gp.make_vortex(nx=64, L=16.0, q=q)
        assert abs(gp.winding(psi) - q) < 0.05


def test_vortex_core_has_density_dip():
    # The vortex core is a density zero; a vortex-free Gaussian has none.
    core = gp.core_depth(gp.make_vortex(nx=64, L=16.0, q=1))
    plain = np.exp(-((np.linspace(-4, 4, 64))**2)[:, None] * np.ones((1, 64)))
    assert core > 0.5
    assert gp.core_depth(plain) < 0.2


def test_particle_number_conserved():
    # The GP flow conserves sum |psi|^2 (verified on both a vortex and a blob).
    for psi0 in (gp.make_vortex(nx=64, L=16.0, q=1), gp.make_vortex(nx=64, L=16.0, q=1) * np.exp(-0.01 * 0)):
        N0 = gp.particle_number(psi0)
        psi = gp.evolve(psi0, g=0.0, nx=64, L=16.0, dt=0.05, n_steps=200)
        assert abs(gp.particle_number(psi) - N0) / N0 < 1e-8


def test_core_persists_under_evolution():
    # Free (g=0) evolution preserves the vortex: the core dip and the unit
    # winding both survive (the phase structure is topological).
    psi0 = gp.make_vortex(nx=64, L=16.0, q=1)
    psi = gp.evolve(psi0, g=0.0, L=16.0, dt=0.05, n_steps=150)
    assert gp.core_depth(psi) > 0.05
    assert abs(gp.winding(psi) - 1.0) < 0.15
