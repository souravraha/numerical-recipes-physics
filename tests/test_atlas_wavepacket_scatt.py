import numpy as np

from numerical_recipes.atlas import wavepacket_scatt as ws

L, N, DX = 80.0, 2048, 80.0 / 2048


def _packet(k0=3.0, x0=-15.0, sigma=2.0):
    return ws.make_packet(N, L, x0, k0, sigma)


def test_probability_conserved():
    # Unitary evolution conserves total probability to machine precision.
    x, psi0 = _packet()
    V = ws.barrier_potential(x, 40.0, 3.0)
    _, hist = ws.scatter(psi0, V, DX, 0.05, 240, record=True)
    probs = [ws.total_probability(h, DX) for h in hist]
    assert max(abs(p - probs[0]) for p in probs) / probs[0] < 1e-6


def test_t_plus_r_equals_one():
    x, psi0 = _packet()
    V = ws.barrier_potential(x, 10.0, 3.0)
    psi = ws.scatter(psi0, V, DX, 0.05, 240)
    T, R = ws.transmission_reflection(psi, x, 0.0, DX)
    assert abs(T + R - 1.0) < 1e-6


def test_high_barrier_reflects():
    x, psi0 = _packet()
    V = ws.barrier_potential(x, 60.0, 3.0)
    psi = ws.scatter(psi0, V, DX, 0.05, 240)
    T, R = ws.transmission_reflection(psi, x, 0.0, DX)
    assert R > 0.6


def test_low_barrier_transmits():
    x, psi0 = _packet()
    V = ws.barrier_potential(x, 2.0, 3.0)
    psi = ws.scatter(psi0, V, DX, 0.05, 240)
    T, R = ws.transmission_reflection(psi, x, 0.0, DX)
    assert T > 0.6


def test_free_packet_drifts_right():
    x, psi0 = _packet(k0=3.0)
    psi = ws.scatter(psi0, np.zeros_like(x), DX, 0.05, 100)
    c0 = float(np.average(x, weights=np.abs(psi0) ** 2))
    c1 = float(np.average(x, weights=np.abs(psi) ** 2))
    assert c1 > c0 + 1.0

# --- lattice of identical barriers: analytic (Kronig-Penney) reference ---
def test_single_barrier_unitarity():
    # Conservation: |t|^2 + |r|^2 = 1 for a single barrier.
    t, r = ws.single_barrier(500.0, 100.0, 0.3)
    assert abs(abs(t) ** 2 + abs(r) ** 2 - 1.0) < 1e-12


def test_barrier_lattice_unitarity():
    # Conservation: |T|^2 + |R|^2 = 1 for a lattice of barriers.
    T, R = ws.barrier_lattice(500.0, 100.0, 0.2, 0.6, 5)
    assert abs(abs(T) ** 2 + abs(R) ** 2 - 1.0) < 1e-10


def test_lattice_reduces_to_single_barrier():
    # Independent consistency: one barrier (n=1) reproduces the single result.
    t, r = ws.single_barrier(500.0, 100.0, 0.2)
    T, R = ws.barrier_lattice(500.0, 100.0, 0.2, 0.6, 1)
    assert np.allclose(abs(T), abs(t))
    assert np.allclose(abs(R), abs(r))


def test_barrier_lattice_band_gaps():
    # Physics: |T(E)|^2 oscillates with E and develops gaps (suppressed
    # transmission) for a lattice of several barriers.
    E = np.linspace(150.0, 1000.0, 400)
    T2 = [abs(ws.barrier_lattice(e, 100.0, 0.2, 0.6, 6)[0]) ** 2 for e in E]
    assert max(T2) > 0.9          # near-transparent resonances
    assert min(T2) < 0.5          # and suppressed regions (band gaps)


def test_barrier_lattice_potential_shape():
    # Structure: n square barriers of height V0, zero elsewhere.
    x = np.linspace(0.0, 30.0, 1000)
    V = ws.barrier_lattice_potential(x, n=6, a=0.2, b=0.6, x_start=10.0, V0=100.0)
    assert np.all(np.isfinite(V))
    assert np.max(V) == 100.0
    assert np.sum(V > 0) > 0
