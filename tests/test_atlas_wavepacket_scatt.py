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
