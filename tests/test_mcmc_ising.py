import numpy as np

from numerical_recipes import mcmc_ising as mi


def test_onsager_tc_value():
    # kTc/J = 2 / ln(1+sqrt(2)) = 2.2691899...
    assert abs(mi.onsager_tc() - 2.2691899) < 1e-5


def test_low_temperature_orders():
    # At low T the magnetisation per spin should be close to +-1 (ordered).
    n = 8
    r = mi.ising_thermo(n=n, T=1.0, n_sweeps=400, equilibration=100, seed=0)
    assert r["magnetization_per_spin"] > 0.8


def test_high_temperature_disorders():
    # At high T the (absolute) magnetisation should be small (disordered).
    n = 8
    r = mi.ising_thermo(n=n, T=5.0, n_sweeps=400, equilibration=100, seed=0)
    assert r["magnetization_per_spin"] < 0.4


def test_specific_heat_peaks_near_tc():
    # The specific heat should peak near Onsager Tc ~ 2.269.
    n = 12
    Ts = [1.5, 2.0, 2.269, 2.5, 3.0]
    Cs = []
    for T in Ts:
        r = mi.ising_thermo(n=n, T=T, n_sweeps=600, equilibration=150, seed=1, method="metropolis")
        Cs.append(r["specific_heat_per_spin"])
    i_pk = int(np.argmax(Cs))
    # Peak should be at or adjacent to the Onsager-Tc entry.
    assert abs(Ts[i_pk] - 2.269) < 0.6
    # The peak should be clearly above the low-T and high-T values.
    assert Cs[i_pk] > Cs[0] and Cs[i_pk] > Cs[-1]


def test_metropolis_vs_wolff_agree():
    # Both MCMC schemes must give the same equilibrium energy/spin at a fixed T.
    n = 10
    T = 2.2
    r_m = mi.ising_thermo(n=n, T=T, n_sweeps=500, equilibration=150, seed=2, method="metropolis")
    r_w = mi.ising_thermo(n=n, T=T, n_sweeps=500, equilibration=150, seed=2, method="wolff")
    assert abs(r_m["energy_per_spin"] - r_w["energy_per_spin"]) < 0.15


def test_energy_is_finite_and_reasonable():
    n = 6
    r = mi.ising_thermo(n=n, T=2.2, n_sweeps=200, equilibration=50, seed=3)
    e = r["energy_per_spin"]
    assert -4.0 <= e <= 4.0  # |E|/spin <= 4 (4 bonds per spin)
    assert np.isfinite(e)


def test_sw_sweep_conserves_energy_distribution():
    # Swendsen-Wang and Wolff are both cluster algorithms and must converge to
    # the same Boltzmann distribution (independent reference: their energies
    # agree near Tc, unlike the much slower Metropolis autocorrelation).
    n = 12
    T = 2.2

    def mean_energy(sweep):
        rng = np.random.default_rng(0)
        s = rng.integers(0, 2, size=(n, n)) * 2 - 1
        for _ in range(400):
            s, _ = mi.sw_sweep(s, T, rng=rng)
        es = []
        for _ in range(400):
            s, _ = mi.sw_sweep(s, T, rng=rng)
            es.append(mi.ising_energy(s) / (n * n))
        return np.mean(es)

    e_sw = mean_energy(mi.sw_sweep)
    e_w = mean_energy(mi.wolff_sweep)
    assert abs(e_sw - e_w) < 0.2


def test_sw_sweep_ordered_below_tc():
    n = 16
    rng = np.random.default_rng(1)
    s = rng.integers(0, 2, size=(n, n)) * 2 - 1
    for _ in range(300):
        s, _ = mi.sw_sweep(s, 1.5, rng=rng)
    assert abs(np.mean(s)) > 0.5  # net magnetisation at low T


def test_sw_sweep_cluster_count_positive():
    n = 8
    s = np.ones((n, n), dtype=int)
    s, nc = mi.sw_sweep(s, 2.2, rng=np.random.default_rng(2))
    assert nc >= 1
