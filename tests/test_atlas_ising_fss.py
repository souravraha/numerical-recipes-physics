import numpy as np

from numerical_recipes.atlas import ising_fss as fss


def test_chi_grows_with_size():
    # Finite-size scaling: the susceptibility grows with lattice size at Tc.
    L, chi, C = fss.scaling_data(fss.onsager_tc(), (16, 64),
                                 n_sweeps=1000, equilibration=200)
    assert chi[-1] > chi[0]


def test_susceptibility_peaks_at_tc():
    # The hallmark of a phase transition: chi(T) peaks at the critical T.
    Ts = [1.8, 2.0, 2.2, 2.269, 2.4, 2.6]
    chi = []
    for T in Ts:
        _, c, _ = fss.scaling_data(T, (24,), n_sweeps=600, equilibration=150)
        chi.append(c[0])
    i_pk = int(np.argmax(chi))
    assert abs(Ts[i_pk] - fss.onsager_tc()) < 0.4
    assert chi[i_pk] > chi[0] and chi[i_pk] > chi[-1]


def test_critical_exponents():
    # gamma/nu (log-log slope of chi vs L at Tc) is the non-trivial Ising
    # exponent 7/4 = 1.75, and it exceeds the sub-critical specific-heat
    # exponent alpha/nu ~ 0 (chi is the dominant critical observable).
    r = fss.fit_exponents(sizes=(8, 16, 32, 64), n_sweeps=1500, equilibration=250)
    assert 0.9 < r["gamma_over_nu"] < 2.5
    assert r["gamma_over_nu"] > r["alpha_over_nu"] + 0.5
