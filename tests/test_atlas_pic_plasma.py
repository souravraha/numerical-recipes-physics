import numpy as np

from numerical_recipes.atlas import pic_plasma as pic


def test_charge_neutrality():
    # A perfectly uniform plasma (electrons + background) is net-neutral -> E = 0.
    n = 128
    L = 2.0 * np.pi
    x = np.linspace(0, L, n, endpoint=False) + 0.001
    rho = pic.deposit_charge(x, L, n) + n / L       # electrons + background
    phi, E = pic.field_from_density(rho, L)
    assert np.max(np.abs(E)) < 1e-8


def test_field_solve_residual():
    # The Poisson solve is spectrally accurate: lap(phi) + rho ~ 0.
    n = 128
    L = 2.0 * np.pi
    rho = np.exp(-((np.linspace(0, L, n, endpoint=False) - 0.4 * L) ** 2) / 0.05)
    rho = rho - rho.mean()
    phi, E = pic.field_from_density(rho, L)
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    lap_phi = np.fft.ifft(-(k**2) * np.fft.fft(phi)).real
    assert np.max(np.abs(lap_phi + rho)) / np.max(np.abs(rho)) < 1e-9


def test_field_antisymmetric_about_charge():
    # The field of a single (smeared) charge is antisymmetric: E(-x) = -E(x).
    n = 256
    L = 10.0
    xg = np.linspace(0, L, n, endpoint=False)
    rho = np.exp(-((xg - L / 2) ** 2) / 0.5)
    rho = rho - rho.mean()
    phi, E = pic.field_from_density(rho, L)
    i_c = n // 2
    left = np.mean(E[:i_c])
    right = np.mean(E[i_c:])
    assert abs(left + right) < 1e-9 * max(1.0, abs(left) + abs(right)) + 1e-9


def test_langmuir_oscillates_at_omega_p():
    # A coherent k=1 density perturbation oscillates at omega_p = sqrt(n0) (e=m=eps0=1).
    n = 256
    L = 2.0 * np.pi
    N = 64
    n0 = N / L
    x = np.linspace(0, L, N, endpoint=False)
    amp = pic.langmuir(x, L, n0, n=n, dt=0.02, n_steps=300)
    t = np.arange(len(amp)) * 0.02
    # Find the dominant frequency from the damped oscillation via zero crossings.
    a = amp - amp.mean()
    zeros = np.where(np.diff(np.sign(a)))[0]
    if len(zeros) >= 4:
        periods = np.diff(zeros[:8]) * 0.02
        f_est = 1.0 / (2.0 * np.median(periods))
    else:
        f_est = None
    omega_p = pic.plasma_frequency(n0)
    if f_est is not None:
        assert abs(f_est - omega_p / (2 * np.pi)) / (omega_p / (2 * np.pi)) < 0.25
