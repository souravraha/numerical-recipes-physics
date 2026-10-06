import numpy as np

from numerical_recipes import spectral_fft as sf


def test_dft_matches_numpy():
    rng = np.random.default_rng(0)
    n = 256
    x = rng.normal(size=n)
    assert np.max(np.abs(sf.dft(x) - np.fft.fft(x))) < 1e-10


def test_spectrum_recovers_peak_amplitudes():
    # A signal = sum of two sine waves; the spectrum should show peaks at the
    # right frequencies with the right amplitudes.
    fs = 1000.0
    dt = 1.0 / fs
    t = np.arange(int(0.1 / dt)) * dt
    sig = 0.6 * np.sin(2 * np.pi * 50 * t) + 1.0 * np.sin(2 * np.pi * 120 * t)
    freq, amp, phase = sf.spectrum(sig, dt)
    i50 = np.argmin(np.abs(freq - 50))
    i120 = np.argmin(np.abs(freq - 120))
    assert abs(amp[i50] - 0.6) < 0.05
    assert abs(amp[i120] - 1.0) < 0.05


def test_poisson_fft_solves_sine_source():
    # lap(u) = -rho, periodic.  For rho = cos(2 pi x / L) (single mode, k=1),
    # u = rho / (2 pi / L)^2.
    n = 64
    L = 1.0
    x = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    rho = np.cos(2 * np.pi * X / L)
    u = sf.poisson_periodic_2d(rho, L=L)
    # The y-dependence should vanish (rho independent of y), and u = cos(2pi x)/ (2pi)^2.
    k2 = (2 * np.pi / L) ** 2
    u_exact = np.cos(2 * np.pi * X / L) / k2
    assert np.max(np.abs(u - u_exact)) < 1e-6


def test_poisson_fft_residual_small():
    rng = np.random.default_rng(1)
    n = 32
    L = 1.0
    rho = rng.normal(size=(n, n))
    rho = rho - rho.mean()  # zero mean (solvable)
    dx = L / n
    u = sf.poisson_periodic_2d(rho, L=L)
    # Numerical laplacian via spectral multipliers should recover -rho.
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    lap_u = np.fft.ifft2(-(KX**2 + KY**2) * np.fft.fft2(u)).real
    # lap(u) = -rho  ->  lap_u + rho ~ 0
    assert np.max(np.abs(lap_u + (rho - rho.mean()))) / np.max(np.abs(rho)) < 1e-8


def test_split_operator_preserves_norm():
    x, psi = sf.gaussian_wavepacket(n=1024, L=40.0, x0=0.0, k0=4.0, sigma=1.5)
    dx = x[1] - x[0]
    V = 0.5 * x**2  # harmonic potential
    n0 = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi_end = sf.tdschrodinger_split_operator(psi, V, dx, dt=0.02, n_steps=200)
    n1 = np.sqrt(np.sum(np.abs(psi_end) ** 2) * dx)
    assert abs(n1 - n0) < 1e-8  # unitary -> norm conserved


def test_split_operator_free_particle_momentum_conserved():
    x, psi = sf.gaussian_wavepacket(n=1024, L=40.0, x0=0.0, k0=3.0, sigma=2.0)
    dx = x[1] - x[0]
    V = np.zeros_like(x)
    psi_end = sf.tdschrodinger_split_operator(psi, V, dx, dt=0.02, n_steps=300)
    # <p> = <k> conserved for free particle (no force).
    k = np.fft.fftfreq(1024, d=dx) * 2 * np.pi
    def mean_p(psi_):
        ph = np.fft.fft(psi_)
        return np.sum(k * (np.abs(ph) ** 2)) / np.sum(np.abs(ph) ** 2)
    assert abs(mean_p(psi_end) - 3.0) < 0.05


def test_poisson_1d_solves_sine_source():
    # lap(u) = -rho on a periodic line; rho = sin(2 pi x / L) -> u = rho/(2pi/L)^2.
    L = 10.0
    n = 256
    x = np.linspace(0, L, n, endpoint=False)
    rho = np.sin(2 * np.pi * x / L)
    u = sf.poisson_periodic_1d(rho, L)
    u_exact = rho / (2 * np.pi / L) ** 2
    assert np.max(np.abs(u - u_exact)) / np.max(np.abs(u)) < 1e-8


def test_poisson_1d_zero_mode_removed():
    n = 64
    rho = np.ones(n) * 3.0  # pure mean, no solvable part
    u = sf.poisson_periodic_1d(rho, L=1.0)
    assert np.max(np.abs(u)) < 1e-12  # only the zero mode -> forced to 0


def test_nonlinear_g0_matches_linear_number():
    # With g=0 the nonlinear propagator must conserve the integrated
    # particle number exactly (it reduces to the linear split-operator).
    ny = nx = 64
    X, Y = np.meshgrid(np.linspace(-5, 5, nx), np.linspace(-5, 5, ny))
    psi0 = np.exp(-(X**2 + Y**2) / 2.0)
    N0 = np.sum(np.abs(psi0)**2)
    psi = sf.tdschrodinger_nonlinear_2d(psi0.copy(), np.zeros((ny, nx)), 0.0,
                                        10.0/64, 10.0/64, 0.02, 200)
    assert abs(np.sum(np.abs(psi)**2) - N0) / N0 < 1e-9


def test_nonlinear_sign_changes_spreading():
    # The sign of g is the defining GP physics: an attractive (g<0) gas
    # collapses (peak density rises), a repulsive (g>0) gas resists the free
    # diffraction (peak stays higher than the free packet), and the free
    # packet (g=0) spreads the most.  So at a fixed time:
    #     peak(g<0)  >  peak(g>0)  >  peak(g=0),
    # with the particle number conserved in all three cases.
    ny = nx = 64
    X, Y = np.meshgrid(np.linspace(-5, 5, nx), np.linspace(-5, 5, ny))
    psi0 = np.exp(-(X**2 + Y**2) / 2.0)
    N0 = np.sum(np.abs(psi0)**2)

    def run(g):
        return sf.tdschrodinger_nonlinear_2d(psi0.copy(), np.zeros((ny, nx)), g,
                                             10.0/64, 10.0/64, 0.05, 400)

    free, rep, attr = run(0.0), run(1.0), run(-1.0)
    p_free = np.max(np.abs(free)**2)
    p_rep = np.max(np.abs(rep)**2)
    p_attr = np.max(np.abs(attr)**2)
    assert p_attr > p_rep > p_free
    for p in (free, rep, attr):
        assert abs(np.sum(np.abs(p)**2) - N0) / N0 < 1e-6
