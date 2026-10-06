/* Spectral methods and the FFT (mirrors numerical_recipes.spectral_fft):
 * radix-2 FFT, naive DFT, one-sided amplitude/phase spectrum, FFT-based 1D and
 * 2D periodic Poisson solvers, split-operator Schrodinger propagator (linear
 * and 2D nonlinear Gross-Pitaevskii), and the spectral 1D wave equation.
 * Complex data uses C99 `double complex`.  Only `-lm` is required. */
/*
 * References (see REFERENCES.md):
 *   - J. W. Cooley, J. W. T. Tukey, Math. Comp. 19, 297 (1965) [FFT].
 *   - A. Feit, F. A. Fleischer, J. Comput. Phys. 14, 235 (1973) [split-operator].
 *   - A. Kosloff, Phys. Rep. 194, 195 (1990) [TDSE methods].
 *   - L. P. Gross, E. Pitaevskii, J. Phys. (Moscow) 18, 914 (1948) [Gross-Pitaevskii].
 */

#ifndef NR_SPECTRAL_FFT_H
#define NR_SPECTRAL_FFT_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* In-place radix-2 FFT of z (length n).  inverse != 0 computes the inverse
 * transform (with 1/n normalisation).  n must be a power of two; returns -1
 * otherwise.  X[k] = sum_j x[j] exp(-2 pi i j k / n). */
int nr_fft(double complex *z, int n, int inverse);

/* Naive O(n^2) DFT (any n): X[k] = sum_j x[j] exp(-2 pi i j k / n). */
int nr_dft(double complex *z, int n);

/* One-sided amplitude + phase spectrum of a real signal x (length n).
 * Outputs of length n/2+1: freq[k] = k/(n dt); amp[0] = |X0|/n and
 * amp[k>0] = 2|Xk|/n (DC term not doubled); phase[k] = arg(X[k]). */
int nr_spectrum(const double *x, int n, double dt,
                double *freq, double *amp, double *phase);

/* Solve laplacian(u) = -rho on a periodic [0,L] line via the FFT.
 * rho and u are length n; the zero (mean) mode of u is set to 0.
 * n must be a power of two; returns -1 otherwise. */
int nr_poisson_periodic_1d(const double *rho, int n, double L, double *u);

/* Solve laplacian(u) = -rho on a periodic [0,L]^2 box via the FFT.
 * rho and u are n x n row-major; the zero (mean) mode of u is set to 0.
 * n must be a power of two; returns -1 otherwise. */
int nr_poisson_periodic_2d(const double *rho, int n, double L, double *u);

/* Normalised Gaussian wavepacket centred at x0 with mean momentum k0.
 * x (length n, filled with the grid [-L/2, L/2)) and psi (length n, complex). */
void nr_gaussian_wavepacket(double *x, double complex *psi,
                            int n, double L, double x0, double k0, double sigma);

/* Split-operator (Strang/Trotter) propagator for i dpsi/dt = H psi with
 * H = -1/2 d^2/dx^2 + V  (hbar = m = 1).  psi (length n, complex) is updated
 * in place.  psi_hist, when non-NULL, stores (n_steps+1) x n snapshots
 * (initial state first).  n must be a power of two. */
void nr_tdschrodinger_split_operator(double complex *psi, const double *V,
                                     int n, double dx, double dt, int n_steps,
                                     double complex *psi_hist);

/* Split-operator (Strang) propagator for the 2D Gross-Pitaevskii equation
 * i dpsi/dt = [-1/2 (d2/dx2 + d2/dy2) + V(x,y) + g |psi|^2] psi  (hbar = m = 1).
 * psi is updated in place.  psi and V are ny x nx row-major (complex psi,
 * real V); g is the (real) interaction strength.  The kinetic step is exact in
 * Fourier space (multiply by exp(-i (kx^2 + ky^2) dt/2)); the external +
 * interaction step is exact in real space (exp(-i (V + g|psi|^2) dt)).
 * nx and ny must be powers of two. */
void nr_tdschrodinger_nonlinear_2d(double complex *psi, const double *V, double g,
                                   int nx, int ny, double dx, double dy,
                                   double dt, int n_steps);

/* Spectral (FFT) leapfrog time-stepping of u_tt = c^2 u_xx with periodic BCs.
 * u and v (length n) are updated in place.  u_hist, when non-NULL, stores
 * (n_steps+1) x n snapshots (initial state first). */
void nr_spectral_wave_1d(double *u, double *v, int n, double dx, double dt,
                         int n_steps, double c, double *u_hist);

#ifdef __cplusplus
}
#endif
#endif
