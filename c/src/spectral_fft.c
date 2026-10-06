#include "nr/spectral_fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Imaginary unit as a `double complex` value (avoids relying on the `I` macro). */
#define NR_I _Complex_I

static int is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

int nr_fft(double complex *z, int n, int inverse) {
    if (!is_pow2(n)) return -1;
    if (n == 1) return 0;

    /* Bit-reversal permutation. */
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) { double complex t = z[i]; z[i] = z[j]; z[j] = t; }
    }

    double sign = inverse ? 1.0 : -1.0;
    for (int len = 2; len <= n; len <<= 1) {
        double ang = sign * 2.0 * M_PI / (double)len;
        double complex wlen = cexp(NR_I * ang);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            double complex w = 1.0;
            for (int jj = 0; jj < half; jj++) {
                double complex u = z[i + jj];
                double complex v = z[i + jj + half] * w;
                z[i + jj] = u + v;
                z[i + jj + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        double inv = 1.0 / (double)n;
        for (int i = 0; i < n; i++) z[i] *= inv;
    }
    return 0;
}

int nr_dft(double complex *z, int n) {
    if (n < 1) return -1;
    double complex *out = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    if (!out) return -1;
    for (int k = 0; k < n; k++) {
        double complex sum = 0.0;
        for (int jj = 0; jj < n; jj++) {
            double ang = -2.0 * M_PI * (double)jj * (double)k / (double)n;
            sum += z[jj] * cexp(NR_I * ang);
        }
        out[k] = sum;
    }
    memcpy(z, out, sizeof(double complex) * (size_t)n);
    free(out);
    return 0;
}

int nr_spectrum(const double *x, int n, double dt,
                double *freq, double *amp, double *phase) {
    if (n < 1 || !(dt > 0.0)) return -1;
    int m = n / 2 + 1;
    double complex *X = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    if (!X) return -1;
    for (int jj = 0; jj < n; jj++) X[jj] = x[jj];   /* imaginary part = 0 */
    int rc = is_pow2(n) ? nr_fft(X, n, 0) : nr_dft(X, n);
    if (rc != 0) { free(X); return -1; }
    for (int k = 0; k < m; k++) {
        freq[k] = (double)k / ((double)n * dt);
        double a = cabs(X[k]);
        amp[k] = (k == 0) ? (a / (double)n) : (2.0 * a / (double)n);
        phase[k] = carg(X[k]);
    }
    free(X);
    return 0;
}

int nr_poisson_periodic_2d(const double *rho, int n, double L, double *u) {
    if (!is_pow2(n) || n < 2 || !(L > 0.0)) return -1;
    size_t total = (size_t)n * (size_t)n;
    double complex *ah = (double complex *)malloc(sizeof(double complex) * total);
    double complex *col = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double *k = (double *)malloc(sizeof(double) * (size_t)n);
    if (!ah || !col || !k) { free(ah); free(col); free(k); return -1; }

    /* Load rho (imaginary part 0). */
    for (size_t idx = 0; idx < total; idx++) ah[idx] = rho[idx];

    /* Forward FFT along rows, then columns (separable, order independent). */
    for (int i = 0; i < n; i++) nr_fft(ah + (size_t)i * n, n, 0);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = ah[(size_t)i * n + j];
        nr_fft(col, n, 0);
        for (int i = 0; i < n; i++) ah[(size_t)i * n + j] = col[i];
    }

    /* Wavenumbers: k = 2 pi * fftfreq(n, dx), dx = L/n. */
    double dx = L / (double)n;
    for (int j = 0; j < n; j++) {
        int f = (j <= n / 2) ? j : j - n;
        k[j] = 2.0 * M_PI * (double)f / ((double)n * dx);
    }

    /* u_hat = rho_hat / (kx^2 + ky^2); zero mode (mean) forced to 0. */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double k2 = k[i] * k[i] + k[j] * k[j];
            ah[(size_t)i * n + j] = (k2 > 0.0) ? (ah[(size_t)i * n + j] / k2) : 0.0;
        }
    }

    /* Inverse FFT along columns, then rows. */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = ah[(size_t)i * n + j];
        nr_fft(col, n, 1);
        for (int i = 0; i < n; i++) ah[(size_t)i * n + j] = col[i];
    }
    for (int i = 0; i < n; i++) nr_fft(ah + (size_t)i * n, n, 1);

    for (size_t idx = 0; idx < total; idx++) u[idx] = creal(ah[idx]);
    free(ah); free(col); free(k);
    return 0;
}

void nr_gaussian_wavepacket(double *x, double complex *psi,
                            int n, double L, double x0, double k0, double sigma) {
    double dx = L / (double)n;
    for (int j = 0; j < n; j++) {
        x[j] = -L / 2.0 + (double)j * dx;
        double g = exp(-((x[j] - x0) * (x[j] - x0)) / (4.0 * sigma * sigma));
        psi[j] = g * cexp(NR_I * k0 * x[j]);
    }
    double s2 = 0.0;
    for (int j = 0; j < n; j++) { double m = cabs(psi[j]); s2 += m * m; }
    double scale = 1.0 / sqrt(s2 * dx);
    for (int j = 0; j < n; j++) psi[j] *= scale;
}

void nr_tdschrodinger_split_operator(double complex *psi, const double *V,
                                     int n, double dx, double dt, int n_steps,
                                     double complex *psi_hist) {
    double *k = (double *)malloc(sizeof(double) * (size_t)n);
    double complex *half_kin = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double complex *potential = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double complex *work = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    if (!k || !half_kin || !potential || !work) abort();

    if (psi_hist) memcpy(psi_hist, psi, sizeof(double complex) * (size_t)n);
    for (int j = 0; j < n; j++) {
        int f = (j <= n / 2) ? j : j - n;
        k[j] = 2.0 * M_PI * (double)f / ((double)n * dx);
        half_kin[j] = cexp(NR_I * (-0.5 * k[j] * k[j] * (dt / 2.0)));
        potential[j] = cexp(NR_I * (-V[j] * dt));
    }

    for (int s = 0; s < n_steps; s++) {
        /* kinetic half-step in Fourier space */
        memcpy(work, psi, sizeof(double complex) * (size_t)n);
        nr_fft(work, n, 0);
        for (int j = 0; j < n; j++) work[j] *= half_kin[j];
        nr_fft(work, n, 1);
        /* full potential step in real space */
        for (int j = 0; j < n; j++) work[j] *= potential[j];
        /* second kinetic half-step in Fourier space */
        nr_fft(work, n, 0);
        for (int j = 0; j < n; j++) work[j] *= half_kin[j];
        nr_fft(work, n, 1);

        memcpy(psi, work, sizeof(double complex) * (size_t)n);
        if (psi_hist) memcpy(psi_hist + (size_t)(s + 1) * n, work,
                             sizeof(double complex) * (size_t)n);
    }
    free(k); free(half_kin); free(potential); free(work);
}

void nr_spectral_wave_1d(double *u, double *v, int n, double dx, double dt,
                         int n_steps, double c, double *u_hist) {
    double *k = (double *)malloc(sizeof(double) * (size_t)n);
    double complex *uhat = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double complex *acc = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    if (!k || !uhat || !acc) abort();

    if (u_hist) memcpy(u_hist, u, sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) {
        int f = (j <= n / 2) ? j : j - n;
        k[j] = 2.0 * M_PI * (double)f / ((double)n * dx);
    }
    double c2 = c * c;
    for (int s = 0; s < n_steps; s++) {
        for (int j = 0; j < n; j++) uhat[j] = u[j];
        nr_fft(uhat, n, 0);
        for (int j = 0; j < n; j++) acc[j] = -c2 * k[j] * k[j] * uhat[j];
        nr_fft(acc, n, 1);
        for (int j = 0; j < n; j++) {
            v[j] = v[j] + dt * creal(acc[j]);
            u[j] = u[j] + dt * v[j];
        }
        if (u_hist) memcpy(u_hist + (size_t)(s + 1) * n, u, sizeof(double) * (size_t)n);
    }
    free(k); free(uhat); free(acc);
}

/* ---- 2D FFT helpers (separable via nr_fft; nx, ny must be powers of two) ---- */
static void fft2_rows(double complex *a, int nx, int ny, int inv) {
    for (int i = 0; i < ny; i++) nr_fft(a + (size_t)i * nx, nx, inv);
}
static void fft2_cols(double complex *a, int nx, int ny, int inv) {
    double complex *col = (double complex *)malloc(sizeof(double complex) * (size_t)ny);
    if (!col) abort();
    for (int j = 0; j < nx; j++) {
        for (int i = 0; i < ny; i++) col[i] = a[(size_t)i * nx + j];
        nr_fft(col, ny, inv);
        for (int i = 0; i < ny; i++) a[(size_t)i * nx + j] = col[i];
    }
    free(col);
}
static void fft2(double complex *a, int nx, int ny, int inv) {
    fft2_rows(a, nx, ny, inv);
    fft2_cols(a, nx, ny, inv);
}

int nr_poisson_periodic_1d(const double *rho, int n, double L, double *u) {
    if (!is_pow2(n)) return -1;
    double complex *z = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double *k = (double *)malloc(sizeof(double) * (size_t)n);
    if (!z || !k) { free(z); free(k); return -1; }
    for (int i = 0; i < n; i++) z[i] = rho[i];
    nr_fft(z, n, 0);
    double dx = L / (double)n;
    for (int j = 0; j < n; j++) {
        int f = (j <= n / 2) ? j : j - n;
        k[j] = 2.0 * M_PI * (double)f / ((double)n * dx);
    }
    for (int i = 0; i < n; i++) {
        double k2 = k[i] * k[i];
        z[i] = (k2 > 0.0) ? (z[i] / k2) : 0.0;   /* zero (mean) mode -> 0 */
    }
    nr_fft(z, n, 1);
    for (int i = 0; i < n; i++) u[i] = creal(z[i]);
    free(z); free(k);
    return 0;
}

void nr_tdschrodinger_nonlinear_2d(double complex *psi, const double *V, double g,
                                   int nx, int ny, double dx, double dy,
                                   double dt, int n_steps) {
    if (!is_pow2(nx) || !is_pow2(ny)) abort();
    double *kx = (double *)malloc(sizeof(double) * (size_t)nx);
    double *ky = (double *)malloc(sizeof(double) * (size_t)ny);
    double complex *half_kin =
        (double complex *)malloc(sizeof(double complex) * (size_t)nx * (size_t)ny);
    if (!kx || !ky || !half_kin) abort();
    for (int j = 0; j < nx; j++) {
        int f = (j <= nx / 2) ? j : j - nx;
        kx[j] = 2.0 * M_PI * (double)f / ((double)nx * dx);
    }
    for (int i = 0; i < ny; i++) {
        int f = (i <= ny / 2) ? i : i - ny;
        ky[i] = 2.0 * M_PI * (double)f / ((double)ny * dy);
    }
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++) {
            double k2 = kx[j] * kx[j] + ky[i] * ky[i];
            half_kin[(size_t)i * nx + j] = cexp(NR_I * (-0.5 * k2 * (dt / 2.0)));
        }
    for (int s = 0; s < n_steps; s++) {
        /* kinetic half-step (Fourier space) */
        fft2(psi, nx, ny, 0);
        for (int i = 0; i < ny * nx; i++) psi[i] *= half_kin[i];
        fft2(psi, nx, ny, 1);
        /* external + interaction full-step (real space) */
        for (int i = 0; i < ny * nx; i++) {
            double m = cabs(psi[i]);
            psi[i] *= cexp(NR_I * (-(V[i] + g * m * m) * dt));
        }
        /* second kinetic half-step (Fourier space) */
        fft2(psi, nx, ny, 0);
        for (int i = 0; i < ny * nx; i++) psi[i] *= half_kin[i];
        fft2(psi, nx, ny, 1);
    }
    free(kx); free(ky); free(half_kin);
}
