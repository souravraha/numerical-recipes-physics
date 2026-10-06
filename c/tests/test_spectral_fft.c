#include "nr/common.h"
#include "nr/spectral_fft.h"
#include "test_harness.h"

#include <complex.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double cmaxdiff(const double complex *a, const double complex *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = cabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void test_dft_matches_fft(void) {
    int n = 256;
    nr_rng rng;
    nr_rng_seed(&rng, 0);
    double complex *a = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double complex *b = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    for (int i = 0; i < n; i++) a[i] = nr_rng_normal(&rng);
    memcpy(b, a, sizeof(double complex) * (size_t)n);
    int rc_dft = nr_dft(a, n);
    int rc_fft = nr_fft(b, n, 0);
    CHECK(rc_dft == 0 && rc_fft == 0);
    CHECK(cmaxdiff(a, b, n) < 1e-10);
    free(a);
    free(b);
}

static void test_spectrum_recovers_peak_amplitudes(void) {
    double fs = 1000.0;
    double dt = 1.0 / fs;
    int n = (int)(0.1 / dt);   /* = 100 samples over 0.1 s */
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) {
        double t = (double)i * dt;
        x[i] = 0.6 * sin(2.0 * M_PI * 50.0 * t) + 1.0 * sin(2.0 * M_PI * 120.0 * t);
    }
    int m = n / 2 + 1;
    double *freq = (double *)malloc(sizeof(double) * (size_t)m);
    double *amp = (double *)malloc(sizeof(double) * (size_t)m);
    double *phase = (double *)malloc(sizeof(double) * (size_t)m);
    CHECK(nr_spectrum(x, n, dt, freq, amp, phase) == 0);
    int i50 = 0, i120 = 0;
    for (int i = 0; i < m; i++) {
        if (fabs(freq[i] - 50.0) < fabs(freq[i50] - 50.0)) i50 = i;
        if (fabs(freq[i] - 120.0) < fabs(freq[i120] - 120.0)) i120 = i;
    }
    CHECK(fabs(amp[i50] - 0.6) < 0.05);
    CHECK(fabs(amp[i120] - 1.0) < 0.05);
    free(x); free(freq); free(amp); free(phase);
}

static void test_poisson_fft_solves_sine_source(void) {
    int n = 64;
    double L = 1.0;
    double *rho = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *u = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            rho[i * n + j] = cos(2.0 * M_PI * ((double)j * L / n) / L);
    CHECK(nr_poisson_periodic_2d(rho, n, L, u) == 0);
    double k2 = (2.0 * M_PI / L) * (2.0 * M_PI / L);
    double maxerr = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double exact = cos(2.0 * M_PI * ((double)j / n)) / k2;
            double d = fabs(u[i * n + j] - exact);
            if (d > maxerr) maxerr = d;
        }
    CHECK(maxerr < 1e-6);
    free(rho); free(u);
}

static void test_poisson_fft_residual_small(void) {
    int n = 32;
    double L = 1.0;
    nr_rng rng;
    nr_rng_seed(&rng, 1);
    double *rho = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double mean = 0.0;
    for (int i = 0; i < n * n; i++) { rho[i] = nr_rng_normal(&rng); mean += rho[i]; }
    mean /= (double)(n * n);
    for (int i = 0; i < n * n; i++) rho[i] -= mean;
    double *u = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    CHECK(nr_poisson_periodic_2d(rho, n, L, u) == 0);

    /* Spectral laplacian of u should recover -rho: lap(u) + rho ~ 0. */
    double dx = L / (double)n;
    double *k = (double *)malloc(sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) {
        int f = (j <= n / 2) ? j : j - n;
        k[j] = 2.0 * M_PI * (double)f / ((double)n * dx);
    }
    double complex *uh = (double complex *)malloc(sizeof(double complex) * (size_t)n * (size_t)n);
    double complex *col = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    for (int i = 0; i < n * n; i++) uh[i] = u[i];
    for (int i = 0; i < n; i++) nr_fft(uh + (size_t)i * n, n, 0);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = uh[(size_t)i * n + j];
        nr_fft(col, n, 0);
        for (int i = 0; i < n; i++) uh[(size_t)i * n + j] = col[i];
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double k2 = k[i] * k[i] + k[j] * k[j];
            uh[(size_t)i * n + j] = uh[(size_t)i * n + j] * (-k2);
        }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = uh[(size_t)i * n + j];
        nr_fft(col, n, 1);
        for (int i = 0; i < n; i++) uh[(size_t)i * n + j] = col[i];
    }
    for (int i = 0; i < n; i++) nr_fft(uh + (size_t)i * n, n, 1);
    double maxres = 0.0, scale = 0.0;
    for (int i = 0; i < n * n; i++) {
        double r = fabs(creal(uh[i]) + rho[i]);
        if (r > maxres) maxres = r;
        double s = fabs(rho[i]);
        if (s > scale) scale = s;
    }
    CHECK(maxres / scale < 1e-8);
    free(rho); free(u); free(k); free(uh); free(col);
}

static double norm_psi(const double complex *psi, int n, double dx) {
    double s = 0.0;
    for (int i = 0; i < n; i++) { double m = cabs(psi[i]); s += m * m; }
    return sqrt(s * dx);
}

static void test_split_operator_preserves_norm(void) {
    int n = 1024;
    double L = 40.0;
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double complex *psi = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double *V = (double *)malloc(sizeof(double) * (size_t)n);
    nr_gaussian_wavepacket(x, psi, n, L, 0.0, 4.0, 1.5);
    double dx = x[1] - x[0];
    for (int i = 0; i < n; i++) V[i] = 0.5 * x[i] * x[i];
    double n0 = norm_psi(psi, n, dx);
    nr_tdschrodinger_split_operator(psi, V, n, dx, 0.02, 200, NULL);
    double n1 = norm_psi(psi, n, dx);
    CHECK(fabs(n1 - n0) < 1e-8);
    free(x); free(psi); free(V);
}

static double mean_p(const double complex *psi, int n, double dx) {
    double *k = (double *)malloc(sizeof(double) * (size_t)n);
    double complex *ph = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    for (int j = 0; j < n; j++) {
        int f = (j <= n / 2) ? j : j - n;
        k[j] = 2.0 * M_PI * (double)f / ((double)n * dx);
    }
    memcpy(ph, psi, sizeof(double complex) * (size_t)n);
    nr_fft(ph, n, 0);
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; j++) {
        double w = cabs(ph[j]); w *= w;
        num += k[j] * w;
        den += w;
    }
    free(k); free(ph);
    return num / den;
}

static void test_split_operator_free_particle_momentum_conserved(void) {
    int n = 1024;
    double L = 40.0;
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double complex *psi = (double complex *)malloc(sizeof(double complex) * (size_t)n);
    double *V = (double *)calloc(n, sizeof(double));
    nr_gaussian_wavepacket(x, psi, n, L, 0.0, 3.0, 2.0);
    double dx = x[1] - x[0];
    nr_tdschrodinger_split_operator(psi, V, n, dx, 0.02, 300, NULL);
    CHECK(fabs(mean_p(psi, n, dx) - 3.0) < 0.05);
    free(x); free(psi); free(V);
}


static void test_poisson_1d_solves_sine_source(void) {
    /* lap(u) = -rho on a periodic line; rho = sin(2 pi x / L) -> u = rho/(2pi/L)^2. */
    int n = 256; double L = 10.0;
    double *rho = malloc(sizeof(double) * (size_t)n);
    double *u   = malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) rho[i] = sin(2.0 * M_PI * (double)i / (double)n);
    CHECK(nr_poisson_periodic_1d(rho, n, L, u) == 0);
    double k2 = (2.0 * M_PI / L) * (2.0 * M_PI / L);
    double maxabs = 0.0, maxu = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(u[i] - rho[i] / k2); if (a > maxabs) maxabs = a;
        double b = fabs(u[i]); if (b > maxu) maxu = b;
    }
    CHECK(maxabs / maxu < 1e-8);
    free(rho); free(u);
}

static void test_poisson_1d_zero_mode_removed(void) {
    int n = 64;
    double *rho = malloc(sizeof(double) * (size_t)n);
    double *u   = malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) rho[i] = 3.0;
    CHECK(nr_poisson_periodic_1d(rho, n, 1.0, u) == 0);
    double mx = 0.0;
    for (int i = 0; i < n; i++) { double a = fabs(u[i]); if (a > mx) mx = a; }
    CHECK(mx < 1e-12);   /* pure mean -> only the zero mode -> forced to 0 */
    free(rho); free(u);
}

static void test_nonlinear_number_conserved_g0(void) {
    /* With g = 0 the nonlinear propagator conserves the integrated particle
     * number exactly (it reduces to the linear split-operator). */
    int nx = 64, ny = 64; double L = 10.0, dx = L/nx, dy = L/ny;
    size_t tot = (size_t)nx * ny;
    double complex *psi = malloc(sizeof(double complex) * tot);
    double *V = calloc(tot, sizeof(double));
    for (int i = 0; i < ny; i++) for (int j = 0; j < nx; j++) {
        double x = -L/2.0 + j*dx, y = -L/2.0 + i*dy;
        psi[i*nx + j] = exp(-(x*x + y*y) / 2.0);
    }
    double N0 = 0.0; for (size_t i = 0; i < tot; i++) { double m = cabs(psi[i]); N0 += m*m; }
    nr_tdschrodinger_nonlinear_2d(psi, V, 0.0, nx, ny, dx, dy, 0.02, 200);
    double N1 = 0.0; for (size_t i = 0; i < tot; i++) { double m = cabs(psi[i]); N1 += m*m; }
    CHECK(fabs(N1 - N0) / N0 < 1e-9);
    free(psi); free(V);
}

static void test_nonlinear_sign_changes_spreading(void) {
    /* The sign of g is the defining GP physics: an attractive (g<0) gas
     * collapses (peak density rises), a repulsive (g>0) gas resists the free
     * diffraction (peak higher than the free packet), and the free packet
     * (g=0) spreads the most.  At a fixed time: peak(g<0) > peak(g>0) > peak(g=0). */
    int nx = 64, ny = 64; double L = 10.0, dx = L/nx, dy = L/ny;
    size_t tot = (size_t)nx * ny;
    double complex *psi = malloc(sizeof(double complex) * tot);
    double *V = calloc(tot, sizeof(double));
    double gs[3] = {0.0, 1.0, -1.0};   /* free, repulsive, attractive */
    double peaks[3], driftnum[3];
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < ny; i++) for (int j = 0; j < nx; j++) {
            double x = -L/2.0 + j*dx, y = -L/2.0 + i*dy;
            psi[i*nx + j] = exp(-(x*x + y*y) / 2.0);
        }
        double N0 = 0.0; for (size_t i = 0; i < tot; i++) { double m = cabs(psi[i]); N0 += m*m; }
        nr_tdschrodinger_nonlinear_2d(psi, V, gs[c], nx, ny, dx, dy, 0.05, 400);
        double pk = 0.0, N1 = 0.0;
        for (size_t i = 0; i < tot; i++) { double m = cabs(psi[i]); double d = m*m; if (d > pk) pk = d; N1 += d; }
        peaks[c] = pk; driftnum[c] = fabs(N1 - N0) / N0;
    }
    CHECK(peaks[2] > peaks[1]);   /* attractive (g<0) > repulsive (g>0) */
    CHECK(peaks[1] > peaks[0]);   /* repulsive (g>0) > free (g=0) */
    CHECK(driftnum[0] < 1e-6 && driftnum[1] < 1e-6 && driftnum[2] < 1e-6);
    free(psi); free(V);
}

int main(void) {
    fprintf(stderr, ">> dft_matches_fft\n"); test_dft_matches_fft();
    fprintf(stderr, ">> spectrum_recovers_peak_amplitudes\n"); test_spectrum_recovers_peak_amplitudes();
    fprintf(stderr, ">> poisson_fft_solves_sine_source\n"); test_poisson_fft_solves_sine_source();
    fprintf(stderr, ">> poisson_fft_residual_small\n"); test_poisson_fft_residual_small();
    fprintf(stderr, ">> split_operator_preserves_norm\n"); test_split_operator_preserves_norm();
    fprintf(stderr, ">> split_operator_free_particle_momentum_conserved\n");
    test_split_operator_free_particle_momentum_conserved();
    
    test_poisson_1d_solves_sine_source();
    test_poisson_1d_zero_mode_removed();
    test_nonlinear_number_conserved_g0();
    test_nonlinear_sign_changes_spreading();    return nr_report("spectral_fft");
}
