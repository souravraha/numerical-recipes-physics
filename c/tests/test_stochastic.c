#include "nr/common.h"
#include "nr/stochastic.h"
#include "test_harness.h"

#include <stdlib.h>

static double popvar(const double *a, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += a[i];
    mean /= (double)n;
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - mean; s += d * d; }
    return s / (double)n;
}

/* Least-squares slope of y on t (matches np.polyfit(t, y, 1)[0]). */
static double lls_slope(const double *t, const double *y, int m) {
    double st = 0.0, sy = 0.0, stt = 0.0, sty = 0.0;
    for (int i = 0; i < m; i++) { st += t[i]; sy += y[i]; stt += t[i] * t[i]; sty += t[i] * y[i]; }
    return (m * sty - st * sy) / (m * stt - st * st);
}

/* Stream particle-by-particle to keep memory bounded (~n_steps doubles, not
 * n_particles * n_steps).  Each particle is an independent path (distinct seed). */
static double brownian_msd_slope(double dt, int n_steps, double D, int n_particles,
                                 double t_lo, double t_hi) {
    int len = n_steps + 1;
    double *t = (double *)malloc(sizeof(double) * (size_t)len);
    double *sum = (double *)calloc((size_t)len, sizeof(double));
    double *row = (double *)malloc(sizeof(double) * (size_t)len);
    for (int i = 0; i < len; i++) t[i] = (double)i * dt;
    for (int p = 0; p < n_particles; p++) {
        nr_brownian_motion(dt, n_steps, D, 0.0, 1, (uint64_t)1000000 + (uint64_t)p, row);
        for (int i = 0; i < len; i++) { double d = row[i]; sum[i] += d * d; }
    }
    for (int i = 0; i < len; i++) sum[i] /= (double)n_particles;
    int first = -1, last = -1;
    for (int i = 0; i < len; i++)
        if (t[i] > t_lo && t[i] < t_hi) { if (first < 0) first = i; last = i; }
    double slope = lls_slope(t + first, sum + first, last - first + 1);
    free(t); free(sum); free(row);
    return slope;
}

static void test_brownian_msd_linear(void) {
    double slope = brownian_msd_slope(1e-4, 20000, 1.0, 2000, 0.3, 1.7);
    CHECK(fabs(slope - 2.0) / 2.0 < 0.10);
}

static void test_brownian_var_grows_linearly(void) {
    double slope = brownian_msd_slope(1e-3, 20000, 1.0, 3000, -1.0, 2.0);
    CHECK(fabs(slope - 2.0) / 2.0 < 0.12);
}

static void test_langevin_equipartition(void) {
    int n_steps = 120000;
    double *x = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    double *v = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    double var_x = 0.0, var_v = 0.0;
    nr_langevin_oscillator(1e-3, n_steps, 1.0, 1.0, 1.0, 1.0, 0, x, v, &var_x, &var_v);
    CHECK(fabs(var_v - 1.0) < 0.20);   /* T/m = 1 */
    CHECK(fabs(var_x - 1.0) < 0.25);   /* T/k = 1 */
    free(x); free(v);
}

static void test_langevin_temperature_scaling(void) {
    int n_steps = 60000;
    double *x = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    double *v = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    double vx1 = 0.0, vv1 = 0.0, vx2 = 0.0, vv2 = 0.0;
    nr_langevin_oscillator(1e-3, n_steps, 1.0, 1.0, 0.5, 1.0, 8, x, v, &vx1, &vv1);
    nr_langevin_oscillator(1e-3, n_steps, 1.0, 1.0, 0.5, 2.0, 8, x, v, &vx2, &vv2);
    CHECK(vv2 > vv1 && vx2 > vx1);
    CHECK(vv2 / vv1 > 1.0 && vv2 / vv1 < 3.0);
    free(x); free(v);
}

static void test_ou_stationary_variance(void) {
    int n_steps = 200000;
    double *x = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    nr_ornstein_uhlenbeck(1.0, 1.0, 0.0, 1e-3, n_steps, 9, x);
    double var = popvar(x + 50000, n_steps + 1 - 50000);
    CHECK(fabs(var - 0.5) / 0.5 < 0.15);   /* sigma^2/(2 theta) = 0.5 */
    free(x);
}

static void test_ou_autocorrelation_exponential(void) {
    int n_steps = 300000;
    int start = 100000;
    int M = n_steps + 1 - start;
    double *x = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    double *y = (double *)malloc(sizeof(double) * (size_t)M);
    nr_ornstein_uhlenbeck(1.0, 1.0, 0.0, 1e-3, n_steps, 10, x);
    double mean = 0.0;
    for (int i = 0; i < M; i++) mean += x[start + i];
    mean /= (double)M;
    for (int i = 0; i < M; i++) y[i] = x[start + i] - mean;
    int lag = (int)(1.0 / 1e-3);   /* 1000 steps = 1.0 s */
    double num = 0.0, den = 0.0;
    for (int i = 0; i < M; i++) den += y[i] * y[i];
    for (int i = 0; i < M - lag; i++) num += y[i + lag] * y[i];
    double acf = (num / (double)(M - lag)) / (den / (double)M);
    CHECK(fabs(acf - exp(-1.0)) < 0.12);
    free(x); free(y);
}

static double f_ou(double x, void *ud) { (void)ud; return -x; }
static double g_ou(double x, void *ud) { (void)x; (void)ud; return 1.0; }

static void test_euler_maruyama_ou_matches_stationary(void) {
    int n_steps = 200000;
    double *traj = (double *)malloc(sizeof(double) * (size_t)(n_steps + 1));
    nr_euler_maruyama(f_ou, g_ou, NULL, 0.0, 1e-3, n_steps, 11, traj);
    double var = popvar(traj + 50000, n_steps + 1 - 50000);
    CHECK(fabs(var - 0.5) / 0.5 < 0.15);
    free(traj);
}

int main(void) {
    fprintf(stderr, ">> brownian_msd_linear\n"); test_brownian_msd_linear();
    fprintf(stderr, ">> brownian_var_grows_linearly\n"); test_brownian_var_grows_linearly();
    fprintf(stderr, ">> langevin_equipartition\n"); test_langevin_equipartition();
    fprintf(stderr, ">> langevin_temperature_scaling\n"); test_langevin_temperature_scaling();
    fprintf(stderr, ">> ou_stationary_variance\n"); test_ou_stationary_variance();
    fprintf(stderr, ">> ou_autocorrelation_exponential\n"); test_ou_autocorrelation_exponential();
    fprintf(stderr, ">> euler_maruyama_ou_matches_stationary\n"); test_euler_maruyama_ou_matches_stationary();
    return nr_report("stochastic");
}
