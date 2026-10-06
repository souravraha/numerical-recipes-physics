/* Symplectic integrators: bounded energy error, 2nd-order convergence,
 * and the leapfrog-vs-RK4 secular-drift comparison on Kepler orbits. */
#include "nr/ode_symplectic.h"
#include "nr/verify.h"
#include "nr/stochastic.h"
#include "nr/common.h"
#include "test_harness.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- harmonic oscillator (k = m = 1) ---- */
static void ho_accel(const double *x, const double *v, double *a, void *ud) {
    (void)v; (void)ud;
    for (int i = 0; i < 2; i++) a[i] = -x[i];
}
static double ho_energy(const double *x, const double *v) {
    return 0.5 * (v[0] * v[0] + v[1] * v[1]) + 0.5 * (x[0] * x[0] + x[1] * x[1]);
}

static void test_harmonic_energy_bounded(void) {
    double x0[2] = {1.0, 0.0}, v0[2] = {0.0, 1.0};
    double E0 = ho_energy(x0, v0);
    double dt = 0.1;
    int n_steps = 4000;
    double x[2], v[2];
    for (int method = 0; method < 2; method++) {
        x[0] = x0[0]; x[1] = x0[1]; v[0] = v0[0]; v[1] = v0[1];
        if (method == 0) nr_leapfrog(x, v, 2, dt, ho_accel, NULL, n_steps, NULL);
        else             nr_velocity_verlet(x, v, 2, dt, ho_accel, NULL, n_steps, NULL);
        double drift = fabs(ho_energy(x, v) - E0) / E0;
        CHECK(drift < 5e-3);
    }
}

static double ho_err_at(double h) {
    double T = 1.5;
    int n = (int)floor(T / h + 0.5);
    double x[2] = {1.0, 0.0}, v[2] = {0.0, 0.0};
    nr_leapfrog(x, v, 2, h, ho_accel, NULL, n, NULL);
    return fabs(x[0] - cos(T));
}
static void test_second_order_convergence(void) {
    double T = 1.5;
    double p = nr_convergence_order(ho_err_at, T / 8.0, 4, 2.0);
    CHECK(p > 1.8 && p < 2.2);
}

/* ---- RK4 on the exact circular Kepler orbit (replaces the SciPy reference) ---- */
static void test_rk4_circular_kepler(void) {
    double x[2] = {1.0, 0.0}, v[2] = {0.0, 1.0};
    double T = 1.0, h = 0.01;
    int n_steps = (int)floor(T / h + 0.5);
    nr_rk4(x, v, 2, h, nr_kepler_accel, NULL, n_steps, NULL);
    CHECK(fabs(x[0] - cos(T)) < 1e-5);
    CHECK(fabs(x[1] - sin(T)) < 1e-5);
    CHECK(fabs(v[0] + sin(T)) < 1e-5);
    CHECK(fabs(v[1] - cos(T)) < 1e-5);
}

/* ---- bounded (symplectic) vs secular (RK4) Kepler energy drift ---- */
typedef void (*integrator_fn)(double *, double *, int, double, nr_accel_fn, void *, int, double *);
static double kepler_drift(integrator_fn integrator, int n_periods, double dt) {
    double x[2] = {1.0, 0.0}, v[2] = {0.0, 1.0};
    double E0 = nr_kepler_energy(x, v, 2, 1.0);
    int n_steps = (int)floor(n_periods * 2.0 * M_PI / dt + 0.5);
    integrator(x, v, 2, dt, nr_kepler_accel, NULL, n_steps, NULL);
    return fabs(nr_kepler_energy(x, v, 2, 1.0) - E0) / fabs(E0);
}
static void test_kepler_drift_leapfrog_vs_rk4(void) {
    double dt = 0.02;
    double leap_short = kepler_drift(nr_leapfrog, 20, dt);
    double leap_long  = kepler_drift(nr_leapfrog, 400, dt);
    double rk4_short  = kepler_drift(nr_rk4, 20, dt);
    double rk4_long   = kepler_drift(nr_rk4, 400, dt);
    CHECK(leap_long < 1e-3);
    CHECK(leap_long < 2.0 * leap_short);
    CHECK(rk4_long > 3.0 * rk4_short);
}

/* ---- harmonic chain energy conservation ---- */
static void test_chain_energy(void) {
    nr_rng rng;
    nr_rng_seed(&rng, 0);
    int n = 16;
    double *u = (double *)malloc(sizeof(double) * (size_t)n);
    double *v = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) { u[i] = nr_rng_normal(&rng); v[i] = nr_rng_normal(&rng); }
    double E0 = nr_chain_energy(u, v, n, 1.0, 1.0);
    nr_chain_params p = {1.0, 1.0, n};
    nr_leapfrog(u, v, n, 0.05, nr_chain_accel, &p, 4000, NULL);
    double drift = fabs(nr_chain_energy(u, v, n, 1.0, 1.0) - E0) / fabs(E0);
    CHECK(drift < 5e-3);
    free(u);
    free(v);
}

/* ---- double pendulum butterfly effect ---- */
static void pendulum_run(const double *th0, double *th_out) {
    double th[2] = {th0[0], th0[1]};
    double om[2] = {0.0, 0.0};
    nr_rk4(th, om, 2, 0.01, nr_double_pendulum, NULL, 3000, NULL);
    th_out[0] = th[0]; th_out[1] = th[1];
}
static void test_butterfly_effect(void) {
    double base[2] = {M_PI / 2.0, M_PI / 4.0};
    double pert[2] = {base[0] + 1e-6, base[1]};
    double th_a[2], th_b[2];
    pendulum_run(base, th_a);
    pendulum_run(pert, th_b);
    double sep = sqrt((th_a[0] - th_b[0]) * (th_a[0] - th_b[0]) + (th_a[1] - th_b[1]) * (th_a[1] - th_b[1]));
    CHECK(sep > 1e-3);
}


/* ---- Langevin thermostat: canonical equilibration of an arbitrary force field ---- */
static void ho1_accel(const double *x, const double *v, double *a, void *ud) {
    (void)v; (void)ud; a[0] = -x[0];
}
static void test_langevin_verlet_equilibrates_harmonic_oscillator(void) {
    /* Strong friction -> short correlation time -> tight equipartition check.
     * Independent reference: <x^2> = T/k and <v^2> = T/m (m = k = T = 1). */
    int n = 1, n_steps = 100000; double dt = 0.005, gamma = 1.0, T = 1.0, m = 1.0;
    double x[1] = {0.0}, v[1] = {1.0};
    double *hist = malloc(sizeof(double) * (size_t)(n_steps + 1));
    nr_langevin_verlet(x, v, n, dt, ho1_accel, NULL, gamma, T, n_steps, 7, m, hist);
    int burn = 20000, N = n_steps - burn; double mx2 = 0.0;
    for (int i = burn; i <= n_steps; i++) { double h = hist[i]; mx2 += h*h; }
    mx2 /= (double)N;
    CHECK(fabs(mx2 - 1.0) < 0.15);   /* <x^2> = T/k */
    free(hist);
}
static void test_langevin_verlet_matches_oscillator(void) {
    /* Cross-check the generic thermostat against the dedicated harmonic-
     * oscillator integrator: both must give <x^2> ~ T/k at the same T. */
    int n = 1, n_steps = 60000; double dt = 0.01, gamma = 1.0, T = 1.0, m = 1.0, k = 1.0;
    double x[1] = {0.0}, v[1] = {1.0};
    double *hist = malloc(sizeof(double) * (size_t)(n_steps + 1));
    nr_langevin_verlet(x, v, n, dt, ho1_accel, NULL, gamma, T, n_steps, 3, m, hist);
    int burn = 15000, N = n_steps - burn; double mx2 = 0.0;
    for (int i = burn; i <= n_steps; i++) { double h = hist[i]; mx2 += h*h; }
    mx2 /= (double)N;
    double var_x = 0.0, var_v = 0.0;
    double *rx = malloc(sizeof(double) * (size_t)(n_steps + 1));
    double *rv = malloc(sizeof(double) * (size_t)(n_steps + 1));
    nr_langevin_oscillator(dt, n_steps, m, k, gamma, T, 3, rx, rv, &var_x, &var_v);
    CHECK(fabs(mx2 - var_x) < 0.2);
    free(hist); free(rx); free(rv);
}

int main(void) {
    test_harmonic_energy_bounded();
    test_second_order_convergence();
    test_rk4_circular_kepler();
    test_kepler_drift_leapfrog_vs_rk4();
    test_chain_energy();
    test_butterfly_effect();
    
    test_langevin_verlet_equilibrates_harmonic_oscillator();
    test_langevin_verlet_matches_oscillator();    return nr_report("ode_symplectic");
}
