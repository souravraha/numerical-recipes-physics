#include "nr/ode_symplectic.h"
#include "nr/common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void record_hist(double *x_hist, const double *x, int n, int step) {
    if (x_hist) memcpy(&x_hist[(size_t)step * n], x, sizeof(double) * (size_t)n);
}

void nr_leapfrog(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
                 int n_steps, double *x_hist) {
    double *a = (double *)malloc(sizeof(double) * (size_t)n);
    if (!a) abort();
    record_hist(x_hist, x, n, 0);
    accel(x, v, a, ud);
    for (int i = 0; i < n; i++) v[i] += 0.5 * dt * a[i];   /* initial half kick */
    for (int s = 0; s < n_steps; s++) {
        for (int i = 0; i < n; i++) x[i] += dt * v[i];      /* drift */
        record_hist(x_hist, x, n, s + 1);
        accel(x, v, a, ud);                                 /* kick */
        for (int i = 0; i < n; i++) v[i] += dt * a[i];
    }
    free(a);
}

void nr_velocity_verlet(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
                        int n_steps, double *x_hist) {
    double *a = (double *)malloc(sizeof(double) * (size_t)n);
    double *a_new = (double *)malloc(sizeof(double) * (size_t)n);
    if (!a || !a_new) abort();
    accel(x, v, a, ud);
    record_hist(x_hist, x, n, 0);
    for (int s = 0; s < n_steps; s++) {
        for (int i = 0; i < n; i++) x[i] += dt * v[i] + 0.5 * dt * dt * a[i];
        record_hist(x_hist, x, n, s + 1);
        accel(x, v, a_new, ud);
        for (int i = 0; i < n; i++) v[i] += 0.5 * dt * (a[i] + a_new[i]);
        for (int i = 0; i < n; i++) a[i] = a_new[i];
    }
    free(a);
    free(a_new);
}

static void rk4_rhs(const double *y, int n, nr_accel_fn accel, void *ud, double *out) {
    for (int i = 0; i < n; i++) out[i] = y[n + i];
    accel(y, y + n, out + n, ud);
}

void nr_rk4(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
            int n_steps, double *x_hist) {
    int tot = 2 * n;
    double *y = (double *)malloc(sizeof(double) * (size_t)tot);
    double *k1 = (double *)malloc(sizeof(double) * (size_t)tot);
    double *k2 = (double *)malloc(sizeof(double) * (size_t)tot);
    double *k3 = (double *)malloc(sizeof(double) * (size_t)tot);
    double *k4 = (double *)malloc(sizeof(double) * (size_t)tot);
    double *yp = (double *)malloc(sizeof(double) * (size_t)tot);
    if (!y || !k1 || !k2 || !k3 || !k4 || !yp) abort();

    for (int i = 0; i < n; i++) { y[i] = x[i]; y[n + i] = v[i]; }
    record_hist(x_hist, x, n, 0);
    for (int s = 0; s < n_steps; s++) {
        rk4_rhs(y, n, accel, ud, k1);
        for (int i = 0; i < tot; i++) yp[i] = y[i] + 0.5 * dt * k1[i];
        rk4_rhs(yp, n, accel, ud, k2);
        for (int i = 0; i < tot; i++) yp[i] = y[i] + 0.5 * dt * k2[i];
        rk4_rhs(yp, n, accel, ud, k3);
        for (int i = 0; i < tot; i++) yp[i] = y[i] + dt * k3[i];
        rk4_rhs(yp, n, accel, ud, k4);
        for (int i = 0; i < tot; i++) y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    for (int i = 0; i < n; i++) { x[i] = y[i]; v[i] = y[n + i]; }
    free(y); free(k1); free(k2); free(k3); free(k4); free(yp);
}

/* Kepler two-body (2D): a = -GM r / |r|^3. */
void nr_kepler_accel(const double *x, const double *v, double *a, void *ud) {
    (void)v;
    double GM = ud ? *(const double *)ud : 1.0;
    double r = sqrt(x[0] * x[0] + x[1] * x[1]);
    double f = -GM / (r * r * r);
    a[0] = f * x[0];
    a[1] = f * x[1];
}

double nr_kepler_energy(const double *x, const double *v, int n, double GM) {
    double r2 = 0.0, v2 = 0.0;
    for (int i = 0; i < n; i++) { r2 += x[i] * x[i]; v2 += v[i] * v[i]; }
    return 0.5 * v2 - GM / sqrt(r2);
}

/* Harmonic chain: length n is carried in the params struct because the
 * accel callback signature has no n. */
void nr_chain_accel(const double *u, const double *v, double *a, void *ud) {
    (void)v;
    nr_chain_params p = ud ? *(const nr_chain_params *)ud : (nr_chain_params){1.0, 1.0, 16};
    int n = (int)p.n;
    double km = p.k / p.m;
    if (n == 1) { a[0] = 0.0; return; }
    a[0] = km * (u[1] - 2.0 * u[0]);
    a[n - 1] = km * (u[n - 2] - 2.0 * u[n - 1]);
    for (int i = 1; i < n - 1; i++) a[i] = km * (u[i - 1] + u[i + 1] - 2.0 * u[i]);
}

double nr_chain_energy(const double *u, const double *v, int n, double k, double m) {
    double kinetic = 0.0;
    double potential = 0.5 * k * (u[0] * u[0] + u[n - 1] * u[n - 1]);
    for (int i = 0; i < n; i++) {
        kinetic += 0.5 * m * v[i] * v[i];
        if (i > 0) potential += 0.5 * k * (u[i] - u[i - 1]) * (u[i] - u[i - 1]);
    }
    return kinetic + potential;
}

/* Double pendulum (2 angles). */
void nr_double_pendulum(const double *theta, const double *omega, double *a, void *ud) {
    nr_pendulum_params p = ud ? *(const nr_pendulum_params *)ud
                              : (nr_pendulum_params){1.0, 1.0, 1.0, 1.0, 9.81};
    double d = theta[0] - theta[1];
    double cd = cos(d), sd = sin(d);
    double den = 2.0 * p.m1 + p.m2 - p.m2 * cos(2.0 * d);
    a[0] = (-p.g * (2.0 * p.m1 + p.m2) * sin(theta[0])
            - p.m2 * p.g * sin(theta[0] - 2.0 * theta[1])
            - 2.0 * sd * p.m2 * (omega[1] * omega[1] * p.l2 + omega[0] * omega[0] * p.l1 * cd))
           / (p.l1 * den);
    a[1] = (2.0 * sd * (omega[0] * omega[0] * p.l1 * (p.m1 + p.m2)
                        + p.g * (p.m1 + p.m2) * cos(theta[0])
                        + omega[1] * omega[1] * p.l2 * p.m2 * cd))
           / (p.l2 * den);
}

double nr_double_pendulum_energy(const double *theta, const double *omega,
                                 double m1, double m2, double l1, double l2, double g) {
    double th1 = theta[0], th2 = theta[1], w1 = omega[0], w2 = omega[1];
    double v1_sq = l1 * l1 * w1 * w1;
    double v2_sq = v1_sq + l2 * l2 * w2 * w2 + 2.0 * l1 * l2 * w1 * w2 * cos(th1 - th2);
    double kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq;
    double potential = -m1 * g * l1 * cos(th1) - m2 * g * (l1 * cos(th1) + l2 * cos(th2));
    return kinetic + potential;
}

void nr_langevin_verlet(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
                        double gamma, double T, int n_steps, uint64_t seed, double m,
                        double *x_hist) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    double *a = (double *)malloc(sizeof(double) * (size_t)n);
    if (!a) abort();
    double sigma_half = sqrt(gamma * T * dt / m);   /* per half-kick (FDT) amplitude */
    accel(x, v, a, ud);
    record_hist(x_hist, x, n, 0);
    for (int s = 0; s < n_steps; s++) {
        for (int i = 0; i < n; i++)
            v[i] += 0.5 * dt * (a[i] - gamma * v[i]) + sigma_half * nr_rng_normal(&rng);
        for (int i = 0; i < n; i++) x[i] += dt * v[i];
        record_hist(x_hist, x, n, s + 1);
        accel(x, v, a, ud);
        for (int i = 0; i < n; i++)
            v[i] += 0.5 * dt * (a[i] - gamma * v[i]) + sigma_half * nr_rng_normal(&rng);
    }
    free(a);
}
