/* Finite-difference PDE solvers, all against analytic references. */
#include "nr/pde_fdm.h"
#include "nr/common.h"
#include "test_harness.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void test_thomas(void) {
    int n = 50;
    double *diag = (double *)malloc(sizeof(double) * (size_t)n);
    double *sub = (double *)malloc(sizeof(double) * (size_t)(n - 1));
    double *sup = (double *)malloc(sizeof(double) * (size_t)(n - 1));
    double *rhs = (double *)malloc(sizeof(double) * (size_t)n);
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    nr_rng rng;
    nr_rng_seed(&rng, 0);
    for (int i = 0; i < n; i++) diag[i] = 4.0;
    for (int i = 0; i < n - 1; i++) sub[i] = sup[i] = -1.0;
    for (int i = 0; i < n; i++) rhs[i] = nr_rng_normal(&rng);
    nr_thomas(diag, sub, sup, rhs, x, n);
    double maxres = 0.0;
    for (int i = 0; i < n; i++) {
        double ax = 4.0 * x[i];
        if (i > 0) ax -= x[i - 1];
        if (i < n - 1) ax -= x[i + 1];
        double d = fabs(ax - rhs[i]);
        if (d > maxres) maxres = d;
    }
    CHECK(maxres < 1e-10);
    free(diag); free(sub); free(sup); free(rhs); free(x);
}

static void test_heat_cn(void) {
    int n = 61;
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double *u = (double *)malloc(sizeof(double) * (size_t)n);
    double dx = 1.0 / (n - 1);
    for (int i = 0; i < n; i++) { x[i] = i * dx; u[i] = sin(M_PI * x[i]); }
    double t_end = 0.05, dt = 0.01 * dx * dx;
    int n_steps = (int)floor(t_end / dt + 0.5);
    CHECK(nr_heat_1d_cn(u, n, dt, dx, n_steps, 1.0, 0.0, 0.0, NULL) == 0);
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(u[i] - exp(-M_PI * M_PI * t_end) * sin(M_PI * x[i]));
        if (d > maxerr) maxerr = d;
    }
    CHECK(maxerr < 2e-3);
    free(x); free(u);
}

static void test_heat_explicit_rejects_unstable(void) {
    int n = 21;
    double *u = (double *)malloc(sizeof(double) * (size_t)n);
    double dx = 1.0 / (n - 1);
    for (int i = 0; i < n; i++) u[i] = sin(M_PI * i * dx);
    CHECK(nr_heat_1d_explicit(u, n, 0.8 * dx * dx, dx, 50, 1.0, 0.0, 0.0, NULL) == -1);
    free(u);
}

static void test_wave(void) {
    int n = 81;
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double *u = (double *)malloc(sizeof(double) * (size_t)n);
    double *v = (double *)calloc((size_t)n, sizeof(double));
    double dx = 1.0 / (n - 1);
    for (int i = 0; i < n; i++) { x[i] = i * dx; u[i] = sin(M_PI * x[i]); }
    double dt = 0.5 * dx;
    int n_steps = (int)floor(0.25 / dt + 0.5);
    CHECK(nr_wave_1d(u, v, n, dt, dx, n_steps, 1.0, NULL) == 0);
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(u[i] - sin(M_PI * x[i]) * cos(M_PI * 0.25));
        if (d > maxerr) maxerr = d;
    }
    CHECK(maxerr < 3e-2);
    free(x); free(u); free(v);
}

static void test_advection(void) {
    int n = 201;
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double *u = (double *)malloc(sizeof(double) * (size_t)n);
    double dx = 1.0 / (n - 1);
    for (int i = 0; i < n; i++) {
        x[i] = i * dx;
        double d = (x[i] - 0.3) / 0.05;
        u[i] = exp(-d * d);
    }
    double dt = 0.5 * dx;
    int n_steps = (int)floor(0.2 / dt + 0.5);
    CHECK(nr_advection_1d_upwind(u, n, dt, dx, n_steps, 1.0, NULL) == 0);
    int ipk = 0;
    for (int i = 1; i < n; i++) if (u[i] > u[ipk]) ipk = i;
    CHECK(x[ipk] > 0.45 && x[ipk] < 0.65);
    free(x); free(u);
}

static void make_rho(int n, double dx, double *rho) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            rho[(size_t)i * n + j] = 2.0 * M_PI * M_PI * sin(M_PI * i * dx) * sin(M_PI * j * dx);
}

static void test_poisson_jacobi(void) {
    int n = 33;
    double dx = 1.0 / (n - 1);
    double *rho = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *u = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    make_rho(n, dx, rho);
    int it;
    CHECK(nr_poisson_2d_jacobi(rho, n, dx, 1e-9, 200000, u, &it) == 0);
    double maxerr = 0.0;
    for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++) {
            double d = fabs(u[(size_t)i * n + j] - sin(M_PI * i * dx) * sin(M_PI * j * dx));
            if (d > maxerr) maxerr = d;
        }
    CHECK(maxerr < 2e-2);
    free(rho); free(u);
}

static void test_poisson_sor_faster(void) {
    int n = 33;
    double dx = 1.0 / (n - 1);
    double *rho = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *u_j = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *u_s = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    make_rho(n, dx, rho);
    int it_j, it_s;
    CHECK(nr_poisson_2d_jacobi(rho, n, dx, 1e-8, 200000, u_j, &it_j) == 0);
    CHECK(nr_poisson_2d_sor(rho, n, dx, 0.0, 1e-8, 200000, u_s, &it_s) == 0);
    double maxdiff = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        double d = fabs(u_j[i] - u_s[i]);
        if (d > maxdiff) maxdiff = d;
    }
    CHECK(maxdiff < 1e-4);
    CHECK(it_s < it_j);
    free(rho); free(u_j); free(u_s);
}

int main(void) {
    test_thomas();
    test_heat_cn();
    test_heat_explicit_rejects_unstable();
    test_wave();
    test_advection();
    test_poisson_jacobi();
    test_poisson_sor_faster();
    return nr_report("pde_fdm");
}
