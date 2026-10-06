#include "nr/pde_fdm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void nr_thomas(const double *diag, const double *sub, const double *sup,
               const double *rhs, double *x, int n) {
    double *c = (double *)malloc(sizeof(double) * (size_t)(n > 1 ? n - 1 : 1));
    double *d = (double *)malloc(sizeof(double) * (size_t)n);
    if (!c || !d) abort();
    c[0] = sup[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for (int i = 1; i < n; i++) {
        double denom = diag[i] - sub[i - 1] * c[i - 1];
        if (i < n - 1) c[i] = sup[i] / denom;
        d[i] = (rhs[i] - sub[i - 1] * d[i - 1]) / denom;
    }
    x[n - 1] = d[n - 1];
    for (int i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
    free(c);
    free(d);
}

int nr_heat_1d_explicit(double *u, int n, double dt, double dx, int n_steps,
                        double D, double u_left, double u_right, double *u_hist) {
    double r = D * dt / (dx * dx);
    if (r > 0.5 + 1e-12) return -1;
    double *u_new = (double *)malloc(sizeof(double) * (size_t)n);
    if (!u_new) abort();
    if (u_hist) memcpy(u_hist, u, sizeof(double) * (size_t)n);
    for (int s = 0; s < n_steps; s++) {
        memcpy(u_new, u, sizeof(double) * (size_t)n);
        for (int i = 1; i < n - 1; i++)
            u_new[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        u_new[0] = u_left;
        u_new[n - 1] = u_right;
        memcpy(u, u_new, sizeof(double) * (size_t)n);
        if (u_hist) memcpy(&u_hist[(size_t)(s + 1) * n], u, sizeof(double) * (size_t)n);
    }
    free(u_new);
    return 0;
}

int nr_heat_1d_cn(double *u, int n, double dt, double dx, int n_steps,
                  double D, double u_left, double u_right, double *u_hist) {
    double r = D * dt / (dx * dx);
    double *diag = (double *)malloc(sizeof(double) * (size_t)n);
    double *sub = (double *)malloc(sizeof(double) * (size_t)(n > 1 ? n - 1 : 1));
    double *sup = (double *)malloc(sizeof(double) * (size_t)(n > 1 ? n - 1 : 1));
    double *rhs = (double *)malloc(sizeof(double) * (size_t)n);
    double *tmp = (double *)malloc(sizeof(double) * (size_t)n);
    if (!diag || !sub || !sup || !rhs || !tmp) abort();
    if (u_hist) memcpy(u_hist, u, sizeof(double) * (size_t)n);
    for (int s = 0; s < n_steps; s++) {
        for (int i = 0; i < n; i++) {
            diag[i] = 1.0 + r;
            rhs[i] = (1.0 - r) * u[i] + (r / 2.0) * (u[i > 0 ? i - 1 : 0] + u[i < n - 1 ? i + 1 : n - 1]);
        }
        for (int i = 0; i < n - 1; i++) sub[i] = sup[i] = -r / 2.0;
        /* Dirichlet BCs absorbed into the RHS. */
        diag[0] = 1.0; sub[0] = 0.0;
        rhs[0] = u_left + (r / 2.0) * u[1];
        diag[n - 1] = 1.0; sup[n - 2] = 0.0;
        rhs[n - 1] = u_right + (r / 2.0) * u[n - 2];
        nr_thomas(diag, sub, sup, rhs, tmp, n);
        tmp[0] = u_left;
        tmp[n - 1] = u_right;
        memcpy(u, tmp, sizeof(double) * (size_t)n);
        if (u_hist) memcpy(&u_hist[(size_t)(s + 1) * n], u, sizeof(double) * (size_t)n);
    }
    free(diag); free(sub); free(sup); free(rhs); free(tmp);
    return 0;
}

int nr_wave_1d(double *u, double *v, int n, double dt, double dx, int n_steps,
               double c, double *u_hist) {
    if (c * dt / dx > 1.0 + 1e-12) return -1;
    double *a = (double *)malloc(sizeof(double) * (size_t)n);
    if (!a) abort();
    if (u_hist) memcpy(u_hist, u, sizeof(double) * (size_t)n);
    for (int s = 0; s < n_steps; s++) {
        for (int i = 0; i < n; i++) a[i] = 0.0;
        for (int i = 1; i < n - 1; i++) a[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx);
        for (int i = 1; i < n - 1; i++) v[i] += dt * c * c * a[i];
        for (int i = 0; i < n; i++) u[i] += dt * v[i];
        u[0] = u[n - 1] = 0.0;
        v[0] = v[n - 1] = 0.0;
        if (u_hist) memcpy(&u_hist[(size_t)(s + 1) * n], u, sizeof(double) * (size_t)n);
    }
    free(a);
    return 0;
}

int nr_advection_1d_upwind(double *u, int n, double dt, double dx, int n_steps,
                           double a, double *u_hist) {
    double nu = a * dt / dx;
    if (nu > 1.0 + 1e-12) return -1;
    double *u_new = (double *)malloc(sizeof(double) * (size_t)n);
    if (!u_new) abort();
    if (u_hist) memcpy(u_hist, u, sizeof(double) * (size_t)n);
    for (int s = 0; s < n_steps; s++) {
        memcpy(u_new, u, sizeof(double) * (size_t)n);
        for (int i = 1; i < n; i++) u_new[i] = u[i] - nu * (u[i] - u[i - 1]);
        memcpy(u, u_new, sizeof(double) * (size_t)n);
        if (u_hist) memcpy(&u_hist[(size_t)(s + 1) * n], u, sizeof(double) * (size_t)n);
    }
    free(u_new);
    return 0;
}

int nr_poisson_2d_jacobi(const double *rho, int n, double dx, double tol,
                         int max_iter, double *u, int *iters_out) {
    double dx2 = dx * dx;
    double *u_new = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    if (!u_new) abort();
    memset(u, 0, sizeof(double) * (size_t)n * (size_t)n);
    int it = max_iter;
    for (it = 1; it <= max_iter; it++) {
        memcpy(u_new, u, sizeof(double) * (size_t)n * (size_t)n);
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++) {
                size_t idx = (size_t)i * n + j;
                u_new[idx] = 0.25 * (u[idx + n] + u[idx - n] + u[idx + 1] + u[idx - 1]
                                     + dx2 * rho[idx]);
            }
        double maxdiff = 0.0;
        for (size_t i = 0; i < (size_t)n * n; i++) {
            double d = fabs(u_new[i] - u[i]);
            if (d > maxdiff) maxdiff = d;
        }
        memcpy(u, u_new, sizeof(double) * (size_t)n * (size_t)n);
        if (maxdiff < tol) break;
    }
    free(u_new);
    *iters_out = it;
    return 0;
}

int nr_poisson_2d_sor(const double *rho, int n, double dx, double omega,
                      double tol, int max_iter, double *u, int *iters_out) {
    double dx2 = dx * dx;
    if (omega <= 0.0) {
        omega = 2.0 / (1.0 + sin(M_PI * dx));
        if (omega > 1.9999) omega = 1.9999;
    }
    size_t nn = (size_t)n * n;
    double *u_old = (double *)malloc(sizeof(double) * nn);
    if (!u_old) abort();
    memset(u, 0, sizeof(double) * nn);
    int it = max_iter;
    for (it = 1; it <= max_iter; it++) {
        memcpy(u_old, u, sizeof(double) * nn);
        for (int parity = 0; parity < 2; parity++) {
            for (int i = 1; i < n - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    if (((i + j) & 1) != parity) continue;
                    size_t idx = (size_t)i * n + j;
                    u[idx] = (1.0 - omega) * u[idx]
                             + (omega / 4.0) * (u[idx - n] + u[idx + n] + u[idx - 1] + u[idx + 1]
                                               + dx2 * rho[idx]);
                }
            }
        }
        double maxdiff = 0.0;
        for (size_t i = 0; i < nn; i++) {
            double d = fabs(u[i] - u_old[i]);
            if (d > maxdiff) maxdiff = d;
        }
        if (maxdiff < tol) break;
    }
    free(u_old);
    *iters_out = it;
    return 0;
}
