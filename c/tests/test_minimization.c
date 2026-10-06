#include "nr/minimization.h"
#include "nr/common.h"
#include "test_harness.h"

#include <math.h>
#include <string.h>

/* ---- well-conditioned quadratic: 0.5 x' A x - b' x, A=[[4,1],[1,3]], b=[2,1] ---- */
static double _A[2][2] = {{4.0, 1.0}, {1.0, 3.0}};
static double _b[2] = {2.0, 1.0};

static double quad(const double *x, void *ud) {
    (void)ud;
    return 0.5 * (x[0] * x[0] * _A[0][0] + x[0] * x[1] * (_A[0][1] + _A[1][0]) + x[1] * x[1] * _A[1][1])
         - (x[0] * _b[0] + x[1] * _b[1]);
}
static void quad_g(const double *x, void *ud, double *out) {
    (void)ud;
    out[0] = _A[0][0] * x[0] + _A[0][1] * x[1] - _b[0];
    out[1] = _A[1][0] * x[0] + _A[1][1] * x[1] - _b[1];
}
static void quad_H(const double *x, void *ud, double *out) {
    (void)x; (void)ud;
    for (int i = 0; i < 4; i++) out[i] = 0.0;
    out[0] = _A[0][0]; out[1] = _A[0][1]; out[2] = _A[1][0]; out[3] = _A[1][1];
}
/* x* = A^-1 b */
static double xstar[2];

/* ---- value-wrapper shims (nr_scalar_fn takes a void* ud) ---- */
static double lj_wrap(const double *x, void *ud) { (void)ud; return nr_lennard_jones_dimer(x); }
static double rosen_wrap(const double *x, void *ud) { (void)ud; return nr_rosenbrock(x); }

/* ---- Lennard-Jones dimer ---- */
static void lj_g(const double *x, void *ud, double *out) {
    (void)ud;
    double r[2] = {x[2] - x[0], x[3] - x[1]};
    double d = sqrt(r[0] * r[0] + r[1] * r[1]);
    double dr = 4.0 * (-12.0 * pow(d, -13) + 6.0 * pow(d, -7));
    out[0] = -dr * r[0] / d; out[1] = -dr * r[1] / d;
    out[2] = dr * r[0] / d;  out[3] = dr * r[1] / d;
}
static void lj_H(const double *x, void *ud, double *out) {
    (void)ud;
    double eps = 1e-5;
    double *xp = (double *)malloc(4 * sizeof(double));
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) out[(size_t)i * 4 + j] = 0.0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                double *xa = (double *)malloc(4 * sizeof(double));
                double *xb = (double *)malloc(4 * sizeof(double));
                memcpy(xa, x, 4 * sizeof(double)); memcpy(xb, x, 4 * sizeof(double));
                xa[i] += eps; xb[i] -= eps;
                double h = (nr_lennard_jones_dimer(xa) - 2.0 * nr_lennard_jones_dimer(x) + nr_lennard_jones_dimer(xb)) / (eps * eps);
                out[(size_t)i * 4 + j] = h;
                free(xa); free(xb);
            } else {
                double h = 0.0;
                double *c = (double *)malloc(4 * sizeof(double));
                double *a = (double *)malloc(4 * sizeof(double));
                double *b = (double *)malloc(4 * sizeof(double));
                double *e = (double *)malloc(4 * sizeof(double));
                memcpy(a, x, 4 * sizeof(double)); a[i] += eps; a[j] += eps;
                memcpy(b, x, 4 * sizeof(double)); b[i] += eps; b[j] -= eps;
                memcpy(e, x, 4 * sizeof(double)); e[i] -= eps; e[j] += eps;
                memcpy(c, x, 4 * sizeof(double)); c[i] -= eps; c[j] -= eps;
                h = (nr_lennard_jones_dimer(a) - nr_lennard_jones_dimer(b) - nr_lennard_jones_dimer(e) + nr_lennard_jones_dimer(c)) / (4.0 * eps * eps);
                out[(size_t)i * 4 + j] = h;
                free(a); free(b); free(e); free(c);
            }
        }
    }
    free(xp);
}
static void check_lj_minimum(const double *x) {
    double r = sqrt((x[2] - x[0]) * (x[2] - x[0]) + (x[3] - x[1]) * (x[3] - x[1]));
    CHECK(fabs(r - pow(2.0, 1.0 / 6.0)) < 1e-3);
    CHECK(fabs(nr_lennard_jones_dimer(x) + 1.0) < 1e-3);
}

/* ---- Rosenbrock ---- */
static void rosen_g(const double *x, void *ud, double *out) {
    (void)ud;
    out[0] = 2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
    out[1] = 200.0 * (x[1] - x[0] * x[0]);
}
static void rosen_H(const double *x, void *ud, double *out) {
    (void)ud;
    out[0] = 2.0 + 400.0 * x[0] - 1200.0 * x[0] * x[0];
    out[1] = -400.0 * x[0];
    out[2] = -400.0 * x[0];
    out[3] = 200.0;
}

static void test_all_methods_quadratic(void) {
    double x0[2] = {5.0, -3.0};
    double *x = (double *)malloc(2 * sizeof(double));
    nr_gradient_descent(quad, quad_g, NULL, x0, 2, 0.3, 1e-8, 2000, x);
    CHECK(nr_norm_diff(x, xstar, 2) < 1e-4);
    free(x);
    x = (double *)malloc(2 * sizeof(double));
    nr_newton(quad, quad_g, quad_H, NULL, x0, 2, 1e-10, 100, x);
    CHECK(nr_norm_diff(x, xstar, 2) < 1e-4);
    free(x);
    x = (double *)malloc(2 * sizeof(double));
    nr_bfgs(quad, quad_g, NULL, x0, 2, 1e-9, 500, x);
    CHECK(nr_norm_diff(x, xstar, 2) < 1e-4);
    free(x);
    x = (double *)malloc(2 * sizeof(double));
    nr_nelder_mead(quad, NULL, x0, 2, 1e-8, 2000, x);
    CHECK(nr_norm_diff(x, xstar, 2) < 1e-4);
    free(x);
}

static void test_newton_quadratic_fast(void) {
    double x0[2] = {5.0, -3.0};
    double *x = (double *)malloc(2 * sizeof(double));
    int it = nr_newton(quad, quad_g, quad_H, NULL, x0, 2, 1e-10, 100, x);
    double r = fabs(_A[0][0] * x[0] + _A[0][1] * x[1] - _b[0]);
    double r2 = fabs(_A[1][0] * x[0] + _A[1][1] * x[1] - _b[1]);
    CHECK(fmax(r, r2) < 1e-8);
    CHECK(it <= 6);
    free(x);
}

static void test_newton_lj(void) {
    double x0[4] = {0.0, 0.0, 1.2, 0.3};
    double *x = (double *)malloc(4 * sizeof(double));
    nr_newton(lj_wrap, lj_g, lj_H, NULL, x0, 4, 1e-10, 100, x);
    check_lj_minimum(x);
    free(x);
}

static void test_bfgs_lj(void) {
    double x0[4] = {0.0, 0.0, 1.2, 0.3};
    double *x = (double *)malloc(4 * sizeof(double));
    nr_bfgs(lj_wrap, lj_g, NULL, x0, 4, 1e-9, 500, x);
    check_lj_minimum(x);
    free(x);
}

static void test_nelder_mead_lj(void) {
    double x0[4] = {0.0, 0.0, 1.2, 0.3};
    double *x = (double *)malloc(4 * sizeof(double));
    nr_nelder_mead(lj_wrap, NULL, x0, 4, 1e-8, 5000, x);
    check_lj_minimum(x);
    double g[4];
    lj_g(x, NULL, g);
    CHECK(nr_norm(g, 4) < 1e-3);
    free(x);
}

static void test_rosenbrock_minimum(void) {
    double x0[2] = {-1.0, 1.5};
    double *x = (double *)malloc(2 * sizeof(double));
    nr_newton(rosen_wrap, rosen_g, rosen_H, NULL, x0, 2, 1e-10, 200, x);
    CHECK(fabs(x[0] - 1.0) < 5e-3);
    CHECK(fabs(x[1] - 1.0) < 5e-3);
    free(x);
    x = (double *)malloc(2 * sizeof(double));
    nr_bfgs(rosen_wrap, rosen_g, NULL, x0, 2, 1e-9, 500, x);
    CHECK(fabs(x[0] - 1.0) < 5e-3);
    CHECK(fabs(x[1] - 1.0) < 5e-3);
    free(x);
    x = (double *)malloc(2 * sizeof(double));
    nr_nelder_mead(rosen_wrap, NULL, x0, 2, 1e-8, 6000, x);
    CHECK(fabs(x[0] - 1.0) < 5e-3);
    CHECK(fabs(x[1] - 1.0) < 5e-3);
    free(x);
}

static void test_armijo_reduces(void) {
    double x[2] = {-1.0, 1.5};
    double g[2], d[2];
    rosen_g(x, NULL, g);
    d[0] = -g[0]; d[1] = -g[1];
    double t = nr_armijo(rosen_wrap, NULL, 2, x, d, g, 0.5, 1e-4, 50);
    double xn[2] = {x[0] + t * d[0], x[1] + t * d[1]};
    CHECK(nr_rosenbrock(xn) < nr_rosenbrock(x));
}

int main(void) {
    /* x* = A^-1 b */
    {
        double A[4] = {_A[0][0], _A[0][1], _A[1][0], _A[1][1]};
        double b[2] = {_b[0], _b[1]};
        nr_solve(A, 2, b, xstar);
    }
    test_all_methods_quadratic();
    test_newton_quadratic_fast();
    test_newton_lj();
    test_bfgs_lj();
    test_nelder_mead_lj();
    test_rosenbrock_minimum();
    test_armijo_reduces();
    return nr_report("minimization");
}
