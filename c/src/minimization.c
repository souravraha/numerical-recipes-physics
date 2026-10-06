#include "nr/minimization.h"
#include "nr/common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

double nr_armijo(nr_scalar_fn f, void *ud, int n, const double *x, const double *d,
                 const double *g, double alpha, double c1, int max_iter) {
    double f0 = f(x, ud);
    double slope = nr_dot(g, d, n);
    if (slope >= 0.0) return 1e-4; /* not a descent direction; tiny step */
    double t = 1.0;
    double f_new = f0;
    for (int i = 0; i < max_iter; i++) {
        double *xp = (double *)malloc(sizeof(double) * (size_t)n);
        for (int j = 0; j < n; j++) xp[j] = x[j] + t * d[j];
        f_new = f(xp, ud);
        free(xp);
        if (f_new <= f0 + c1 * t * slope) return t;
        t *= alpha;
    }
    return t;
}

int nr_gradient_descent(nr_scalar_fn f, nr_grad_fn g, void *ud, const double *x0, int n,
                        double lr, double tol, int max_iter, double *x_out) {
    (void)f;
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double *grad = (double *)malloc(sizeof(double) * (size_t)n);
    memcpy(x, x0, sizeof(double) * (size_t)n);
    int it = 0;
    for (it = 1; it <= max_iter; it++) {
        g(x, ud, grad);
        if (nr_norm(grad, n) < tol) break;
        for (int i = 0; i < n; i++) x[i] -= lr * grad[i];
    }
    memcpy(x_out, x, sizeof(double) * (size_t)n);
    free(x);
    free(grad);
    return it;
}

int nr_newton(nr_scalar_fn f, nr_grad_fn g, nr_hess_fn hess, void *ud, const double *x0, int n,
              double tol, int max_iter, double *x_out) {
    /* Newton with Armijo; when the Hessian is indefinite (Newton step not a
     * descent direction) fall back to steepest descent, then Armijo.  This is
     * what makes it globally robust on the Rosenbrock valley while retaining
     * quadratic convergence near a positive-definite minimum. */
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double *grad = (double *)malloc(sizeof(double) * (size_t)n);
    double *H = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *d = (double *)malloc(sizeof(double) * (size_t)n);
    memcpy(x, x0, sizeof(double) * (size_t)n);
    int it = 0;
    for (it = 1; it <= max_iter; it++) {
        g(x, ud, grad);
        if (nr_norm(grad, n) < tol) break;
        int use_newton = 0;
        hess(x, ud, H);
        if (nr_solve(H, n, grad, d) == 0) {
            for (int i = 0; i < n; i++) d[i] = -d[i];
            if (nr_dot(grad, d, n) < 0.0) use_newton = 1;
        }
        if (!use_newton) {
            for (int i = 0; i < n; i++) d[i] = -grad[i];
        }
        double t = nr_armijo(f, ud, n, x, d, grad, 0.5, 1e-4, 50);
        for (int i = 0; i < n; i++) x[i] += t * d[i];
    }
    memcpy(x_out, x, sizeof(double) * (size_t)n);
    free(x);
    free(grad);
    free(H);
    free(d);
    return it;
}

int nr_bfgs(nr_scalar_fn f, nr_grad_fn g, void *ud, const double *x0, int n,
            double tol, int max_iter, double *x_out) {
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    double *g0 = (double *)malloc(sizeof(double) * (size_t)n);
    double *B = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *d = (double *)malloc(sizeof(double) * (size_t)n);
    double *x_new = (double *)malloc(sizeof(double) * (size_t)n);
    double *g_new = (double *)malloc(sizeof(double) * (size_t)n);
    double *s = (double *)malloc(sizeof(double) * (size_t)n);
    double *y = (double *)malloc(sizeof(double) * (size_t)n);
    double *Bs = (double *)malloc(sizeof(double) * (size_t)n);
    memcpy(x, x0, sizeof(double) * (size_t)n);
    g(x, ud, g0);
    for (int i = 0; i < n * n; i++) B[i] = 0.0;
    for (int i = 0; i < n; i++) B[(size_t)i * n + i] = 1.0;
    int it = 0;
    for (it = 1; it <= max_iter; it++) {
        if (nr_norm(g0, n) < tol) break;
        if (nr_solve(B, n, g0, d) != 0) {
            for (int i = 0; i < n; i++) d[i] = -g0[i];
        } else {
            for (int i = 0; i < n; i++) d[i] = -d[i];
        }
        double t = nr_armijo(f, ud, n, x, d, g0, 0.5, 1e-4, 50);
        for (int i = 0; i < n; i++) x_new[i] = x[i] + t * d[i];
        g(x_new, ud, g_new);
        for (int i = 0; i < n; i++) { s[i] = x_new[i] - x[i]; y[i] = g_new[i] - g0[i]; }
        double sy = nr_dot(s, y, n);
        if (sy > 1e-12) {
            for (int i = 0; i < n; i++) Bs[i] = 0.0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) Bs[i] += B[(size_t)i * n + j] * s[j];
            double sBs = nr_dot(s, Bs, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    B[(size_t)i * n + j] -= Bs[i] * Bs[j] / sBs;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    B[(size_t)i * n + j] += y[i] * y[j] / sy;
        }
        memcpy(x, x_new, sizeof(double) * (size_t)n);
        memcpy(g0, g_new, sizeof(double) * (size_t)n);
    }
    memcpy(x_out, x, sizeof(double) * (size_t)n);
    free(x); free(g0); free(B); free(d); free(x_new); free(g_new); free(s); free(y); free(Bs);
    return it;
}

int nr_nelder_mead(nr_scalar_fn f, void *ud, const double *x0, int n,
                   double tol, int max_iter, double *x_out) {
    double **simplex = (double **)calloc((size_t)(n + 1), sizeof(double *));
    if (!simplex) abort();
    double *vals = (double *)malloc(sizeof(double) * (size_t)(n + 1));
    for (int i = 0; i <= n; i++) simplex[i] = (double *)malloc(sizeof(double) * (size_t)n);
    for (int i = 0; i < n; i++) simplex[0][i] = x0[i];
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < n; j++) simplex[i][j] = x0[j];
        double step = (fabs(x0[i - 1]) < 1e-8) ? 0.05 : 0.05 * fabs(x0[i - 1]);
        simplex[i][i - 1] += step;
    }
    for (int i = 0; i <= n; i++) vals[i] = f(simplex[i], ud);
    int it = 0;
    double *centroid = (double *)malloc(sizeof(double) * (size_t)n);
    for (it = 1; it <= max_iter; it++) {
        /* argsort by vals (build a permutation, then apply via temp arrays) */
        int *order = (int *)malloc(sizeof(int) * (size_t)(n + 1));
        for (int i = 0; i <= n; i++) order[i] = i;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j <= n; j++)
                if (vals[order[j]] < vals[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
        double **simplex2 = (double **)malloc(sizeof(double *) * (size_t)(n + 1));
        double *vals2 = (double *)malloc(sizeof(double) * (size_t)(n + 1));
        for (int i = 0; i <= n; i++) { simplex2[i] = simplex[order[i]]; vals2[i] = vals[order[i]]; }
        for (int i = 0; i <= n; i++) simplex[i] = simplex2[i];
        memcpy(vals, vals2, sizeof(double) * (size_t)(n + 1));
        free(simplex2);
        free(vals2);
        free(order);
        if ((vals[n] - vals[0]) < tol) break;
        for (int i = 0; i < n; i++) centroid[i] = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) centroid[j] += simplex[i][j];
        for (int j = 0; j < n; j++) centroid[j] /= n;
        const double *worst = simplex[n];
        double *xr = (double *)malloc(sizeof(double) * (size_t)n);
        for (int j = 0; j < n; j++) xr[j] = centroid[j] + 1.0 * (centroid[j] - worst[j]);
        double fr = f(xr, ud);
        if (fr < vals[0]) {
            double *xe = (double *)malloc(sizeof(double) * (size_t)n);
            for (int j = 0; j < n; j++) xe[j] = centroid[j] + 2.0 * (centroid[j] - worst[j]);
            double fe = f(xe, ud);
            if (fe < fr) { memcpy(simplex[n], xe, sizeof(double) * (size_t)n); vals[n] = fe; }
            else { memcpy(simplex[n], xr, sizeof(double) * (size_t)n); vals[n] = fr; }
            free(xe);
        } else if (fr < vals[n - 1]) {
            memcpy(simplex[n], xr, sizeof(double) * (size_t)n);
            vals[n] = fr;
        } else {
            double *xc = (double *)malloc(sizeof(double) * (size_t)n);
            for (int j = 0; j < n; j++) xc[j] = centroid[j] + 0.5 * (worst[j] - centroid[j]);
            double fc = f(xc, ud);
            if (fc < vals[n]) { memcpy(simplex[n], xc, sizeof(double) * (size_t)n); vals[n] = fc; }
            else {
                for (int i = 1; i <= n; i++) {
                    for (int j = 0; j < n; j++) simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                    vals[i] = f(simplex[i], ud);
                }
            }
            free(xc);
        }
        free(xr);
    }
    int best = 0;
    for (int i = 1; i <= n; i++) if (vals[i] < vals[best]) best = i;
    memcpy(x_out, simplex[best], sizeof(double) * (size_t)n);
    for (int i = 0; i <= n; i++) free(simplex[i]);
    free(simplex);
    free(vals);
    free(centroid);
    return it;
}

double nr_lennard_jones_dimer(const double *x) {
    double dx = x[2] - x[0], dy = x[3] - x[1];
    double r = sqrt(dx * dx + dy * dy);
    return 4.0 * (pow(r, -12) - pow(r, -6));
}

double nr_rosenbrock(const double *x) {
    double a = x[0] - 1.0, b = x[1] - x[0] * x[0];
    return a * a + 100.0 * b * b;
}
