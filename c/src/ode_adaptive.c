#include "nr/ode_adaptive.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Dormand-Prince 5(4) Butcher tableau (6 stages). */
static const double DP_C[6] = {0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0};
static const double DP_A[6][6] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0 / 5, 0.0, 0.0, 0.0, 0.0, 0.0},
    {3.0 / 40, 9.0 / 40, 0.0, 0.0, 0.0, 0.0},
    {44.0 / 45, -56.0 / 15, 32.0 / 9, 0.0, 0.0, 0.0},
    {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0.0, 0.0},
    {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656, 0.0},
};
static const double DP_B5[6] = {35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84};
static const double DP_B4[6] = {5321.0 / 57600, 0.0, 7429.0 / 16695, 1321.0 / 1920, -126603.0 / 339200, 207.0 / 1400};

static void dp54_compute(double t, const double *y, double h, nr_ode_fn f, void *ud, int n,
                         double *y5_out, double *y4_out) {
    double *k = (double *)malloc(sizeof(double) * 6 * (size_t)n);
    double *ys = (double *)malloc(sizeof(double) * (size_t)n);
    if (!k || !ys) abort();
    f(t, y, k, ud);
    for (int i = 1; i < 6; i++) {
        for (int j = 0; j < n; j++) {
            double s = y[j];
            for (int m = 0; m < i; m++) s += h * DP_A[i][m] * k[(size_t)m * n + j];
            ys[j] = s;
        }
        f(t + DP_C[i] * h, ys, k + (size_t)i * n, ud);
    }
    for (int j = 0; j < n; j++) {
        double y5 = y[j], y4 = y[j];
        for (int i = 0; i < 6; i++) {
            y5 += h * DP_B5[i] * k[(size_t)i * n + j];
            y4 += h * DP_B4[i] * k[(size_t)i * n + j];
        }
        y5_out[j] = y5;
        y4_out[j] = y4;
    }
    free(k);
    free(ys);
}

static double norm_scaled(const double *x, const double *scale, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (x[i] / scale[i]) * (x[i] / scale[i]);
    return sqrt(s);
}

int nr_integrate_adaptive(nr_ode_fn f, void *ud, const double *y0, int n,
                          double t0, double t1, double rtol, double atol,
                          double h0, double max_step, double min_step,
                          double *t, double *y, int max_points,
                          int *n_steps_out, int *rejections_out) {
    if (max_points < 1) return -1;
    double *cur_y = (double *)malloc(sizeof(double) * (size_t)n);
    double *y5 = (double *)malloc(sizeof(double) * (size_t)n);
    double *y4 = (double *)malloc(sizeof(double) * (size_t)n);
    if (!cur_y || !y5 || !y4) abort();
    memcpy(cur_y, y0, sizeof(double) * (size_t)n);

    t[0] = t0;
    memcpy(y, y0, sizeof(double) * (size_t)n);
    int n_points = 1;
    int n_steps = 0, rejections = 0;
    if (t1 == t0) {
        *n_steps_out = 0;
        *rejections_out = 0;
        free(cur_y); free(y5); free(y4);
        return 0;
    }

    int direction = t1 > t0 ? 1 : -1;
    if (max_step <= 0.0) max_step = fabs(t1 - t0) / 10.0;

    /* Heuristic first step (Hairer-Norsett-Wanner). */
    double h;
    if (h0 != 0.0) {
        h = h0;
    } else {
        double *scale = (double *)malloc(sizeof(double) * (size_t)n);
        double *f0 = (double *)malloc(sizeof(double) * (size_t)n);
        if (!scale || !f0) abort();
        int any_nonzero = 0;
        for (int i = 0; i < n; i++) {
            scale[i] = atol + rtol * fabs(y0[i]);
            if (y0[i] != 0.0) any_nonzero = 1;
        }
        f(t0, y0, f0, ud);
        double d0 = any_nonzero ? norm_scaled(y0, scale, n) : 1.0;
        double d1 = norm_scaled(f0, scale, n);
        double mx = d0 > d1 ? d0 : d1;
        h = mx > 0.0 ? 0.01 / pow(mx, 0.2) : max_step;
        if (h < min_step) h = min_step;
        if (h > max_step) h = max_step;
        free(scale);
        free(f0);
    }

    double cur_t = t0;
    while ((direction > 0 && cur_t < t1) || (direction < 0 && cur_t > t1)) {
        if (direction > 0) { if (h > t1 - cur_t) h = t1 - cur_t; }
        else { if (h < t1 - cur_t) h = t1 - cur_t; }
        if (fabs(h) < min_step) break;

        dp54_compute(cur_t, cur_y, h, f, ud, n, y5, y4);
        double err = 0.0;
        for (int i = 0; i < n; i++) {
            double scale = fabs(y5[i]) > fabs(y4[i]) ? fabs(y5[i]) : fabs(y4[i]);
            double e = fabs(y5[i] - y4[i]) / (atol + rtol * scale);
            if (e > err) err = e;
        }

        if (err <= 1.0 || fabs(h) <= min_step) {
            cur_t = cur_t + direction * fabs(h);
            memcpy(cur_y, y5, sizeof(double) * (size_t)n);
            if (n_points >= max_points) { free(cur_y); free(y5); free(y4); return -1; }
            t[n_points] = cur_t;
            memcpy(&y[(size_t)n_points * n], cur_y, sizeof(double) * (size_t)n);
            n_points++;
            n_steps++;
        } else {
            rejections++;
        }
        double factor = (err == 0.0) ? 5.0 : 0.9 * pow(err, -0.2);
        if (factor > 5.0) factor = 5.0;
        if (factor < 0.2) factor = 0.2;
        h = direction * fabs(h) * factor;
        if (h > max_step) h = max_step;
        if (h < -max_step) h = -max_step;
        h = direction * (fabs(h) > min_step ? fabs(h) : min_step);
    }

    *n_steps_out = n_steps;
    *rejections_out = rejections;
    free(cur_y); free(y5); free(y4);
    return 0;
}

/* Series RLC circuit: q'' + (R/L) q' + (1/(LC)) q = 0. */
static void rlc_rhs(double t, const double *y, double *f, void *ud) {
    (void)t;
    const double *p = (const double *)ud;
    double R = p[0], L = p[1], C = p[2];
    f[0] = y[1];
    f[1] = -(R / L) * y[1] - (1.0 / (L * C)) * y[0];
}
int nr_rlc_circuit(double R, double L, double C, double q0, double t_end,
                   double *t, double *y, int max_points,
                   int *n_steps_out, int *rejections_out) {
    double p[3] = {R, L, C};
    double y0[2] = {q0, 0.0};
    return nr_integrate_adaptive(rlc_rhs, p, y0, 2, 0.0, t_end, 1e-7, 1e-9,
                                 0.0, 0.0, 1e-12, t, y, max_points, n_steps_out, rejections_out);
}

/* Van der Pol oscillator: x' = v, v' = mu (1 - x^2) v - x. */
static void vdp_rhs(double t, const double *y, double *f, void *ud) {
    (void)t;
    double mu = *(const double *)ud;
    f[0] = y[1];
    f[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
}
int nr_van_der_pol(double mu, double x0, double y0, double t_end,
                   double *t, double *y, int max_points,
                   int *n_steps_out, int *rejections_out) {
    double y0v[2] = {x0, y0};
    return nr_integrate_adaptive(vdp_rhs, &mu, y0v, 2, 0.0, t_end,
                                 1e-7, 1e-9, 0.0, 0.0, 1e-12, t, y, max_points,
                                 n_steps_out, rejections_out);
}
