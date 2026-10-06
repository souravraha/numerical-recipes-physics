#include "nr/mcmc.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Metropolis accept/reject decision from a log-likelihood difference. */
int nr_metropolis_accept(double old, double new, nr_rng *rng);

/* ---- Metropolis-Hastings (Gaussian random walk, diagonal proposal) ---- */
void nr_metropolis_hastings(const double *x0, int d,
                            nr_log_target_fn logt, void *ud,
                            const double *cov,
                            int n_steps, uint64_t seed,
                            double *x_final, double *history, double *n_accept) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    double *x = (double *)malloc(sizeof(double) * (size_t)d);
    memcpy(x, x0, sizeof(double) * (size_t)d);

    double cur = logt(x, d, ud);
    long accepted = 0;
    if (history) {
        for (int j = 0; j < d; j++) history[j] = x0[j];
    }
    for (int step = 0; step < n_steps; step++) {
        double *prop = (double *)malloc(sizeof(double) * (size_t)d);
        for (int j = 0; j < d; j++) {
            double sd = (cov != NULL) ? sqrt(cov[j]) : 1.0;
            prop[j] = x[j] + sd * nr_rng_normal(&rng);
        }
        double lp = logt(prop, d, ud);
        if (nr_metropolis_accept(cur, lp, &rng)) {
            memcpy(x, prop, sizeof(double) * (size_t)d);
            cur = lp;
            accepted++;
        }
        free(prop);
        if (history) {
            for (int j = 0; j < d; j++) history[(size_t)(step + 1) * d + j] = x[j];
        }
    }
    if (x_final) memcpy(x_final, x, sizeof(double) * (size_t)d);
    if (n_accept) *n_accept = (n_steps > 0) ? (double)accepted / (double)n_steps : 0.0;
    free(x);
}

/* ---- HMC (fixed-step leapfrog + Metropolis accept/reject) ---- */
void nr_hmc(const double *x0, int d,
            nr_log_target_fn logt, nr_grad_fn grad, void *ud,
            double step_size, int n_leapfrog,
            int n_steps, uint64_t seed,
            double *x_final, double *history, double *n_accept) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    double *x = (double *)malloc(sizeof(double) * (size_t)d);
    double *q = (double *)malloc(sizeof(double) * (size_t)d);
    double *p = (double *)malloc(sizeof(double) * (size_t)d);
    double *g = (double *)malloc(sizeof(double) * (size_t)d);
    memcpy(x, x0, sizeof(double) * (size_t)d);

    long accepted = 0;
    if (history) {
        for (int j = 0; j < d; j++) history[j] = x0[j];
    }
    for (int step = 0; step < n_steps; step++) {
        for (int j = 0; j < d; j++) p[j] = nr_rng_normal(&rng);
        double H_old = 0.0;
        for (int j = 0; j < d; j++) H_old += 0.5 * p[j] * p[j];
        H_old -= logt(x, d, ud);

        /* leapfrog */
        memcpy(q, x, sizeof(double) * (size_t)d);
        grad(q, d, g, ud);
        for (int j = 0; j < d; j++) p[j] += 0.5 * step_size * g[j];
        for (int i = 0; i < n_leapfrog; i++) {
            for (int j = 0; j < d; j++) q[j] += step_size * p[j];
            if (i != n_leapfrog - 1) {
                grad(q, d, g, ud);
                for (int j = 0; j < d; j++) p[j] += step_size * g[j];
            }
        }
        grad(q, d, g, ud);
        for (int j = 0; j < d; j++) p[j] += 0.5 * step_size * g[j];
        for (int j = 0; j < d; j++) p[j] = -p[j];   /* reverse momentum */

        double H_new = 0.0;
        for (int j = 0; j < d; j++) H_new += 0.5 * p[j] * p[j];
        H_new -= logt(q, d, ud);

        if (nr_metropolis_accept(H_old, H_new, &rng)) {
            memcpy(x, q, sizeof(double) * (size_t)d);
            accepted++;
        }
        if (history) {
            for (int j = 0; j < d; j++) history[(size_t)(step + 1) * d + j] = x[j];
        }
    }
    if (x_final) memcpy(x_final, x, sizeof(double) * (size_t)d);
    if (n_accept) *n_accept = (n_steps > 0) ? (double)accepted / (double)n_steps : 0.0;
    free(x); free(q); free(p); free(g);
}

int nr_metropolis_accept(double old, double new, nr_rng *rng) {
    if (new == -INFINITY) return 0;
    double delta = new - old;      /* accept if log u < delta */
    if (delta >= 0.0) return 1;
    return log(nr_rng_double(rng)) < delta;
}

/* ---- Diagnostics ---- */
void nr_autocorrelation(const double *x, int n, int max_lag, double *acf) {
    if (max_lag >= n) max_lag = n - 1;
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    double c0 = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; c0 += d * d; }
    for (int k = 0; k <= max_lag; k++) {
        double s = 0.0;
        for (int i = 0; i < n - k; i++) s += (x[i] - mean) * (x[i + k] - mean);
        acf[k] = (c0 > 0.0) ? s / c0 : 1.0;
    }
    acf[0] = 1.0;
}

double nr_integrated_autocorrelation_time(const double *x, int n, int max_lag) {
    if (max_lag <= 0 || max_lag >= n) max_lag = n / 2;
    double *acf = (double *)malloc(sizeof(double) * (size_t)(max_lag + 1));
    nr_autocorrelation(x, n, max_lag, acf);
    double best = 1.0, s = 0.0;
    for (int k = 1; k <= max_lag; k++) {
        s += acf[k];
        double tau = 1.0 + 2.0 * s;
        if (tau < best) break;
        best = tau;
    }
    free(acf);
    return (best < 1.0) ? 1.0 : best;
}

double nr_effective_sample_size(const double *x, int n) {
    double tau = nr_integrated_autocorrelation_time(x, n, 0);
    return (double)n / tau;
}

double nr_gelman_rubin(const double *chains, int m, int n) {
    if (m < 2) return 1.0;
    /* means and variances per chain */
    double *mean_t = (double *)malloc(sizeof(double) * (size_t)m);
    double *var_t = (double *)malloc(sizeof(double) * (size_t)m);
    double overall = 0.0;
    for (int c = 0; c < m; c++) {
        double mu = 0.0;
        for (int i = 0; i < n; i++) mu += chains[(size_t)c * n + i];
        mu /= (double)n;
        mean_t[c] = mu;
        overall += mu;
        double va = 0.0;
        for (int i = 0; i < n; i++) { double d = chains[(size_t)c * n + i] - mu; va += d * d; }
        var_t[c] = (n > 1) ? va / (double)(n - 1) : 0.0;
    }
    overall /= (double)m;
    double B = 0.0;
    for (int c = 0; c < m; c++) { double d = mean_t[c] - overall; B += d * d; }
    B = (double)n * B / (double)(m - 1);
    double W = 0.0;
    for (int c = 0; c < m; c++) W += var_t[c];
    W /= (double)m;
    double var_c = ((double)(n - 1) / (double)n) * W + B / (double)n;
    double r2 = (W > 0.0) ? var_c / W : 1.0;
    free(mean_t); free(var_t);
    return sqrt(r2);
}
