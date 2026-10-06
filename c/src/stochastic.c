#include "nr/stochastic.h"

#include "nr/common.h"

#include <math.h>
#include <stdlib.h>

static double popvar(const double *a, int n) {
    if (n < 1) return NAN;
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += a[i];
    mean /= (double)n;
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - mean; s += d * d; }
    return s / (double)n;
}

void nr_euler_maruyama(nr_sde_drift_fn f, nr_sde_diff_fn g, void *ud, double x0,
                       double dt, int n_steps, uint64_t seed, double *traj) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    double x = x0;
    traj[0] = x;
    double sqrt_dt = sqrt(dt);
    for (int i = 0; i < n_steps; i++) {
        double dW = nr_rng_normal(&rng) * sqrt_dt;
        x = x + f(x, ud) * dt + g(x, ud) * dW;
        traj[i + 1] = x;
    }
}

void nr_brownian_motion(double dt, int n_steps, double D, double x0,
                        int n_particles, uint64_t seed, double *traj) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    double step = sqrt(2.0 * D * dt);
    for (int p = 0; p < n_particles; p++) {
        double *row = traj + (size_t)p * (size_t)(n_steps + 1);
        row[0] = x0;
        double x = x0;
        for (int i = 1; i <= n_steps; i++) {
            x += nr_rng_normal(&rng) * step;
            row[i] = x;
        }
    }
}

void nr_langevin_oscillator(double dt, int n_steps, double m, double k, double gamma,
                            double T, uint64_t seed, double *x, double *v,
                            double *var_x, double *var_v) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    int n = n_steps;
    double c = gamma * dt / m;
    double noise_amp = sqrt(2.0 * gamma * T * dt / m);   /* fluctuation-dissipation */
    x[0] = sqrt(T / k);      /* start near the stationary distribution */
    v[0] = sqrt(T / m);
    for (int i = 0; i < n; i++) {
        double eta = nr_rng_normal(&rng) * noise_amp;
        double v1 = (v[i] - x[i] * k * dt / m + eta) / (1.0 + c);
        double x1 = x[i] + 0.5 * dt * (v[i] + v1);
        x[i + 1] = x1;
        v[i + 1] = v1;
    }
    int burn = n / 4;
    if (burn < 1) burn = 1;
    int cnt = n + 1 - burn;
    *var_x = popvar(x + burn, cnt);
    *var_v = popvar(v + burn, cnt);
}

void nr_ornstein_uhlenbeck(double theta, double sigma, double x0, double dt,
                           int n_steps, uint64_t seed, double *x) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    x[0] = x0;
    double sqrt_dt = sqrt(dt);
    for (int i = 0; i < n_steps; i++) {
        x[i + 1] = x[i] - theta * x[i] * dt + sigma * nr_rng_normal(&rng) * sqrt_dt;
    }
}
