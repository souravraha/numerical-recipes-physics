/* Stochastic differential equations and Langevin dynamics
 * (mirrors numerical_recipes.stochastic): Euler-Maruyama, 1D Brownian motion,
 * the underdamped Langevin harmonic oscillator (Bao-Wu / exact-OU velocity
 * update + trapezoidal drift), and the Ornstein-Uhlenbeck process.  Noise is
 * drawn from the seeded xoshiro256** RNG in common.h (fluctuation-dissipation
 * sets the amplitude, not a free parameter). */
/*
 * References (see REFERENCES.md):
 *   - G. Maruyama, J. Math. Soc. Japan 21, 347 (1969) [Euler-Maruyama].
 *   - P. Langevin, C. R. Acad. Sci. 146, 1099 (1908) [Langevin / FDT].
 *   - L. S. Ornstein, G. E. Uhlenbeck, Phys. Rev. 36, 783 (1930) [Ornstein-Uhlenbeck].
 *   - C. Bao, H. Wu, SIAM J. Numer. Anal. (2010) [Langevin thermostat].
 */

#ifndef NR_STOCHASTIC_H
#define NR_STOCHASTIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*nr_sde_drift_fn)(double x, void *ud);   /* f(x) */
typedef double (*nr_sde_diff_fn)(double x, void *ud);    /* g(x) */

/* Euler-Maruyama for dx = f(x) dt + g(x) dW.  traj has length n_steps+1. */
void nr_euler_maruyama(nr_sde_drift_fn f, nr_sde_diff_fn g, void *ud, double x0,
                       double dt, int n_steps, uint64_t seed, double *traj);

/* 1D Brownian motion (f=0, g=sqrt(2D)).  traj is (n_particles) x (n_steps+1)
 * row-major. */
void nr_brownian_motion(double dt, int n_steps, double D, double x0,
                        int n_particles, uint64_t seed, double *traj);

/* Underdamped Langevin harmonic oscillator.  x and v have length n_steps+1.
 * var_x and var_v are the stationary variances (transient discarded).  Theory:
 * var_x = T/k, var_v = T/m. */
void nr_langevin_oscillator(double dt, int n_steps, double m, double k, double gamma,
                            double T, uint64_t seed, double *x, double *v,
                            double *var_x, double *var_v);

/* Ornstein-Uhlenbeck: dx = -theta x dt + sigma dW.  x has length n_steps+1. */
void nr_ornstein_uhlenbeck(double theta, double sigma, double x0, double dt,
                           int n_steps, uint64_t seed, double *x);

#ifdef __cplusplus
}
#endif
#endif
