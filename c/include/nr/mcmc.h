/* General-purpose Markov-chain Monte Carlo (mirrors numerical_recipes.mcmc):
 * Gaussian random-walk Metropolis-Hastings, fixed-step Hamiltonian Monte
 * Carlo, and chain diagnostics (autocorrelation, integrated autocorrelation
 * time, effective sample size, Gelman-Rubin R-hat).  Targets are defined by
 * a log-density callback (and a gradient callback for HMC); callbacks carry a
 * `void *ud` user-data payload.  RNG is the shared xoshiro256** from common.h.
 * The C Metropolis-Hastings uses a *diagonal* proposal covariance (pass a
 * length-d array of variances, or NULL for the unit variance). */
/*
 * References (see REFERENCES.md):
 *   - N. Metropolis et al., J. Chem. Phys. 21, 1087 (1953) [Metropolis].
 *   - W. K. Hastings, Biometrika 56, 97 (1970) [Metropolis-Hastings].
 *   - R. M. Neal, "MCMC using Hamiltonian dynamics," Handbook of MCMC (2011).
 *   - R. E. Gelman, D. B. Rubin, Stat. Sci. 7, 473 (1992) [R-hat].
 */

#ifndef NR_MCMC_H
#define NR_MCMC_H

#include <stdint.h>
#include "nr/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Log of the (unnormalised) target density at x (length d).  May return
 * -INFINITY in regions of zero weight. */
typedef double (*nr_log_target_fn)(const double *x, int d, void *ud);
/* Gradient of the log-target at x, written to out (length d). */
typedef void (*nr_grad_fn)(const double *x, int d, double *out, void *ud);

/* Gaussian random-walk Metropolis-Hastings.
 * x0 (length d) is the start; cov (length d, diagonal variances, or NULL for
 * unit) is the proposal; x_final (length d) is the end state; history, when
 * non-NULL, is (n_steps+1) x d row-major (initial state first); n_accept,
 * when non-NULL, receives the acceptance rate. */
void nr_metropolis_hastings(const double *x0, int d,
                            nr_log_target_fn logt, void *ud,
                            const double *cov,
                            int n_steps, uint64_t seed,
                            double *x_final, double *history, double *n_accept);

/* Fixed-step HMC (leapfrog + Metropolis accept/reject).  Same I/O contract
 * as nr_metropolis_hastings, plus step_size, n_leapfrog and the gradient. */
void nr_hmc(const double *x0, int d,
            nr_log_target_fn logt, nr_grad_fn grad, void *ud,
            double step_size, int n_leapfrog,
            int n_steps, uint64_t seed,
            double *x_final, double *history, double *n_accept);

/* Sample autocorrelation acf[0..max_lag] of x (length n); acf[0] = 1. */
void nr_autocorrelation(const double *x, int n, int max_lag, double *acf);

/* Truncated integrated autocorrelation time (>= 1). */
double nr_integrated_autocorrelation_time(const double *x, int n, int max_lag);

/* Effective sample size = n / tau_int. */
double nr_effective_sample_size(const double *x, int n);

/* Gelman-Rubin R-hat for m independent chains of length n (chains is
 * m x n row-major).  Returns the potential scale reduction factor. */
double nr_gelman_rubin(const double *chains, int m, int n);

#ifdef __cplusplus
}
#endif
#endif
