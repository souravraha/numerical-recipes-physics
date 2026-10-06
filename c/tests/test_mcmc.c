/* General-purpose MCMC: Metropolis-Hastings, HMC, and chain diagnostics.
 * The target is a known 2-D Gaussian, so every statistic has an independent
 * analytic reference.  Mirrors tests/test_mcmc.py. */
#include "nr/mcmc.h"
#include "nr/common.h"
#include "test_harness.h"

#include <math.h>
#include <stdlib.h>

/* 2-D Gaussian target: mu = {1,-2}, Sigma = {{1,.3},{.3,2}}, Prec = inv(Sigma). */
static const double MU[2] = {1.0, -2.0};
static const double PREC[2][2] = {
    { 2.0 / 1.91, -0.3 / 1.91 },
    { -0.3 / 1.91,  1.0 / 1.91 } };

static double log_t(const double *x, int d, void *ud) {
    (void)d; (void)ud;
    double dx = x[0] - MU[0], dy = x[1] - MU[1];
    return -0.5 * (PREC[0][0]*dx*dx + (PREC[0][1] + PREC[1][0])*dx*dy + PREC[1][1]*dy*dy);
}
static void grad_t(const double *x, int d, double *out, void *ud) {
    (void)d; (void)ud;
    double dx = x[0] - MU[0], dy = x[1] - MU[1];
    out[0] = -(PREC[0][0]*dx + PREC[0][1]*dy);
    out[1] = -(PREC[1][0]*dx + PREC[1][1]*dy);
}

static void sample_stats(const double *hist, int n, int d, double *mean, double *cov) {
    for (int j = 0; j < d; j++) mean[j] = 0.0;
    for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) mean[j] += hist[i*d + j];
    for (int j = 0; j < d; j++) mean[j] /= (double)n;
    for (int a = 0; a < d; a++) for (int b = 0; b < d; b++) {
        double s = 0.0;
        for (int i = 0; i < n; i++) { double da = hist[i*d+a]-mean[a], db = hist[i*d+b]-mean[b]; s += da*db; }
        cov[a*d + b] = s / (double)(n - 1);
    }
}
static double maxabs2x2(const double *cov, const double *ref) {
    double m = 0.0;
    for (int i = 0; i < 4; i++) { double d = fabs(cov[i] - ref[i]); if (d > m) m = d; }
    return m;
}

static void test_metropolis_recovers_mean_cov(void) {
    double x0[2] = {0.0, 0.0}, cov[2] = {1.5, 1.5};
    int d = 2, n = 20000, burn = 2000;
    double *hist = malloc(sizeof(double) * (size_t)(n + 1) * d);
    double acc = 0.0;
    nr_metropolis_hastings(x0, d, log_t, NULL, cov, n, 1, NULL, hist, &acc);
    const double SIG[4] = {1.0, 0.3, 0.3, 2.0};
    int ns = n - burn;
    double mean[2], covm[4];
    sample_stats(hist + burn*d, ns, d, mean, covm);
    double em = fabs(mean[0]-MU[0]) > fabs(mean[1]-MU[1]) ? fabs(mean[0]-MU[0]) : fabs(mean[1]-MU[1]);
    CHECK(em < 0.1);
    CHECK(maxabs2x2(covm, SIG) < 0.3);
    CHECK(acc > 0.2 && acc < 0.95);
    free(hist);
}

static void test_hmc_recovers_mean(void) {
    double x0[2] = {0.0, 0.0};
    int d = 2, n = 2000, burn = 100;
    double *hist = malloc(sizeof(double) * (size_t)(n + 1) * d);
    double acc = 0.0;
    nr_hmc(x0, d, log_t, grad_t, NULL, 0.5, 15, n, 1, NULL, hist, &acc);
    double mean[2], covm[4];
    sample_stats(hist + burn*d, n - burn, d, mean, covm);
    CHECK(fabs(mean[0]-MU[0]) < 0.1);
    CHECK(fabs(mean[1]-MU[1]) < 0.1);
    CHECK(acc > 0.5);
    free(hist);
}

static void test_metropolis_and_hmc_agree(void) {
    double x0[2] = {0.0, 0.0}, cov[2] = {1.5, 1.5};
    int d = 2;
    double *hm = malloc(sizeof(double) * (size_t)30001 * d);
    double *hh = malloc(sizeof(double) * (size_t)4001 * d);
    double acc = 0.0;
    nr_metropolis_hastings(x0, d, log_t, NULL, cov, 30000, 2, NULL, hm, &acc);
    nr_hmc(x0, d, log_t, grad_t, NULL, 0.5, 15, 4000, 2, NULL, hh, &acc);
    double mm[2], mh[2], c1[4], c2[4];
    sample_stats(hm + 5000*d, 25000, d, mm, c1);
    sample_stats(hh + 500*d, 3500, d, mh, c2);
    CHECK(fabs(mm[0]-mh[0]) < 0.2);
    CHECK(fabs(mm[1]-mh[1]) < 0.2);
    free(hm); free(hh);
}

static void test_hmc_high_acceptance_small_step(void) {
    /* Tiny step -> near-Hamiltonian trajectory -> high acceptance rate.
     * A broken leapfrog would reject almost every proposal. */
    double x0[2] = {0.0, 0.0};
    double x_final[2], acc = 0.0;
    nr_hmc(x0, 2, log_t, grad_t, NULL, 0.05, 10, 2000, 0, x_final, NULL, &acc);
    CHECK(acc > 0.9);
}

static void gen_ar1(double *x, int n, double rho, uint64_t seed) {
    nr_rng rng; nr_rng_seed(&rng, seed);
    x[0] = 0.0;
    for (int i = 1; i < n; i++) x[i] = rho*x[i-1] + sqrt(1.0 - rho*rho) * nr_rng_normal(&rng);
}

static void test_autocorrelation_iid_and_ar1(void) {
    int n = 5000;
    double *iid = malloc(sizeof(double) * (size_t)n);
    nr_rng rng; nr_rng_seed(&rng, 0);
    for (int i = 0; i < n; i++) iid[i] = nr_rng_normal(&rng);
    double acf[6];
    nr_autocorrelation(iid, n, 5, acf);
    CHECK(fabs(acf[0]-1.0) < 1e-12);
    CHECK(fabs(acf[1]) < 0.05);
    free(iid);
    int N = 20000;
    double *ar = malloc(sizeof(double) * (size_t)N);
    gen_ar1(ar, N, 0.9, 0);
    double acf2[4];
    nr_autocorrelation(ar, N, 3, acf2);
    CHECK(fabs(acf2[1]-0.9) < 0.05);
    free(ar);
}

static void test_effective_sample_size_ar1_theory(void) {
    int N = 20000;
    double *ar = malloc(sizeof(double) * (size_t)N);
    gen_ar1(ar, N, 0.9, 0);
    double ess = nr_effective_sample_size(ar, N);
    double theory = (double)N * (1.0 - 0.9) / (1.0 + 0.9);
    CHECK(0.6*theory < ess && ess < 1.5*theory);
    free(ar);
}

static void test_effective_sample_size_iid_large(void) {
    int n = 5000;
    double *iid = malloc(sizeof(double) * (size_t)n);
    nr_rng rng; nr_rng_seed(&rng, 1);
    for (int i = 0; i < n; i++) iid[i] = nr_rng_normal(&rng);
    CHECK(nr_effective_sample_size(iid, n) > 3000);
    free(iid);
}

static void fill_chains(double *chains, int m, int n, const double *means, uint64_t seed) {
    nr_rng rng; nr_rng_seed(&rng, seed);
    for (int c = 0; c < m; c++)
        for (int i = 0; i < n; i++) chains[c*n + i] = means[c] + nr_rng_normal(&rng);
}

static void test_gelman_rubin_near_one(void) {
    int m = 4, n = 4000;
    double *chains = malloc(sizeof(double) * (size_t)m * n);
    double means[4] = {1.0, 1.0, 1.0, 1.0};
    fill_chains(chains, m, n, means, 0);
    double R = nr_gelman_rubin(chains, m, n);
    CHECK(R > 0.9 && R < 1.1);
    free(chains);
}

static void test_gelman_rubin_grows_when_divergent(void) {
    int m = 4, n = 4000;
    double *good = malloc(sizeof(double) * (size_t)m * n);
    double *bad  = malloc(sizeof(double) * (size_t)m * n);
    double means_good[4] = {1.0, 1.0, 1.0, 1.0};
    double means_bad[4]  = {0.0, 2.0, 4.0, 6.0};
    fill_chains(good, m, n, means_good, 0);
    fill_chains(bad,  m, n, means_bad, 0);
    double Rg = nr_gelman_rubin(good, m, n), Rb = nr_gelman_rubin(bad, m, n);
    CHECK(Rb > 2.0 * Rg);
    free(good); free(bad);
}

int main(void) {
    fprintf(stderr, ">> metropolis_recovers_mean_cov\n"); test_metropolis_recovers_mean_cov();
    fprintf(stderr, ">> hmc_recovers_mean\n"); test_hmc_recovers_mean();
    fprintf(stderr, ">> metropolis_and_hmc_agree\n"); test_metropolis_and_hmc_agree();
    fprintf(stderr, ">> hmc_high_acceptance_small_step\n"); test_hmc_high_acceptance_small_step();
    fprintf(stderr, ">> autocorrelation_iid_and_ar1\n"); test_autocorrelation_iid_and_ar1();
    fprintf(stderr, ">> effective_sample_size_ar1_theory\n"); test_effective_sample_size_ar1_theory();
    fprintf(stderr, ">> effective_sample_size_iid_large\n"); test_effective_sample_size_iid_large();
    fprintf(stderr, ">> gelman_rubin_near_one\n"); test_gelman_rubin_near_one();
    fprintf(stderr, ">> gelman_rubin_grows_when_divergent\n"); test_gelman_rubin_grows_when_divergent();
    return nr_report("mcmc");
}
