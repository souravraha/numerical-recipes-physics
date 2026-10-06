/* Symmetric eigensolvers: Householder reconstruction, Sturm bracketing,
 * constructed A = Q D Q^T, physics spectra, and Lanczos extremes. */
#include "nr/linalg_eig.h"
#include "nr/common.h"
#include "test_harness.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double max_abs_diff(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m = fmax(m, fabs(a[i] - b[i]));
    return m;
}

static void test_householder_recovers_A(void) {
    nr_rng rng;
    nr_rng_seed(&rng, 0);
    int n = 20;
    double *A = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *B = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) B[(size_t)i * n + j] = nr_rng_normal(&rng);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) A[(size_t)i * n + j] = 0.5 * (B[(size_t)i * n + j] + B[(size_t)j * n + i]);
    free(B);
    double *T = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *Q = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *rec = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *qq = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    nr_householder_tridiagonalize(A, n, T, Q);
    /* rec = Q T Q^T (two-pass to keep the transpose explicit) */
    double *qt = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += Q[(size_t)i * n + k] * T[(size_t)k * n + j];
            qt[(size_t)i * n + j] = s;
        }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += qt[(size_t)i * n + k] * Q[(size_t)j * n + k];
            rec[(size_t)i * n + j] = s;
        }
    free(qt);
    CHECK(max_abs_diff(rec, A, n * n) < 1e-10);
    /* T is tridiagonal */
    int tridiag_ok = 1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (j - i > 1 || i - j > 1) tridiag_ok = tridiag_ok && fabs(T[(size_t)i * n + j]) < 1e-12;
    CHECK(tridiag_ok);
    /* Q orthogonal: Q Q^T = I */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += Q[(size_t)i * n + k] * Q[(size_t)j * n + k];
            CHECK(fabs(s - (i == j ? 1.0 : 0.0)) < 1e-12);
        }
    free(A); free(T); free(Q); free(rec); free(qq);
}

static void test_sturm_brackets(void) {
    double d[3] = {3.0, 2.0, 1.0};
    double e[2] = {1.0, 1.0};
    /* eigenvalues: 2 - sqrt(3), 2, 2 + sqrt(3) */
    double w[3] = {2.0 - sqrt(3.0), 2.0, 2.0 + sqrt(3.0)};
    CHECK(nr_sturm_count(d, e, 3, w[0] - 0.05) == 0);
    CHECK(nr_sturm_count(d, e, 3, w[0] + 0.05) == 1);
    CHECK(nr_sturm_count(d, e, 3, w[1] - 0.05) == 1);
    CHECK(nr_sturm_count(d, e, 3, w[1] + 0.05) == 2);
    CHECK(nr_sturm_count(d, e, 3, w[2] - 0.05) == 2);
    CHECK(nr_sturm_count(d, e, 3, w[2] + 0.05) == 3);
}

/* eigh on a constructed A = Q D Q^T (Q orthogonal, D with distinct entries). */
static void test_eigh_constructed(void) {
    int n = 30;
    /* deterministic symmetric B */
    double *B = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) B[(size_t)i * n + j] = sin(0.3 + 0.1 * (i + 3 * j));
    double *Qr = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *T = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *Qb = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    nr_householder_tridiagonalize(B, n, T, Qb);
    double *d = (double *)malloc(sizeof(double) * (size_t)n);
    double *dsorted = (double *)malloc(sizeof(double) * (size_t)n);
    for (int j = 0; j < n; j++) d[j] = 0.5 * (j + 1) + 0.3 * sin(1.7 * j + 0.4);
    memcpy(dsorted, d, sizeof(double) * (size_t)n);
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (dsorted[j] < dsorted[i]) { double t = dsorted[i]; dsorted[i] = dsorted[j]; dsorted[j] = t; }
    /* A = Q D Q^T */
    double *A = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += Qb[(size_t)i * n + k] * d[k] * Qb[(size_t)j * n + k];
            A[(size_t)i * n + j] = s;
        }
    double *w = (double *)malloc(sizeof(double) * (size_t)n);
    double *V = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    nr_eigh_symmetric(A, n, w, V);
    CHECK(max_abs_diff(w, dsorted, n) < 1e-7);
    /* residual A v_j = w_j v_j */
    double *Av = (double *)malloc(sizeof(double) * (size_t)n);
    double maxrel = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * V[(size_t)k * n + j];
            Av[i] = s;
        }
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; i++) {
            num += fabs(Av[i] - w[j] * V[(size_t)i * n + j]);
            den += fabs(Av[i]);
        }
        maxrel = fmax(maxrel, num / den);
    }
    CHECK(maxrel < 1e-6);
    /* orthonormality */
    int ortho_ok = 1;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += V[(size_t)i * n + k] * V[(size_t)j * n + k];
            if (fabs(s - (i == j ? 1.0 : 0.0)) > 1e-8) ortho_ok = 0;
        }
    CHECK(ortho_ok);
    free(B); free(Qr); free(T); free(Qb); free(d); free(dsorted); free(A); free(w); free(V); free(Av);
}

static void test_particle_in_box(void) {
    int n = 60;
    double *H = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *exact = (double *)malloc(sizeof(double) * (size_t)n);
    nr_particle_in_box(n, 1.0, H, exact);
    double *w = (double *)malloc(sizeof(double) * (size_t)n);
    double *V = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    nr_eigh_symmetric(H, n, w, V);
    double dx = 1.0 / (n + 1);
    double *exact_fd = (double *)malloc(sizeof(double) * (size_t)n);
    double wmax = 0.0;
    for (int k = 1; k <= n; k++) {
        double s = sin(k * M_PI / (2.0 * (n + 1)));
        exact_fd[k - 1] = 4.0 * s * s / (dx * dx);
        wmax = fmax(wmax, exact_fd[k - 1]);
    }
    double rel = max_abs_diff(w, exact_fd, n) / wmax;
    CHECK(rel < 1e-7);
    /* low modes approach the continuum (k pi)^2 spectrum */
    double lowerr = 0.0, lowmax = 0.0;
    for (int k = 0; k < 8; k++) {
        lowerr = fmax(lowerr, fabs(w[k] - exact[k]));
        lowmax = fmax(lowmax, exact[k]);
    }
    CHECK(lowerr / lowmax < 5e-2);
    CHECK(w[0] > 0.0);
    int increasing = 1;
    for (int i = 1; i < n; i++) increasing = increasing && (w[i] - w[i - 1]) > 0.0;
    CHECK(increasing);
    free(H); free(exact); free(w); free(V); free(exact_fd);
}

static void test_phonon_chain(void) {
    int n = 40;
    double *D = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *exact = (double *)malloc(sizeof(double) * (size_t)n);
    nr_phonon_chain(n, 1.0, 1.0, D, exact);
    double *w = (double *)malloc(sizeof(double) * (size_t)n);
    double *V = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    nr_eigh_symmetric(D, n, w, V);
    CHECK(fabs(w[0]) < 1e-8);
    int nonneg = 1;
    for (int i = 1; i < n; i++) nonneg = nonneg && w[i] >= -1e-9;
    CHECK(nonneg);
    double *nz = (double *)malloc(sizeof(double) * (size_t)(n - 1));
    double *ref = (double *)malloc(sizeof(double) * (size_t)(n - 1));
    memcpy(nz, w + 1, sizeof(double) * (size_t)(n - 1));
    memcpy(ref, exact + 1, sizeof(double) * (size_t)(n - 1));
    for (int i = 0; i < n - 2; i++)
        for (int j = i + 1; j < n - 1; j++)
            if (ref[j] < ref[i]) { double t = ref[i]; ref[i] = ref[j]; ref[j] = t; }
    double rmax = 0.0, refmax = 0.0;
    for (int i = 0; i < n - 1; i++) {
        rmax = fmax(rmax, fabs(nz[i] - ref[i]));
        refmax = fmax(refmax, ref[i]);
    }
    CHECK(rmax / refmax < 1e-4);
    free(D); free(exact); free(w); free(V); free(nz); free(ref);
}

typedef struct { double *d; double *e; int n; } tridiag_params;

/* Lanczos extremes of a large tridiagonal operator (reference: exact spectrum). */
static void tridiag_matvec(const double *x, double *y, void *ud) {
    tridiag_params *p = (tridiag_params *)ud;
    int n = p->n;
    y[0] = p->d[0] * x[0] + (n > 1 ? p->e[0] * x[1] : 0.0);
    for (int i = 1; i < n - 1; i++) y[i] = p->e[i - 1] * x[i - 1] + p->d[i] * x[i] + p->e[i] * x[i + 1];
    y[n - 1] = p->e[n - 2] * x[n - 2] + p->d[n - 1] * x[n - 1];
}

static void test_lanczos_extremes(void) {
    int n = 400;
    nr_rng rng;
    nr_rng_seed(&rng, 3);
    double *d = (double *)malloc(sizeof(double) * (size_t)n);
    double *e = (double *)malloc(sizeof(double) * (size_t)(n - 1));
    for (int i = 0; i < n; i++) d[i] = fabs(nr_rng_normal(&rng)) + 1.0;
    for (int i = 0; i < n - 1; i++) e[i] = 0.5 * nr_rng_normal(&rng);
    double *w_exact = (double *)malloc(sizeof(double) * (size_t)n);
    nr_tridiagonal_eigenvalues(d, e, n, 1e-13, 300, w_exact);
    /* top 5 and bottom 5 by magnitude */
    int k = 5;
    double cand[10];
    for (int i = 0; i < k; i++) cand[i] = w_exact[i];
    for (int i = 0; i < k; i++) cand[k + i] = w_exact[n - 1 - i];
    for (int i = 0; i < 9; i++)
        for (int j = i + 1; j < 10; j++)
            if (fabs(cand[j]) < fabs(cand[i])) { double t = cand[i]; cand[i] = cand[j]; cand[j] = t; }
    double extremes[5];
    for (int i = 0; i < k; i++) extremes[i] = cand[10 - k + i];
    tridiag_params p = {d, e, n};
    double *w_l = (double *)malloc(sizeof(double) * (size_t)k);
    double *V_l = (double *)malloc(sizeof(double) * (size_t)n * (size_t)k);
    nr_lanczos(tridiag_matvec, &p, n, k, 200, NULL, 0, w_l, V_l);
    double captured[5];
    for (int i = 0; i < k; i++) captured[i] = w_l[i];
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 5; j++)
            if (fabs(captured[j]) < fabs(captured[i])) { double t = captured[i]; captured[i] = captured[j]; captured[j] = t; }
    double maxerr = 0.0;
    for (int i = 0; i < k; i++) maxerr = fmax(maxerr, fabs(captured[i] - extremes[i]));
    CHECK(maxerr < 1e-3);
    free(d); free(e); free(w_exact); free(w_l); free(V_l);
}

int main(void) {
    fprintf(stderr, ">> householder\n"); test_householder_recovers_A();
    fprintf(stderr, ">> sturm\n"); test_sturm_brackets();
    fprintf(stderr, ">> eigh_constructed\n"); test_eigh_constructed();
    fprintf(stderr, ">> particle_in_box\n"); test_particle_in_box();
    fprintf(stderr, ">> phonon_chain\n"); test_phonon_chain();
    fprintf(stderr, ">> lanczos_extremes\n"); test_lanczos_extremes();
    return nr_report("linalg_eig");
}
