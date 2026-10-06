/* Shared core for the numerical-recipes C port: small matrix helpers and a
 * seeded RNG (xoshiro256** + Box-Muller).  No external dependencies. */
/*
 * References (see REFERENCES.md):
 *   - D. Blackman, S. Vigna, xoshiro (arXiv:1805.04832, 2018) [xoshiro256**].
 *   - G. E. P. Box, G. M. Muller, Ann. Math. Statist. 29, 610 (1958) [Box-Muller].
 */

#ifndef NR_COMMON_H
#define NR_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Row-major matrix of doubles.  Zero-initialised. */
typedef struct {
    int n;        /* rows */
    int m;        /* cols */
    double *data;
} nr_mat;

nr_mat nr_mat_new(int n, int m);
void   nr_mat_free(nr_mat *a);
void   nr_mat_identity(nr_mat *a, int n);
void   nr_mat_copy(const nr_mat *src, nr_mat *dst);

/* Solve A x = b for a square matrix A (n x n, row-major) with partial
 * pivoting.  Returns 0 on success, -1 if a zero pivot was encountered. */
int nr_solve(const double *A, int n, const double *b, double *x);

/* Householder QR: A (m x n, row-major, m >= n) = Q (m x m) R (m x n). */
void nr_householder_qr(const double *A, int m, int n, double *Q, double *R);

/* Orthonormalisation of columns: in-place QR of V (n x k), returns Q. */
void nr_qr_orthonormalize(double *V, int n, int k);

/* ---- Seeded RNG ---- */
typedef struct {
    uint64_t s[4];
    int have_spare_normal;
    double spare_normal;
} nr_rng;

void   nr_rng_seed(nr_rng *r, uint64_t seed);
double nr_rng_double(nr_rng *r);   /* uniform on [0, 1) */
double nr_rng_normal(nr_rng *r);   /* standard normal (Box-Muller, cached pair) */

/* Basic vector norms/dots. */
double nr_norm(const double *v, int n);
double nr_dot(const double *a, const double *b, int n);
double nr_norm_diff(const double *a, const double *b, int n);

#ifdef __cplusplus
}
#endif
#endif
