/* Symmetric eigenvalue machinery (mirrors numerical_recipes.linalg_eig):
 * Householder tridiagonalisation, Sturm bisection, inverse iteration, and
 * Lanczos for large sparse operators. */
/*
 * References (see REFERENCES.md):
 *   - A. S. Householder, Numer. Math. 9, 409 (1958) [tridiagonalisation].
 *   - C. Sturm, Ann. Math. 1, 244 (1829) [Sturm sequence].
 *   - E. T. Lanczos, J. Math. Phys. 2, 205 (1951) [Lanczos].
 *   - G. H. Golub, C. F. Van Loan, Matrix Computations (2013).
 */

#ifndef NR_LINALG_EIG_H
#define NR_LINALG_EIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A = Q T Q^T, T tridiagonal, Q orthogonal.  A is not modified. */
void nr_householder_tridiagonalize(const double *A, int n, double *T, double *Q);

/* Number of eigenvalues of the tridiagonal (diag d, off-diag e, n-1)
 * strictly below x (Sturm sequence). */
int nr_sturm_count(const double *d, const double *e, int n, double x);

/* All eigenvalues, ascending, by Sturm bisection. */
void nr_tridiagonal_eigenvalues(const double *d, const double *e, int n,
                                double tol, int max_iter, double *w);

/* Thomas solve for a symmetric tridiagonal system (off-diag length n-1). */
void nr_tridiag_solve(const double *d, const double *e, int n, const double *rhs, double *x);

/* Eigenvector for eigenvalue w (inverse iteration, deterministic sign fix). */
void nr_inverse_iteration(const double *d, const double *e, int n, double w,
                          int n_iter, uint64_t seed, double *v);

/* Full eigensolver: eigenvalues ascending in w, orthonormal eigenvectors as
 * columns of V (V is n x n row-major). */
void nr_eigh_symmetric(const double *A, int n, double *w, double *V);

/* k extreme Ritz eigenpairs of a symmetric operator M x (callback form).
 * w_ritz ascending, Ritz vectors as columns of V_ritz (n x k). */
typedef void (*nr_matvec_fn)(const double *x, double *y, void *ud);
void nr_lanczos(nr_matvec_fn M, void *ud, int n, int k, int n_iter,
                const double *v0, uint64_t seed, double *w_ritz, double *V_ritz);

/* Physics Hamiltonians (dense matrices, row-major) with exact spectra. */
void nr_particle_in_box(int n, double L, double *H, double *exact);
void nr_phonon_chain(int n, double k, double m, double *D, double *exact);
void nr_truss_stiffness(int n, double *S);

#ifdef __cplusplus
}
#endif
#endif
