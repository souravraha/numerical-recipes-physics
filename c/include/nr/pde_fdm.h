/* Finite-difference PDE solvers (mirrors numerical_recipes.pde_fdm).
 * CFL/stability violations return -1 instead of raising; converged Poisson
 * solvers return 0 with the iteration count in *iters_out. */
/*
 * References (see REFERENCES.md):
 *   - L. Courant, K. Friedrichs, R. Lewy, Math. Ann. 100, 32 (1928) [CFL].
 *   - J. Crank, P. Nicolson, Math. Proc. Camb. Phil. Soc. 42, 50 (1946) [Crank-Nicolson].
 *   - L. H. Thomas, J. Opt. Soc. Am. 24, 595 (1934) [tridiagonal solve].
 *   - R. S. Varga, Matrix Iterative Analysis (1962) [SOR].
 */

#ifndef NR_PDE_FDM_H
#define NR_PDE_FDM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Thomas algorithm for a tridiagonal system (main diag + sub/sup of length n-1). */
void nr_thomas(const double *diag, const double *sub, const double *sup,
               const double *rhs, double *x, int n);

/* 1D heat equation u_t = D u_xx, Dirichlet ends (u_left/u_right). */
int nr_heat_1d_explicit(double *u, int n, double dt, double dx, int n_steps,
                        double D, double u_left, double u_right, double *u_hist);
int nr_heat_1d_cn(double *u, int n, double dt, double dx, int n_steps,
                  double D, double u_left, double u_right, double *u_hist);

/* 1D wave equation u_tt = c^2 u_xx, fixed ends. */
int nr_wave_1d(double *u, double *v, int n, double dt, double dx, int n_steps,
               double c, double *u_hist);

/* 1D advection u_t + a u_x = 0 (upwind, stable for a > 0). */
int nr_advection_1d_upwind(double *u, int n, double dt, double dx, int n_steps,
                           double a, double *u_hist);

/* 2D Poisson lap(u) = -rho, zero boundary; rho/u are n x n row-major.
 * omega <= 0 selects the near-optimal 2/(1 + sin(pi dx)). */
int nr_poisson_2d_jacobi(const double *rho, int n, double dx, double tol,
                         int max_iter, double *u, int *iters_out);
int nr_poisson_2d_sor(const double *rho, int n, double dx, double omega,
                      double tol, int max_iter, double *u, int *iters_out);

#ifdef __cplusplus
}
#endif
#endif
