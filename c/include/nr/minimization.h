/* Nonlinear minimisation (mirrors numerical_recipes.minimization).
 *
 * Callbacks:
 *   scalar_fn f(x, ud) -> value
 *   grad_fn   g(x, ud, out) -> gradient into out (n)
 *   hess_fn   H(x, ud, out) -> Hessian into out (n x n, row-major)
 *
 * All optimisers write the minimiser into a fresh malloc'd array (caller
 * frees) and return the iteration count. */
/*
 * References (see REFERENCES.md):
 *   - R. Fletcher (1970); D. Goldfarb (1970); R. Broyden (1970); D. Shanno (1970) [BFGS].
 *   - J. A. Nelder, R. Mead, Comput. J. 7, 308 (1965) [Nelder-Mead].
 *   - L. Armijo et al., Tech. Rep. AFOSR-TR-69-16 (1969) [Armijo line search].
 */

#ifndef NR_MINIMIZATION_H
#define NR_MINIMIZATION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef double (*nr_scalar_fn)(const double *x, void *ud);
typedef void (*nr_grad_fn)(const double *x, void *ud, double *out);
typedef void (*nr_hess_fn)(const double *x, void *ud, double *out);

/* Armijo backtracking: returns step size t. */
double nr_armijo(nr_scalar_fn f, void *ud, int n, const double *x, const double *d,
                 const double *g, double alpha, double c1, int max_iter);

int nr_gradient_descent(nr_scalar_fn f, nr_grad_fn g, void *ud, const double *x0, int n,
                        double lr, double tol, int max_iter, double *x_out);
int nr_newton(nr_scalar_fn f, nr_grad_fn g, nr_hess_fn hess, void *ud, const double *x0, int n,
              double tol, int max_iter, double *x_out);
int nr_bfgs(nr_scalar_fn f, nr_grad_fn g, void *ud, const double *x0, int n,
            double tol, int max_iter, double *x_out);
int nr_nelder_mead(nr_scalar_fn f, void *ud, const double *x0, int n,
                   double tol, int max_iter, double *x_out);

double nr_lennard_jones_dimer(const double *x);
double nr_rosenbrock(const double *x);

#ifdef __cplusplus
}
#endif
#endif
