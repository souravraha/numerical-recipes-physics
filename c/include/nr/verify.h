/* Shared verification helpers (mirrors numerical_recipes.verify). */
/*
 * References (see REFERENCES.md):
 *   - G. H. Golub, C. F. Van Loan, Matrix Computations (2013).
 *   - N. J. Higham, Accuracy and Stability of Numerical Algorithms (2002).
 */

#ifndef NR_VERIFY_H
#define NR_VERIFY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Estimated order p of error_fn(h) ~ C h^p from the finest two points.
 * Returns NaN if any error is non-positive or non-finite (Python raises). */
double nr_convergence_order(double (*error_fn)(double h), double h, int n_points, double halving);

/* max |approx - exact| / max(1, max |exact|). */
double nr_max_relative_error(const double *approx, const double *exact, int n);

/* Max relative drift of an energy history: max |E - E0| / |E0|. */
double nr_energy_drift(const double *energies, int n);

#ifdef __cplusplus
}
#endif
#endif
