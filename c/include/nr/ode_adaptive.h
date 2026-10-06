/* Adaptive-step Runge-Kutta 4(5) (Dormand-Prince), as in
 * numerical_recipes.ode_adaptive.
 *
 * The trajectory is written into caller-provided buffers: t (max_points)
 * and y (max_points * n, row-major).  t[0] = t0, y[0] = y0.  Returns 0 on
 * success, -1 if the trajectory needs more than max_points points. */
/*
 * References (see REFERENCES.md):
 *   - J. P. Dormand, P. J. Prince, J. Comput. Appl. Math. 6, 67 (1980) [Dormand-Prince 5(4)].
 *   - E. Hairer, S. Norsett, G. Wanner, Solving Ordinary Differential Equations I (2008).
 */

#ifndef NR_ODE_ADAPTIVE_H
#define NR_ODE_ADAPTIVE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nr_ode_fn)(double t, const double *y, double *f, void *ud);

/* h0 = 0 selects the Hairer-Norsett-Wanner first-step heuristic;
 * max_step = 0 defaults to |t1 - t0| / 10. */
int nr_integrate_adaptive(nr_ode_fn f, void *ud, const double *y0, int n,
                          double t0, double t1, double rtol, double atol,
                          double h0, double max_step, double min_step,
                          double *t, double *y, int max_points,
                          int *n_steps_out, int *rejections_out);

/* Reference systems: series RLC (state [q, i]) and van der Pol (state [x, v]). */
int nr_rlc_circuit(double R, double L, double C, double q0, double t_end,
                   double *t, double *y, int max_points,
                   int *n_steps_out, int *rejections_out);
int nr_van_der_pol(double mu, double x0, double y0, double t_end,
                   double *t, double *y, int max_points,
                   int *n_steps_out, int *rejections_out);

#ifdef __cplusplus
}
#endif
#endif
