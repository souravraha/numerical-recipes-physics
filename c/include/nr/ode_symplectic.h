/* Symplectic ODE integrators and reference Hamiltonian systems
 * (mirrors numerical_recipes.ode_symplectic).
 *
 * All integrators take x'' = accel(x, v, ud) and work in-place on the
 * caller's x and v arrays.  If x_hist is non-NULL it must hold
 * (n_steps + 1) * n doubles (row-major); x_hist[0] is the initial state
 * and x_hist[i] the state after step i (the v staggering of leapfrog is
 * invisible in the recorded positions).
 *
 * Langevin adds a friction term and a random kick fixed by the
 * fluctuation-dissipation relation, driving an arbitrary force field to
 * the canonical (NVT) ensemble. */
/*
 * References (see REFERENCES.md):
 *   - L. Verlet, Phys. Rev. 159, 98 (1967) [leapfrog / velocity-Verlet].
 *   - C. Runge (1895); W. Kutta, Z. Math. Phys. 46, 43 (1901) [RK4].
 *   - E. Hairer, C. Lubich, L. Wanner, Geometric Numerical Integration (2006).
 *   - P. Langevin, Acad. Sci. Paris 146, 1052 (1908) [Langevin thermostat].
 */

#ifndef NR_ODE_SYMPLECTIC_H
#define NR_ODE_SYMPLECTIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*nr_accel_fn)(const double *x, const double *v, double *a, void *ud);

void nr_leapfrog(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
                 int n_steps, double *x_hist);
void nr_velocity_verlet(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
                        int n_steps, double *x_hist);
void nr_rk4(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
            int n_steps, double *x_hist);

/* Langevin (velocity-Verlet) thermostatted integrator for an NVT ensemble.
 * Integrates m x'' = F(x) - gamma m x' + sqrt(2 gamma T / m) xi(t) with
 * F(x) = m * accel(x, v), k_B = 1, driving an arbitrary force field to the
 * canonical distribution (equipartition <v^2> = T/m per degree of freedom).
 * gamma is the friction, T the temperature, seed the RNG, m the mass. */
void nr_langevin_verlet(double *x, double *v, int n, double dt, nr_accel_fn accel, void *ud,
                        double gamma, double T, int n_steps, uint64_t seed, double m,
                        double *x_hist);

/* Kepler two-body: a = -GM x / |x|^3.  ud may point at a double GM
 * (NULL means GM = 1). */
void   nr_kepler_accel(const double *x, const double *v, double *a, void *ud);
double nr_kepler_energy(const double *x, const double *v, int n, double GM);

/* 1D harmonic chain, fixed ends.  ud may point at a nr_chain_params
 * (NULL means k = m = 1). */
typedef struct { double k, m; int n; } nr_chain_params;
void   nr_chain_accel(const double *u, const double *v, double *a, void *ud);
double nr_chain_energy(const double *u, const double *v, int n, double k, double m);

/* Double pendulum (2 angles).  ud may point at a nr_pendulum_params
 * (NULL means m1 = m2 = l1 = l2 = 1, g = 9.81). */
typedef struct { double m1, m2, l1, l2, g; } nr_pendulum_params;
void   nr_double_pendulum(const double *theta, const double *omega, double *a, void *ud);
double nr_double_pendulum_energy(const double *theta, const double *omega,
                                 double m1, double m2, double l1, double l2, double g);

#ifdef __cplusplus
}
#endif
#endif
