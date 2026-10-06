/* Monte Carlo for the 2D Ising model (mirrors numerical_recipes.mcmc_ising):
 * the canonical second-order phase transition.  Metropolis single-spin sweeps
 * and Wolff cluster sweeps both obey detailed balance and converge to the same
 * Boltzmann distribution (they differ only in speed).  Spins are int in {+1,-1},
 * stored n x n row-major.  RNG is the shared xoshiro256** from common.h. */
/*
 * References (see REFERENCES.md):
 *   - N. Metropolis et al., J. Chem. Phys. 21, 1087 (1953) [Metropolis].
 *   - U. Wolff, Phys. Rev. Lett. 56, 116 (1986) [Wolff cluster].
 *   - L. Onsager, Phys. Rev. 65, 117 (1944) [exact critical temperature].
 */

#ifndef NR_MCMC_ISING_H
#define NR_MCMC_ISING_H

#include <stdint.h>
#include "nr/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double energy_per_spin;
    double magnetization_per_spin;   /* mean |M|/N over samples */
    double specific_heat_per_spin;   /* (<E^2>-<E>^2)/(kT^2 N) */
    double susceptibility_per_spin;  /* (<M^2>-<M>^2) N, M=|M|/N */
    int n_samples;
} nr_ising_thermo_result;

/* Method selector for nr_ising_thermo. */
enum { NR_ISING_METROPOLIS = 0, NR_ISING_WOLFF = 1, NR_ISING_SW = 2 };

/* Onsager's exact critical temperature: kTc/J = 2 / ln(1 + sqrt(2)). */
double nr_onsager_tc(double J, double k_B);

/* Total energy E = -J sum_{<ij>} s_i s_j. */
double nr_ising_energy(const int *s, int n, double J, int periodic);

/* Magnetisation per spin: (1/N) sum s_i. */
double nr_ising_magnetization(const int *s, int n);

/* One Metropolis sweep (one attempted flip per spin).  Returns # accepted. */
int nr_metropolis_sweep(int *s, int n, double kT, double J, int periodic,
                        nr_rng *rng);

/* One Wolff cluster sweep (grow a cluster from a random seed, flip it).
 * Returns the cluster size. */
int nr_wolff_sweep(int *s, int n, double kT, double J, int periodic,
                   nr_rng *rng);

/* One Swendsen-Wang cluster sweep (identify all clusters at once via
 * percolation, flip each with probability 1/2).  Returns the # clusters. */
int nr_sw_sweep(int *s, int n, double kT, double J, int periodic, nr_rng *rng);

/* Equilibrate (discard `equilibration` sweeps) then sample; return thermo data. */
nr_ising_thermo_result nr_ising_thermo(int n, double T, int n_sweeps, int equilibration,
                                       uint64_t seed, double J, int method, int periodic);

#ifdef __cplusplus
}
#endif
#endif
