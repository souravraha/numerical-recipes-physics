#include "nr/mcmc_ising.h"
#include "test_harness.h"

#include <stdlib.h>

static void test_onsager_tc_value(void) {
    CHECK(fabs(nr_onsager_tc(1.0, 1.0) - 2.2691899) < 1e-5);
}

static void test_low_temperature_orders(void) {
    nr_ising_thermo_result r =
        nr_ising_thermo(8, 1.0, 400, 100, 0, 1.0, NR_ISING_METROPOLIS, 1);
    CHECK(r.magnetization_per_spin > 0.8);
}

static void test_high_temperature_disorders(void) {
    nr_ising_thermo_result r =
        nr_ising_thermo(8, 5.0, 400, 100, 0, 1.0, NR_ISING_METROPOLIS, 1);
    CHECK(r.magnetization_per_spin < 0.4);
}

static void test_specific_heat_peaks_near_tc(void) {
    double Ts[5] = {1.5, 2.0, 2.269, 2.5, 3.0};
    double Cs[5];
    for (int i = 0; i < 5; i++) {
        nr_ising_thermo_result r =
            nr_ising_thermo(12, Ts[i], 600, 150, 1, 1.0, NR_ISING_METROPOLIS, 1);
        Cs[i] = r.specific_heat_per_spin;
    }
    int i_pk = 0;
    for (int i = 1; i < 5; i++) if (Cs[i] > Cs[i_pk]) i_pk = i;
    CHECK(fabs(Ts[i_pk] - 2.269) < 0.6);
    CHECK(Cs[i_pk] > Cs[0] && Cs[i_pk] > Cs[4]);
}

static void test_metropolis_vs_wolff_agree(void) {
    nr_ising_thermo_result rm =
        nr_ising_thermo(10, 2.2, 500, 150, 2, 1.0, NR_ISING_METROPOLIS, 1);
    nr_ising_thermo_result rw =
        nr_ising_thermo(10, 2.2, 500, 150, 2, 1.0, NR_ISING_WOLFF, 1);
    CHECK(fabs(rm.energy_per_spin - rw.energy_per_spin) < 0.15);
}

static void test_energy_is_finite_and_reasonable(void) {
    nr_ising_thermo_result r =
        nr_ising_thermo(6, 2.2, 200, 50, 3, 1.0, NR_ISING_METROPOLIS, 1);
    double e = r.energy_per_spin;
    CHECK(-4.0 <= e && e <= 4.0);
    CHECK(isfinite(e));
}


static void test_sw_matches_wolff(void) {
    /* Swendsen-Wang (all clusters at once) and Wolff (single cluster) must
     * converge to the same Boltzmann distribution -- only the path through
     * phase space differs, so the equilibrium observables agree. */
    nr_ising_thermo_result rw =
        nr_ising_thermo(10, 2.2, 500, 150, 5, 1.0, NR_ISING_WOLFF, 1);
    nr_ising_thermo_result rs =
        nr_ising_thermo(10, 2.2, 500, 150, 5, 1.0, NR_ISING_SW, 1);
    CHECK(fabs(rw.energy_per_spin - rs.energy_per_spin) < 0.15);
    CHECK(fabs(rw.magnetization_per_spin - rs.magnetization_per_spin) < 0.15);
}

int main(void) {
    fprintf(stderr, ">> onsager_tc_value\n"); test_onsager_tc_value();
    fprintf(stderr, ">> low_temperature_orders\n"); test_low_temperature_orders();
    fprintf(stderr, ">> high_temperature_disorders\n"); test_high_temperature_disorders();
    fprintf(stderr, ">> specific_heat_peaks_near_tc\n"); test_specific_heat_peaks_near_tc();
    fprintf(stderr, ">> metropolis_vs_wolff_agree\n"); test_metropolis_vs_wolff_agree();
    fprintf(stderr, ">> energy_is_finite_and_reasonable\n"); test_energy_is_finite_and_reasonable();
    
    test_sw_matches_wolff();    return nr_report("mcmc_ising");
}
