/* Adaptive RK45: accuracy against analytic solutions (replacing the SciPy
 * references), damping/limit-cycle behaviour, and tolerance responsiveness. */
#include "nr/ode_adaptive.h"
#include "test_harness.h"

#include <math.h>
#include <stdlib.h>

/* y' = -2y + cos(t), y(0) = 1: y(t) = (3/5) e^{-2t} + (2/5) cos t + (1/5) sin t. */
static void linear_rhs(double t, const double *y, double *f, void *ud) {
    (void)y; (void)ud;
    f[0] = -2.0 * y[0] + cos(t);
}
static void test_linear_analytic(void) {
    double y0 = 1.0, T = 3.0;
    double t[4096], y[4096];
    int n_steps, rejections;
    CHECK(nr_integrate_adaptive(linear_rhs, NULL, &y0, 1, 0.0, T, 1e-9, 1e-11,
                                0.0, 0.0, 1e-12, t, y, 4096, &n_steps, &rejections) == 0);
    double exact = 0.6 * exp(-2.0 * T) + 0.4 * cos(T) + 0.2 * sin(T);
    CHECK(fabs(y[n_steps] - exact) < 1e-6);
}

/* y'' = -9y - 2y': y(t) = e^{-t} (cos(sqrt(8) t) + sin(sqrt(8) t)/sqrt(8)). */
static void osc_rhs(double t, const double *y, double *f, void *ud) {
    (void)t; (void)ud;
    f[0] = y[1];
    f[1] = -9.0 * y[0] - 2.0 * y[1];
}
static void test_oscillator_analytic(void) {
    double y0[2] = {1.0, 0.0}, T = 10.0;
    double t[8192], y[2 * 8192];
    int n_steps, rejections;
    CHECK(nr_integrate_adaptive(osc_rhs, NULL, y0, 2, 0.0, T, 1e-8, 1e-10,
                                0.0, 0.0, 1e-12, t, y, 8192, &n_steps, &rejections) == 0);
    double w = sqrt(8.0);
    double exact = exp(-T) * (cos(w * T) + sin(w * T) / w);
    CHECK(fabs(y[2 * n_steps] - exact) < 1e-5);
}

/* RLC: energy must decay, never blow up; peaks shrink over time. */
static void test_rlc_energy_decays(void) {
    double t[8192], y[2 * 8192];
    int n_steps, rejections;
    CHECK(nr_rlc_circuit(0.5, 1.0, 1.0, 1.0, 20.0, t, y, 8192, &n_steps, &rejections) == 0);
    int npts = n_steps + 1;
    double e0 = 0.5 * y[0] * y[0] + 0.5 * y[1] * y[1];
    double eN = 0.5 * y[2 * n_steps] * y[2 * n_steps] + 0.5 * y[2 * n_steps + 1] * y[2 * n_steps + 1];
    CHECK(eN < e0);
    int finite = 1;
    for (int i = 0; i < npts; i++) finite = finite && isfinite(y[2 * i]) && isfinite(y[2 * i + 1]);
    CHECK(finite);
    double early = 0.0, late = 0.0;
    for (int i = 0; i < npts / 4; i++) early = fmax(early, fabs(y[2 * i]));
    for (int i = 3 * npts / 4; i < npts; i++) late = fmax(late, fabs(y[2 * i]));
    CHECK(late < early);
}

/* Van der Pol (mu = 1): settles to a bounded limit cycle, oscillating. */
static void test_van_der_pol_limit_cycle(void) {
    double t[16384], y[2 * 16384];
    int n_steps, rejections;
    CHECK(nr_van_der_pol(1.0, 1.0, 0.0, 40.0, t, y, 16384, &n_steps, &rejections) == 0);
    double xmax = 0.0, xmin = 1e300, xmaxabs = 0.0;
    for (int i = 0; i <= n_steps; i++) {
        double x = y[2 * i];
        if (x > xmax) xmax = x;
        if (x < xmin) xmin = x;
        if (fabs(x) > xmaxabs) xmaxabs = fabs(x);
    }
    CHECK(xmaxabs < 5.0);
    CHECK(xmin < 0.0 && xmax > 0.0);
}

/* Tighter tolerance must take more steps than loose tolerance. */
static void rot_rhs(double t, const double *y, double *f, void *ud) {
    (void)t; (void)ud;
    f[0] = y[1];
    f[1] = -y[0];
}
static void test_step_count_respects_tolerance(void) {
    double y0[2] = {1.0, 0.0};
    double t[16384], y[2 * 16384];
    int nt, rt, nl, rl;
    CHECK(nr_integrate_adaptive(rot_rhs, NULL, y0, 2, 0.0, 10.0, 1e-10, 1e-12,
                                0.0, 0.0, 1e-12, t, y, 16384, &nt, &rt) == 0);
    CHECK(nr_integrate_adaptive(rot_rhs, NULL, y0, 2, 0.0, 10.0, 1e-4, 1e-6,
                                0.0, 0.0, 1e-12, t, y, 16384, &nl, &rl) == 0);
    CHECK(nt > nl);
}

int main(void) {
    test_linear_analytic();
    test_oscillator_analytic();
    test_rlc_energy_decays();
    test_van_der_pol_limit_cycle();
    test_step_count_respects_tolerance();
    return nr_report("ode_adaptive");
}
