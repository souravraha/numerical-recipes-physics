#include "nr/verify.h"

#include <math.h>
#include <stdlib.h>

double nr_convergence_order(double (*error_fn)(double h), double h, int n_points, double halving) {
    if (n_points < 2) return NAN;
    double *hv = (double *)malloc(sizeof(double) * (size_t)n_points);
    double *ev = (double *)malloc(sizeof(double) * (size_t)n_points);
    if (!hv || !ev) { free(hv); free(ev); return NAN; }
    for (int i = 0; i < n_points; i++) {
        hv[i] = h * pow(halving, -(double)i);
        ev[i] = error_fn(hv[i]);
    }
    for (int i = 0; i < n_points; i++) {
        if (!(ev[i] > 0.0) || !isfinite(ev[i])) {
            free(hv);
            free(ev);
            return NAN;
        }
    }
    double ratio = hv[1] / hv[0];
    double p = (log(ev[n_points - 1]) - log(ev[n_points - 2])) / log(ratio);
    free(hv);
    free(ev);
    return p;
}

double nr_max_relative_error(const double *approx, const double *exact, int n) {
    double maxdiff = 0.0, scale = 1.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(approx[i] - exact[i]);
        if (d > maxdiff) maxdiff = d;
        double a = fabs(exact[i]);
        if (a > scale) scale = a;
    }
    return maxdiff / scale;
}

double nr_energy_drift(const double *energies, int n) {
    double e0 = energies[0], maxd = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(energies[i] - e0);
        if (d > maxd) maxd = d;
    }
    if (e0 == 0.0) return maxd;
    return maxd / fabs(e0);
}
