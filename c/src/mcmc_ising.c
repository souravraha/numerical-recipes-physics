#include "nr/mcmc_ising.h"

#include <math.h>
#include <stdlib.h>

double nr_onsager_tc(double J, double k_B) {
    return J / (k_B * 0.5 * log(1.0 + sqrt(2.0)));   /* = 2 / ln(1+sqrt(2)) for J=k_B=1 */
}

double nr_ising_energy(const int *s, int n, double J, int periodic) {
    double E = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int si = s[i * n + j];
            if (periodic) {
                E += -J * si * (s[((i + 1) % n) * n + j] + s[((i - 1 + n) % n) * n + j]
                               + s[i * n + (j + 1) % n] + s[i * n + (j - 1 + n) % n]);
            } else {
                if (i + 1 < n) E += -J * si * s[(i + 1) * n + j];
                if (i - 1 >= 0) E += -J * si * s[(i - 1) * n + j];
                if (j + 1 < n) E += -J * si * s[i * n + (j + 1)];
                if (j - 1 >= 0) E += -J * si * s[i * n + (j - 1)];
            }
        }
    }
    return E / 2.0;   /* each bond counted twice */
}

double nr_ising_magnetization(const int *s, int n) {
    long sum = 0;
    int N = n * n;
    for (int i = 0; i < N; i++) sum += s[i];
    return (double)sum / (double)N;
}

int nr_metropolis_sweep(int *s, int n, double kT, double J, int periodic, nr_rng *rng) {
    int accepted = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int nbr = 0;
            if (periodic) {
                nbr = s[((i + 1) % n) * n + j] + s[((i - 1 + n) % n) * n + j]
                    + s[i * n + (j + 1) % n] + s[i * n + (j - 1 + n) % n];
            } else {
                if (i + 1 < n) nbr += s[(i + 1) * n + j];
                if (i - 1 >= 0) nbr += s[(i - 1) * n + j];
                if (j + 1 < n) nbr += s[i * n + (j + 1)];
                if (j - 1 >= 0) nbr += s[i * n + (j - 1)];
            }
            double dE = 2.0 * J * s[i * n + j] * (double)nbr;
            if (dE <= 0.0 || nr_rng_double(rng) < exp(-dE / kT)) {
                s[i * n + j] *= -1;
                accepted++;
            }
        }
    }
    return accepted;
}

int nr_wolff_sweep(int *s, int n, double kT, double J, int periodic, nr_rng *rng) {
    int N = n * n;
    int *in_cluster = (int *)calloc((size_t)N, sizeof(int));
    int *stack = (int *)malloc(sizeof(int) * (size_t)N);
    int *cluster = (int *)malloc(sizeof(int) * (size_t)N);
    if (!in_cluster || !stack || !cluster) { free(in_cluster); free(stack); free(cluster); return -1; }

    int i0 = (int)(nr_rng_double(rng) * (double)n); if (i0 >= n) i0 = n - 1;
    int j0 = (int)(nr_rng_double(rng) * (double)n); if (j0 >= n) j0 = n - 1;
    double p = (kT > 0.0) ? (1.0 - exp(-2.0 * J / kT)) : 1.0;

    int top = 0, clen = 0;
    int seed_idx = i0 * n + j0;
    cluster[clen++] = seed_idx;
    in_cluster[seed_idx] = 1;
    stack[0] = seed_idx;

    while (top >= 0) {
        int idx = stack[top--];
        int i = idx / n, j = idx % n;
        int si = s[idx];
        int nbrs[4];
        int nc = 0;
        if (periodic) {
            nbrs[nc++] = ((i + 1) % n) * n + j;
            nbrs[nc++] = ((i - 1 + n) % n) * n + j;
            nbrs[nc++] = i * n + (j + 1) % n;
            nbrs[nc++] = i * n + (j - 1 + n) % n;
        } else {
            if (i + 1 < n) nbrs[nc++] = (i + 1) * n + j;
            if (i - 1 >= 0) nbrs[nc++] = (i - 1) * n + j;
            if (j + 1 < n) nbrs[nc++] = i * n + (j + 1);
            if (j - 1 >= 0) nbrs[nc++] = i * n + (j - 1);
        }
        for (int c = 0; c < nc; c++) {
            int ni = nbrs[c];
            if (!in_cluster[ni] && s[ni] == si) {
                if (kT <= 0.0 || nr_rng_double(rng) < p) {
                    in_cluster[ni] = 1;
                    cluster[clen++] = ni;
                    stack[++top] = ni;
                }
            }
        }
    }
    for (int c = 0; c < clen; c++) s[cluster[c]] *= -1;
    free(in_cluster); free(stack); free(cluster);
    return clen;
}

static int sw_find(int *parent, int a) {
    int root = a;
    while (parent[root] != root) root = parent[root];
    while (parent[a] != root) { int t = parent[a]; parent[a] = root; a = t; }
    return root;
}

int nr_sw_sweep(int *s, int n, double kT, double J, int periodic, nr_rng *rng) {
    int N = n * n;
    double p = (kT > 0.0) ? (1.0 - exp(-2.0 * J / kT)) : 1.0;
    int *parent = (int *)malloc(sizeof(int) * (size_t)N);
    if (!parent) return -1;
    for (int i = 0; i < N; i++) parent[i] = i;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int a = i * n + j;
            int nbrs[2]; int nc = 0;
            if (periodic) {
                nbrs[nc++] = ((i + 1) % n) * n + j;
                nbrs[nc++] = i * n + (j + 1) % n;
            } else {
                if (i + 1 < n) nbrs[nc++] = (i + 1) * n + j;
                if (j + 1 < n) nbrs[nc++] = i * n + (j + 1);
            }
            for (int c = 0; c < nc; c++) {
                int b = nbrs[c];
                if (s[b] == s[a] && nr_rng_double(rng) < p) {
                    int ra = sw_find(parent, a), rb = sw_find(parent, b);
                    if (ra != rb) parent[ra] = rb;
                }
            }
        }
    }
    /* group by root and flip each cluster with probability 1/2 */
    int *label = (int *)malloc(sizeof(int) * (size_t)N);
    int *seen = (int *)calloc((size_t)N, sizeof(int));
    if (!label || !seen) { free(parent); free(label); free(seen); return -1; }
    int n_clusters = 0;
    for (int i = 0; i < N; i++) {
        int r = sw_find(parent, i);
        if (!seen[r]) {
            seen[r] = 1;
            n_clusters++;
            if (nr_rng_double(rng) < 0.5) {
                for (int j = 0; j < N; j++) if (sw_find(parent, j) == r) s[j] *= -1;
            }
        }
    }
    free(parent); free(label); free(seen);
    return n_clusters;
}

nr_ising_thermo_result nr_ising_thermo(int n, double T, int n_sweeps, int equilibration,
                                       uint64_t seed, double J, int method, int periodic) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    int N = n * n;
    int *s = (int *)malloc(sizeof(int) * (size_t)N);
    for (int i = 0; i < N; i++) s[i] = (nr_rng_double(&rng) < 0.5) ? -1 : 1;

    int n_keep = n_sweeps - equilibration;
    if (n_keep < 1) n_keep = n_sweeps;
    double *E = (double *)malloc(sizeof(double) * (size_t)n_keep);
    double *M = (double *)malloc(sizeof(double) * (size_t)n_keep);
    int nsamp = 0;
    for (int sw = 0; sw < n_sweeps; sw++) {
        if (method == NR_ISING_WOLFF) nr_wolff_sweep(s, n, T, J, periodic, &rng);
        else if (method == NR_ISING_SW) nr_sw_sweep(s, n, T, J, periodic, &rng);
        else nr_metropolis_sweep(s, n, T, J, periodic, &rng);
        if (sw < equilibration) continue;
        E[nsamp] = nr_ising_energy(s, n, J, periodic);
        M[nsamp] = fabs(nr_ising_magnetization(s, n));
        nsamp++;
    }

    double meanE = 0.0, meanM = 0.0, meanM2 = 0.0;
    for (int i = 0; i < nsamp; i++) { meanE += E[i]; meanM += M[i]; meanM2 += M[i] * M[i]; }
    meanE /= (double)nsamp; meanM /= (double)nsamp; meanM2 /= (double)nsamp;
    double varE = 0.0;
    for (int i = 0; i < nsamp; i++) { double d = E[i] - meanE; varE += d * d; }
    varE /= (double)nsamp;

    nr_ising_thermo_result r;
    r.energy_per_spin = meanE / (double)N;
    r.magnetization_per_spin = meanM;
    r.specific_heat_per_spin = varE / (T * T * (double)N);
    r.susceptibility_per_spin = (meanM2 - meanM * meanM) * (double)N;
    r.n_samples = nsamp;
    free(s); free(E); free(M);
    return r;
}
