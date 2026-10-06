#include "nr/common.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>
#include <string.h>

nr_mat nr_mat_new(int n, int m) {
    nr_mat a;
    a.n = n;
    a.m = m;
    a.data = (double *)malloc(sizeof(double) * (size_t)n * (size_t)m);
    if (!a.data) abort();
    memset(a.data, 0, sizeof(double) * (size_t)n * (size_t)m);
    return a;
}

void nr_mat_free(nr_mat *a) {
    if (a && a->data) {
        free(a->data);
        a->data = NULL;
    }
}

void nr_mat_identity(nr_mat *a, int n) {
    for (int i = 0; i < n; i++) a->data[(size_t)i * n + i] = 1.0;
}

void nr_mat_copy(const nr_mat *src, nr_mat *dst) {
    dst->n = src->n;
    dst->m = src->m;
    dst->data = (double *)malloc(sizeof(double) * (size_t)src->n * (size_t)src->m);
    if (!dst->data) abort();
    memcpy(dst->data, src->data, sizeof(double) * (size_t)src->n * (size_t)src->m);
}

int nr_solve(const double *A, int n, const double *b, double *x) {
    double *L = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *rhs = (double *)malloc(sizeof(double) * (size_t)n);
    if (!L || !rhs) { free(L); free(rhs); return -1; }
    memcpy(L, A, sizeof(double) * (size_t)n * (size_t)n);
    memcpy(rhs, b, sizeof(double) * (size_t)n);
    int *piv = (int *)malloc(sizeof(int) * (size_t)n);
    if (!piv) { free(L); free(rhs); return -1; }
    for (int i = 0; i < n; i++) piv[i] = i;

    for (int col = 0; col < n; col++) {
        int best = col;
        double bestabs = fabs(L[(size_t)col * n + col]);
        for (int i = col + 1; i < n; i++) {
            double v = fabs(L[(size_t)i * n + col]);
            if (v > bestabs) { bestabs = v; best = i; }
        }
        if (bestabs < 1e-300) { free(L); free(rhs); free(piv); return -1; }
        if (best != col) {
            for (int j = 0; j < n; j++) {
                double t = L[(size_t)col * n + j];
                L[(size_t)col * n + j] = L[(size_t)best * n + j];
                L[(size_t)best * n + j] = t;
            }
            double t = rhs[col];
            rhs[col] = rhs[best];
            rhs[best] = t;
            int ti = piv[col];
            piv[col] = piv[best];
            piv[best] = ti;
        }
        for (int i = col + 1; i < n; i++) {
            double f = L[(size_t)i * n + col] / L[(size_t)col * n + col];
            for (int j = col; j < n; j++) L[(size_t)i * n + j] -= f * L[(size_t)col * n + j];
            rhs[i] -= f * rhs[col];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = rhs[i];
        for (int j = i + 1; j < n; j++) s -= L[(size_t)i * n + j] * x[j];
        x[i] = s / L[(size_t)i * n + i];
    }
    free(L);
    free(rhs);
    free(piv);
    return 0;
}

void nr_householder_qr(const double *A, int m, int n, double *Q, double *R) {
    /* A (m x n) = Q (m x m) R (m x n).  Q built up as product of reflectors. */
    for (int i = 0; i < m * m; i++) Q[i] = 0.0;
    for (int i = 0; i < m; i++) Q[(size_t)i * m + i] = 1.0;
    memcpy(R, A, sizeof(double) * (size_t)m * (size_t)n);
    for (int k = 0; k < n; k++) {
        double x0 = R[(size_t)k * n + k];
        double norm = 0.0;
        for (int i = k; i < m; i++) norm += R[(size_t)i * n + k] * R[(size_t)i * n + k];
        norm = sqrt(norm);
        if (norm < 1e-300) continue;
        double sign = x0 >= 0.0 ? 1.0 : -1.0;
        double *v = (double *)malloc(sizeof(double) * (size_t)(m - k));
        for (int i = k; i < m; i++) v[i - k] = R[(size_t)i * n + k];
        v[0] += sign * norm;
        double nv = 0.0;
        for (int i = k; i < m; i++) nv += v[i - k] * v[i - k];
        nv = sqrt(nv);
        if (nv < 1e-300) { free(v); continue; }
        for (int i = k; i < m; i++) v[i - k] /= nv;
        /* R <- (I - 2 v v^T) R on rows k..m-1 */
        for (int j = k; j < n; j++) {
            double s = 0.0;
            for (int i = k; i < m; i++) s += v[i - k] * R[(size_t)i * n + j];
            for (int i = k; i < m; i++) R[(size_t)i * n + j] -= 2.0 * v[i - k] * s;
        }
        /* Q <- Q (I - 2 v v^T) on cols k..m-1 */
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            for (int i = k; i < m; i++) s += v[i - k] * Q[(size_t)j * m + i];
            for (int i = k; i < m; i++) Q[(size_t)j * m + i] -= 2.0 * v[i - k] * s;
        }
        free(v);
    }
}

void nr_qr_orthonormalize(double *V, int n, int k) {
    double *Q = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *R = (double *)malloc(sizeof(double) * (size_t)n * (size_t)k);
    nr_householder_qr(V, n, k, Q, R);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) V[(size_t)i * k + j] = Q[(size_t)i * n + j];
    free(Q);
    free(R);
}

/* ---------------- RNG: xoshiro256** + Box-Muller ---------------- */

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void nr_rng_seed(nr_rng *r, uint64_t seed) {
    uint64_t sm = seed;
    for (int i = 0; i < 4; i++) {
        r->s[i] = splitmix64(&sm);
        if (r->s[i] == 0) r->s[i] = 0x1234567890ABCDEFULL;
    }
    r->have_spare_normal = 0;
    r->spare_normal = 0.0;
}

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t xoshiro_next(nr_rng *r) {
    const uint64_t result = rotl64(r->s[1] * 5, 7) * 9;
    const uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl64(r->s[3], 45);
    return result;
}

double nr_rng_double(nr_rng *r) {
    return (double)(xoshiro_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

double nr_rng_normal(nr_rng *r) {
    if (r->have_spare_normal) {
        r->have_spare_normal = 0;
        return r->spare_normal;
    }
    double u1 = nr_rng_double(r);
    while (u1 < 1e-300) u1 = nr_rng_double(r);
    double u2 = nr_rng_double(r);
    double rho = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    r->spare_normal = rho * sin(theta);
    r->have_spare_normal = 1;
    return rho * cos(theta);
}

double nr_norm(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrt(s);
}

double nr_dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

double nr_norm_diff(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
    return sqrt(s);
}
