#include "nr/linalg_eig.h"
#include "nr/common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double max_abs(const double *v, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m = fmax(m, fabs(v[i]));
    return m;
}

void nr_householder_tridiagonalize(const double *A, int n, double *T, double *Q) {
    double *Am = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    if (!Am) abort();
    memcpy(Am, A, sizeof(double) * (size_t)n * (size_t)n);
    memset(Q, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) Q[(size_t)i * n + i] = 1.0;

    for (int m = 0; m < n - 2; m++) {
        int L = n - m - 1;
        double *v = (double *)malloc(sizeof(double) * (size_t)L);
        if (!v) abort();
        double norm = 0.0;
        for (int i = 0; i < L; i++) norm += Am[(size_t)(m + 1 + i) * n + m] * Am[(size_t)(m + 1 + i) * n + m];
        norm = sqrt(norm);
        if (norm < 1e-14) { free(v); continue; }
        double x0 = Am[(size_t)(m + 1) * n + m];
        double sgn = (x0 != 0.0) ? (x0 > 0.0 ? 1.0 : -1.0) : 1.0;
        for (int i = 0; i < L; i++) v[i] = Am[(size_t)(m + 1 + i) * n + m];
        v[0] += sgn * norm;
        double nv = sqrt(0.0);
        for (int i = 0; i < L; i++) nv += v[i] * v[i];
        nv = sqrt(nv);
        if (nv < 1e-14) { free(v); continue; }
        for (int i = 0; i < L; i++) v[i] /= nv;

        /* A <- P A (rows m+1..n-1), for columns j = m..n-1 */
        for (int j = m; j < n; j++) {
            double s = 0.0;
            for (int i = 0; i < L; i++) s += v[i] * Am[(size_t)(m + 1 + i) * n + j];
            for (int i = 0; i < L; i++) Am[(size_t)(m + 1 + i) * n + j] -= 2.0 * v[i] * s;
        }
        /* A <- A P (cols m+1..n-1), for rows i = m..n-1 */
        for (int i = m; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < L; j++) s += v[j] * Am[(size_t)i * n + m + 1 + j];
            for (int j = 0; j < L; j++) Am[(size_t)i * n + m + 1 + j] -= 2.0 * s * v[j];
        }
        /* Q <- Q P (cols m+1..n-1) */
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k2 = 0; k2 < L; k2++) s += v[k2] * Q[(size_t)j * n + m + 1 + k2];
            for (int i = 0; i < L; i++) Q[(size_t)j * n + m + 1 + i] -= 2.0 * v[i] * s;
        }
        free(v);
    }
    memset(T, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) T[(size_t)i * n + i] = 0.5 * (Am[(size_t)i * n + i] + Am[(size_t)i * n + i]);
    for (int i = 0; i + 1 < n; i++) {
        double off = 0.5 * (Am[(size_t)i * n + i + 1] + Am[(size_t)(i + 1) * n + i]);
        T[(size_t)i * n + i + 1] = off;
        T[(size_t)(i + 1) * n + i] = off;
    }
    free(Am);
}

int nr_sturm_count(const double *d, const double *e, int n, double x) {
    double dprev = d[0] - x;
    int count = dprev < 0.0 ? 1 : 0;
    for (int i = 1; i < n; i++) {
        double dcur = d[i] - x - (e[i - 1] * e[i - 1]) / dprev;
        count += dcur < 0.0 ? 1 : 0;
        dprev = dcur;
    }
    return count;
}

void nr_tridiagonal_eigenvalues(const double *d, const double *e, int n,
                                double tol, int max_iter, double *w) {
    if (n == 1) { w[0] = d[0]; return; }
    double emax = 0.0;
    for (int i = 0; i < n - 1; i++) emax = fmax(emax, fabs(e[i]));
    double lo0 = 1e300, hi0 = -1e300;
    for (int i = 0; i < n; i++) { lo0 = fmin(lo0, d[i]); hi0 = fmax(hi0, d[i]); }
    lo0 -= 2.0 * emax;
    hi0 += 2.0 * emax;
    for (int k = 0; k < n; k++) {
        double lo = lo0, hi = hi0;
        for (int it = 0; it < max_iter; it++) {
            double mid = 0.5 * (lo + hi);
            if (nr_sturm_count(d, e, n, mid) <= k) lo = mid;
            else hi = mid;
            if ((hi - lo) < tol * fmax(1.0, fabs(mid))) break;
        }
        w[k] = 0.5 * (lo + hi);
    }
}

void nr_tridiag_solve(const double *d, const double *e, int n, const double *rhs, double *x) {
    if (n == 1) { x[0] = rhs[0] / d[0]; return; }
    double *c = (double *)malloc(sizeof(double) * (size_t)(n - 1));
    double *dd = (double *)malloc(sizeof(double) * (size_t)n);
    if (!c || !dd) abort();
    c[0] = e[0] / d[0];
    dd[0] = rhs[0] / d[0];
    for (int i = 1; i < n; i++) {
        double denom = d[i] - e[i - 1] * c[i - 1];
        if (i < n - 1) c[i] = e[i] / denom;
        dd[i] = (rhs[i] - e[i - 1] * dd[i - 1]) / denom;
    }
    x[n - 1] = dd[n - 1];
    for (int i = n - 2; i >= 0; i--) x[i] = dd[i] - c[i] * x[i + 1];
    free(c);
    free(dd);
}

void nr_inverse_iteration(const double *d, const double *e, int n, double w,
                          int n_iter, uint64_t seed, double *v) {
    nr_rng rng;
    nr_rng_seed(&rng, seed);
    double scale = max_abs(d, n) + 1.0;
    double sigma = w + 1e-7 * scale;
    double *ds = (double *)malloc(sizeof(double) * (size_t)n);
    double *x = (double *)malloc(sizeof(double) * (size_t)n);
    if (!ds || !x) abort();
    for (int i = 0; i < n; i++) ds[i] = d[i] - sigma;
    for (int i = 0; i < n; i++) v[i] = nr_rng_normal(&rng);
    for (int it = 0; it < n_iter; it++) {
        nr_tridiag_solve(ds, e, n, v, x);
        double norm = nr_norm(x, n);
        if (norm == 0.0) break;
        for (int i = 0; i < n; i++) v[i] = x[i] / norm;
    }
    int imax = 0;
    for (int i = 1; i < n; i++) if (fabs(v[i]) > fabs(v[imax])) imax = i;
    if (v[imax] < 0.0) for (int i = 0; i < n; i++) v[i] = -v[i];
    free(ds);
    free(x);
}

static void sort_ascending(double *w, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (w[j] < w[i]) { double t = w[i]; w[i] = w[j]; w[j] = t; }
}

void nr_eigh_symmetric(const double *A, int n, double *w, double *V) {
    double *T = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *Q = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *d = (double *)calloc((size_t)n, sizeof(double));
    double *e = (double *)calloc((size_t)(n > 1 ? n - 1 : 1), sizeof(double));
    double *Vt = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    if (!T || !Q || !d || !e || !Vt) abort();
    nr_householder_tridiagonalize(A, n, T, Q);
    for (int i = 0; i < n; i++) d[i] = T[(size_t)i * n + i];
    for (int i = 0; i + 1 < n; i++) e[i] = T[(size_t)i * n + i + 1];
    nr_tridiagonal_eigenvalues(d, e, n, 1e-13, 300, w);
    sort_ascending(w, n);
    for (int idx = 0; idx < n; idx++)
        nr_inverse_iteration(d, e, n, w[idx], 12, 1000 + (uint64_t)idx, &Vt[(size_t)idx * n]);
    /* V = Q Vt (Vt stores the vectors as rows: Vt[idx * n + i] = component i of vector idx) */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += Q[(size_t)i * n + k] * Vt[(size_t)j * n + k];
            V[(size_t)i * n + j] = s;
        }
    nr_qr_orthonormalize(V, n, n);
    for (int j = 0; j < n; j++) {
        int imax = 0;
        for (int i = 1; i < n; i++) if (fabs(V[(size_t)i * n + j]) > fabs(V[(size_t)imax * n + j])) imax = i;
        if (V[(size_t)imax * n + j] < 0.0)
            for (int i = 0; i < n; i++) V[(size_t)i * n + j] = -V[(size_t)i * n + j];
    }
    free(T); free(Q); free(d); free(e); free(Vt);
}

void nr_lanczos(nr_matvec_fn M, void *ud, int n, int k, int n_iter,
                const double *v0, uint64_t seed, double *w_ritz, double *V_ritz) {
    int m = n_iter > 0 ? n_iter : (2 * k > 20 ? 2 * k : 20);
    if (m > n) m = n;
    size_t qcols = (size_t)m + 1;
    double *Q = (double *)malloc(sizeof(double) * (size_t)n * qcols);
    double *alpha = (double *)malloc(sizeof(double) * (size_t)m);
    double *beta = (double *)malloc(sizeof(double) * (size_t)m);
    double *w = (double *)malloc(sizeof(double) * (size_t)n);
    double *u = (double *)malloc(sizeof(double) * (size_t)n);
    double *tv = (double *)malloc(sizeof(double) * (size_t)n);
    if (!Q || !alpha || !beta || !w || !u || !tv) abort();
    memset(Q, 0, sizeof(double) * (size_t)n * qcols);

    nr_rng rng;
    nr_rng_seed(&rng, seed);
    if (v0) {
        for (int i = 0; i < n; i++) Q[i] = v0[i];
    } else {
        for (int i = 0; i < n; i++) Q[i] = nr_rng_normal(&rng);
    }
    double nrm = nr_norm(Q, n);
    for (int i = 0; i < n; i++) Q[i] /= nrm;

    int m_used = m;
    for (int j = 0; j < m; j++) {
        M(&Q[(size_t)j * n], u, ud);
        double a = nr_dot(&Q[(size_t)j * n], u, n);
        alpha[j] = a;
        for (int i = 0; i < n; i++) {
            double s = u[i] - a * Q[(size_t)j * n + i];
            if (j > 0) s -= beta[j - 1] * Q[(size_t)(j - 1) * n + i];
            w[i] = s;
        }
        /* Full reorthogonalisation against Q[:, :j+1]. */
        for (int p = 0; p <= j; p++) {
            double s = nr_dot(&Q[(size_t)p * n], w, n);
            for (int i = 0; i < n; i++) w[i] -= s * Q[(size_t)p * n + i];
        }
        double b = nr_norm(w, n);
        beta[j] = b;
        if (b < 1e-14) { m_used = j + 1; break; }
        for (int i = 0; i < n; i++) Q[(size_t)(j + 1) * n + i] = w[i] / b;
    }

    int em = m_used - 1;
    double *d_small = (double *)malloc(sizeof(double) * (size_t)m_used);
    double *e_small = (double *)malloc(sizeof(double) * (size_t)(em > 0 ? em : 1));
    if (!d_small || !e_small) abort();
    for (int i = 0; i < m_used; i++) d_small[i] = alpha[i];
    for (int i = 0; i < em; i++) e_small[i] = beta[i];
    double *w_small = (double *)malloc(sizeof(double) * (size_t)m_used);
    if (!w_small) abort();
    nr_tridiagonal_eigenvalues(d_small, e_small, m_used, 1e-13, 300, w_small);

    /* Pick the k largest-magnitude Ritz values, output them ascending. */
    int *order = (int *)malloc(sizeof(int) * (size_t)m_used);
    if (!order) abort();
    for (int i = 0; i < m_used; i++) order[i] = i;
    for (int i = 0; i < m_used; i++)
        for (int j2 = i + 1; j2 < m_used; j2++)
            if (fabs(w_small[order[j2]]) > fabs(w_small[order[i]])) {
                int t = order[i]; order[i] = order[j2]; order[j2] = t;
            }
    int kuse = k < m_used ? k : m_used;
    double *sel = (double *)malloc(sizeof(double) * (size_t)kuse);
    if (!sel) abort();
    for (int i = 0; i < kuse; i++) sel[i] = w_small[order[i]];
    sort_ascending(sel, kuse);

    for (int r = 0; r < kuse; r++) {
        nr_inverse_iteration(d_small, e_small, m_used, sel[r], 12, 2000 + (uint64_t)r, tv);
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j2 = 0; j2 < m_used; j2++) s += Q[(size_t)j2 * n + i] * tv[j2];
            V_ritz[(size_t)i * k + (size_t)r] = s;
        }
    }
    for (int i = 0; i < kuse; i++) w_ritz[i] = sel[i];
    free(Q); free(alpha); free(beta); free(w); free(u); free(tv);
    free(d_small); free(e_small); free(w_small); free(order); free(sel);
}

void nr_particle_in_box(int n, double L, double *H, double *exact) {
    double dx = L / (n + 1);
    double h = 2.0 / (dx * dx), t = -1.0 / (dx * dx);
    memset(H, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) H[(size_t)i * n + i] = h;
    for (int i = 0; i + 1 < n; i++) {
        H[(size_t)i * n + i + 1] = t;
        H[(size_t)(i + 1) * n + i] = t;
    }
    for (int k2 = 1; k2 <= n; k2++) exact[k2 - 1] = (k2 * M_PI / L) * (k2 * M_PI / L);
}

void nr_phonon_chain(int n, double k, double m, double *D, double *exact) {
    double km = k / m;
    memset(D, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) D[(size_t)i * n + i] = 2.0 * km;
    for (int i = 0; i + 1 < n; i++) {
        D[(size_t)i * n + i + 1] = -km;
        D[(size_t)(i + 1) * n + i] = -km;
    }
    D[0] = km;
    D[(size_t)(n - 1) * n + (n - 1)] = km;
    for (int j = 0; j < n; j++) {
        double s = sin(j * M_PI / (2.0 * n));
        exact[j] = 4.0 * km * s * s;
    }
}

void nr_truss_stiffness(int n, double *S) {
    memset(S, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) S[(size_t)i * n + i] = 2.0;
    for (int i = 0; i + 1 < n; i++) {
        S[(size_t)i * n + i + 1] = -1.0;
        S[(size_t)(i + 1) * n + i] = -1.0;
    }
}
