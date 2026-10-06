"""Eigenvalue routines for symmetric matrices, written from scratch.

Most "eigenvalue problems" in physics are *symmetric* (or Hermitian): a
discretised kinetic-energy operator, a molecular Hessian, a tridiagonal
Hamiltonian from a finite-difference or basis expansion.  Symmetric matrices
have real eigenvalues and orthonormal eigenvectors, and the standard robust
pipeline is:

  1. **Householder tridiagonalisation** -- reduce ``A`` to a tridiagonal
     ``T`` with an orthogonal ``Q`` so that ``A = Q T Q^T``.  Only tridiagonal
     work is needed afterwards, which is why real eigensolvers do this.
  2. **Sturm bisection** -- use the sign sequence of the LDL^T factorisation
     of ``T - x I`` (the Sturm sequence) to count how many eigenvalues lie
     below ``x``; bracketing each eigenvalue this way and bisecting is
     robust and needs no pivoting.
  3. **Inverse iteration** -- solve ``(T - w I) v = b`` repeatedly to
     recover the eigenvector for an eigenvalue ``w``.

For *large sparse* problems (the realistic case in HPC, and the routine that
scales to million-degree-of-freedom molecular dynamics and lattice QCD) the
workhorse is **Lanczos**, which builds an orthonormal Krylov basis and
reduces the problem to a small tridiagonal matrix whose Ritz values
approximate the extremes of the full spectrum.


References (see REFERENCES.md):
  - A. S. Householder, Numer. Math. 9, 409 (1958) [tridiagonalisation].
  - C. Sturm, Ann. Math. 1, 244 (1829) [Sturm sequence].
  - E. T. Lanczos, J. Math. Phys. 2, 205 (1951) [Lanczos].
  - G. H. Golub, C. F. Van Loan, Matrix Computations, 4th ed. (2013)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "householder_tridiagonalize",
    "sturm_count",
    "tridiagonal_eigenvalues",
    "tridiag_solve",
    "inverse_iteration",
    "eigh_symmetric",
    "lanczos",
    "particle_in_box",
    "phonon_chain",
    "truss_stiffness",
]


# --------------------------------------------------------------------------- #
# 1. Householder tridiagonalisation of a real symmetric matrix
# --------------------------------------------------------------------------- #
def householder_tridiagonalize(A):
    """Reduce a real symmetric matrix to tridiagonal form.

    Returns ``(T, Q)`` with ``A = Q @ T @ Q.T`` and ``T`` tridiagonal.

    For each column ``m`` we zero ``A[m+2:, m]`` with a Householder
    reflector acting on the ``(n-m-1)``-dimensional subspace (rows/cols
    ``m+1..n-1``); the subdiagonal ``A[m+1, m]`` is preserved (replaced by
    ``-sign(x0)||x||``).  The reflectors are accumulated into ``Q`` so that
    ``A = Q T Q^T``.
    """
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    Q = np.eye(n)
    for m in range(n - 2):
        x = A[m + 1 :, m].copy()          # length n-m-1 (rows m+1..n-1)
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-14:
            continue
        # v such that (I - 2 v v^T / v^T v) x = -sign(x0) ||x|| e1, the
        # stable choice (avoids cancellation when x0 is large).
        v = x.copy()
        v[0] += np.sign(x[0] if x[0] != 0 else 1.0) * norm_x
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            continue
        v /= nv
        H = np.eye(x.size) - 2.0 * np.outer(v, v)   # (n-m-1) x (n-m-1)
        P = np.eye(n)
        P[m + 1 :, m + 1 :] = H                      # act on rows/cols m+1..n-1
        A = P @ A @ P
        Q = Q @ P
    T = np.zeros((n, n))
    T[np.diag_indices(n)] = np.diag(A)
    if n > 1:
        i = np.arange(n - 1)
        off = np.diag(A, 1)
        T[i, i + 1] = off
        T[i + 1, i] = off
    return 0.5 * (T + T.T), Q


# --------------------------------------------------------------------------- #
# 2. Sturm sequence + bisection for tridiagonal eigenvalues
# --------------------------------------------------------------------------- #
def sturm_count(d, e, x):
    """Number of eigenvalues of the tridiagonal (diag d, off-diag e) below x.

    Counts the negative pivots of the LDL^T factorisation of ``T - x I``.
    A zero pivot (``x`` exactly an eigenvalue of a leading submatrix) is
    handled by IEEE signed-zero semantics: the reciprocal diverges to
    ``+/-inf`` and the pivot count stays well-defined, so the count is
    consistent with "eigenvalues strictly below x".
    """
    n = d.size
    dprev = d[0] - x
    count = 1 if dprev < 0 else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(1, n):
            dcur = d[i] - x - (e[i - 1] ** 2) / dprev
            count += 1 if dcur < 0 else 0
            dprev = dcur
    return count


def tridiagonal_eigenvalues(d, e, tol=1e-13, max_iter=300):
    """All eigenvalues (ascending) of a tridiagonal via Sturm bisection."""
    n = d.size
    if n == 1:
        return d.copy()
    emax = np.max(np.abs(e)) if e.size else 0.0
    # A tridiagonal row has at most two off-diagonal neighbours, so the
    # spectrum is contained in [min(d) - 2*emax, max(d) + 2*emax]
    # (Gershgorin).  A tighter bound would miss eigenvalues that sit
    # outside max(d)+emax when a row couples to two large off-diagonals.
    lo0 = float(np.min(d)) - 2.0 * emax
    hi0 = float(np.max(d)) + 2.0 * emax
    w = np.zeros(n)
    for k in range(n):
        lo, hi = lo0, hi0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if sturm_count(d, e, mid) <= k:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol * max(1.0, abs(mid)):
                break
        w[k] = 0.5 * (lo + hi)
    return w


# --------------------------------------------------------------------------- #
# 3. Tridiagonal linear solve (Thomas) + inverse iteration
# --------------------------------------------------------------------------- #
def tridiag_solve(d, e, rhs):
    """Solve a tridiagonal system (diag d, off-diag e) by the Thomas algorithm."""
    n = d.size
    c = np.zeros(n - 1)
    dd = np.zeros(n)
    c[0] = e[0] / d[0]
    dd[0] = rhs[0] / d[0]
    for i in range(1, n):
        denom = d[i] - e[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = e[i] / denom
        dd[i] = (rhs[i] - e[i - 1] * dd[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - c[i] * x[i + 1]
    return x


def inverse_iteration(d, e, w, n_iter=12, seed=0):
    """Eigenvector for eigenvalue w of a tridiagonal, by inverse iteration."""
    rng = np.random.default_rng(seed)
    n = d.size
    scale = float(np.max(np.abs(d))) + 1.0
    sigma = w + 1e-7 * scale  # small shift avoids a singular system
    d_shifted = d - sigma
    b = rng.normal(size=n)
    x = b
    for _ in range(n_iter):
        x = tridiag_solve(d_shifted, e, x)
        norm = np.linalg.norm(x)
        if norm == 0.0:
            break
        x = x / norm
    # Deterministic sign: largest-magnitude component positive.
    if x[np.argmax(np.abs(x))] < 0:
        x = -x
    return x


# --------------------------------------------------------------------------- #
# Full symmetric eigensolver (pipeline)
# --------------------------------------------------------------------------- #
def eigh_symmetric(A):
    """Eigenvalues (ascending) + orthonormal eigenvectors of a real symmetric
    matrix, via Householder tridiagonalisation + Sturm bisection + inverse
    iteration.

    Returns ``(w, V)`` with ``A @ V[:, k] = w[k] * V[:, k]``.
    """
    A = np.asarray(A, dtype=float)
    T, Q = householder_tridiagonalize(A)
    n = T.shape[0]
    d = np.diag(T).copy()
    e = np.diag(T, 1).copy()

    w = tridiagonal_eigenvalues(d, e)
    order = np.argsort(w)
    w = w[order]

    Vt = np.zeros((n, n))
    for idx, wk in enumerate(w):
        Vt[:, idx] = inverse_iteration(d, e, wk, seed=1000 + idx)
    V = Q @ Vt

    # Re-orthonormalise (guards against near-degenerate eigenvector mixing).
    V, _ = np.linalg.qr(V)
    # Align each column's sign deterministically.
    for j in range(n):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    return w, V


# --------------------------------------------------------------------------- #
# Lanczos for large sparse symmetric matrices
# --------------------------------------------------------------------------- #
def lanczos(M, k=6, n_iter=None, v0=None, seed=0):
    """k extreme eigenpairs of a large symmetric operator via Lanczos.

    ``M`` is a callable ``M(x) -> Mx`` or a 2D array.  Returns the ``k``
    largest-magnitude Ritz values and their eigenvectors.  Full
    reorthogonalisation is used for stability.
    """
    rng = np.random.default_rng(seed)
    if hasattr(M, "shape"):
        n = M.shape[0]

        def apply(x):
            return M @ x
    else:
        probe = np.asarray(M(np.array([1.0] * 1)))
        n = probe.size
        apply = lambda x: np.asarray(M(x))  # noqa: E731

    m = n_iter if n_iter is not None else max(2 * k, 20)
    m = int(min(m, n))

    q = np.zeros((n, m + 1))
    q[:, 0] = v0 if v0 is not None else rng.normal(size=n)
    q[:, 0] /= np.linalg.norm(q[:, 0])

    alpha = np.zeros(m)
    beta = np.zeros(m)

    for j in range(m):
        w = apply(q[:, j])
        alpha[j] = float(q[:, j] @ w)
        w = w - alpha[j] * q[:, j]
        if j > 0:
            w = w - beta[j - 1] * q[:, j - 1]
        # Full reorthogonalisation.
        w = w - q[:, : j + 1] @ (q[:, : j + 1].T @ w)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-14:
            m = j + 1
            break
        q[:, j + 1] = w / beta[j]

    T = np.diag(alpha[:m])
    if m > 1:
        T = T + np.diag(beta[: m - 1], 1) + np.diag(beta[: m - 1], -1)
    w_small, v_small = np.linalg.eigh(T)
    idx = np.argsort(np.abs(w_small))[::-1][:k]
    idx = idx[np.argsort(w_small[idx])]
    ritz_vecs = q[:, :m] @ v_small[:, idx]
    return w_small[idx], ritz_vecs


# --------------------------------------------------------------------------- #
# Physics Hamiltonians: "the same tridiagonal, different boundary conditions"
# --------------------------------------------------------------------------- #
def particle_in_box(n=64, L=1.0):
    """Discretised -d^2/dx^2 on [0, L] with Dirichlet BCs (infinite well).

    Exact spectrum: E_k = (k pi / L)^2 for k = 1, 2, ...
    Returns ``(H, exact)``.
    """
    dx = L / (n + 1)
    h = 2.0 / dx**2
    t = -1.0 / dx**2
    H = h * np.eye(n) + t * (np.eye(n, k=1) + np.eye(n, k=-1))
    k = np.arange(1, n + 1)
    exact = (k * np.pi / L) ** 2
    return H, exact


def phonon_chain(n=32, k=1.0, m=1.0):
    """1D monatomic chain with *free* ends: dynamical matrix K M^-1.

    Normal-mode frequencies omega_j^2 = (4 k / m) sin^2(j pi / (2 n)),
    j = 0 .. n-1 (j=0 is the zero-energy translation).  Returns ``(D, exact)``.
    """
    D = (k / m) * (2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    D[0, 0] = k / m
    D[-1, -1] = k / m
    j = np.arange(n)
    exact = (4.0 * k / m) * np.sin(j * np.pi / (2 * n)) ** 2
    return D, exact


def truss_stiffness(n=8):
    """1D truss / finite-element stiffness matrix (Dirichlet ends).

    Eigenvalues are the squared natural frequencies.
    """
    k = 1.0
    S = k * (2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    return S
