"""Non-reversible anchored Langevin dynamics (NALD).

    a(x)       = exp(U(x) - U0(x))
    c(x)       = exp(U(x)) J(x) grad psi(x)
    b_alpha(x) = -a(x) grad U0(x) + alpha c(x)
    dX_t       = b_alpha(X_t) dt + sqrt(2 a(X_t)) dW_t

With the canonical psi = exp(-U0) this is

    dX_t = -a(X_t) (I + alpha J(X_t)) grad U0(X_t) dt + sqrt(2 a(X_t)) dW_t,

and grad U is never evaluated -- only the smooth anchor's gradient.

Shared by the notebooks in notebooks/; the definitions are lifted verbatim from
notebooks/nald_laplace_experiments.ipynb.
"""
import math
import numpy as np

class EllipticLaplace:
    r"""Target A.   U(x)  = sqrt(x' S^{-1} x)                (elliptical Laplace, cusp at 0)
                      U0(x) = sqrt(x' S^{-1} x + delta^2)     (C^infty hyperbolic anchor)"""
    key, label = "A", "Elliptical Laplace"

    def __init__(self, Sigma, delta):
        self.Sigma = np.asarray(Sigma, float)
        self.d     = self.Sigma.shape[0]
        self.delta = float(delta)
        self.Sinv  = np.linalg.inv(self.Sigma)
        self.L     = np.linalg.cholesky(self.Sigma)          # L L' = Sigma

    # --- q(x) = x' Sigma^{-1} x -------------------------------------------------
    def q(self, x):      return np.einsum('...i,ij,...j->...', x, self.Sinv, x)

    # --- the two potentials -----------------------------------------------------
    def U(self, x):      return np.sqrt(self.q(x))                       #  NOT differentiable at 0
    def U0(self, x):     return np.sqrt(self.q(x) + self.delta**2)       #  C^infty anchor

    # --- everything the sampler needs ------------------------------------------
    def gradU0(self, x): return (x @ self.Sinv) / np.sqrt(self.q(x) + self.delta**2)[..., None]

    def log_a(self, x):                                  # log a = U - U0, stable form
        r = np.sqrt(self.q(x))
        return -self.delta**2 / (r + np.sqrt(r * r + self.delta**2))

    # --- exact i.i.d. sampling and exact moments (ground truth) -----------------
    def sample(self, n, rng):
        r = rng.gamma(self.d, 1.0, size=n)
        u = rng.standard_normal((n, self.d)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        return (r[:, None] * u) @ self.L.T

    def cov_exact(self):  return (self.d + 1) * self.Sigma
    def EU_exact(self):   return float(self.d)
    def a_bounds(self):   return math.exp(-self.delta), 1.0


class L1Laplace:
    r"""Target B.   U(x)  = sum_i |x_i| / b_i                       (kinks on all coordinate planes)
                      U0(x) = sum_i sqrt(x_i^2 + delta^2) / b_i      (C^infty anchor)"""
    key, label = "B", r"$\ell^1$ Laplace"

    def __init__(self, b, delta):
        self.b     = np.asarray(b, float)
        self.d     = len(self.b)
        self.delta = float(delta)

    # --- the two potentials -----------------------------------------------------
    def U(self, x):      return np.sum(np.abs(x) / self.b, axis=-1)                     # non-smooth
    def U0(self, x):     return np.sum(np.sqrt(x * x + self.delta**2) / self.b, axis=-1)  # C^infty

    # --- everything the sampler needs ------------------------------------------
    def gradU0(self, x): return x / (self.b * np.sqrt(x * x + self.delta**2))

    def log_a(self, x):                                  # log a = U - U0, stable form
        ax = np.abs(x)
        return -np.sum(self.delta**2 / (self.b * (ax + np.sqrt(ax * ax + self.delta**2))), axis=-1)

    # --- exact i.i.d. sampling and exact moments (ground truth) -----------------
    def sample(self, n, rng): return rng.laplace(0.0, self.b, size=(n, self.d))
    def cov_exact(self):      return np.diag(2 * self.b**2)
    def EU_exact(self):       return float(self.d)
    def a_bounds(self):       return math.exp(-self.delta * np.sum(1 / self.b)), 1.0


def hat(u):
    """Cross-product (skew) matrix:  hat(u) v = u x v."""
    u = np.asarray(u, float)
    return np.array([[0.0, -u[2],  u[1]],
                     [u[2],  0.0, -u[0]],
                     [-u[1], u[0],  0.0]])

U_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
J_A    = hat(U_AXIS)                      # the constant skew-symmetric perturbation used throughout


def J_s_matrix(x, s):
    """The state-dependent perturbation, exactly as written in the text:  J_s(x) = s * hat(x)."""
    return np.array([[0.0,   -s*x[2],  s*x[1]],
                     [s*x[2],  0.0,   -s*x[0]],
                     [-s*x[1], s*x[0],  0.0  ]])


def state_scale(tgt, n=400_000, seed=5):
    """s = (E_pi ||X||^2)^{-1/2}, so that ||J_s(X)|| ~ 1 for a typical draw."""
    return float(1.0/np.sqrt((tgt.sample(n, np.random.default_rng(seed))**2).sum(1).mean()))



def _cross(a, b):
    out = np.empty_like(a)
    a0, a1, a2 = a[:,0], a[:,1], a[:,2]; b0, b1, b2 = b[:,0], b[:,1], b[:,2]
    out[:,0] = a1*b2 - a2*b1; out[:,1] = a2*b0 - a0*b2; out[:,2] = a0*b1 - a1*b0
    return out

def rodrigues(X, w, t):
    """Exact flow of  dX/dt = w x X  over time t (w frozen).  Norm preserving."""
    nrm  = np.sqrt(w[:,0]**2 + w[:,1]**2 + w[:,2]**2)
    th   = nrm * t
    k    = w / np.where(nrm > 1e-300, nrm, 1.0)[:, None]
    ct   = np.cos(th)[:, None]; st = np.sin(th)[:, None]
    return X*ct + _cross(k, X)*st + k*(k*X).sum(1)[:, None]*(1.0 - ct)


def nald(tgt, J="none", alpha=0.0, s=0.0, Ja=J_A, h=0.01, n_steps=100_000, n_chains=48,
         seed=0, burn=None, thin=10, x0=None, rotation="exact", block=2000,
         record=None, store=True):
    r"""Simulate  dX = -a(X)(I + alpha J(X)) grad U0(X) dt + sqrt(2 a(X)) dW   (canonical psi = e^{-U0}).

    J        : "none" | "const" (uses Ja) | "state" (uses s, J_s(x) = s*hat(x))
    rotation : for J="state" only -- "exact" (Rodrigues splitting) or "euler" (plain Euler-Maruyama)
    record   : callable  states (n_chains,d) -> (n_chains, m)  of test functions to store.
               Default records (x_1, ..., x_d, U(x)).
    Returns  (trace, X_final);  trace has shape (n_kept, n_chains, m).
    """
    rng = np.random.default_rng(seed)
    d   = tgt.d
    X   = tgt.sample(n_chains, rng) if x0 is None else np.array(x0, float, copy=True)
    n_chains = X.shape[0]
    if burn is None: burn = n_steps // 5
    if record is None:
        record = lambda z: np.concatenate([z, tgt.U(z)[:, None]], axis=1)
    sh = np.sqrt(2.0 * h)
    M  = np.ascontiguousarray(np.asarray(Ja, float).T)          # so that  g @ M == (Ja g)
    m  = record(X).shape[1]
    keep = ((n_steps - burn) + thin - 1)//thin if store else 0
    out  = np.empty((keep, n_chains, m)) if store else None
    kk, bi, nz = 0, block, None

    for n in range(n_steps):
        if bi == block:
            nz = rng.standard_normal((block, n_chains, d)); bi = 0
        a = np.exp(tgt.log_a(X))[:, None]
        g = tgt.gradU0(X)
        Y = X - h*(a*g) + (sh*np.sqrt(a))*nz[bi]; bi += 1
        if J == "const":
            Y -= (h*alpha) * (a * (g @ M))
        elif J == "state":
            w = (alpha*s) * (a*g)                                # dX/dt = w x X
            Y = rodrigues(Y, w, h) if rotation == "exact" else Y + h*_cross(w, Y)
        X = Y
        if store and n >= burn and (n - burn) % thin == 0:
            out[kk] = record(X); kk += 1
    return (out[:kk] if store else None), X


# ---------------------------------------------------------------------------
# Designing the constant perturbation J_a
# ---------------------------------------------------------------------------
def surrogate_gap(u, S, alpha):
    """Spectral gap  min Re spec( (I + alpha*hat(u)) S )  of the Gaussian surrogate.

    For a target whose covariance is Cov, take S = Cov^{-1}.  Because tr(hat(u) S) = 0 for
    skew hat(u) and symmetric S, the three eigenvalues always sum to tr(S), so

        min Re spec( (I + alpha J) S )  <=  tr(S)/3

    for EVERY skew J and every alpha -- a hard ceiling, attained when the perturbation
    equalises the three relaxation rates.  The unperturbed gap is min diag(S) in the
    principal frame, so the best achievable improvement is tr(S)/3 / lambda_min(S).
    """
    u = np.asarray(u, float)
    n = np.linalg.norm(u)
    if n < 1e-12:
        return float(np.min(np.linalg.eigvals(S).real))
    return float(np.min(np.linalg.eigvals((np.eye(3) + alpha*hat(u/n)) @ S).real))


def optimal_axis(Cov, n_grid=4000, seed=3, tol=1e-8):
    """Axis u* and the smallest alpha* at which hat(u*) attains the tr(S)/3 ceiling.

    Returns (alpha_star, u_star, ceiling, unperturbed_gap).  The surrogate fixes the
    DIRECTION; the strength alpha is then chosen by measurement, since the ceiling is
    attained on a whole family of (alpha, u) and the surrogate cannot separate them.
    """
    from scipy.optimize import minimize
    S = np.linalg.inv(np.asarray(Cov, float))
    ceiling, base = float(np.trace(S)/3), float(np.min(np.linalg.eigvals(S).real))
    G = np.random.default_rng(seed).standard_normal((n_grid, 3))
    G /= np.linalg.norm(G, axis=1, keepdims=True)

    def best(alpha):
        g = np.array([surrogate_gap(u, S, alpha) for u in G])
        r = minimize(lambda u: -surrogate_gap(u, S, alpha), G[int(np.argmax(g))],
                     method="Nelder-Mead", options=dict(xatol=1e-11, fatol=1e-14, maxiter=3000))
        u = r.x/np.linalg.norm(r.x)
        return -float(r.fun), (u if u[-1] >= 0 else -u)

    lo, hi = 0.25, 16.0
    for _ in range(26):
        mid = 0.5*(lo + hi)
        if best(mid)[0] >= ceiling*(1 - tol): hi = mid
        else:                                 lo = mid
    _, u_star = best(hi)
    return hi, u_star, ceiling, base
