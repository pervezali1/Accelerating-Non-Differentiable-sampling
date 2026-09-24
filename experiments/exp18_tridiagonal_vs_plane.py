import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---- parameters ---------------------------------------------------------
D, NU, KAPPA = 3, 6.0, 100.0        # Student-t core
LAM, A_MCP, EPS = 0.25, 2.0, 0.1    # MCP: lambda, a, and the smoothing eps
A = 1.75                            # field strength: J_a has spectral norm A.
# Both LAM and A are set together: the skew field trades on anisotropy, and the
# smoothed penalty destroys anisotropy by adding curvature LAM/EPS isotropically.
# Effective condition number, and the best this matrix can then do at equal bias:
#
#     LAM        0      0.25    0.5    0.75     1
#     cond     100.0    37.4   21.4    15.1   11.8
#     best      4.39x   2.03x  1.46x  1.25x  1.16x   at a = 2.95, 1.75, 1.25, 1.03, 0.85
#
# Measured on this notebook, five chains x 20000 steps, normal10 / uniform5:
#     LAM = 1      0.89x / 1.00x  -- a 1.16x ceiling is inside the resolution
#     LAM = 0.5    1.25x / 1.56x
#     LAM = 0.25   1.74x / 1.74x
# EPS is the stronger knob of the two, since it enters only through LAM/EPS.
N, REPS, STEPS = 5000, 5, 20000     # particles, replications, iterations
RAMP = 20.0                         # warm-up of the skew field, in fast relaxations.
# 20, not the 10 that suffices on the smooth target: measured on this target the
# uniform prior rises 1.399x out of a trough at ramp 10 and 1.000x at ramp 20,
# at the same 991 iterations either way -- so the longer hold is free. The
# Gaussian prior is flat at both (794 iterations). A ramp calibrated on one
# prior and one target does not automatically transfer to another.
PRIORS = ("normal10", "uniform5")   # X_0 ~ N(0, 10 I)  and  X_0 ~ Uniform(-5, 5)^d

# Stepsizes: the equal-bias values of the SMOOTH core, from the exact
# second-moment recursion at 2% stationary covariance bias.  The composite
# potential is not log-quadratic, so its own recursion is not exact and the
# equal-bias property is approximate -- this is the protocol of exp4.
ETA = {"zero": 1.098755e-03, "const": 5.855338e-04}

class StudentT:
    """Student-t pi ~ q^{-iota} with the log-quadratic anchor U0 = beta log q.

        q(x)  = 1 + (1/nu) x' Sigma^-1 x
        U     = iota log q,   iota = (d + nu) / 2
        U0    = beta log q,   beta = iota - 1
        e^{U-U0} = q,  grad U0 = (2 beta/nu) Sigma^-1 x / q
        drift b = -e^{U-U0} grad U0 = -(2 beta/nu) Sigma^-1 x     (linear: q cancels)
        sigma(x) = sqrt(q)
    """

    def __init__(self, d, nu, kappa):
        self.d, self.nu = d, float(nu)
        # log-spaced eigenvalues spanning the condition number, largest = 1
        self.Sigma = np.diag(np.logspace(-np.log10(kappa), 0.0, d))
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        ev, V = np.linalg.eigh(self.Sigma)
        self.Sigma_evals, self.Sigma_evecs = ev, V
        self.Sigma_half = (V * np.sqrt(ev)) @ V.T
        self.iota = 0.5 * (d + self.nu)
        self.beta = self.iota - 1.0
        self.coef = 2.0 * self.beta / self.nu

    def q(self, x):
        x = np.atleast_2d(x)
        return 1.0 + np.einsum("ni,ij,nj->n", x, self.Sigma_inv, x) / self.nu

    def grad_U0(self, x):
        x = np.atleast_2d(x)
        return self.coef * (x @ self.Sigma_inv) / self.q(x)[:, None]

    def anchor_scale(self, x):
        return self.q(x)

    def sigma(self, x):
        return np.sqrt(self.q(x))

    def sample(self, n, rng):
        z = rng.standard_normal((n, self.d))
        g = rng.chisquare(self.nu, size=n)
        return (z @ self.Sigma_half) * np.sqrt(self.nu / g)[:, None]

    def projected_scale(self, theta):
        return float(np.sqrt(theta @ self.Sigma @ theta))

    def projected_ppf(self, u, theta):
        return stats.t.ppf(u, df=self.nu, scale=self.projected_scale(theta))

    def projected_isf(self, w, theta):
        return stats.t.isf(w, df=self.nu, scale=self.projected_scale(theta))


def mcp(t, lam, a):
    """MCP of Zhang (2010): non-differentiable at 0, and bounded."""
    t = np.abs(np.asarray(t, float))
    return np.where(t <= a * lam, lam * t - t * t / (2.0 * a), 0.5 * a * lam * lam)


def mcp_smoothed(t, lam, a, eps):
    """The paper's Eq. 67 smoothing of the MCP."""
    t = np.asarray(t, float)
    r = np.sqrt(a * a * lam * lam + eps * eps)
    inner = lam * np.sqrt(t * t + eps * eps) - lam * t * t / (2.0 * r)
    outer = lam * (a * a * lam * lam + 2.0 * eps * eps) / (2.0 * r)
    return np.where(np.abs(t) <= a * lam, inner, outer)


def mcp_smoothed_grad(t, lam, a, eps):
    t = np.asarray(t, float)
    r = np.sqrt(a * a * lam * lam + eps * eps)
    inner = lam * t / np.sqrt(t * t + eps * eps) - lam * t / r
    return np.where(np.abs(t) <= a * lam, inner, 0.0)


class StudentTMCP(StudentT):
    """U  = iota log q + sum_i p_lam(x_i)        non-differentiable at x_i = 0
       U0 = beta log q + sum_i p^eps_lam(x_i)    C^1, the anchor"""

    def __init__(self, d, nu, kappa, lam=1.0, a=2.0, eps=0.1):
        super().__init__(d, nu, kappa)
        self.lam, self.a, self.eps = float(lam), float(a), float(eps)

    def penalty(self, x):
        return np.sum(mcp(x, self.lam, self.a), axis=-1)

    def penalty_smoothed(self, x):
        return np.sum(mcp_smoothed(x, self.lam, self.a, self.eps), axis=-1)

    def grad_U0(self, x):
        x = np.atleast_2d(x)
        return super().grad_U0(x) + mcp_smoothed_grad(x, self.lam, self.a, self.eps)

    def anchor_scale(self, x):
        x = np.atleast_2d(x)
        return self.q(x) * np.exp(self.penalty(x) - self.penalty_smoothed(x))

    def sigma(self, x):
        return np.sqrt(self.anchor_scale(x))

    def sample(self, n, rng, max_rounds=200):
        """Exact i.i.d. draws by rejection from the Student-t core.
        exp(-penalty) <= 1 bounds the ratio, so the acceptance is exact."""
        out, got = [], 0
        for _ in range(max_rounds):
            m = max(int(1.6 * (n - got)), 1024)
            prop = super().sample(m, rng)
            acc = prop[rng.random(m) < np.exp(-self.penalty(prop))]
            out.append(acc); got += acc.shape[0]
            if got >= n:
                break
        else:
            raise RuntimeError("rejection sampler did not reach n draws")
        return np.concatenate(out, axis=0)[:n]


def stiff_soft(Sigma, a):
    """Rotation generator in the (stiffest, softest) eigenplane of Sigma."""
    ev, V = np.linalg.eigh(np.asarray(Sigma, float))
    u, v = V[:, int(np.argmin(ev))], V[:, int(np.argmax(ev))]
    return a * (np.outer(u, v) - np.outer(v, u))


def make_prior(kind, d, n, rng):
    """The two starting ensembles.

        normal10   X_0 ~ N(0, 10 I_d)          covariance 10 I
        uniform5   X_0 ~ Uniform(-5, 5)^d      covariance (100/12) I = 8.33 I
    """
    if kind == "normal10":
        return rng.standard_normal((n, d)) * np.sqrt(10.0)
    if kind == "uniform5":
        return rng.uniform(-5.0, 5.0, size=(n, d))
    raise ValueError(f"unknown prior {kind!r}")


def warmup_schedule(target, eta, n_relax=10.0):
    """Smoothstep over n_relax relaxations of the fastest direction."""
    k_relax = 1.0 / (eta * target.coef / target.Sigma_evals.min())
    K = max(1.0, n_relax * k_relax)

    def sched(k):
        u = min(1.0, k / K)
        return u * u * (3.0 - 2.0 * u)
    return sched


def anchored_step(target, eta, J=None, schedule=None):
    """Euler-Maruyama for dX = [-I + J] e^{U-U0} grad U0 dt + sqrt(2) sigma dW."""
    sq = np.sqrt(2.0 * eta)
    counter = {"k": 0}

    def step(x, rng):
        s_k = 1.0 if schedule is None else float(schedule(counter["k"]))
        counter["k"] += 1
        g = target.grad_U0(x)
        rot = g if J is None else g @ J.T
        drift = target.anchor_scale(x)[:, None] * ((0.0 if J is None else s_k) * rot - g)
        return x + eta * drift + sq * (target.sigma(x)[:, None] * rng.standard_normal(x.shape))
    return step


def w2_sq_1d_two_sample(x, y):
    """W2^2 between two equal-size empirical samples on the line."""
    x = np.sort(np.asarray(x, float).ravel())
    y = np.sort(np.asarray(y, float).ravel())
    if x.size != y.size:
        n = min(x.size, y.size)
        u = (np.arange(n) + 0.5) / n
        x, y = np.quantile(x, u), np.quantile(y, u)
    return float(np.mean((x - y) ** 2))


def sliced_w2_two_sample(x, y, d):
    """The MCP target has no closed-form projected quantiles, so the reference
    side is an exact sample rather than an exact quantile function."""
    total = sum(w2_sq_1d_two_sample(x @ th, y @ th) for th in np.eye(d))
    return float(np.sqrt(total / d))


def log_schedule(n_steps, n_points=90):
    pts = np.unique(np.round(np.geomspace(1, n_steps, n_points)).astype(int))
    return sorted(set([0] + pts.tolist() + [n_steps]))


target = StudentTMCP(D, NU, KAPPA, lam=LAM, a=A_MCP, eps=EPS)
J_a = stiff_soft(target.Sigma, A)                   # spectral norm exactly A
record_at = log_schedule(STEPS, 90)

rng = np.random.default_rng(0)
prop = StudentT(D, NU, KAPPA).sample(200_000, rng)
print(f"Sigma eigenvalues        : {np.sort(target.Sigma_evals)}")
print(f"iota = {target.iota}, beta = {target.beta}")
print(f"rejection acceptance rate: {np.mean(np.exp(-target.penalty(prop))):.3f}")

# the reference sample the curves are measured against, and the estimator floor
reference = target.sample(N, rng)
floor = np.mean([sliced_w2_two_sample(target.sample(N, rng), target.sample(N, rng), D)
                 for _ in range(6)])
print(f"two-sample sliced-W2 floor at n={N}: {floor:.4f}")
print(f"\nJ_a (spectral norm {np.linalg.norm(J_a, 2):.2f}):\n{np.round(J_a, 4)}")
# ---------------------------------------------------------------- three fields
def tridiagonal(a):
    """The paper's J_a: equal strength on (1,2) and (2,3), nothing on (1,3)."""
    J = np.zeros((3, 3))
    J[0, 1] = J[1, 2] = a
    J[1, 0] = J[2, 1] = -a
    return J

A_TRI = 1.75
FIELDS = [("zero",  r"$J = 0$",                              None,                      ETA["zero"],     0.0),
          ("tri",   rf"$J_a$ tridiagonal,  $a = {A_TRI:g}$", tridiagonal(A_TRI),        4.185021e-04,   RAMP),
          ("plane", rf"$J_a$ stiff$\leftrightarrow$soft,  $a = {A:g}$",
                                                             stiff_soft(target.Sigma, A), ETA["const"], RAMP)]
print("\nfields, all at the same 2% equal-bias level:")
for k, lab, J, eta, _ in FIELDS:
    n = 0.0 if J is None else float(np.linalg.norm(J, 2))
    print(f"   {k:>6}  spectral norm {n:4.2f}   eta {eta:.6e}")

def curve(J, eta, prior):
    out = []
    for r in range(REPS):
        x = make_prior(prior, D, N, np.random.default_rng(500 + r))
        sched = warmup_schedule(target, eta, RAMP) if J is not None else None
        step = anchored_step(target, eta, J, sched)
        rng = np.random.default_rng(9000 + r)
        w, k = [], 0
        for t in record_at:
            while k < t:
                x = step(x, rng); k += 1
            w.append(sliced_w2_two_sample(x, reference, D))
        out.append(w)
    return np.mean(out, axis=0)

curves, hits = {}, {}
for prior in PRIORS:
    curves[prior] = {k: curve(J, eta, prior) for k, _, J, eta, _ in FIELDS}
    hits[prior] = {k: next((t for t, v in zip(record_at, curves[prior][k]) if v <= 2 * floor), -1)
                   for k, *_ in FIELDS}
    z = hits[prior]["zero"]
    print(f"{prior}: " + ",  ".join(
        f"{k} {hits[prior][k]}" + (f" ({z/hits[prior][k]:.2f}x)" if k != "zero" and hits[prior][k] > 0 else "")
        for k, *_ in FIELDS), flush=True)

PRIOR_TITLE = {"normal10": r"$X_0 \sim N(0,\,10\,I_d)$",
               "uniform5": r"$X_0 \sim \mathrm{Uniform}(-5,5)^d$"}
it = np.asarray(record_at, dtype=float); it[0] = it[1] / 2
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)
for ax, prior in zip(axes, PRIORS):
    z = hits[prior]["zero"]
    for i, (k, lab, J, eta, _) in enumerate(FIELDS):
        sp = f"   $\\bf{{{hits[prior][k]}}}$ it, {z/hits[prior][k]:.2f}$\\times$" if hits[prior][k] > 0 else ""
        ax.plot(it, curves[prior][k], lw=2.2, ls="--" if k == "tri" else "-", label=lab + sp)
    ax.axhline(2 * floor, color="0.5", lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_title(PRIOR_TITLE[prior], loc="left", fontsize=11)
    ax.grid(True, which="major", lw=0.6, alpha=0.45)
    ax.grid(True, which="minor", lw=0.4, alpha=0.20)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.6, loc="upper right")
axes[0].set_ylabel("Wasserstein distance")
axes[0].set_ylim(top=axes[0].get_ylim()[1] * 2.2)
fig.tight_layout()
fig.savefig("mcp_tridiagonal_vs_plane.png", dpi=200)
print("wrote mcp_tridiagonal_vs_plane.png")
