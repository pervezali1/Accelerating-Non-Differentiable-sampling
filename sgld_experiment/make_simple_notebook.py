"""Build ``simple_nonreversible_vs_reversible.ipynb`` -- a self-contained, explained notebook.

Everything (data, U = f + g, U_0 = f + g_0, a(w), J_s, the update, the figures) is
defined in the notebook itself; the only thing imported from the repository is the
constraint set (uniform sampling on K and the Euclidean projection onto K).  The
Colab variant additionally writes ``anchored_sgld.py`` next to itself.

    python make_simple_notebook.py
    jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 \
        simple_nonreversible_vs_reversible.ipynb --output simple_nonreversible_vs_reversible.executed.ipynb
"""
from __future__ import annotations

import nbformat as nbf

M = []   # (cell_type, source)

M.append(("markdown", r"""# Non-reversible vs reversible anchored Langevin — a self-contained walk-through

**Question.** Does the non-reversible anchored Langevin sampler (block skew-symmetric
$J_s$, $\alpha=1$) reach good predictions faster than the reversible one ($\alpha=0$) on a
constrained Bayesian logistic regression whose target has a **non-differentiable** LASSO
term?

**Answer (in the figures below).** Yes, by a wide margin: with the design described in
section 1, the non-reversible chain is at $\approx0.72$ test accuracy after 100 iterations
while the reversible chain is at $\approx0.56$; both end at the same plateau. The margin
survives held-out random seeds ($t\approx29$).

**How to read this notebook.** Every ingredient is written out in its own cell:
the data (1), the target $U=f+g$ (2), the anchor $U_0=f+g_0$ and $a(w)=e^{U-U_0}$ (3),
the constraint sets and $J_s$ (4), the update rule (5), the runs (6), the figures (7),
the numbers (8) and the explanation (9). The only thing imported from the repository is
the constraint set: uniform sampling on $K$ and the Euclidean projection onto $K$
(`anchored_sgld.BallGeometry`, `anchored_sgld.L1SmoothBallGeometry`).

Notation: $d=9$ parameters, $w=(w_0,w_1,\dots,w_8)$ with $w_0$ the intercept;
$x_j\in\mathbb R^9$ has a leading 1; $R$ replicate chains run in parallel as the rows of a
$(R,d)$ array."""))

M.append(("code", '''import os, sys, time, hashlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
import anchored_sgld as nral          # ONLY for the constraint sets: sample_uniform(K) and project(K)

QUICK           = os.environ.get("NRAL_QUICK", "0") == "1"
R               = 20 if QUICK else 100        # replicate chains
N_ITERATIONS    = 300 if QUICK else 1000
CHECKPOINT      = 10                          # accuracy is recorded every CHECKPOINT iterations
LAMBDA_LASSO    = 2.0                         # strength of the non-differentiable g
DELTA           = 0.02                        # smoothing of g inside the anchor U_0
SIGMA_INTERCEPT = 5.0                         # sd of the Gaussian prior on the intercept w_0
EPSILON         = 0.2                         # smoothing of the L1 constraint (soft-sign width)
HELD_OUT        = (101,) if QUICK else (101, 202, 303, 404)   # extra sampler seeds for confirmation
OUT             = "results_beat"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("QUICK MODE" if QUICK else "FULL", f"| R = {R}, {N_ITERATIONS} iterations")'''))

M.append(("markdown", r"""## 1. Data — unstandardised covariates, with the slow direction where $J$ can reach it

Synthetic logistic regression, $n=2000$ observations (80% train / 20% test), intercept
plus 8 slopes. The slopes come in the triples $(1,2)$, $(3,4,5)$, $(6,7,8)$ because $J_s$
is block diagonal on the coordinate triples $(0,1,2),(3,4,5),(6,7,8)$ — see section 4.

Inside each of the two full triples the covariance is

$$\Sigma_I = v_{\rm axis}\,q_1q_1^\top + v_{\rm fast}\,q_2q_2^\top + v_{\rm slow}\,q_3q_3^\top,\qquad
q_1=\tfrac{(1,1,1)}{\sqrt3},\; q_2=\tfrac{(0,1,-1)}{\sqrt2},\; q_3=\tfrac{(2,-1,-1)}{\sqrt6},$$

with $\beta_I=(b,e,e)$, $b>e>0$. Three facts about this choice, proved in section 9:

* $q_1$ is the axis about which the $L_1$ block rotation turns (the soft-sign of a
  positive $\beta_I$ is $\propto(1,1,1)$), so $q_2,q_3$ span the plane that $J$ rotates;
* $q_2\cdot\beta_I=0$: the fast direction carries no signal, so its large variance does
  not saturate the logits;
* $q_3\cdot\beta_I=\tfrac{2(b-e)}{\sqrt6}\neq0$ and $v_{\rm slow}\ll v_{\rm fast}$: the slow
  posterior direction carries signal and lies in the rotated plane.

The reversible chain needs $\sim1/(\eta a\lambda_{\rm slow})$ iterations along $q_3$; the
non-reversible one needs $\sim2/(\eta a\lambda_{\rm fast})$. That is the whole effect.

The unit ball has radius 1, so for it $\beta$ is halved and the variances quadrupled
(identical logits, four times the curvature)."""))

M.append(("code", '''def make_data(v_axis, v_fast, v_slow, b, e, b1, block1_variance, n_total=2000, data_seed=2026, split_seed=2027):
    q1 = np.ones(3) / np.sqrt(3.0); q2 = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0); q3 = np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0)
    block = v_axis * np.outer(q1, q1) + v_fast * np.outer(q2, q2) + v_slow * np.outer(q3, q3)
    Sigma = block1_variance * np.eye(8)                      # slopes 1,2 (with the intercept in block 1): isotropic
    Sigma[2:5, 2:5] = block; Sigma[5:8, 5:8] = block         # slopes 3-5 and 6-8: the structured triples
    beta = np.array([0.0, b1, b1, b, e, e, b, e, e])         # intercept first
    rng = np.random.default_rng(data_seed)
    Z = rng.multivariate_normal(np.zeros(8), Sigma, size=n_total)
    X = np.hstack([np.ones((n_total, 1)), Z])                # column 0 is the intercept
    y = (rng.uniform(size=n_total) <= expit(X @ beta)).astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=split_seed, stratify=y, shuffle=True)
    return dict(X_train=np.ascontiguousarray(X_tr), y_train=y_tr, X_test=np.ascontiguousarray(X_te), y_test=y_te, beta=beta, Sigma=Sigma)

DATA = {
    "l1":   make_data(v_axis=1.0, v_fast=64.0,  v_slow=2.0, b=1.5, e=0.25,  b1=0.3,  block1_variance=1.0),
    "ball": make_data(v_axis=4.0, v_fast=256.0, v_slow=4.0, b=0.5, e=0.125, b1=0.15, block1_variance=4.0),
}
for tag, D in DATA.items():
    z = D["X_test"] @ D["beta"]
    ceiling = np.mean(np.maximum(expit(z), 1 - expit(z)))      # accuracy of the TRUE beta = the Bayes ceiling
    print(f"{tag:5s} n_train {len(D['y_train'])}, n_test {len(D['y_test'])}, positives {D['y_train'].mean():.3f}, "
          f"|beta|_1 {np.abs(D['beta']).sum():.2f}, |beta|_2 {np.linalg.norm(D['beta']):.2f}, Bayes ceiling {ceiling:.4f}")
    print("      feature sd:", np.round(D["X_train"][:, 1:].std(axis=0), 2), " <- unequal scales inside each triple")'''))

M.append(("markdown", r"""## 2. The target $U=f+g$

$$U(w)=\underbrace{\sum_{j=1}^{n}\Bigl[\log\bigl(1+e^{x_j^\top w}\bigr)-y_j\,x_j^\top w\Bigr]+\frac{w_0^2}{2\sigma^2}}_{f(w)\ \text{— differentiable}}
\;+\;\underbrace{\lambda_{\rm lasso}\sum_{j=1}^{8}|w_j|}_{g(w)\ \text{— NOT differentiable at } w_j=0}$$

$f$ is the logistic negative log-likelihood plus a weak Gaussian prior on the intercept;
$g$ is the LASSO penalty on the slopes (the intercept is not penalised). The posterior
we want to sample is $\pi(w)\propto e^{-U(w)}\mathbf 1_K(w)$. Because $\nabla g$ does not
exist on the coordinate hyperplanes, $\nabla U$ cannot be used in a Langevin drift — that
is what the anchor in section 3 is for."""))

M.append(("code", '''def f(w, X, y):
    """Differentiable part: logistic negative log-likelihood + Gaussian prior on the intercept.  w: (R, d)."""
    z = X @ w.T                                                   # (n, R) linear predictors
    return (np.logaddexp(0.0, z) - y[:, None] * z).sum(axis=0) + w[:, 0] ** 2 / (2.0 * SIGMA_INTERCEPT ** 2)

def g(w):
    """NON-differentiable part: LASSO penalty on the slopes w_1..w_8 (the intercept w_0 is not penalised)."""
    return LAMBDA_LASSO * np.abs(w[:, 1:]).sum(axis=1)

def U(w, X, y):
    """The target potential: pi(w) is proportional to exp(-U(w)) on K."""
    return f(w, X, y) + g(w)'''))

M.append(("markdown", r"""## 3. The anchor $U_0=f+g_0$ and the state-dependent scale $a(w)$

Replace $|w_j|$ by the smooth $\sqrt{w_j^2+\delta^2}$:

$$g_0(w)=\lambda_{\rm lasso}\sum_{j=1}^{8}\sqrt{w_j^2+\delta^2},\qquad U_0=f+g_0,\qquad
a(w)=e^{U(w)-U_0(w)}=e^{\,g(w)-g_0(w)}\in\bigl[e^{-8\lambda\delta},\,1\bigr].$$

$U_0$ is $C^\infty$ and its exact gradient is available everywhere. The anchored diffusion
$dw=-a(w)\nabla U_0(w)\,dt+\sqrt{2a(w)}\,dB_t$ has invariant density
$\propto e^{-U_0}/a=e^{-U}$ — the *true* non-differentiable target — even though only $U_0$ is
ever differentiated. The cost of a large $\lambda\delta$ is a small $a$, i.e. a slow
clock, so $\delta$ is kept small."""))

M.append(("code", '''def g0(w):
    """Smooth stand-in for g: lambda * sum_j sqrt(w_j^2 + delta^2) over the slopes."""
    return LAMBDA_LASSO * np.sqrt(w[:, 1:] ** 2 + DELTA ** 2).sum(axis=1)

def U0(w, X, y):
    """The anchor: f + g_0, smooth everywhere."""
    return f(w, X, y) + g0(w)

def grad_U0(w, X, y):
    """EXACT gradient of U_0 (nothing is subsampled).  w: (R, d) -> (R, d)."""
    residual = expit(X @ w.T) - y[:, None]                        # (n, R) = p_j(w) - y_j
    grad = residual.T @ X                                         # (R, d) gradient of the log-likelihood part
    grad[:, 0] += w[:, 0] / SIGMA_INTERCEPT ** 2                  # prior on the intercept
    grad[:, 1:] += LAMBDA_LASSO * w[:, 1:] / np.sqrt(w[:, 1:] ** 2 + DELTA ** 2)   # gradient of g_0
    return grad

def log_a(w):
    """log a(w) = U - U_0 = g - g_0, in [-8 lambda delta, 0]."""
    return LAMBDA_LASSO * (np.abs(w[:, 1:]) - np.sqrt(w[:, 1:] ** 2 + DELTA ** 2)).sum(axis=1)

def a(w):
    return np.exp(log_a(w))'''))

M.append(("markdown", "### Sanity checks\n\n`grad_U0` against centred finite differences of `U0`; $\\log a=U-U_0$; the bounds on $a$."))

M.append(("code", '''X, y = DATA["l1"]["X_train"], DATA["l1"]["y_train"]
rng = np.random.default_rng(7); w = rng.normal(scale=0.3, size=(1, 9)); h = 1e-6
numeric = np.array([(U0(w + h * np.eye(9)[i], X, y) - U0(w - h * np.eye(9)[i], X, y))[0] / (2 * h) for i in range(9)])
print(f"grad_U0 vs finite differences (max relative error): {np.abs(grad_U0(w, X, y)[0] - numeric).max() / np.abs(numeric).max():.2e}")
probe = rng.normal(scale=0.5, size=(500, 9))
print(f"|log a - (U - U_0)| on 500 random points:           {np.abs(log_a(probe) - (U(probe, X, y) - U0(probe, X, y))).max():.2e}")
print(f"a(w) in [{a(probe).min():.4f}, {a(probe).max():.4f}]  (bound exp(-8 lambda delta) = {np.exp(-8 * LAMBDA_LASSO * DELTA):.4f})")'''))

M.append(("markdown", r"""## 4. The constraint sets and the skew-symmetric matrix $J_s(w)$

Two constraint sets $K$: the **unit ball** $\{\|w\|_2\le1\}$ and the **smoothed $L_1$ ball**
$\{\sum_{i=0}^{8}\sqrt{w_i^2+\varepsilon^2}\le\Lambda\}$, $\Lambda=9\varepsilon+r$. After every
step the chain is projected back onto $K$ (Euclidean projection; for the $L_1$ ball it is
computed exactly from the KKT conditions inside `anchored_sgld`). Chains start uniformly
distributed on $K$.

$J_s(w)$ is block diagonal on the coordinate triples $I\in\{(0,1,2),(3,4,5),(6,7,8)\}$ and
each block is a **cross-product matrix** $[v_I]_\times$, i.e. $(J_s\,u)_I=v_I\times u_I$, with

$$v_I=s\,w_I\ \ (\text{ball}),\qquad v_I=-s\,\nabla_I\,\Bigl(\textstyle\sum_i\sqrt{w_i^2+\varepsilon^2}\Bigr)=-s\,\frac{w_I}{\sqrt{w_I^2+\varepsilon^2}}\ \ (L_1\text{ ball}).$$

Properties: $[v]_\times$ is skew-symmetric; $\nabla\!\cdot J=0$ (for the ball
$\partial_iJ_{ij}=s\,\epsilon_{ikj}\partial_iw_k=0$, for the $L_1$ ball it is a contraction of a
symmetric Hessian with $\epsilon_{ikj}$), so no divergence correction is needed and
$e^{-U}$ stays invariant; and $J_s n=0$ because $v_I$ is parallel to the outward normal of $K$
restricted to the block, so the rotation is tangential to $\partial K$. The strength $s$ is the
one tunable knob of the non-reversible method."""))

M.append(("code", '''GEOMETRY = {
    "l1":   nral.L1SmoothBallGeometry(9, EPSILON, float(np.abs(DATA["l1"]["beta"]).sum() + 1.0)),   # radius r = |beta|_1 + 1
    "ball": nral.BallGeometry(9),
}
for tag, geom in GEOMETRY.items():
    print(f"{tag:5s} -> {geom.name:<16} beta_true feasible: {bool(geom.feasible(DATA[tag]['beta']))}")

def block_axes(w, geometry, s):
    """(R, 3, 3): the axis of every coordinate triple: s*w_I (ball) or -s*grad_I sum sqrt(w_i^2+eps^2) (L1)."""
    axes = w if geometry.name == "unit ball" else -w / np.sqrt(w ** 2 + EPSILON ** 2)
    return s * axes.reshape(w.shape[0], 3, 3)

def apply_J(w, u, geometry, s):
    """Matrix-free J_s(w) u: block-wise cross products v_I x u_I.  w, u: (R, d)."""
    return np.cross(block_axes(w, geometry, s), u.reshape(u.shape[0], 3, 3)).reshape(u.shape)

# checks: agrees with the repository's apply_J, and u . J u = 0 (skew-symmetry)
wp, up = rng.normal(size=(50, 9)) * 0.3, rng.normal(size=(50, 9))
for tag, geom in GEOMETRY.items():
    mine, lib = apply_J(wp, up, geom, 3.0), nral.apply_J(wp, up, geom, np.array([3.0, 3.0, 3.0]))
    print(f"{tag:5s} |apply_J - anchored_sgld.apply_J| = {np.abs(mine - lib).max():.1e};   max |u . J u| = {np.abs((up * mine).sum(axis=1)).max():.1e}")'''))

M.append(("markdown", r"""## 5. The update

$$w_{k+1}=\Pi_K\!\Bigl(w_k-\eta\,a(w_k)\nabla U_0(w_k)\;+\;\eta\,\alpha\,a(w_k)\,J_s(w_k)\nabla U_0(w_k)\;+\;\sqrt{2\eta\,a(w_k)}\;\xi_{k+1}\Bigr),\qquad \xi_{k+1}\sim N(0,I_d).$$

$\alpha=0$: **reversible** anchored Langevin. $\alpha=1$: **non-reversible** anchored
Langevin. Within a replicate both chains use the *same* starting point and the *same*
Gaussian increments $\xi_1,\xi_2,\dots$ — the only difference between the two runs is
$\alpha$, so the paired difference of their accuracies is a clean measurement.
Accuracy is *single-iterate*: at checkpoint $k$ each replicate predicts with its own
current $w_k$ (no averaging, no smoothing)."""))

M.append(("code", '''def accuracy(w, X, y):
    """Fraction of correct predictions of every replicate: (R,)."""
    return (((X @ w.T) > 0) == (y[:, None] > 0.5)).mean(axis=0)

def shared_streams(geometry, seed_offset=0):
    """Identical starting points and Gaussian increments for both methods (a reproducible seed per geometry)."""
    tag = int.from_bytes(hashlib.sha256(geometry.name.encode()).digest()[:4], "big")
    init_ss, noise_ss = np.random.SeedSequence([3000 + seed_offset, tag]).spawn(2)
    w_init = geometry.sample_uniform(np.random.default_rng(init_ss), R)                 # uniform on K
    noise = np.stack([np.random.default_rng(sd).standard_normal((N_ITERATIONS, 9)) for sd in noise_ss.spawn(R)])
    return w_init, noise                                                                 # (R, d), (R, N_ITERATIONS, d)

def run_chain(alpha, data, geometry, s, eta, streams):
    X, y, Xt, yt = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    w_init, noise = streams
    w = w_init.copy()
    checkpoints, train, test, n_projected = [0], [accuracy(w, X, y)], [accuracy(w, Xt, yt)], 0
    for k in range(N_ITERATIONS):
        grad = grad_U0(w, X, y)                                    # exact gradient of the anchor
        ak = a(w)[:, None]
        drift = -eta * ak * grad
        if alpha != 0.0:
            drift = drift + eta * alpha * ak * apply_J(w, grad, geometry, s)
        proposal = w + drift + np.sqrt(2.0 * eta * ak) * noise[:, k, :]
        outcome = geometry.project(proposal)                       # back onto K
        w = outcome.beta; n_projected += int(outcome.projected.sum())
        if (k + 1) % CHECKPOINT == 0:
            checkpoints.append(k + 1); train.append(accuracy(w, X, y)); test.append(accuracy(w, Xt, yt))
    return dict(checkpoints=np.array(checkpoints), train=np.array(train), test=np.array(test),
                projection_rate=n_projected / (N_ITERATIONS * R))'''))

M.append(("markdown", r"""## 6. Run both methods on both constraint sets

| constraint set | $s$ | $\eta$ | why |
|---|---|---|---|
| $L_1$-smooth ball | 4 | $7\cdot10^{-6}$ | $\eta a\lambda_{\max}\approx0.14$: stable, and the reversible chain is still converging along $q_3$ for a few hundred iterations |
| unit ball | 16 | $2\cdot10^{-6}$ | four times the curvature $\Rightarrow$ $\eta/4$; the ball axis $s\,w_I$ is short, so $s$ must be larger for the same rotation angle |

Both are inside the stability window derived in section 9; `results_beat/parameter_map.md`
in the repository shows what happens outside it."""))

M.append(("code", '''SETTINGS = {"l1": dict(s=4.0, eta=7e-6), "ball": dict(s=16.0, eta=2e-6)}
METHODS  = (("Reversible anchored Langevin", 0.0), ("Non-reversible anchored Langevin", 1.0))
results  = {}
for tag in ("l1", "ball"):
    streams = shared_streams(GEOMETRY[tag])
    results[tag] = {}
    for name, alpha in METHODS:
        t0 = time.time()
        results[tag][name] = run_chain(alpha, DATA[tag], GEOMETRY[tag], streams=streams, **SETTINGS[tag])
        print(f"{GEOMETRY[tag].name:<16} {name:<34} {time.time() - t0:5.1f} s   projection rate {results[tag][name]['projection_rate']:.4f}")'''))

M.append(("markdown", r"""## 7. Figures — training and test accuracy only

Lines: mean over the $R$ replicates; bands: mean $\pm$ one sample standard deviation
(repeat-run variability, not a confidence interval). The vertical axis is fixed to
$[0.4,0.9]$ on every panel — it is not windowed on the plateau."""))

M.append(("code", '''STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}
for tag in ("l1", "ball"):
    geom, D, p = GEOMETRY[tag], DATA[tag], SETTINGS[tag]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, which, title in ((axes[0], "train", f"Training accuracy  (n = {len(D['y_train'])})"),
                             (axes[1], "test",  f"Test accuracy  (n = {len(D['y_test'])})")):
        for name, run in results[tag].items():
            st = STYLE[name]; mean, sd = run[which].mean(axis=1), run[which].std(axis=1, ddof=1)
            ax.fill_between(run["checkpoints"], np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=st["color"], alpha=0.15, lw=0)
            ax.plot(run["checkpoints"], mean, color=st["color"], ls=st["ls"], lw=st["lw"], label=name)
        ax.set_ylim(0.40, 0.90); ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy"); ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Non-reversible vs reversible anchored Langevin — {geom.name}   (s = {p['s']:g}, eta = {p['eta']:.0e}, R = {R})", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(OUT, f"simple_accuracy_{tag}.png"); fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
    display(Image(filename=png))'''))

M.append(("markdown", r"""## 8. The numbers

Paired per-replicate differences (non-reversible minus reversible) of single-iterate test
accuracy, with the paired $t$-statistic over the $R$ replicates. Then the whole comparison is
repeated on sampler seeds that took no part in choosing the configuration, at the
iteration where the gap peaks."""))

M.append(("code", '''def paired(nr, rev):
    d = nr - rev; se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d.mean() / se if se > 0 else np.nan

rows = []
for tag in ("l1", "ball"):
    rev, nr = results[tag]["Reversible anchored Langevin"], results[tag]["Non-reversible anchored Langevin"]
    for k in (50, 100, 200, 300, 400, 600, 800, N_ITERATIONS):
        i = int(np.argmin(np.abs(rev["checkpoints"] - k)))
        m, se, t = paired(nr["test"][i], rev["test"][i])
        rows.append({"geometry": GEOMETRY[tag].name, "iteration": int(rev["checkpoints"][i]), "reversible": rev["test"][i].mean(),
                     "non_reversible": nr["test"][i].mean(), "paired_diff": m, "paired_se": se, "t_stat": t})
display(pd.DataFrame(rows))

EVALUATE_AT = {"l1": 100, "ball": 90}
held = []
for tag in ("l1", "ball"):
    diffs, ses = [], []
    for off in HELD_OUT:
        streams = shared_streams(GEOMETRY[tag], seed_offset=off)
        rev = run_chain(0.0, DATA[tag], GEOMETRY[tag], streams=streams, **SETTINGS[tag])
        nr  = run_chain(1.0, DATA[tag], GEOMETRY[tag], streams=streams, **SETTINGS[tag])
        i = int(np.argmin(np.abs(rev["checkpoints"] - EVALUATE_AT[tag])))
        m, se, t = paired(nr["test"][i], rev["test"][i]); diffs.append(m); ses.append(se)
        held.append({"geometry": GEOMETRY[tag].name, "seed_offset": off, "iteration": int(rev["checkpoints"][i]),
                     "reversible": rev["test"][i].mean(), "non_reversible": nr["test"][i].mean(), "paired_diff": m, "t_stat": t})
    pooled, pooled_se = np.mean(diffs), np.sqrt(np.sum(np.square(ses))) / len(ses)
    print(f"{GEOMETRY[tag].name:<16} held-out pooled diff {pooled:+.4f} at iteration {EVALUATE_AT[tag]} (t = {pooled / pooled_se:+.1f}), "
          f"all seeds positive: {all(d > 0 for d in diffs)}  ->  {'CONFIRMED' if all(d > 0 for d in diffs) and pooled / pooled_se > 2 else 'NOT confirmed'}")
display(pd.DataFrame(held))'''))

M.append(("markdown", r"""## 9. Why the non-reversible chain is faster here — and where it is not

**Linearise at the mode.** Near the posterior mode $w^*$ the drift is $-\eta a(I-\alpha J)H$
with $H=\nabla^2U_0(w^*)$. Inside one triple $I$, $J_I=[v_I]_\times$ rotates only in the plane
**perpendicular** to its axis $v_I$. Write the block Hessian's eigenpairs as
$\lambda_1\ge\lambda_2\ge\lambda_3$.

* If the slow eigenvector $q_3$ lies in the rotated plane with partner $q_2$, the two rates
  become $\tfrac{\lambda_2+\lambda_3}{2}\pm\sqrt{\tfrac{(\lambda_2-\lambda_3)^2}{4}-\sigma^2\lambda_2\lambda_3}$
  with $\sigma=s|v_I|$. For $\sigma\ge\sigma^*=\tfrac{\lambda_2-\lambda_3}{2\sqrt{\lambda_2\lambda_3}}$
  the slow rate is replaced by the mean $\tfrac{\lambda_2+\lambda_3}{2}$: a speed-up of
  $(\kappa+1)/2$ with $\kappa=\lambda_2/\lambda_3$. In discrete time the pair stays stable
  while $\sigma^2<\tfrac{\lambda_2+\lambda_3}{\eta a\lambda_2\lambda_3}-1$.
* A slow direction **along** the axis is untouched for every $s$; an oblique one gains at
  most $\approx1/\cos^2\theta$.
* On the ball the axis at the mode is $w^*_I$ itself, so the ball can never accelerate the
  direction of the block's dominant coefficient. Under the $L_1$ geometry the axis is the
  soft-sign of $w_I$, $\approx(1,1,1)/\sqrt3$, so $q_3=(2,-1,-1)/\sqrt6$ is in the plane **and**
  carries signal ($q_3\cdot\beta_I\ne0$).

**Why it shows in accuracy.** A residual $\Delta$ along a direction with feature variance
$v$ perturbs the test logits by $\sqrt v\,|\Delta|$, and the accuracy loss is
$\approx f(0)\,\delta^2/4$ for a logit perturbation of standard deviation $\delta$ ($f$ = density of
the true logits at 0). The residual along $q_3$ carries an $O(1)$ share of the logit
variance, so while the reversible chain still has it, its accuracy is visibly lower.

**Where the win lives** (one-factor map in `results_beat/parameter_map.md`): anisotropy
$\kappa\ge4$ (the gap grows with $\kappa$ and saturates near $\kappa\approx32$),
$0.5\le s\le12$, $2\cdot10^{-6}\le\eta\le6\cdot10^{-5}$, any $\lambda_{\rm lasso}$, $\delta$, $n$, $\varepsilon$,
radius tried. **Ties:** $\kappa\le2$, $s=0$, an isotropic design, a signal-free slow
direction, initialisation at the origin, and any iteration after both chains have
converged. **Loses:** $s\ge16$ or $\eta=10^{-4}$ — past the stability edge the rotation
overshoots, the projection fires and the non-reversible chain is worse.

**Scope.** This is a convergence-speed effect on a designed regime (strongly unequal
feature scales with the signal in a low-variance direction that lies in the rotated
plane). Both chains reach the same plateau; the effect is the transient between a uniform
start on $K$ and that plateau."""))


def build(colab: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []
    for i, (kind, src) in enumerate(M):
        if colab and i == 1:
            cells.append(nbf.v4.new_markdown_cell(
                "## Colab bootstrap\n\n`anchored_sgld.py` (the constraint sets) is written next to the notebook so it runs anywhere."))
            with open("anchored_sgld.py") as handle:
                cells.append(nbf.v4.new_code_cell("%%writefile anchored_sgld.py\n" + handle.read()))
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    for colab in (False, True):
        nb = build(colab); nbf.validate(nb)
        path = "simple_nonreversible_vs_reversible_colab.ipynb" if colab else "simple_nonreversible_vs_reversible.ipynb"
        nbf.write(nb, path); print("wrote", path, f"({len(nb.cells)} cells)")
