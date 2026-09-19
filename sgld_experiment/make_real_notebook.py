"""Build ``real_data_nonreversible_vs_reversible.ipynb``: the real-data comparison, self-contained.

Titanic and the MAGIC gamma telescope, both constraint sets, accuracy curves only.
The data files are read from ./data (downloaded if missing); nothing is imported
from the repository.

    python make_real_notebook.py
    jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 \
        real_data_nonreversible_vs_reversible.ipynb --output real_data_nonreversible_vs_reversible.executed.ipynb
"""
from __future__ import annotations

import nbformat as nbf

from make_simple_notebook import M as SIMPLE

def simple_cell(first_line: str) -> str:
    for kind, src in SIMPLE:
        if kind == "code" and src.lstrip().startswith(first_line):
            return src
    raise KeyError(first_line)

M = []

M.append(("markdown", r"""# Non-reversible vs reversible anchored Langevin on real data — Titanic and the MAGIC telescope

The two samplers of the synthetic study, applied to two real binary-classification
problems with the same target $U=f+g$ (logistic likelihood plus a non-differentiable
LASSO term), the same anchor $U_0=f+g_0$, the same block cross-product $J_s$ and both
constraint sets. Only **training and test accuracy against iteration** are plotted.

* **Titanic** — 891 passengers, survival; intercept + 8 engineered columns (`pclass`, `sex`,
  median-imputed `age`, `sibsp`, `parch`, $\log(1+\mathrm{fare})$, `embarked=C`, `embarked=Q`); $d=9$.
* **MAGIC gamma telescope** — 19,020 events, gamma vs hadron; intercept + 10 features + one
  zero column so that $d=12$ is a multiple of 3 (the block structure of $J$); its coefficient
  is pulled to zero by $g$ and costs nothing.

**What the figures show.** With the posterior written in the coordinates described in
section 2, the non-reversible chain reaches $\approx0.73$ test accuracy on Titanic and
$\approx0.74$ on MAGIC within 100–150 iterations while the reversible one is still at
$\approx0.55$; both converge to the logistic-regression reference (0.78 / 0.79). The margin
survives held-out sampler seeds ($t\approx20$–$30$, $R=100$).

**What the figures do not show, said up front.** With plain standardised features the two
samplers tie on both data sets (best gap $\le+0.02$; see the search summary in
`results_real/summary.md`). The win needs a *choice of parametrisation*: an invertible linear
map of the features, fitted once from a pilot logistic regression, that leaves the logits —
and hence the model and its accuracy — unchanged but puts a slow, signal-carrying posterior
direction into the plane that $J$ rotates. Section 2 spells it out; section 9 says where the
win lives and where it does not.

Sections: 1 data · 2 parametrisation · 3 $U=f+g$ · 4 $U_0=f+g_0$ and $a(w)$ · 5 constraint sets
and $J_s$ · 6 the update · 7 runs · 8 figures · 9 numbers, held-out seeds and scope."""))

M.append(("code", '''import os, sys, time, hashlib, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

QUICK           = os.environ.get("NRAL_QUICK", "0") == "1"
R               = 20 if QUICK else 100        # replicate chains
CHECKPOINT      = 10
LAMBDA_LASSO    = 2.0                         # strength of the non-differentiable g
DELTA           = 0.02                        # smoothing of g inside U_0
SIGMA_INTERCEPT = 5.0                         # sd of the Gaussian prior on the intercept
EPSILON         = 0.2                         # smoothing of the L1 constraint
HELD_OUT        = (101,) if QUICK else (101, 202, 303, 404)
DATA_DIR, OUT   = "data", "results_real"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

SOURCES = {"titanic.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
           "magic.tsv.gz": "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/magic/magic.tsv.gz"}
for name, url in SOURCES.items():                                   # download once if the repository's data/ is absent
    if not os.path.exists(os.path.join(DATA_DIR, name)):
        urllib.request.urlretrieve(url, os.path.join(DATA_DIR, name)); print("downloaded", name)
print("QUICK MODE" if QUICK else "FULL", f"| R = {R}")'''))

M.append(("markdown", r"""## 1. Data

Stratified 80/20 split (`random_state = 2027`). Features are z-scored with the training
statistics, then every feature is sign-flipped so that its pilot logistic-regression
coefficient is positive — $g$ and both constraint sets are symmetric under sign flips, so
this changes nothing about the model, but it makes the soft-sign axis of the $L_1$ block
rotation point along $(1,1,1)/\sqrt3$ in every block (section 5). The pilot fit
(scikit-learn, $C=100$) also provides the reference test accuracy."""))

M.append(("code", '''def load(name):
    if name == "titanic":
        t = pd.read_csv(os.path.join(DATA_DIR, "titanic.csv"))
        X = pd.DataFrame({"pclass": t.pclass.astype(float), "sex": (t.sex == "female").astype(float),
                          "age": t.age.fillna(t.age.median()), "sibsp": t.sibsp.astype(float), "parch": t.parch.astype(float),
                          "log_fare": np.log1p(t.fare), "emb_C": (t.embarked == "C").astype(float), "emb_Q": (t.embarked == "Q").astype(float)})
        return X.values.astype(float), t.survived.values.astype(float), list(X.columns)
    m = pd.read_csv(os.path.join(DATA_DIR, "magic.tsv.gz"), sep="\\t"); X = m.drop(columns="target")
    return X.values.astype(float), m.target.values.astype(float), list(X.columns)

def pilot(X, y):
    lr = LogisticRegression(C=100.0, max_iter=5000).fit(X, y)
    return lr.coef_[0].copy(), float(lr.intercept_[0])

def split_and_standardise(name):
    X, y, names = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=2027, stratify=y, shuffle=True)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    coef, _ = pilot(Xtr, ytr)
    flip = np.where(coef < 0, -1.0, 1.0)                                # sign alignment: every pilot coefficient positive
    return Xtr * flip, Xte * flip, ytr, yte, coef * flip, names

for name in ("titanic", "magic"):
    Xtr, Xte, ytr, yte, coef, names = split_and_standardise(name)
    print(f"{name:8s} n_train {len(ytr)}, n_test {len(yte)}, positives {ytr.mean():.3f}, majority-class test accuracy "
          f"{max(yte.mean(), 1 - yte.mean()):.3f}, pilot coefficients:", dict(zip(names, coef.round(2))))'''))

M.append(("markdown", r"""## 2. The parametrisation that lets $J$ work

Section 9 recalls why: inside a coordinate triple, $J_s$ only accelerates a slow
posterior direction that lies in the plane perpendicular to its axis, and the accuracy
gap is large only if that direction also carries signal. Standardised real features do
not have that structure (the block Hessians are nearly isotropic). So the posterior is
written in different coordinates.

For each full slope triple $I$ with training covariance $C$ and pilot coefficients
$\beta_I$, take the invertible map $x'=Ax$ with

$$A=\Sigma^{1/2}\,W\,C^{-1/2},\qquad
\Sigma=v_{\rm axis}q_1q_1^\top+v_{\rm fast}q_2q_2^\top+v_{\rm slow}q_3q_3^\top,\qquad
q_1=\tfrac{(1,1,1)}{\sqrt3},\ q_2=\tfrac{(0,1,-1)}{\sqrt2},\ q_3=\tfrac{(2,-1,-1)}{\sqrt6},$$

and $W$ the Householder reflection that takes the whitened coefficient vector
$u=C^{1/2}\beta_I$ onto the direction of $\Sigma^{1/2}(b,e,e)$. Then
$\mathrm{cov}(x')=\Sigma$ exactly, the coefficients become $\beta'_I=A^{-\top}\beta_I\propto(b,e,e)$
with $|u|$ fixing the scale, and $x'^\top\beta'=x^\top\beta$: **the logits, the model and its
accuracy are unchanged**; only the coordinates are. The six most informative features
(largest pilot coefficients) go to the two full triples; the intercept's triple (and, on
MAGIC, the triple holding the zero column) stays as it is. Here $(v_{\rm axis},v_{\rm fast},v_{\rm slow})=(1,256,1)$
for the $L_1$ runs, $(1,1024,1)$ for the Titanic ball run, and $(b,e)=(1,0.5)$.

This is a data-dependent choice of parametrisation made from a pilot fit. It is the same
posterior; both samplers see identical data; and it is exactly what the synthetic study
said is needed. It is not something the raw features give you for free (section 9)."""))

M.append(("code", '''Q1 = np.ones(3) / np.sqrt(3.0); Q2 = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0); Q3 = np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0)

def sqrt_and_inverse_sqrt(C):
    w, V = np.linalg.eigh(C); w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T

def block_map(Xtr_I, Xte_I, beta_I, v_axis, v_fast, v_slow, b, e):
    C_half, C_inv_half = sqrt_and_inverse_sqrt(np.cov(Xtr_I.T))
    Sigma = v_axis * np.outer(Q1, Q1) + v_fast * np.outer(Q2, Q2) + v_slow * np.outer(Q3, Q3)
    S_half, _ = sqrt_and_inverse_sqrt(Sigma)
    u = C_half @ beta_I                                                 # whitened coefficients
    r = S_half @ np.array([b, e, e]); r *= np.linalg.norm(u) / np.linalg.norm(r)
    u_hat, r_hat = u / np.linalg.norm(u), r / np.linalg.norm(r); v = u_hat - r_hat
    W = np.eye(3) if np.linalg.norm(v) < 1e-12 else np.eye(3) - 2.0 * np.outer(v, v) / (v @ v)
    A = S_half @ W @ C_inv_half
    return Xtr_I @ A.T, Xte_I @ A.T

def make_problem(name, v_axis, v_fast, v_slow, b=1.0, e=0.5, geometry="l1", ball_target=0.8, radius_margin=1.0):
    Xtr, Xte, ytr, yte, coef, names = split_and_standardise(name)
    p = Xtr.shape[1]
    ranked = list(np.argsort(-np.abs(coef))); idx = ranked[6:][:2] + ranked[:6] + ranked[6:][2:]   # top-6 -> the two full triples
    Xtr, Xte, coef, names = Xtr[:, idx], Xte[:, idx], coef[idx], [names[i] for i in idx]
    n_pad = (-(1 + p)) % 3                                              # zero columns so that d is a multiple of 3
    Xtr = np.hstack([Xtr, np.zeros((len(ytr), n_pad))]); Xte = np.hstack([Xte, np.zeros((len(yte), n_pad))])
    names += [f"pad{i}" for i in range(n_pad)]; coef = np.concatenate([coef, np.zeros(n_pad)])
    for start in range(2, p + n_pad - 2, 3):                            # full slope triples (2,3,4), (5,6,7), ...
        block = [start, start + 1, start + 2]
        if max(block) < p:
            Xtr[:, block], Xte[:, block] = block_map(Xtr[:, block], Xte[:, block], coef[block], v_axis, v_fast, v_slow, b, e)
    coef, intercept = pilot(Xtr[:, :p], ytr)                            # pilot fit in the final coordinates
    beta_ref = np.concatenate([[intercept], coef, np.zeros(n_pad)])
    X_train = np.hstack([np.ones((len(ytr), 1)), Xtr]); X_test = np.hstack([np.ones((len(yte), 1)), Xte])
    if geometry == "ball":                                              # scale every column so beta_ref fits in the unit ball
        k = np.linalg.norm(beta_ref) / ball_target; X_train, X_test, beta_ref = X_train * k, X_test * k, beta_ref / k
    d = X_train.shape[1]
    geom = Ball(d) if geometry == "ball" else L1Ball(d, r=float(np.abs(beta_ref).sum() + radius_margin))
    ref_acc = float(((X_test @ beta_ref > 0) == (yte > 0.5)).mean())
    return dict(name=name, X_train=X_train, y_train=ytr, X_test=X_test, y_test=yte, beta_ref=beta_ref, names=names,
                d=d, geometry=geom, reference_accuracy=ref_acc)'''))

M.append(("markdown", r"""## 3. The target $U=f+g$

$$U(w)=\underbrace{\sum_{j}\Bigl[\log\bigl(1+e^{x_j^\top w}\bigr)-y_j\,x_j^\top w\Bigr]+\frac{w_0^2}{2\sigma^2}}_{f\ \text{— differentiable}}
\;+\;\underbrace{\lambda_{\rm lasso}\sum_{j\ge1}|w_j|}_{g\ \text{— NOT differentiable}}$$

$\sigma=5$, $\lambda_{\rm lasso}=2$; the sum runs over the training rows; the posterior is
$\pi(w)\propto e^{-U(w)}\mathbf 1_K(w)$."""))
M.append(("code", simple_cell("def f(w, X, y):")))

M.append(("markdown", r"""## 4. The anchor $U_0=f+g_0$ and $a(w)=e^{U-U_0}$

$g_0(w)=\lambda_{\rm lasso}\sum_{j\ge1}\sqrt{w_j^2+\delta^2}$ with $\delta=0.02$; $a(w)\in[e^{-8\lambda\delta},1]$
on Titanic ($8$ slopes) and $[e^{-11\lambda\delta},1]$ on MAGIC. The anchored diffusion
$dw=-a\nabla U_0\,dt+\sqrt{2a}\,dB$ has invariant density $\propto e^{-U_0}/a=e^{-U}$, the true
non-differentiable target, although only $U_0$ is ever differentiated."""))
M.append(("code", simple_cell("def g0(w):")))

M.append(("markdown", r"""## 5. Constraint sets and the block matrix $J_s$

Unit ball $\{\|w\|_2\le1\}$ or smoothed $L_1$ ball $\{\sum_i\sqrt{w_i^2+\varepsilon^2}\le d\varepsilon+r\}$
with $r=\|\hat\beta\|_1+1$; Euclidean projection after every step (the $L_1$ one solved from
its KKT conditions by bisection); chains start uniformly on $K$. $J_s$ is block diagonal on
the coordinate triples, block $I$ being $[v_I]_\times$ with $v_I=s\,w_I$ (ball) or
$v_I=-s\,w_I/\sqrt{w_I^2+\varepsilon^2}$ ($L_1$): skew-symmetric, divergence-free, tangential to $\partial K$."""))

M.append(("code", '''class Ball:
    name = "unit ball"
    def __init__(self, d):        self.d = d
    def feasible(self, w):        return np.linalg.norm(w) <= 1.0 + 1e-9
    def sample_uniform(self, rng, n):
        Z = rng.standard_normal((n, self.d)); radius = rng.random(n) ** (1.0 / self.d)
        return Z * (radius / np.linalg.norm(Z, axis=1))[:, None]
    def project(self, z):
        norm = np.linalg.norm(z, axis=1); outside = norm > 1.0
        w = z.copy(); w[outside] = z[outside] / norm[outside, None]
        return w, outside

class L1Ball:
    name = "L1-smooth ball"
    def __init__(self, d, r):     self.d, self.Lambda = d, d * EPSILON + r
    def value(self, w):           return np.sqrt(w * w + EPSILON ** 2).sum(axis=-1)
    def feasible(self, w):        return self.value(w) <= self.Lambda + 1e-9
    def sample_uniform(self, rng, n):
        accepted, kept = [], 0                       # uniform on the exact L1 ball of radius Lambda, rejected to K
        while kept < n:
            size = max(n, 512)
            lap = rng.laplace(0.0, 1.0, size=(size, self.d)); ex = rng.exponential(1.0, size=size)
            prop = self.Lambda * lap / (np.abs(lap).sum(axis=1) + ex)[:, None]
            good = prop[self.value(prop) <= self.Lambda]; accepted.append(good); kept += len(good)
        return np.vstack(accepted)[:n]
    def _solve(self, z_abs, mu):
        low, high = np.zeros_like(z_abs), z_abs.copy()
        for _ in range(60):
            mid = 0.5 * (low + high)
            positive = mid + mu * mid / np.sqrt(mid * mid + EPSILON ** 2) > z_abs
            high = np.where(positive, mid, high); low = np.where(positive, low, mid)
        return 0.5 * (low + high)
    def project(self, z):
        outside = self.value(z) > self.Lambda; w = z.copy(); rows = np.nonzero(outside)[0]
        if rows.size:
            zr = z[rows]; sign, z_abs = np.sign(zr), np.abs(zr)
            gap = lambda mu: self.value(self._solve(z_abs, mu[:, None])) - self.Lambda
            low, high = np.zeros(rows.size), np.ones(rows.size)
            for _ in range(200):
                need = gap(high) > 0.0
                if not need.any(): break
                high[need] *= 2.0
            for _ in range(80):
                mid = 0.5 * (low + high); positive = gap(mid) > 0.0
                low = np.where(positive, mid, low); high = np.where(positive, high, mid)
            w[rows] = sign * self._solve(z_abs, (0.5 * (low + high))[:, None])
        return w, outside

def apply_J(w, u, geometry, s):
    """Matrix-free J_s(w) u: block-wise cross products v_I x u_I."""
    axes = w if geometry.name == "unit ball" else -w / np.sqrt(w ** 2 + EPSILON ** 2)
    n_blocks = w.shape[1] // 3
    return np.cross(s * axes.reshape(w.shape[0], n_blocks, 3), u.reshape(u.shape[0], n_blocks, 3)).reshape(u.shape)'''))

M.append(("markdown", r"""## 6. The update

$$w_{k+1}=\Pi_K\!\Bigl(w_k-\eta\,a(w_k)\nabla U_0(w_k)+\eta\,\alpha\,a(w_k)\,J_s(w_k)\nabla U_0(w_k)+\sqrt{2\eta\,a(w_k)}\;\xi_{k+1}\Bigr)$$

$\alpha=0$ reversible, $\alpha=1$ non-reversible; within a replicate both chains share the
starting point and every $\xi_k$. Accuracy is single-iterate (each replicate predicts with
its own current $w_k$)."""))

M.append(("code", '''def accuracy(w, X, y):
    return (((X @ w.T) > 0) == (y[:, None] > 0.5)).mean(axis=0)

def shared_streams(geometry, n_iterations, seed_offset=0):
    tag = int.from_bytes(hashlib.sha256(geometry.name.encode()).digest()[:4], "big")
    init_ss, noise_ss = np.random.SeedSequence([3000 + seed_offset, tag]).spawn(2)
    w_init = geometry.sample_uniform(np.random.default_rng(init_ss), R)
    noise = np.stack([np.random.default_rng(sd).standard_normal((n_iterations, geometry.d)) for sd in noise_ss.spawn(R)])
    return w_init, noise

def run_chain(alpha, P, s, eta, n_iterations, streams, s_warmup=0):
    """s_warmup > 0 ramps the block strength linearly from 0 over the first s_warmup iterations."""
    X, y, Xt, yt, geometry = P["X_train"], P["y_train"], P["X_test"], P["y_test"], P["geometry"]
    w_init, noise = streams; w = w_init.copy()
    checkpoints, train, test, n_projected = [0], [accuracy(w, X, y)], [accuracy(w, Xt, yt)], 0
    for k in range(n_iterations):
        grad = grad_U0(w, X, y); ak = a(w)[:, None]
        drift = -eta * ak * grad
        if alpha != 0.0:
            s_k = s * min(1.0, (k + 1) / s_warmup) if s_warmup else s
            drift = drift + eta * alpha * ak * apply_J(w, grad, geometry, s_k)
        w, projected = geometry.project(w + drift + np.sqrt(2.0 * eta * ak) * noise[:, k, :])
        n_projected += int(projected.sum())
        if (k + 1) % CHECKPOINT == 0:
            checkpoints.append(k + 1); train.append(accuracy(w, X, y)); test.append(accuracy(w, Xt, yt))
    return dict(checkpoints=np.array(checkpoints), train=np.array(train), test=np.array(test), projection_rate=n_projected / (n_iterations * R))'''))

M.append(("markdown", r"""## 7. Runs

Four cases. $\eta$ is set so that the Euler step is stable on the fast direction
($\eta a\lambda_{\max}\approx0.1$; MAGIC has $\sim20\times$ the training rows of Titanic, hence the
smaller $\eta$) and $s$ sits inside the discrete-time stability window (the ball needs a
larger $s$ because its axis $s\,w_I$ is short). The held-out evaluation iteration is the
one where the search-run gap peaked (fixed before the held-out seeds were drawn).

**The early dip on the ball, and how it is avoided.** Started uniformly on $K$, far from
the mode, the ball's rotated drift $s\,w_I\times\nabla_IU_0$ is perpendicular to the gradient
and $s\|w_I\|\approx13$ times longer than the gradient step, so for the first tens of
iterations the non-reversible chain moves *sideways* and its accuracy dips $\approx0.02$
below the reversible one before the contraction takes over. A larger step
($\eta=1.4\cdot10^{-6}$ instead of $7\cdot10^{-7}$, with $s=12$ instead of $16$ so the chain is
not pushed into the boundary) finishes that excursion within the first checkpoint, and
ramping $s$ from $0$ over the first 10 iterations removes what is left: the Titanic ball
curve is then monotone from the start, projected on under 1% of the steps, with no late
deficit — at the price of a smaller peak ($+0.10$ instead of $+0.16$ with the dip). The $L_1$
runs have no dip, and neither does the MAGIC ball run."""))

M.append(("code", '''CASES = {
    "titanic_l1":   dict(name="titanic", geometry="l1",   v_axis=1.0, v_fast=256.0,  v_slow=1.0, s=8.0,  eta=7e-6, n_iterations=3000, evaluate_at=140),
    "titanic_ball": dict(name="titanic", geometry="ball", v_axis=1.0, v_fast=1024.0, v_slow=1.0, s=12.0, eta=1.4e-6, n_iterations=2000, evaluate_at=135, s_warmup=10),
    "magic_l1":     dict(name="magic",   geometry="l1",   v_axis=1.0, v_fast=256.0,  v_slow=1.0, s=8.0,  eta=2e-6, n_iterations=1000, evaluate_at=80),
    "magic_ball":   dict(name="magic",   geometry="ball", v_axis=1.0, v_fast=256.0,  v_slow=1.0, s=16.0, eta=2e-7, n_iterations=1500, evaluate_at=50),
}
if QUICK:
    for c in CASES.values(): c["n_iterations"] = min(c["n_iterations"], 400)
METHODS = (("Reversible anchored Langevin", 0.0), ("Non-reversible anchored Langevin", 1.0))
problems, results = {}, {}
for tag, c in CASES.items():
    P = make_problem(c["name"], c["v_axis"], c["v_fast"], c["v_slow"], geometry=c["geometry"]); problems[tag] = P
    print(f"{tag:13s} d = {P['d']}, n_train = {len(P['y_train'])}, reference test accuracy {P['reference_accuracy']:.4f}, "
          f"beta_ref feasible: {bool(P['geometry'].feasible(P['beta_ref']))}, order: {P['names']}")
    streams = shared_streams(P["geometry"], c["n_iterations"]); results[tag] = {}
    for method, alpha in METHODS:
        t0 = time.time(); results[tag][method] = run_chain(alpha, P, c["s"], c["eta"], c["n_iterations"], streams, c.get("s_warmup", 0))
        print(f"   {method:<34} {time.time() - t0:6.1f} s   projection rate {results[tag][method]['projection_rate']:.4f}", flush=True)'''))

M.append(("markdown", r"""## 8. Figures — training and test accuracy only

Mean over the $R$ replicates, bands $\pm$ one sample sd; vertical axis fixed to $[0.4,0.9]$
on every panel, not windowed."""))

M.append(("code", '''STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}
PRETTY = {"titanic": "Titanic", "magic": "MAGIC gamma telescope"}
for tag, c in CASES.items():
    P, runs = problems[tag], results[tag]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, which, title in ((axes[0], "train", f"Training accuracy  (n = {len(P['y_train'])})"),
                             (axes[1], "test",  f"Test accuracy  (n = {len(P['y_test'])})")):
        for method, run in runs.items():
            st = STYLE[method]; mean, sd = run[which].mean(axis=1), run[which].std(axis=1, ddof=1)
            ax.fill_between(run["checkpoints"], np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=st["color"], alpha=0.15, lw=0)
            ax.plot(run["checkpoints"], mean, color=st["color"], ls=st["ls"], lw=st["lw"], label=method)
        ax.axhline(P["reference_accuracy"], color="grey", lw=0.8, ls=":")
        ax.set_ylim(0.40, 0.90); ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy"); ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"{PRETTY[c['name']]} — {P['geometry'].name}   (s = {c['s']:g}, eta = {c['eta']:.0e}, R = {R}; dotted: reference accuracy {P['reference_accuracy']:.3f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(OUT, f"notebook_{tag}.png"); fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
    display(Image(filename=png))'''))

M.append(("markdown", r"""## 9. The numbers, held-out seeds, and where the win lives

Paired per-replicate differences (non-reversible minus reversible) of single-iterate test
accuracy with the paired $t$ over the $R$ replicates; then the same comparison on four
sampler seeds that took no part in the search, at the pre-registered iteration. `CONFIRMED`
means every held-out seed positive and pooled $t>2$."""))

M.append(("code", '''def paired(nr, rev):
    d = nr - rev; se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d.mean() / se if se > 0 else np.nan

rows = []
for tag, c in CASES.items():
    rev, nr = results[tag]["Reversible anchored Langevin"], results[tag]["Non-reversible anchored Langevin"]
    for k in [k for k in (20, 50, 100, 150, 200, 300, 500, 1000, 2000) if k < c["n_iterations"]] + [c["n_iterations"]]:
        i = int(np.argmin(np.abs(rev["checkpoints"] - k))); m, se, t = paired(nr["test"][i], rev["test"][i])
        rows.append({"case": tag, "iteration": int(rev["checkpoints"][i]), "reversible": rev["test"][i].mean(),
                     "non_reversible": nr["test"][i].mean(), "paired_diff": m, "paired_se": se, "t_stat": t})
display(pd.DataFrame(rows))

held = []
for tag, c in CASES.items():
    P, diffs, ses = problems[tag], [], []
    for off in HELD_OUT:
        streams = shared_streams(P["geometry"], c["evaluate_at"] + CHECKPOINT, seed_offset=off)     # only up to the evaluation iteration
        rev = run_chain(0.0, P, c["s"], c["eta"], c["evaluate_at"], streams, c.get("s_warmup", 0)); nr = run_chain(1.0, P, c["s"], c["eta"], c["evaluate_at"], streams, c.get("s_warmup", 0))
        m, se, t = paired(nr["test"][-1], rev["test"][-1]); diffs.append(m); ses.append(se)
        held.append({"case": tag, "seed_offset": off, "iteration": c["evaluate_at"], "reversible": rev["test"][-1].mean(),
                     "non_reversible": nr["test"][-1].mean(), "paired_diff": m, "t_stat": t})
    pooled, pooled_se = np.mean(diffs), np.sqrt(np.sum(np.square(ses))) / len(ses)
    print(f"{tag:13s} held-out pooled diff {pooled:+.4f} at iteration {c['evaluate_at']} (t = {pooled / pooled_se:+.1f}), all positive: "
          f"{all(d > 0 for d in diffs)}  ->  {'CONFIRMED' if all(d > 0 for d in diffs) and pooled / pooled_se > 2 else 'NOT confirmed'}")
display(pd.DataFrame(held))'''))

M.append(("markdown", r"""### Where the win lives — and where it does not

* **The mechanism** (synthetic study, `results_beat/`): inside a triple, $J_I=[v_I]_\times$
  rotates only in the plane perpendicular to its axis; a slow posterior direction in that
  plane with a fast partner has its rate lifted from $\lambda_{\rm slow}$ to
  $(\lambda_{\rm fast}+\lambda_{\rm slow})/2$ once $|v_I|$ exceeds $\sigma^*$, while a slow direction
  along the axis is untouched. The accuracy gap is large only if the accelerated direction
  also carries signal. The parametrisation of section 2 builds exactly this.
* **Without it** (`results_real/summary.md`, 366 configurations): standardised features
  tie on both data sets (best +0.018, $t\approx2$); equalised coefficients tie; raw MAGIC
  scales give +0.035 ($t=4$) — the anisotropy is there but the slow coordinates sit along
  the $L_1$ axis.
* **It is a convergence-speed effect**: both chains converge to the reference accuracy
  (dotted line); the gap at the end of the runs is $0\pm0.001$ on MAGIC and closes on
  Titanic once the reversible chain has caught up.
* **Past the stability edge** ($s$ too large for the chosen $\eta$) the projection fires on
  more than half of the steps and the non-reversible chain is worse; the windows are the
  ones the synthetic parameter map predicts, shifted by the data set's curvature.
* Titanic's `age` is median-imputed over all 891 rows before the split (a routine, tiny
  leak that affects both samplers identically)."""))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src) for kind, src in M]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    nb = build(); nbf.validate(nb)
    nbf.write(nb, "real_data_nonreversible_vs_reversible.ipynb")
    print("wrote real_data_nonreversible_vs_reversible.ipynb", f"({len(nb.cells)} cells)")
