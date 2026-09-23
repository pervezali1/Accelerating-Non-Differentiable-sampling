"""One paired NR-vs-REV comparison on a REAL data set; prints the same JSON as hunt_helper.

    python real_hunt.py '{"dataset":"titanic","scaling":"standard","s":4,"eta":1e-4,"geometry":"l1"}'

Data (in ./data): MAGIC gamma telescope (19020 x 10, PMLB mirror of UCI magic04) and
Titanic (891 x 8 engineered features).  A column of ones is prepended for the
intercept and zero columns are appended so that d is a multiple of 3 (the block
structure of J); a zero column costs nothing (its coefficient is pulled to 0 by g).

Config keys beyond hunt_helper's (s, eta, lambda, delta, epsilon, n_iter, R, eval_at,
checkpoint, init, observable, held_out, geometry):
  dataset      "titanic" | "magic"
  n_train_sub  subsample the training set to this many rows (speed)   (all)
  scaling      "standard" (z-score) | "raw" (untouched) |
               "equalise" (z-score, then scale each feature by |beta_hat_j| so every
                           coefficient has the same size: a democratic L1 axis) |
               "design"   (z-score, then a per-block linear reparametrisation that gives
                           each full slope triple the synthetic-experiment structure
                           v_axis q1q1' + v_fast q2q2' + v_slow q3q3' and coefficients
                           proportional to (b, e, e); logits are unchanged)
  v_axis, v_fast, v_slow, b, e   parameters of "design"            (1, 64, 2, 1.5, 0.25)
  align        flip feature signs so every pilot coefficient is positive   (true)
  order        "natural" | "importance" (largest |beta_hat| first) |
               "design" (the six most informative features go to the two full triples) |
               explicit list of feature indices
  radius_margin  l1_radius = |beta_hat|_1 + margin                        (1.0)
  ball_target    for geometry=ball, features are scaled so |beta_hat|_2 = this   (0.8)
  s_warmup       ramp the block strength linearly from 0 over this many iterations (0)
The pilot coefficients beta_hat come from scikit-learn logistic regression (C = 100)
on the training rows in the FINAL coordinates; they are used only for the
preprocessing choices and as the 'beta_true' reference (antipodal init, reference
accuracy).  Both samplers see identical data.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import anchored_lasso as lasso
import anchored_sgld as nral
from hunt_helper import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = np.ones(3) / np.sqrt(3.0)
Q2 = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
Q3 = np.array([2.0, -1.0, -1.0]) / np.sqrt(6.0)


def load(name: str):
    if name == "titanic":
        t = pd.read_csv(os.path.join(HERE, "data", "titanic.csv"))
        X = pd.DataFrame({
            "pclass": t.pclass.astype(float), "sex": (t.sex == "female").astype(float),
            "age": t.age.fillna(t.age.median()), "sibsp": t.sibsp.astype(float), "parch": t.parch.astype(float),
            "log_fare": np.log1p(t.fare), "emb_C": (t.embarked == "C").astype(float), "emb_Q": (t.embarked == "Q").astype(float)})
        return X.values.astype(float), t.survived.values.astype(float), list(X.columns)
    if name == "magic":
        m = pd.read_csv(os.path.join(HERE, "data", "magic.tsv.gz"), sep="\t")
        X = m.drop(columns="target")
        return X.values.astype(float), m.target.values.astype(float), list(X.columns)
    raise ValueError(name)


def pilot_fit(X, y):
    lr = LogisticRegression(C=100.0, max_iter=5000).fit(X, y)
    return lr.coef_[0].copy(), float(lr.intercept_[0]), lr


def sqrtm(C):
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def design_transform(Xtr_block, Xte_block, beta_block, v_axis, v_fast, v_slow, b, e):
    """Linear map A with cov(A x) = v_axis q1q1' + v_fast q2q2' + v_slow q3q3' and A^{-T} beta prop. to (b, e, e)."""
    C = np.cov(Xtr_block.T)
    C_half, C_inv_half = sqrtm(C)
    Sigma = v_axis * np.outer(Q1, Q1) + v_fast * np.outer(Q2, Q2) + v_slow * np.outer(Q3, Q3)
    S_half, _ = sqrtm(Sigma)
    u = C_half @ beta_block                                  # whitened coefficients: |u|^2 = the block's logit variance
    target = np.array([b, e, e])
    r = S_half @ target
    r = r * (np.linalg.norm(u) / np.linalg.norm(r))          # same norm as u
    u_hat, r_hat = u / np.linalg.norm(u), r / np.linalg.norm(r)
    v = u_hat - r_hat
    W = np.eye(3) if np.linalg.norm(v) < 1e-12 else np.eye(3) - 2.0 * np.outer(v, v) / (v @ v)   # Householder: W u_hat = r_hat
    A = S_half @ W @ C_inv_half
    return Xtr_block @ A.T, Xte_block @ A.T


def build(cfgd):
    name = cfgd.get("dataset", "titanic")
    X, y, names = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=2027, stratify=y, shuffle=True)
    n_sub = cfgd.get("n_train_sub")
    if n_sub and n_sub < len(ytr):
        keep = np.random.default_rng(2028).choice(len(ytr), int(n_sub), replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    scaling = cfgd.get("scaling", "standard")
    if scaling != "raw":
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    beta_hat, _, _ = pilot_fit(Xtr, ytr)
    if cfgd.get("align", True):                              # all pilot coefficients positive
        flip = np.where(beta_hat < 0, -1.0, 1.0)
        Xtr, Xte, beta_hat = Xtr * flip, Xte * flip, beta_hat * flip
    order = cfgd.get("order", "natural")
    p = X.shape[1]
    if order == "natural":
        idx = list(range(p))
    elif order == "importance":
        idx = list(np.argsort(-np.abs(beta_hat)))
    elif order == "design":                                  # six most informative -> the two full triples (slopes 3..8)
        ranked = list(np.argsort(-np.abs(beta_hat)))
        top6, rest = ranked[:6], ranked[6:]
        idx = rest[:2] + top6 + rest[2:]
    else:
        idx = [int(i) for i in order]
    Xtr, Xte, beta_hat, names = Xtr[:, idx], Xte[:, idx], beta_hat[idx], [names[i] for i in idx]
    if scaling == "equalise":
        c = np.abs(beta_hat) / np.abs(beta_hat).mean()
        Xtr, Xte, beta_hat = Xtr * c, Xte * c, beta_hat / c
    d = 1 + p
    n_pad = (-d) % 3
    Xtr = np.hstack([Xtr, np.zeros((len(ytr), n_pad))]); Xte = np.hstack([Xte, np.zeros((len(yte), n_pad))])
    names = names + [f"pad{i}" for i in range(n_pad)]
    beta_hat = np.concatenate([beta_hat, np.zeros(n_pad)])
    if scaling == "design":
        # slopes occupy columns 0..p+n_pad-1 of Xtr here; full triples are slope blocks (2,3,4), (5,6,7), ... not touching the pads
        for start in range(2, p + n_pad - 2, 3):
            block = [start, start + 1, start + 2]
            if any(bi >= p for bi in block):
                continue
            Xtr[:, block], Xte[:, block] = design_transform(
                Xtr[:, block], Xte[:, block], beta_hat[block], float(cfgd.get("v_axis", 1.0)), float(cfgd.get("v_fast", 64.0)),
                float(cfgd.get("v_slow", 2.0)), float(cfgd.get("b", 1.5)), float(cfgd.get("e", 0.25)))
    # final pilot fit in the final coordinates (intercept separate), reference accuracy
    coef, intercept, lr = pilot_fit(Xtr[:, :p + n_pad], ytr) if n_pad == 0 else pilot_fit(Xtr[:, :p], ytr)
    beta_ref = np.concatenate([[intercept], coef, np.zeros(n_pad)])
    geometry_name = cfgd.get("geometry", "l1")
    X_train = np.hstack([np.ones((len(ytr), 1)), Xtr]); X_test = np.hstack([np.ones((len(yte), 1)), Xte])
    if geometry_name == "ball":                              # unit ball: scale ALL columns (intercept included) by k so that
        k = np.linalg.norm(beta_ref) / float(cfgd.get("ball_target", 0.8))   # beta_ref / k fits inside; logits unchanged
        X_train, X_test, beta_ref = X_train * k, X_test * k, beta_ref / k
    s = cfgd.get("s", 4.0)
    n_blocks = (d + n_pad) // 3
    scales = tuple(float(x) for x in (s if isinstance(s, list) else [s] * n_blocks))
    cfg = lasso.LassoConfig(
        d=d + n_pad, lambda_lasso=float(cfgd.get("lambda", 2.0)), delta_anchor=float(cfgd.get("delta", 0.02)),
        eta=float(cfgd.get("eta", 1e-4)), n_repeats=int(cfgd.get("R", 40)), n_iterations=int(cfgd.get("n_iter", 1000)),
        checkpoint_every=int(cfgd.get("checkpoint", 10)), block_scales=scales, epsilon=float(cfgd.get("epsilon", 0.2)),
        l1_radius=float(np.abs(beta_ref).sum() + float(cfgd.get("radius_margin", 1.0))), n_total=len(y),
        scale_warmup=int(cfgd.get("s_warmup", 0)),
    )
    ds = lasso.Dataset(np.ascontiguousarray(X_train), ytr, np.ascontiguousarray(X_test), yte, beta_ref)
    tg = lasso.LassoTarget(ds.X_train, ds.y_train, cfg.lambda_lasso, cfg.sigma_intercept, cfg.delta_anchor,
                           smoothing=cfgd.get("smoothing", "sqrt"))
    geom = nral.BallGeometry(cfg.d) if geometry_name == "ball" else nral.L1SmoothBallGeometry(cfg.d, cfg.epsilon, cfg.l1_radius)
    sk_acc = float((((ds.X_test @ beta_ref) > 0) == (yte > 0.5)).mean())
    info = {"dataset": name, "d": cfg.d, "n_train": int(len(ytr)), "n_test": int(len(yte)), "features": names,
            "beta_ref": [round(float(v), 4) for v in beta_ref], "reference_test_accuracy": sk_acc,
            "feature_sd": [round(float(v), 3) for v in X_train[:, 1:].std(axis=0)], "feasible": bool(geom.feasible(beta_ref)),
            "l1_radius": cfg.l1_radius, "a_lower_bound": tg.a_lower_bound, "eta_L": float(cfg.eta * tg.lipschitz_constant())}
    return cfg, ds, tg, geom, info


def main():
    cfgd = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cfg, ds, tg, geom, info = build(cfgd)
    result = evaluate(cfg, ds, tg, geom, cfgd)
    result["info"] = info
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
