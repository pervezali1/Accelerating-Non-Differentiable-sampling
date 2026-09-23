"""One paired NR-vs-REV comparison from a JSON config; prints a JSON result.

Used by the parallel search so that every strategy runs the SAME verified
sampler and the SAME statistics.  Example:

    python hunt_helper.py '{"design":"block_aniso","spread":10,"lambda":2,"s":5,
                            "eta":7.87e-6,"geometry":"l1","n_iter":1000,"R":60,
                            "eval_at":1000,"held_out":[101,202]}'

Config keys (all optional except as noted):
  design      "isotropic" | "ar1" | "block_aniso" | "aligned"   (default isotropic)
  rho_x       AR(1) correlation for design=ar1                  (0.99)
  spread      eigenvalue spread for block_aniso: (spread,1,1/spread) (10)
  kappa       for aligned: slow-direction variance = 2/kappa     (100)
  variances   for design=scaled: 8 slope variances (unequal feature scales),
              or a full 8x8 slope covariance (any slow direction you like)
  beta        for design=scaled: the true 9-vector, intercept first
  epsilon     smoothing of the constraint (soft-sign width of grad g)  (0.2)
  init_scale  jitter around the origin for init=origin               (0.02)
  s_warmup    ramp s linearly from 0 over this many iterations        (0)
  smoothing   anchor smoothing of |w_j|: "sqrt" or "gaussian" (E|w_j + delta Z|)   (sqrt)
  lambda      lambda_lasso                                       (2)
  delta       delta_anchor                                       (0.02)
  s           block strength, scalar or [s1,s2,s3]               (5)
  eta         step size                                          (1e-5)
  geometry    "ball" | "l1"                                      (ball)
  l1_radius   L1-smooth radius budget                            (1.9)
  n_iter, R, eval_at, checkpoint                                 (1000,60,1000,20)
  n_total     total observations (train = 80%)                   (2000)
  init        "uniform" | "antipodal" (start at -beta_true dir on the boundary) | "origin"
  observable  "single" (accuracy of w_k) | "ergodic" (accuracy of the running-mean predictor)
  held_out    list of extra sampler-seed offsets to confirm on    ([])
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
from scipy.special import expit

import anchored_lasso as lasso
import anchored_sgld as nral


def build(cfgd):
    n_total = int(cfgd.get("n_total", 2000))
    s = cfgd.get("s", 5.0)
    scales = tuple(float(x) for x in (s if isinstance(s, list) else [s] * 3))
    cfg = lasso.LassoConfig(
        lambda_lasso=float(cfgd.get("lambda", 2.0)), delta_anchor=float(cfgd.get("delta", 0.02)),
        eta=float(cfgd.get("eta", 1e-5)), n_repeats=int(cfgd.get("R", 60)),
        n_iterations=int(cfgd.get("n_iter", 1000)), checkpoint_every=int(cfgd.get("checkpoint", 20)),
        block_scales=scales, l1_radius=float(cfgd.get("l1_radius", 1.9)), n_total=n_total,
        epsilon=float(cfgd.get("epsilon", 0.2)), scale_warmup=int(cfgd.get("s_warmup", 0)),
    )
    design = cfgd.get("design", "isotropic")
    if design == "isotropic":
        ds = lasso.make_dataset(cfg)
    elif design == "ar1":
        ds = lasso.make_dataset(replace(cfg, rho_x=float(cfgd.get("rho_x", 0.99))))
    elif design == "block_aniso":
        sp = float(cfgd.get("spread", 10.0)); ds = lasso.make_block_anisotropic_dataset(cfg, (sp, 1.0, 1.0 / sp))
    elif design == "aligned":
        ds = lasso.make_aligned_design_dataset(cfg, float(cfgd.get("kappa", 100.0)))
    elif design == "scaled":
        ds = lasso.make_scaled_dataset(cfg, cfgd["variances"], cfgd["beta"])
    else:
        raise ValueError(design)
    tg = lasso.LassoTarget(ds.X_train, ds.y_train, cfg.lambda_lasso, cfg.sigma_intercept, cfg.delta_anchor,
                           smoothing=cfgd.get("smoothing", "sqrt"))
    geom = (nral.BallGeometry(cfg.d) if cfgd.get("geometry", "ball") == "ball"
            else nral.L1SmoothBallGeometry(cfg.d, cfg.epsilon, cfg.l1_radius))
    return cfg, ds, tg, geom


def run_pair(cfg, ds, tg, geom, cfgd, offset):
    streams = lasso.make_streams(cfg, geom, offset)
    init = cfgd.get("init", "uniform")
    if init == "antipodal":
        direction = -ds.beta_true / np.linalg.norm(ds.beta_true)
        point = geom.boundary_point(direction) * 0.98
        streams.w_init[:] = point + 0.02 * np.random.default_rng(offset + 7).standard_normal(streams.w_init.shape)
        streams.w_init[:] = geom.project(streams.w_init).beta
    elif init == "origin":
        streams.w_init[:] = float(cfgd.get("init_scale", 0.02)) * np.random.default_rng(offset + 7).standard_normal(streams.w_init.shape)
        streams.w_init[:] = geom.project(streams.w_init).beta
    out = {}
    for name, alpha in lasso.METHODS:
        out[name] = lasso.run_chain(ds, tg, geom, streams, cfg, method=name, alpha=alpha)
    return out


def accuracy_series(run, ds, observable):
    if observable == "single":
        return run.test_accuracy                                  # (n_ckpt, R)
    # accuracy of the running-mean predictive probability, per replicate
    n_ckpt, R, _ = run.w.shape
    out = np.empty((n_ckpt, R))
    for r in range(R):
        acc = np.zeros(ds.X_test.shape[0])
        for k in range(n_ckpt):
            acc += expit(ds.X_test @ run.w[k, r])
            out[k, r] = (((acc / (k + 1)) >= 0.5) == (ds.y_test >= 0.5)).mean()
    return out


def evaluate(cfg, ds, tg, geom, cfgd):
    """Paired comparison + held-out confirmation for an already-built problem; returns the result dict."""
    observable = cfgd.get("observable", "single")
    eval_at = int(cfgd.get("eval_at", cfg.n_iterations))
    result = {"config": cfgd}

    runs = run_pair(cfg, ds, tg, geom, cfgd, 0)
    rev, nr = runs["Reversible anchored Langevin"], runs["Non-reversible anchored Langevin"]
    A_rev, A_nr = accuracy_series(rev, ds, observable), accuracy_series(nr, ds, observable)
    i = int(np.argmin(np.abs(rev.checkpoints - eval_at)))
    m, se, t = lasso.paired_difference(A_nr[i], A_rev[i])
    traj, peak = [], None
    for k in range(len(rev.checkpoints)):
        mm, _, tt = lasso.paired_difference(A_nr[k], A_rev[k])
        row = {"it": int(rev.checkpoints[k]), "rev": float(A_rev[k].mean()),
               "nr": float(A_nr[k].mean()), "diff": float(mm), "t": float(tt)}
        if peak is None or row["diff"] > peak["diff"]:
            peak = row
        traj.append(row)
    keep = set(np.unique(np.linspace(0, len(rev.checkpoints) - 1, 16).astype(int)))
    traj = [row for k, row in enumerate(traj) if k in keep]
    half = len(rev.checkpoints) // 2
    result.update({
        "eval_iteration": int(rev.checkpoints[i]),
        "rev_acc": float(A_rev[i].mean()), "nr_acc": float(A_nr[i].mean()),
        "paired_diff": float(m), "paired_se": float(se), "t": float(t),
        "projection_rate_nr": nr.projection_rate, "projection_rate_rev": rev.projection_rate,
        "nonfinite": int(nr.n_nonfinite + rev.n_nonfinite),
        "rev_still_rising": bool(A_rev[-1].mean() > A_rev[half].mean() + 1e-4),
        "bayes_ceiling": float(np.mean(np.maximum(expit(ds.X_test @ ds.beta_true), 1 - expit(ds.X_test @ ds.beta_true)))),
        "trajectory": traj,
        "peak": peak,
    })
    held = []
    for off in cfgd.get("held_out", []):
        r2 = run_pair(cfg, ds, tg, geom, cfgd, int(off))
        a_r, a_n = accuracy_series(r2["Reversible anchored Langevin"], ds, observable), \
                   accuracy_series(r2["Non-reversible anchored Langevin"], ds, observable)
        mm, ss, tt = lasso.paired_difference(a_n[i], a_r[i])
        held.append({"offset": int(off), "diff": float(mm), "se": float(ss), "t": float(tt)})
    if held:
        ds_ = np.array([h["diff"] for h in held]); ses = np.array([h["se"] for h in held])
        pse = float(np.sqrt((ses ** 2).sum()) / len(ses))
        result["held_out"] = held
        result["held_out_pooled_diff"] = float(ds_.mean())
        result["held_out_pooled_t"] = float(ds_.mean() / pse)
        result["held_out_all_positive"] = bool((ds_ > 0).all())
    return result


def main():
    cfgd = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cfg, ds, tg, geom = build(cfgd)
    print(json.dumps(evaluate(cfg, ds, tg, geom, cfgd), indent=1))


if __name__ == "__main__":
    main()
