#!/usr/bin/env python3
"""Section 3.3 of the SRNSGLD paper, with a lasso regularizer and anchored Langevin.

The paper's Section 3.3 samples a constrained Bayesian logistic posterior whose
only regularisation is the constraint set, by projected SGLD (``J = 0``) against
skew-reflected non-reversible SGLD with a constant ``J_a`` and with a
state-dependent ``J_s(x)`` (centred ball) or ``J_g(x)`` (smoothed ``l_p``
sublevel set).  Its experiments are reproduced here with two changes:

1. a **lasso** term ``lam |x|_1`` is added to the potential, which makes the
   target non-differentiable exactly where the posterior puts mass;
2. the dynamics become **anchored Langevin** -- follow the gradient of a smooth
   majorising anchor ``U_0`` and scale the diffusion by the clock
   ``a = exp(U - U_0)`` -- so the kinked target is still the invariant law.
   With ``J = 0`` that is anchored PSGLD; with ``J != 0`` it is *non-reversible*
   anchored Langevin, and the skew projection at the boundary makes it
   skew-reflected.

Everything else follows the paper: the same datasets and dimensions, the same
constraint sets, the same uniform-on-the-unit-ball initialisation, the same step
size, batch size, walker count, iteration counts and the same ``a`` and ``s``.
Accuracy is the paper's metric -- each walker's own parameter classifies the
training and test sets, and the figures show the mean and standard deviation
across walkers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored_constrained import (  # noqa: E402
    Ball,
    ConstantField,
    LassoLogistic,
    SmoothedLpBall,
    ZeroField,
    ball_axial_field,
    outward_tilt_direction,
    reference_constrained_rwm,
    run_anchored_srnsgld,
    sublevel_axial_field,
    tilted_axial_field,
)
from nds.data import load_nine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# The paper's settings, Section 3.3.  ``a`` and ``s`` are its own choices; the
# ``lp`` entries take the constant amplitude from the figure legends, which use
# a = 2 there, and the ball entries from the text.
# The lasso strength, as a multiple of ``n_train`` because the potential is a
# sum over data points.  Per problem, the largest value on the ladder
# {0.01, 0.03, 0.1, 0.3} that satisfies all three of: the paper's step size
# still resolves the kink (``eta lam <= 0.02``); the clock does not collapse
# (mean clock at the constrained lasso MAP above 0.2); and the constrained
# lasso MAP still lies on the boundary of ``K``, so that the constraint is
# active as it is in the paper -- a stronger lasso shrinks the posterior
# strictly inside ``K`` and the skew reflection stops mattering at all.
# ``experiments/calibrate_anchored_lasso.py`` prints the ladder.
LAM_SCALE = {"synthetic": 0.01, "magic": 0.01, "titanic": 0.01}

SETTINGS = {
    "synthetic": dict(
        pretty="Synthetic ($d$ = 3)", d=3, n=2000, radius_sq=1.0,
        step_size=1e-4, batch=50, n_iter=1000, n_walkers=100,
        ball=dict(a=1.0, s=10.0),
        lp=dict(a=1.0, s=10.0, p=4.0, eps=0.2, level=1.0, n_iter=1000),
    ),
    "magic": dict(
        pretty="MAGIC Gamma Telescope", d=9, radius_sq=2.0,
        step_size=1e-4, batch=30, n_iter=1000, n_walkers=100,
        ball=dict(a=2.0, s=[5.0, 5.0, 5.0]),
        lp=dict(a=2.0, s=[5.0, 5.0, 5.0], p=2.4, eps=0.2, level=4.0, n_iter=1000),
    ),
    "titanic": dict(
        pretty="Titanic (survival)", d=9, radius_sq=2.0,
        step_size=1e-4, batch=30, n_iter=1500, n_walkers=100,
        ball=dict(a=3.0, s=[2.0, 7.0, 2.0]),
        lp=dict(a=2.0, s=[2.0, 7.0, 2.0], p=2.4, eps=0.18, level=4.0, n_iter=2000),
    ),
}


def synthetic_data(n: int, seed: int, beta=(1.0, -0.7, 0.0), test_fraction=0.2) -> dict:
    """The paper's generator: ``X ~ N(0, 2 I_3)``, ``y = 1{p <= sigmoid(b . X)}``.

    The paper does not state the regression coefficient it generates from.  Its
    Section 3.2 uses ``x* = [1, -0.7, -0.5]``; here the last coordinate is set
    to zero instead, so that one feature is genuinely irrelevant and the lasso
    has something to find.  The first two are unchanged, and ``|beta| = 1.22``
    is still outside the constraint ball of radius 1, which mirrors the paper's
    figures where the truth sits outside ``K``.
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, float)
    X = rng.normal(scale=np.sqrt(2.0), size=(n, len(beta)))
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = (rng.random(n) <= p).astype(float)
    cut = int(round(test_fraction * n))
    return {
        "X_train": X[cut:], "y_train": y[cut:],
        "X_test": X[:cut], "y_test": y[:cut],
        "beta_star": beta,
    }



def field_operator_norms(dcfg: dict, d: int, radius: float) -> dict:
    """Operator norms of the paper's fields at ``|x| = radius``, at ``kappa = 1``.

    The tridiagonal ``J_a`` has eigenvalues ``2 i a cos(k pi / (d+1))``, so its
    norm is ``2 a cos(pi / (d+1))``.  A hat map of ``k`` has norm ``|k|``, so
    ``J_s`` has norm ``max_l s_l |x_l|``, which at radius ``r`` with ``m`` equal
    blocks is about ``s r / sqrt(m)``.
    """
    m = d // 3
    s = np.atleast_1d(np.asarray(dcfg["s"], float))
    return {
        "constant": 2.0 * dcfg["a"] * np.cos(np.pi / (d + 1)),
        "state": float(s.max() * radius / np.sqrt(m)),
    }


def calibrated_amplitude_scale(
    target, domain, dcfg, eta: float, n_walkers: int, start_radius: float,
    budget: float, seed: int,
) -> float:
    r"""Scale the paper's amplitudes so the rotation moves a bounded distance.

    The paper's ``a`` and ``s`` are dimensionless -- in ``-(I + J) grad f`` they
    say how large the rotational part of the drift is *relative to* the gradient
    part -- but whether that is usable depends on how far one step actually
    moves.  The rotational displacement of a single step is
    ``eta |J| |grad f|``, and this rule keeps it under ``budget`` times the
    radius of the constraint set, measured at the initial ensemble:

        kappa = min(1, budget * r / (eta max(|J_a|, |J_s|) E|grad U_0|)).

    At the paper's own amplitudes that displacement is a multiple of the whole
    constraint set on the real datasets, because their potential is a sum over
    ``n`` data points; the rule is what makes the comparison mean anything.
    """
    rng = np.random.default_rng(seed)
    W0 = domain.uniform(target.d, n_walkers, rng, radius=start_radius)
    g0 = float(np.mean(np.sqrt((target.anchor_grad(W0, np.arange(target.n)) ** 2).sum(axis=0))))
    radius = getattr(domain, "radius", None)
    if radius is None:  # sublevel set: use the mean radius of the initial ensemble
        radius = float(np.mean(np.sqrt((W0 * W0).sum(axis=0))))
    norms = field_operator_norms(dcfg, target.d, radius)
    worst = max(norms.values())
    return float(min(1.0, budget * radius / (eta * worst * g0)))

def build(problem: str, domain_key: str, args) -> tuple:
    cfg = SETTINGS[problem]
    dcfg = cfg[domain_key]
    if problem == "synthetic":
        data = synthetic_data(cfg["n"], args.split_seed)
    else:
        ds = load_nine(problem, seed=args.split_seed, test_fraction=0.2)
        data = {
            "X_train": ds.X_train, "y_train": ds.y_train,
            "X_test": ds.X_test, "y_test": ds.y_test,
        }
    d = data["X_train"].shape[1]
    n_train = len(data["y_train"])

    if domain_key == "ball":
        domain = Ball(cfg["radius_sq"], squared=True)
    else:
        domain = SmoothedLpBall(dcfg["p"], dcfg["eps"], dcfg["level"])

    scale = args.lam_scale if args.lam_scale is not None else LAM_SCALE[problem]
    lam = args.lam if args.lam is not None else scale * n_train
    delta = args.delta if args.delta is not None else args.clock_budget / max(lam, 1e-12)
    target = LassoLogistic(data["X_train"], data["y_train"], lam=lam, delta=delta)

    kappa = args.amp_scale
    if kappa is None:
        kappa = calibrated_amplitude_scale(
            target, domain, dcfg, args.step_size or cfg["step_size"],
            cfg["n_walkers"], args.start_radius, args.rotation_budget, args.seed,
        )
    # per-field overrides, so a tuned pair can be run without touching the rest
    kappa_a = args.amp_scale_constant if args.amp_scale_constant is not None else kappa
    kappa_s = args.amp_scale_state if args.amp_scale_state is not None else kappa
    a = dcfg["a"] * kappa_a
    s_amp = (np.atleast_1d(np.asarray(dcfg["s"], float)) * kappa_s).tolist()
    fields = [ZeroField(d), ConstantField(d, a)]
    fields[0].label = r"$J = 0$ (anchored PSGLD)"
    fields[1].label = rf"constant $J_a$, $a$ = {a:.3g}"
    name = "J_s" if domain_key == "ball" else "J_g"
    if args.tilt:
        direction = outward_tilt_direction(
            target, domain, start_radius=args.start_radius, seed=args.seed
        )
        state = tilted_axial_field(
            d, s_amp, domain, direction, args.tilt,
            p=2.0 if domain_key == "ball" else dcfg["p"],
            eps=0.0 if domain_key == "ball" else dcfg["eps"],
        )
        state.label = (
            rf"tilted ${name}(x)$, $s$ = {[round(v, 3) for v in s_amp]}, "
            rf"$c$ = {args.tilt:g}"
        )
    elif domain_key == "ball":
        state = ball_axial_field(d, s_amp)
        state.label = rf"state-dependent $J_s(x)$, $s$ = {[round(v, 3) for v in s_amp]}"
    else:
        state = sublevel_axial_field(d, s_amp, dcfg["p"], dcfg["eps"])
        state.label = rf"state-dependent $J_g(x)$, $s$ = {[round(v, 3) for v in s_amp]}"
    fields.append(state)
    kappa = {"constant": kappa_a, "state": kappa_s}
    return cfg, dcfg, data, domain, target, fields, kappa


def reference(problem, domain_key, target, domain, data, args) -> dict:
    tag = f"{problem}_{domain_key}_lam{target.lam:.4g}"
    path = os.path.join(RESULTS, f"reference_anchored_{tag}.npz")
    if os.path.exists(path) and not args.refresh_reference:
        return dict(np.load(path))
    started = time.time()
    ref = reference_constrained_rwm(
        target, domain, data, n_iter=args.ref_iter, n_warmup=args.ref_warmup,
        thin=args.ref_thin, seed=args.seed,
    )
    print(
        f"    reference: train {ref['accuracy_train']:.4f} test {ref['accuracy_test']:.4f} "
        f"(halves {ref['accuracy_test_halves'][0]:.4f}/{ref['accuracy_test_halves'][1]:.4f}) "
        f"acceptance {ref['acceptance']:.2f} |mean| {ref['mean_radius']:.3f} "
        f"near-zero {ref['n_near_zero']}/{target.d}  {time.time() - started:.0f}s",
        flush=True,
    )
    np.savez_compressed(path, **ref)
    return ref


def run(problem: str, domain_key: str, args) -> dict:
    started = time.time()
    cfg, dcfg, data, domain, target, fields, kappa = build(problem, domain_key, args)
    n_iter = args.n_iter or dcfg.get("n_iter", cfg["n_iter"])
    eta = args.step_size or cfg["step_size"]
    print(
        f"[{problem}/{domain_key}] d={target.d} n_train={target.n} "
        f"n_test={len(data['y_test'])} {domain.key} lam={target.lam:.3g} "
        f"delta={target.delta:.4g} clock floor {target.clock_floor:.3g} "
        f"eta={eta:.1e} batch={cfg['batch']} iters={n_iter} walkers={cfg['n_walkers']} "
        f"eta*lam={eta * target.lam:.4f} "
        f"kappa={kappa['constant']:.4g}/{kappa['state']:.4g} tilt={args.tilt:g}",
        flush=True,
    )
    ref = reference(problem, domain_key, target, domain, data, args)

    ref_mean = np.asarray(ref["posterior_mean"], float)
    ref_metric = np.linalg.pinv(np.asarray(ref["posterior_cov"], float))
    runs = []
    for field in fields:
        out = run_anchored_srnsgld(
            target, field, domain, data, n_iter=n_iter,
            n_walkers=cfg["n_walkers"], step_size=eta, batch_size=cfg["batch"],
            seed=args.seed, start_radius=args.start_radius, anchored=not args.no_clock,
            reference_mean=ref_mean, reference_metric=ref_metric,
            score_every=args.score_every,
        )
        runs.append(out)
        print(
            f"    {field.key:9s} train {out['accuracy_train'][-1].mean():.4f} "
            f"test {out['accuracy_test'][-1].mean():.4f} "
            f"+/- {out['accuracy_test'][-1].std():.4f}  "
            f"error {out['mean_error']:.3f}  "
            f"clock {out['clock'].mean():.3f}  boundary {out['boundary_rate']:.3f}  "
            f"failed {out['failed_retractions']}",
            flush=True,
        )

    tag = f"{problem}_{domain_key}" + args.tag_suffix
    np.savez_compressed(
        os.path.join(RESULTS, f"traces_anchored_{tag}.npz"),
        **{f"{r['key']}_{k}": r[k] for r in runs
           for k in ("accuracy_train", "accuracy_test", "clock", "running_error")},
        scored_at=runs[0]["scored_at"],
        **{f"{r['key']}_posterior_mean": r["posterior_mean"] for r in runs},
    )
    meta = {
        "problem": problem,
        "pretty": cfg["pretty"],
        "domain": domain_key,
        "domain_label": domain.label,
        "d": int(target.d),
        "n_train": int(target.n),
        "n_test": int(len(data["y_test"])),
        "lam": target.lam,
        "lam_scale": target.lam / target.n,
        "delta": target.delta,
        "clock_floor": target.clock_floor,
        "step_size": eta,
        "eta_lam": eta * target.lam,
        "batch_size": cfg["batch"],
        "n_iter": int(n_iter),
        "score_every": args.score_every,
        "n_walkers": cfg["n_walkers"],
        "start_radius": args.start_radius,
        "anchored": not args.no_clock,
        "amp_scale": kappa,
        "tilt": args.tilt,
        "amplitudes": {
            "a": dcfg["a"] * kappa["constant"],
            "s": (np.atleast_1d(np.asarray(dcfg["s"], float)) * kappa["state"]).tolist(),
        },
        "operator_norms": field_operator_norms(dcfg, target.d, getattr(domain, "radius", 1.0)),
        "reference": {
            "accuracy_train": float(ref["accuracy_train"]),
            "accuracy_test": float(ref["accuracy_test"]),
            "accuracy_test_sd": float(ref["accuracy_test_sd"]),
            "accuracy_test_halves": np.asarray(ref["accuracy_test_halves"]).tolist(),
            "acceptance": float(ref["acceptance"]),
            "posterior_mean": np.asarray(ref["posterior_mean"]).tolist(),
            "mean_radius": float(ref["mean_radius"]),
            "n_near_zero": int(ref["n_near_zero"]),
        },
        "runs": [
            {
                "key": r["key"],
                "label": r["label"],
                "train_final": float(r["accuracy_train"][-1].mean()),
                "test_final": float(r["accuracy_test"][-1].mean()),
                "test_final_sd": float(r["accuracy_test"][-1].std()),
                "mean_error": float(r["mean_error"]),
                "time_mean": r["time_mean"].tolist(),
                "clock_mean": float(r["clock"].mean()),
                "boundary_rate": r["boundary_rate"],
                "failed_retractions": r["failed_retractions"],
                "posterior_mean": r["posterior_mean"].tolist(),
                "n_near_zero": int((np.abs(r["time_mean"]) < 0.01).sum()),
            }
            for r in runs
        ],
        "runtime_seconds": time.time() - started,
    }
    with open(os.path.join(RESULTS, f"summary_anchored_{tag}.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[{problem}/{domain_key}] done in {meta['runtime_seconds']:.0f}s", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["synthetic", "magic", "titanic"])
    parser.add_argument("--domains", nargs="+", default=["ball", "lp"])
    parser.add_argument("--lam", type=float, default=None,
                        help="lasso strength; the default is --lam-scale times n_train, "
                             "since the potential is a sum over data points")
    parser.add_argument("--lam-scale", type=float, default=None,
                        help="override the per-problem lasso strength in LAM_SCALE")
    parser.add_argument("--delta", type=float, default=None,
                        help="anchor smoothing; the default is --clock-budget / lam, "
                             "which caps the clock's worst case at exp(-budget d)")
    parser.add_argument("--clock-budget", type=float, default=0.5)
    parser.add_argument("--amp-scale", type=float, default=None,
                        help="scales the paper's a and s together; 1.0 is the paper's "
                             "own amplitudes, and the default calibrates it so that one "
                             "step's rotational displacement is at most --rotation-budget "
                             "times the radius")
    parser.add_argument("--rotation-budget", type=float, default=0.1)
    parser.add_argument("--amp-scale-constant", type=float, default=None,
                        help="override --amp-scale for the constant field only")
    parser.add_argument("--amp-scale-state", type=float, default=None,
                        help="override --amp-scale for the state-dependent field only")
    parser.add_argument("--tilt", type=float, default=0.0,
                        help="non-radial h in the paper's recipe, h = 1 + c (u . x), with "
                             "u the closed-form outward direction; 0 is the paper's h = 1")
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--score-every", type=int, default=1,
                        help="score accuracy every k iterations; the figures read this, "
                             "so keep it small unless the runs are long")
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--start-radius", type=float, default=1.0,
                        help="walkers start uniform on the centred ball of this radius")
    parser.add_argument("--no-clock", action="store_true",
                        help="drop the clock, leaving the smoothed-anchor chain")
    parser.add_argument("--ref-iter", type=int, default=100000)
    parser.add_argument("--ref-warmup", type=int, default=20000)
    parser.add_argument("--ref-thin", type=int, default=20)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--tag-suffix", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    for problem in args.problems:
        for domain_key in args.domains:
            run(problem, domain_key, args)


if __name__ == "__main__":
    main()
