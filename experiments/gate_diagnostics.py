#!/usr/bin/env python3
"""Is the gate of ``J_s`` actually open where the chain ends up?

The state-dependent field is ``J_s(w) = s(w) J_a`` with a single scalar gate

    s(w) = 1 / (1 + exp((r(w) - R) / W)),   r(w)^2 = (w - m)^T H (w - m),

and ``run_designed.py`` keys ``R`` and ``W`` to ``r0``, the whitened distance
from the ``w = 0`` start to the warm-up mean: ``R = 0.75 r0``, ``W = 0.1 r0``.
That is a *relative* rule, and it silently assumes the cold start sits outside
the bulk.  In whitened coordinates the bulk of a ``d``-dimensional posterior
sits at radius ``sqrt(d)``, so the assumption is ``r0 >> sqrt(d)`` -- true for a
low-dimensional posterior whose mean is far from the origin, false once ``d``
grows and the origin is already inside the bulk.

This script measures which case each dataset is in: it rebuilds the same warm-up
geometry, then evaluates the gate on draws from the reference posterior, i.e.
where the chain actually lives after it has converged.  A mean gate value near 1
means the rotation is on in the bulk; a mean near 0 means ``J_s`` degenerated to
``J = 0`` running at the design's step size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.data import PRETTY_NAMES, load  # noqa: E402
from nds.design import paired_skew  # noqa: E402
from nds.sampler import warm_up  # noqa: E402
from nds.skew import GatedSkew  # noqa: E402
from nds.target import LogisticPosterior  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def measure(name: str, args) -> dict:
    ds = load(name, seed=args.split_seed)
    target = LogisticPosterior(ds.X_train, ds.y_train, prior_scale=args.prior_scale)
    pilot = warm_up(
        target, n_iter=args.warmup_iter, n_walkers=args.warmup_walkers,
        seed=args.seed, n_rounds=args.warmup_rounds, shrinkage=args.shrinkage,
    )
    precond, m_hat = pilot["geometry"], pilot["mean_state"]
    H_hat = precond.D_inv
    r0 = float(np.sqrt(max(m_hat @ (H_hat @ m_hat), 1e-12)))
    R, W = args.gate_fraction * r0, args.gate_width * r0

    J_a, _ = paired_skew(H_hat, amplitude=1.0)
    gate = GatedSkew(J_a, 1.0, radius=R, width=W, metric=H_hat, center=m_hat)

    ref = np.load(os.path.join(RESULTS, f"reference_{name}.npz"))
    mean, cov = ref["posterior_mean"], ref["posterior_cov"]
    rng = np.random.default_rng(args.seed)
    draws = rng.multivariate_normal(mean, cov, size=args.n_draws).T  # (d, n)

    s_bulk = gate.scale(draws)
    r_bulk = gate._radius(draws)[0]
    # the warm-up ensemble's own radii, as a second read on where the bulk is
    r_warm = gate._radius(pilot["states"])[0] if "states" in pilot else None

    row = {
        "dataset": name,
        "pretty": PRETTY_NAMES.get(name, name),
        "d": int(target.d),
        "sqrt_d": float(np.sqrt(target.d)),
        "r0_start_to_bulk": r0,
        "gate_radius": R,
        "gate_width": W,
        "bulk_radius_median": float(np.median(r_bulk)),
        "gate_mean_in_bulk": float(s_bulk.mean()),
        "gate_median_in_bulk": float(np.median(s_bulk)),
    }
    if r_warm is not None:
        row["warmup_radius_median"] = float(np.median(r_warm))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["titanic", "magic", "breast_cancer", "spambase"]
    )
    parser.add_argument("--n-draws", type=int, default=4000)
    parser.add_argument("--gate-fraction", type=float, default=0.75)
    parser.add_argument("--gate-width", type=float, default=0.1)
    parser.add_argument("--prior-scale", type=float, default=2.0)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--warmup-iter", type=int, default=600)
    parser.add_argument("--warmup-walkers", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    rows = [measure(name, args) for name in args.datasets]
    rows.sort(key=lambda r: r["d"])

    lines = [
        "# Is the gate open where the chain lives?",
        "",
        "`J_s(w) = s(w) J_a` with `s(w) = 1 / (1 + exp((r(w) - R) / W))`,",
        "`R = 0.75 r0`, `W = 0.1 r0`, `r0` = whitened distance from `w = 0` to the",
        "warm-up mean.  Radii are measured in the gate's own metric: centred at the",
        "warm-up mean `m`, in the warm-up inverse covariance `H`.  If that geometry",
        "matched the posterior the bulk would sit at radius `sqrt(d)`, so `sqrt(d)`",
        "is the column to compare the measured bulk radius against -- a bulk radius",
        "far above it means the warm-up never reached the bulk, and then `r0` is not",
        "the distance to the bulk at all.  The gate is open where the chain lives",
        "only if `r0` is comfortably larger than the bulk radius.  Gate values are",
        "averaged over draws from the reference posterior.",
        "",
        "| dataset | `d` | `r0` (start to bulk) | `sqrt(d)` | bulk radius (median) | `R` | `W` | mean `s` in the bulk |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['pretty']} | {r['d']} | {r['r0_start_to_bulk']:.2f} | {r['sqrt_d']:.2f} | "
            f"{r['bulk_radius_median']:.2f} | {r['gate_radius']:.2f} | {r['gate_width']:.2f} | "
            f"{r['gate_mean_in_bulk']:.3g} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "gate_in_the_bulk.md"), "w") as handle:
        handle.write(text)
    with open(os.path.join(RESULTS, "gate_in_the_bulk.json"), "w") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
