#!/usr/bin/env python3
"""Is the speed-up bigger than the run-to-run noise?

The curves in ``plot_srnsgld.py`` are one walker seed each.  This repeats the
three fields over several seeds -- the same seed for all three fields, so the
mini-batch stream and the Brownian increments are common random numbers -- and
reports the ratio of whitened errors at a fixed early iteration, with its spread
over seeds.  The reference posterior and the step size are the cached ones, so
this only re-runs the samplers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.constrained import (  # noqa: E402
    Ball,
    MinibatchLogistic,
    curvature_step_size,
    field_from_rho,
    run_constrained_sgld,
)
from nds.data import PRETTY_NAMES, load_nine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--start-radius", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    lines = [
        "# The same comparison over several seeds",
        "",
        f"Whitened error of the running posterior mean at iteration {args.n_iter}, "
        f"{args.n_walkers} walkers,",
        f"rotation strength rho = {args.rho:g}, {len(args.seeds)} seeds, each seed shared by all",
        "three fields (common random numbers).  `ratio` is the `J = 0` error over this",
        "field's, seed by seed, quoted as mean +/- standard error.",
        "",
    ]
    for name in args.datasets:
        ds = load_nine(name, seed=args.split_seed)
        model = MinibatchLogistic(ds.X_train, ds.y_train)
        ball = Ball(args.radius)
        h, _ = curvature_step_size(model, ball)
        meta = json.load(open(os.path.join(RESULTS, f"summary_{name}_srnsgld.json")))
        ref_mean = np.asarray(meta["reference"]["posterior_mean"], float)
        ref_metric = np.linalg.pinv(
            np.asarray(dict(np.load(os.path.join(
                RESULTS, f"reference_ball_{name}_r{ball.radius:g}.npz")))["posterior_cov"], float)
        )
        batch = meta["batch_size"]

        errors = {}
        for key in ("zero", "constant", "state"):
            rho = 0.0 if key == "zero" else args.rho
            field = field_from_rho(key, model.d, rho, ball.radius)
            errors[key] = [
                run_constrained_sgld(
                    model, field, ball, ds.X_test, ds.y_test, n_iter=args.n_iter,
                    n_walkers=args.n_walkers, step_size=h, batch_size=batch, seed=s,
                    start_radius=args.start_radius, reference_mean=ref_mean,
                    reference_metric=ref_metric,
                )["mean_error"][-1]
                for s in args.seeds
            ]

        lines += [
            f"## {PRETTY_NAMES.get(name, name)}",
            "",
            "| field | error | ratio to `J = 0` |",
            "|---|---|---|",
        ]
        base = np.asarray(errors["zero"])
        for key in ("zero", "constant", "state"):
            e = np.asarray(errors[key])
            ratio = base / e
            se = ratio.std(ddof=1) / np.sqrt(len(ratio))
            lines.append(
                f"| `{key}` | {e.mean():.4f} +/- {e.std(ddof=1) / np.sqrt(len(e)):.4f} | "
                f"{ratio.mean():.3f} +/- {se:.3f} |"
            )
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "srnsgld_seeds.md"), "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
