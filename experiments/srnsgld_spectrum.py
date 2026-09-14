#!/usr/bin/env python3
"""Why the constant field beats the block-diagonal one, in one table.

Near the mode the drift of both samplers linearises to ``-(I + J(x*)) H``, so
the exponential rate of convergence is ``min Re mu`` over the eigenvalues of
``(I + J(x*)) H`` and the stability limit is the spectral radius.  A skew term
cannot add rate -- ``trace((I + J) H) = trace H`` -- it can only move it from
the fast directions to the slow one, and how much it can move depends on how
many pairs of directions it couples.

* The constant ``J_a`` is tridiagonal: it couples the nine coordinates in one
  connected chain ``x_1 - x_2 - ... - x_9`` and has a single null direction
  (odd ``d``).
* The state-dependent ``J_s`` is block diagonal: three disconnected triples,
  and each hat-map block annihilates its own coordinate vector, so there are
  three null directions and no coupling at all between blocks.

The table prints the predicted rate gain ``min Re mu / lambda_min`` beside the
measured convergence ratio from ``run_srnsgld.py``, because the point of the
table is that they agree.
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
)
from nds.data import PRETTY_NAMES, load_nine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def measured(name: str, probe: float = 0.1) -> tuple:
    """Whitened errors a tenth of the way into the sampling runs.

    Not the final errors: the rate being predicted is a rate of *convergence*,
    and by the last iteration every field that converges at all has, so the
    final errors mostly measure the length of the averaging window.  A fixed
    early iteration measures the transient the rate governs.  ``probe`` is that
    iteration as a fraction of the run.
    """
    path = os.path.join(RESULTS, f"traces_{name}_srnsgld.npz")
    if not os.path.exists(path):
        return {}, None
    blob = np.load(path)
    errors = {}
    for key in blob.files:
        if not key.endswith("_mean_error"):
            continue
        field, rho = key[: -len("_mean_error")].split("_rho")
        curve = blob[key]
        t = max(int(round(probe * len(curve))) - 1, 0)
        errors[(field, float(rho))] = float(curve[t])
    return errors, t + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--rho", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--probe", type=float, default=0.1,
                        help="where in the run the measured ratio is read, as a "
                             "fraction of its length")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    lines = [
        "# Rate redistribution: what each field can actually buy",
        "",
        "Eigenvalues of the linearised drift ``(I + J(x*)) H`` at the constrained MAP.",
        "`rate` is `min Re mu`, the exponential rate of the slowest direction; `gain` is",
        "that rate over `lambda_min`, the reversible one; `radius` is `max |mu|`, which is",
        "what limits the step size.  `measured` is the ratio of the `J = 0` whitened error",
        "to this field's, a tenth of the way into the runs of `run_srnsgld.py` -- the",
        "column `gain` is what it is meant to predict.",
        "",
    ]
    for name in args.datasets:
        ds = load_nine(name, seed=args.split_seed)
        model = MinibatchLogistic(ds.X_train, ds.y_train)
        ball = Ball(args.radius)
        _, geom = curvature_step_size(model, ball)
        H = np.asarray(geom["hessian"], float)
        x_star = np.asarray(geom["map"], float)
        ev = np.linalg.eigvalsh(H)
        errs, probe = measured(name, args.probe)
        base = errs.get(("zero", 0.0))

        lines += [
            f"## {PRETTY_NAMES.get(name, name)}",
            "",
            f"`lambda_min` = {ev.min():.1f}, `lambda_max` = {ev.max():.1f}, "
            f"condition number {ev.max() / ev.min():.1f}, `|x*|` = "
            f"{np.linalg.norm(x_star):.3f} of `r` = {ball.radius:g}.",
            "",
            f"| field | rho | null directions of `J` | rate | gain | radius | "
            f"measured at {probe} |" if probe else
            "| field | rho | null directions of `J` | rate | gain | radius | measured |",
            "|---|---|---|---|---|---|---|",
            f"| `J = 0` | 0 | 9 | {ev.min():.1f} | 1.00 | {ev.max():.0f} | 1.00 |",
        ]
        for key in ("constant", "state"):
            for rho in args.rho:
                field = field_from_rho(key, model.d, rho, ball.radius)
                J = field.matrix(x_star)
                mu = np.linalg.eigvals((np.eye(model.d) + J) @ H)
                rate = float(mu.real.min())
                null = model.d - int(np.linalg.matrix_rank(J))
                err = errs.get((key, rho))
                shown = f"{base / err:.2f}" if (base and err) else "--"
                lines.append(
                    f"| `{key}` | {rho:g} | {null} | {rate:.1f} | "
                    f"{rate / ev.min():.2f} | {np.abs(mu).max():.0f} | {shown} |"
                )
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "srnsgld_spectrum.md"), "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
