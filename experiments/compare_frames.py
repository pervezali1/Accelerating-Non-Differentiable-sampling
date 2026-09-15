#!/usr/bin/env python3
"""Does the searched frame beat the paper's, and does the rate predictor call it?

For every problem, constraint set, penalty and step size in the sweeps, this
lines up four things per field: the amplitude selected by the usual rule, the
accuracy gap left at the end of the budget, the whitened error of the running
posterior mean, and -- for the fields whose matrix is known at the constrained
MAP -- the slowest rate of ``(I + J) H``, which is what the linearised interior
descent says should happen.  The point of putting them side by side is that the
rate is computed without simulating anything, so agreement is evidence that the
mechanism is the one named rather than a tuning artefact.

``--frames columns spectral`` compares the paper's block order against the
signed permutation chosen by :func:`nds.constrained.spectral_frame`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "experiments"))

PRETTY = {"titanic": "Titanic", "magic": "MAGIC", "synthetic": "Synthetic"}
SET = {"ball": "ball", "lp": "sublevel"}
NAME = {"zero": "`J = 0`", "constant": "`J_a`", "state": "`J_s` / `J_g`"}


def ratio(base: dict, pick: dict, stat: str) -> tuple:
    b, sb = base[stat], base.get(stat + "_se", 0.0)
    p, sp = max(pick[stat], 1e-12), pick.get(stat + "_se", 0.0)
    r = b / p
    return r, r * np.sqrt((sb / max(b, 1e-12)) ** 2 + (sp / p) ** 2)


def main() -> None:
    from select_anchored import select  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--frames", nargs="+", default=["columns", "spectral"])
    parser.add_argument("--max-fail-rate", type=float, default=0.001)
    parser.add_argument("--guard", default="error",
                        choices=["error", "error_running"],
                        help="which error the selection guard protects; see "
                             "select_anchored.select")
    parser.add_argument("--out", default="anchored_frames.md")
    args = parser.parse_args()

    # (problem, domain, penalty, eta, seeds) -> {frame: rows}
    groups: dict = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_anchored_*.json"))):
        meta = json.load(open(path))
        order = meta.get("block_order", "columns")
        if order not in args.frames:
            continue
        for r in meta["rows"]:
            if r["n_iter"] != args.n_iter or "gap_tail" not in r:
                continue
            key = (meta["problem"], meta["domain"], meta.get("regularizer", "l1"),
                   round(r["eta"], 15), tuple(meta["seeds"]))
            groups.setdefault(key, {}).setdefault(order, []).append(r)

    lines = [
        f"# The frame the state-dependent field is read in, {args.n_iter} iterations",
        "",
        "Which coordinates share a `3`-block of `J_s` is free: conjugating by an",
        "orthogonal `Q` keeps the field skew and divergence free and keeps `J(x) x = 0`,",
        "so it is the paper's construction in another frame.  `Q` also has to keep the",
        "constraint set invariant, which on the smoothed `l_p` set means signed",
        "permutations -- so that is the group searched.  `columns` is the paper's own",
        "order; `spectral` is the signed permutation maximising the slowest rate of",
        "`(I + J) H` at the constrained MAP (`nds.constrained.spectral_frame`).",
        "",
        "`rate` is that predicted ratio, computed with no simulation at all; `gap` and",
        "`error` are measured, ratios of `J = 0` over the field, so above 1 the field",
        "wins.  Amplitudes are selected by the rule in `select_anchored.py` against the",
        "gap.  Uncertainties are standard errors over the walker seeds.",
        "",
        f"| problem | set | penalty | eta | field | frame | kappa | c | rate | "
        f"gap ratio | {args.guard} ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(groups):
        problem, domain, reg, eta, seeds = key
        per_frame = groups[key]
        base = None
        for order in args.frames:
            base = next((r for r in per_frame.get(order, []) if r["field"] == "zero"),
                        base)
        if base is None:
            continue
        rate0 = base.get("slowest_rate")
        shown_baseline = False
        for order in args.frames:
            rows = per_frame.get(order)
            if not rows:
                continue
            for field in ("constant", "state"):
                if field == "constant" and order != args.frames[0]:
                    continue  # the constant field has no frame to choose
                pick, clean = select(rows, field, base, args.max_fail_rate,
                                     "gap", args.guard)
                if pick is None:
                    continue
                if not shown_baseline:
                    lines.append(
                        f"| {PRETTY[problem]} | {SET[domain]} | {reg} | {eta:.3g} | "
                        f"`J = 0` | -- | -- | -- | 1.00x | 1.00x | 1.00x |"
                    )
                    shown_baseline = True
                g, gs = ratio(base, pick, "gap_tail")
                e, es = ratio(base, pick, args.guard)
                pr = pick.get("slowest_rate")
                rate = f"{pr / rate0:.2f}x" if (pr and rate0) else "--"
                flag = "" if clean else " (bias traded)"
                lines.append(
                    f"| {PRETTY[problem]} | {SET[domain]} | {reg} | {eta:.3g} | "
                    f"{NAME[field]}{flag} | "
                    f"{'--' if field == 'constant' else order} | {pick['kappa']:g} | "
                    f"{pick.get('tilt', 0.0):g} | {rate} | "
                    f"{g:.2f}x +/- {gs:.2f} | {e:.2f}x +/- {es:.2f} |"
                )
    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(RESULTS, args.out), "w").write(text)
    print("wrote", os.path.join(RESULTS, args.out))


if __name__ == "__main__":
    main()
