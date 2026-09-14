#!/usr/bin/env python3
"""Pick the amplitude for each field from the sweeps, by a stated rule.

Reads every ``results/sweep_anchored_*.json`` and, for each problem, constraint
set and step size, selects for each field the row that

1. does not lose stationary accuracy -- its final whitened error is within one
   standard error of the ``J = 0`` row's, and
2. does not lean on the fallback -- fewer than 0.1% of its boundary pushes miss,
   so the oblique projection is doing what it claims, and
3. among those, has the smallest ``band``: the fewest iterations to come within
   0.005 of the exact constrained lasso posterior's test accuracy.

If no row satisfies (1) and (2), the best band is reported anyway and flagged,
so that a case where the rotation can only trade bias for speed is visible as
such rather than quietly selected.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def select(rows: list, field: str, base: dict, max_fail_rate: float) -> tuple:
    cands = [r for r in rows if r["field"] == field]
    if not cands:
        return None, False
    steps = base["n_iter"] * 1.0
    ok = [
        r for r in cands
        if r["error"] <= base["error"] + base["error_se"]
        and r["failed"] <= max_fail_rate * steps
        and r["reached_all"]
    ]
    if ok:
        return min(ok, key=lambda r: r["band"]), True
    return min(cands, key=lambda r: r["band"]), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fail-rate", type=float, default=0.001)
    parser.add_argument("--out", default="anchored_selected_amplitudes.md")
    args = parser.parse_args()

    lines = [
        "# The amplitude each field wants",
        "",
        "Selected from the sweeps by the rule in `experiments/select_anchored.py`: the",
        "fewest iterations to the accuracy band, among the amplitudes that do not lose",
        "stationary accuracy and do not fall back on a missed boundary push.  `band` is",
        "iterations to come within 0.005 of the exact constrained lasso posterior's test",
        "accuracy, `error` the whitened distance of the time-averaged mean to its mean,",
        "`error at n/4` the same a quarter of the way in.  `c` is the tilt of the",
        "non-radial `h`; `c = 0` is the paper's own field.  Uncertainties are standard",
        "errors over the seeds.",
        "",
    ]
    table = [
        "| problem | set | eta | seeds | field | kappa | c | band | speed-up | error | error at n/4 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    # pool rows across sweep files, but only where the seed set matches, so the
    # baseline a row is compared against was run on the same walker seeds
    groups: dict = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_anchored_*.json"))):
        meta = json.load(open(path))
        for r in meta["rows"]:
            key = (meta["problem"], meta["domain"], r["eta"], tuple(meta["seeds"]))
            groups.setdefault(key, []).append(r)

    for (problem, domain, eta, seeds) in sorted(groups):
        here = groups[(problem, domain, eta, seeds)]
        base = next((r for r in here if r["field"] == "zero"), None)
        if base is None:
            continue
        table.append(
            f"| {problem} | {domain} | {eta:.2e} | {len(seeds)} | `J = 0` | -- | -- | "
            f"{base['band']:.0f} +/- {base['band_se']:.0f} | 1.00x | "
            f"{base['error']:.3f} +/- {base['error_se']:.3f} | "
            f"{base['error_quarter']:.3f} |"
        )
        for field, name in (("constant", "`J_a`"), ("state", "`J_s` / `J_g`")):
            pick, clean = select(here, field, base, args.max_fail_rate)
            if pick is None:
                continue
            flag = "" if clean else " **(bias traded)**"
            table.append(
                f"| {problem} | {domain} | {eta:.2e} | {len(seeds)} | {name}{flag} | "
                f"{pick['kappa']:g} | {pick.get('tilt', 0.0):g} | "
                f"{pick['band']:.0f} +/- {pick['band_se']:.0f} | "
                f"**{base['band'] / max(pick['band'], 1e-9):.2f}x** | "
                f"{pick['error']:.3f} +/- {pick['error_se']:.3f} | "
                f"{pick['error_quarter']:.3f} |"
            )
            if not clean:
                notes.append(
                    f"* {problem}/{domain} at `eta` = {eta:.2e}: no amplitude of {name} "
                    f"improves the band without giving up stationary accuracy; the row "
                    f"shown is the best band."
                )
    lines += table + [""]
    if notes:
        lines += ["Where the rule could not be met:", ""] + notes + [""]
    text = "\n".join(lines) + "\n"
    print(text)
    open(os.path.join(RESULTS, args.out), "w").write(text)


if __name__ == "__main__":
    main()
