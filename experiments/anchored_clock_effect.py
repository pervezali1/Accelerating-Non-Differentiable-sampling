#!/usr/bin/env python3
"""What the clock is worth: anchored Langevin against the bare smoothed anchor.

Dropping the clock leaves a chain whose invariant law is ``e^{-U_0} 1_K`` --
the *smoothed* lasso, not the lasso.  Keeping it costs mixing, because the
effective step size becomes ``eta a(x)`` with ``a <= 1``.  So the clock is worth
having only where the difference between the two laws is larger than the mixing
it costs, and that is what this table measures: the whitened distance from each
chain's time-averaged mean to the exact constrained lasso posterior's mean,
side by side, together with the per-coordinate errors on the coordinates the
lasso actually drives to zero, which is where the two laws differ most.

Reads the summaries written by ``run_anchored_srnsgld.py`` with and without
``--no-clock`` (tag suffix ``_noclock``).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load(problem: str, domain: str, suffix: str):
    path = os.path.join(RESULTS, f"summary_anchored_{problem}_{domain}{suffix}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["synthetic", "magic", "titanic"])
    parser.add_argument("--domain", default="lp")
    parser.add_argument("--zero-tol", type=float, default=0.05)
    args = parser.parse_args()

    lines = [
        "# What the clock is worth",
        "",
        "`error` is the whitened distance from the time-averaged mean to the exact",
        "constrained lasso posterior's mean.  `anchored` keeps the clock",
        "`a = exp(U - U_0)`, so the invariant law is the kinked one; `anchor only` drops",
        "it, so the invariant law is the smoothed one.  `near-zero error` is the largest",
        "absolute error over the coordinates whose exact posterior mean is within",
        f"{args.zero_tol} of zero -- the coordinates the lasso is acting on, where the two",
        "laws differ most.",
        "",
        "| problem | field | error, anchored | error, anchor only | near-zero coords | "
        "near-zero error, anchored | near-zero error, anchor only |",
        "|---|---|---|---|---|---|---|",
    ]
    for problem in args.problems:
        clocked = load(problem, args.domain, "")
        bare = load(problem, args.domain, "_noclock")
        if not (clocked and bare):
            print(f"missing pair for {problem}/{args.domain}; skipping")
            continue
        exact = np.asarray(clocked["reference"]["posterior_mean"], float)
        which = np.abs(exact) < args.zero_tol
        for a, b in zip(clocked["runs"], bare["runs"]):
            assert a["key"] == b["key"]
            ma = np.asarray(a["time_mean"], float)
            mb = np.asarray(b["time_mean"], float)
            near_a = f"{np.abs(ma - exact)[which].max():.4f}" if which.any() else "--"
            near_b = f"{np.abs(mb - exact)[which].max():.4f}" if which.any() else "--"
            lines.append(
                f"| {problem} | `{a['key']}` | {a['mean_error']:.3f} | "
                f"{b['mean_error']:.3f} | {int(which.sum())}/{clocked['d']} | "
                f"{near_a} | {near_b} |"
            )
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, f"anchored_clock_effect_{args.domain}.md"), "w") as h:
        h.write(text)


if __name__ == "__main__":
    main()
