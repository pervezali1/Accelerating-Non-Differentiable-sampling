#!/usr/bin/env python3
"""Is any variant actually different from the reversible baseline?

All variants share walker seeds, so walker ``j`` of one variant and walker ``j``
of another see the same noise stream and can be differenced pairwise.  This
prints the paired mean difference against ``J = 0`` at the final iteration, with
its standard error over walkers, for accuracy and for the whitened error of the
running posterior mean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VARIANTS = ["constant", "state", "state_nocorr"]


def paired(a: np.ndarray, b: np.ndarray) -> tuple:
    diff = a - b
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["titanic", "magic", "breast_cancer", "spambase"]
    )
    parser.add_argument("--geometry", default="warmup")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    lines = [
        "Paired differences against the J = 0 baseline at the final iteration "
        "(same walker seeds, so differences are paired).",
        "A difference is only meaningful if it is a few standard errors from zero.",
        "",
        "| dataset | variant | accuracy difference | error difference |",
        "|---|---|---|---|",
    ]
    for name in args.datasets:
        path = os.path.join(
            ROOT, "results", f"traces_{name}_{args.geometry}_alpha{args.alpha:g}.npz"
        )
        if not os.path.exists(path):
            continue
        blob = np.load(path, allow_pickle=False)
        acc0 = blob["accuracy_zero"][-1]
        err0 = blob["error_zero"][-1]
        for variant in VARIANTS:
            da, sa = paired(blob[f"accuracy_{variant}"][-1], acc0)
            de, se = paired(blob[f"error_{variant}"][-1], err0)
            lines.append(
                f"| {name} | {variant} | {da:+.4f} +/- {sa:.4f} | {de:+.3f} +/- {se:.3f} |"
            )
    out = os.path.join(ROOT, "results", "paired_comparison.md")
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
