#!/usr/bin/env python3
"""Choosing the lasso strength for the constrained anchored experiments.

Three things pull against each other, and the ladder below shows all of them:

* a **larger** ``lam`` makes the non-differentiable term matter -- coordinates
  sit at the kink, and the clock ``a = exp(U - U_0)`` is what keeps the target
  right -- but it also shrinks the posterior *inside* the constraint set, and a
  constraint that never binds makes the skew reflection irrelevant, so the
  paper's experiment would no longer be the paper's experiment;
* the anchor's smoothing ``delta`` has to resolve the kink at the paper's step
  size, which needs ``eta lam`` small compared with the posterior's own scale;
* ``delta`` also sets the clock's floor, ``exp(-lam d delta)``: with the rule
  ``delta = budget / lam`` used here the floor is ``exp(-budget d)``, free of
  ``lam``, but the clock measured at the constrained lasso MAP still falls as
  more coordinates reach the kink.

The strength used by ``run_anchored_srnsgld.py`` is the largest on the ladder
with ``eta lam <= 0.02``, clock at the MAP above 0.2, and the MAP still on the
boundary of ``K``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored_constrained import (  # noqa: E402
    Ball,
    RegularisedLogistic,
    SmoothedLpBall,
    ZeroField,
    delta_for,
    make_regularizer,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def constrained_lasso_map(target, domain, n_iter: int = 4000) -> np.ndarray:
    """Projected gradient descent on the smooth anchor, retracted into ``K``."""
    L = 0.25 * np.linalg.eigvalsh(target.X.T @ target.X).max() + target.lam / target.delta
    idx = np.arange(target.n)
    w = np.zeros((target.d, 1))
    zero = ZeroField(target.d)
    for _ in range(n_iter):
        w = domain.retract(w - target.anchor_grad(w, idx) / L, zero)[0]
    return w.ravel()


def main() -> None:
    from run_anchored_srnsgld import LAM_SCALE, SETTINGS, synthetic_data  # noqa: E402
    from nds.data import load_nine  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["synthetic", "magic", "titanic"])
    parser.add_argument("--ladder", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--clock-budget", type=float, default=0.5)
    parser.add_argument("--regularizer", default="l1",
                        choices=["l1", "group", "tv", "linf"])
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    lines = [
        f"# Choosing the penalty strength ({args.regularizer})",
        "",
        "`lam = c n_train`, `delta = 0.5 / lam`.  `|x*|` is the norm of the constrained",
        "lasso MAP and `on boundary` says whether the constraint is still active there;",
        "`zeros` counts coordinates within 0.01 of zero; `clock` is `exp(U - U_0)` at that",
        "MAP; `eta lam` is the step size times the lasso strength, which is roughly the",
        "jitter the explicit scheme leaves on a coordinate sitting at the kink.",
        "",
        "| problem | set | c | lam | delta | eta lam | \\|x*\\| | on boundary | zeros | clock |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for problem in args.problems:
        cfg = SETTINGS[problem]
        if problem == "synthetic":
            data = synthetic_data(cfg["n"], args.split_seed)
        else:
            ds = load_nine(problem, seed=args.split_seed, test_fraction=0.2)
            data = {"X_train": ds.X_train, "y_train": ds.y_train}
        X, y = data["X_train"], data["y_train"]
        n, d = X.shape
        eta = cfg["step_size"]
        domains = [
            ("ball", Ball(cfg["radius_sq"], squared=True)),
            ("lp", SmoothedLpBall(cfg["lp"]["p"], cfg["lp"]["eps"], cfg["lp"]["level"])),
        ]
        for dname, domain in domains:
            for c in args.ladder:
                lam = c * n
                delta = delta_for(args.regularizer, lam, d, args.clock_budget)
                target = RegularisedLogistic(
                    X, y, make_regularizer(args.regularizer, lam, delta, d)
                )
                w = constrained_lasso_map(target, domain)
                on_bd = not bool(domain.contains(w[:, None] * 1.0001)[0])
                chosen = " **used**" if abs(c - LAM_SCALE[problem]) < 1e-12 else ""
                lines.append(
                    f"| {problem}{chosen} | {dname} | {c:g} | {lam:.1f} | "
                    f"{target.delta:.5f} | {eta * lam:.4f} | {np.linalg.norm(w):.3f} | "
                    f"{'yes' if on_bd else 'no'} | {(np.abs(w) < 0.01).sum()}/{d} | "
                    f"{target.clock(w[:, None])[0]:.3f} |"
                )
    text = "\n".join(lines) + "\n"
    print(text)
    suffix = "" if args.regularizer == "l1" else f"_{args.regularizer}"
    with open(os.path.join(RESULTS, f"anchored_lasso_calibration{suffix}.md"), "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
