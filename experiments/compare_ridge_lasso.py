#!/usr/bin/env python3
"""Ridge against lasso: does swapping the regularizer change the conclusions?

Two comparisons, both written to ``results/ridge_vs_lasso.md``.

**Sampler against sampler.**  The same protocol -- 200 iterations, 24 walkers
from ``w = 0``, common seeds, three fields, each at its own step size -- was run
on the smooth ridge target (`notebooks/titanic_magic.ipynb`) and on the
non-differentiable lasso target with anchored Langevin
(`notebooks/anchored_l1_titanic_magic.ipynb`).  The two priors give different
posteriors, so absolute accuracies and losses are not comparable; what is
comparable is each rotation's improvement over the reversible baseline *within*
its own target.

**Sampler against the lasso point estimate.**  The lasso is usually a
minimisation, and the minimiser of the same objective is what scikit-learn
returns for ``penalty="l1"`` and ``C = 1 / lambda``.  Comparing it with the
posterior mean says how much of the answer the sparsity of the point estimate
throws away: under a Laplace prior the posterior mean has no exact zeros, only
coefficients pulled towards them.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored import LassoLogistic  # noqa: E402
from nds.data import PRETTY_NAMES, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
NOTEBOOKS = os.path.join(ROOT, "notebooks")
FIELDS = ["J = 0", "constant J_a", "gated J_s"]


def markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """Pipe table, so the script needs nothing beyond ``requirements.txt``."""
    frame = frame.reset_index() if index else frame
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def sampler_table() -> pd.DataFrame:
    ridge = pd.read_csv(os.path.join(NOTEBOOKS, "titanic_magic_summary.csv"))
    lasso = pd.read_csv(os.path.join(NOTEBOOKS, "anchored_l1_summary.csv"))
    rows = []
    for dataset in ridge["dataset"].unique():
        for prior, table in (("ridge", ridge), ("lasso", lasso)):
            sub = table[table["dataset"] == dataset].set_index("field")
            base = sub.loc["J = 0"]
            for field in FIELDS:
                row = sub.loc[field]
                rows.append({
                    "dataset": dataset,
                    "prior": prior,
                    "field": field,
                    "loss gap": round(float(row["loss - ref"]), 4),
                    "iters to loss band": int(row["iters to loss band"]),
                    "speedup over J = 0": round(
                        float(base["iters to loss band"]) / float(row["iters to loss band"]), 2),
                    "mean error": round(float(row["mean error @200"]), 3),
                    "error ratio to J = 0": round(
                        float(base["mean error @200"]) / float(row["mean error @200"]), 2),
                    "step vs J = 0": round(float(row["step size"]) / float(base["step size"]), 2),
                })
    return pd.DataFrame(rows)


def point_estimate_table(penalty: float, delta: float) -> pd.DataFrame:
    """The lasso minimiser against the posterior mean of the same objective."""
    from sklearn.linear_model import LogisticRegression

    rows = []
    for name in ("titanic", "magic"):
        ds = load(name, seed=0)
        target = LassoLogistic(ds.X_train, ds.y_train, penalty=penalty, delta=delta)
        ref = np.load(os.path.join(RESULTS, f"reference_lasso_{name}_p{penalty:g}.npz"))
        post_mean, predictive = ref["posterior_mean"], ref["predictive"]

        # scikit-learn minimises C * sum_i loss_i + ||w||_1, so C = 1 / lambda is the
        # same objective as U.  The intercept column is dropped from the design and
        # left to scikit-learn, which does not penalise its own intercept -- matching
        # this target.  liblinear would penalise it, so saga is the solver to use.
        fit = LogisticRegression(C=1.0 / penalty, l1_ratio=1.0, solver="saga",
                                 fit_intercept=True, tol=1e-10, max_iter=50000)
        fit.fit(ds.X_train[:, 1:], ds.y_train)
        w_map = np.concatenate([fit.intercept_, fit.coef_.ravel()])

        # a minimiser of U, not merely of something similar: nudging any coordinate
        # either way must not lower U
        U_map = float(target.potential(w_map[:, None])[0])
        probe = np.repeat(w_map[:, None], 2 * len(w_map), axis=1)
        for j in range(len(w_map)):
            probe[j, 2 * j] += 1e-3
            probe[j, 2 * j + 1] -= 1e-3
        assert target.potential(probe).min() >= U_map - 1e-8, "not a local minimum of U"

        def metrics(p):
            acc = float(((p > 0.5) * ds.y_test + (p < 0.5) * (1 - ds.y_test)
                         + (p == 0.5) * 0.5).mean())
            q = np.clip(p, 1e-12, 1 - 1e-12)
            loss = float(-(ds.y_test * np.log(q) + (1 - ds.y_test) * np.log1p(-q)).mean())
            return acc, loss

        p_map = 1.0 / (1.0 + np.exp(-(ds.X_test @ w_map)))
        acc_map, loss_map = metrics(p_map)
        acc_post, loss_post = metrics(predictive)
        rows.append({
            "dataset": PRETTY_NAMES[name],
            "U at the lasso minimiser": round(U_map, 2),
            "U at the posterior mean": round(float(target.potential(post_mean[:, None])[0]), 2),
            "exact zeros, minimiser": int(np.sum(np.abs(w_map[1:]) < 1e-10)),
            "exact zeros, posterior mean": int(np.sum(np.abs(post_mean[1:]) < 1e-10)),
            "coefficients": len(w_map) - 1,
            "||w||_1, minimiser": round(float(np.abs(w_map[1:]).sum()), 3),
            "||w||_1, posterior mean": round(float(np.abs(post_mean[1:]).sum()), 3),
            "test accuracy, minimiser": round(acc_map, 4),
            "test accuracy, posterior": round(acc_post, 4),
            "test log-loss, minimiser": round(loss_map, 4),
            "test log-loss, posterior": round(loss_post, 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=0.02)
    args = parser.parse_args()

    samplers = sampler_table()
    points = point_estimate_table(args.penalty, args.delta)
    print(samplers.to_string(index=False))
    print()
    print(points.set_index("dataset").T.to_string())

    lines = [
        "# Ridge against lasso",
        "",
        "Same protocol on both targets: 200 iterations, 24 walkers from w = 0, common",
        "seeds, three fields, each at its own step size. The ridge target is smooth and",
        "sampled with the plain scheme; the lasso target is non-differentiable and sampled",
        "with anchored Langevin. The two priors give different posteriors, so compare each",
        "rotation with the reversible baseline inside its own target, not across targets.",
        "",
        markdown(samplers),
        "",
        "## The lasso as a minimisation",
        "",
        f"scikit-learn with penalty='l1' and C = 1/lambda = {1 / args.penalty:g} minimises the same",
        "objective U, so its solution is the MAP of the posterior that was sampled.",
        "",
        markdown(points.set_index("dataset").T, index=True),
        "",
    ]
    out = os.path.join(RESULTS, "ridge_vs_lasso.md")
    with open(out, "w") as handle:
        handle.write("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
