#!/usr/bin/env python3
"""Does the answer survive a change of regularizer?

One panel per non-differentiable penalty, one bar group per problem and
constraint set, two bars per group: how much closer to the exact constrained
posterior each skew field gets than ``J = 0`` does, at the end of a fixed
iteration budget.  Above 1 means the field wins.

Both readings are drawn: the accuracy gap left at the end of the budget (the
paper's metric) and the whitened error of the running posterior mean (which
keeps resolving after accuracy has saturated).  Amplitudes come from the same
selection rule as ``select_anchored.py --objective gap``; error bars are
propagated from the seed standard errors of the two quantities being divided.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
sys.path.insert(0, os.path.join(ROOT, "experiments"))

PENALTY = {
    "l1": r"lasso  $\lambda\|x\|_1$",
    "group": r"group lasso  $\lambda\sum_g\|x_g\|_2$",
    "tv": r"total variation  $\lambda\sum_j|x_{j+1}-x_j|$",
    "linf": r"max norm  $\lambda\|x\|_\infty$",
}
STYLE = {"constant": "#2B5FD9", "state": "#0E9F5C"}
LABEL = {"constant": r"constant $J_a$", "state": r"state-dependent $J_s$ / $J_g$"}
INK, MUTED = "#1f2328", "#6b7280"
PRETTY = {"titanic": "Titanic", "magic": "MAGIC", "synthetic": "Synthetic"}
SET = {"ball": "ball", "lp": "sublevel"}


def frame(ax) -> None:
    ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


def ratio_with_error(base: dict, pick: dict, stat: str) -> tuple:
    """``base / pick`` and its standard error, by propagation."""
    b, sb = base[stat], base.get(stat + "_se", 0.0)
    p, sp = max(pick[stat], 1e-12), pick.get(stat + "_se", 0.0)
    r = b / p
    return r, r * np.sqrt((sb / max(b, 1e-12)) ** 2 + (sp / p) ** 2)


def main() -> None:
    from select_anchored import select  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--penalties", nargs="+", default=["l1", "group", "tv"])
    parser.add_argument("--etas", nargs="+", type=float,
                        default=[1e-4, 6.25e-6],
                        help="only groups at one of these step sizes are drawn")
    parser.add_argument("--max-fail-rate", type=float, default=0.001)
    parser.add_argument("--out", default="anchored_regularizers.png")
    args = parser.parse_args()

    groups: dict = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_anchored_*.json"))):
        meta = json.load(open(path))
        reg = meta.get("regularizer", "l1")
        for r in meta["rows"]:
            if r["n_iter"] != args.n_iter or "gap_tail" not in r:
                continue
            if not any(abs(r["eta"] - e) < 1e-12 for e in args.etas):
                continue
            key = (reg, meta["problem"], meta["domain"], r["eta"], tuple(meta["seeds"]))
            groups.setdefault(key, []).append(r)

    # one column per (problem, set), in a fixed order
    columns = [(p, d) for p in ("titanic", "magic", "synthetic") for d in ("ball", "lp")]
    picked: dict = {}
    for (reg, problem, domain, eta, seeds), rows in groups.items():
        base = next((r for r in rows if r["field"] == "zero"), None)
        if base is None:
            continue
        for field in ("constant", "state"):
            pick, clean = select(rows, field, base, args.max_fail_rate, "gap")
            if pick is None:
                continue
            key = (reg, problem, domain, field)
            entry = {
                "gap": ratio_with_error(base, pick, "gap_tail"),
                "err": ratio_with_error(base, pick, "error"),
                "kappa": pick["kappa"],
                "tilt": pick.get("tilt", 0.0),
                "clean": clean,
                "seeds": len(seeds),
                "eta": eta,
            }
            # with several seed sets or step sizes for the same cell, keep the
            # one with the most seeds, then the larger step
            old = picked.get(key)
            if old is None or (entry["seeds"], entry["eta"]) > (old["seeds"], old["eta"]):
                picked[key] = entry

    fig, axes = plt.subplots(
        2, len(args.penalties), figsize=(4.9 * len(args.penalties), 7.2),
        squeeze=False, layout="constrained", sharey="row",
    )
    x = np.arange(len(columns))
    width = 0.36
    for col, reg in enumerate(args.penalties):
        for row, stat in enumerate(("gap", "err")):
            ax = axes[row][col]
            frame(ax)
            for k, field in enumerate(("constant", "state")):
                vals, errs = [], []
                for problem, domain in columns:
                    e = picked.get((reg, problem, domain, field))
                    vals.append(e[stat][0] if e else np.nan)
                    errs.append(e[stat][1] if e else 0.0)
                ax.bar(
                    x + (k - 0.5) * width, vals, width, yerr=errs, capsize=2,
                    color=STYLE[field], alpha=0.9, label=LABEL[field],
                    error_kw=dict(lw=1.0, ecolor=MUTED),
                )
            ax.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=5)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{PRETTY[p]}\n{SET[d]}" for p, d in columns], fontsize=8.5
            )
            ax.set_ylabel(
                "accuracy gap:  $J = 0$ / field" if stat == "gap"
                else "posterior-mean error:  $J = 0$ / field",
                color=INK, fontsize=9.5,
            )
            if row == 0:
                ax.set_title(PENALTY[reg], color=INK, fontsize=11, pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False,
               fontsize=10)
    fig.suptitle(
        f"Three non-differentiable regularizers, {args.n_iter} iterations: "
        "how much closer than $J = 0$\n"
        "(above the dashed line the skew field wins; bars are the selected amplitude, "
        "error bars from the seed spread)",
        fontsize=12, color=INK,
    )
    out = os.path.join(FIGURES, args.out)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)

    lines = [
        f"# Three regularizers, {args.n_iter} iterations",
        "",
        "Ratios above 1 mean the skew field is closer to the exact constrained posterior",
        "than `J = 0` at the end of the budget.  `gap` is the accuracy gap, `error` the",
        "whitened error of the running posterior mean; amplitudes are selected by",
        "`select_anchored.py --objective gap`.  A cell marked `(bias traded)` is one where",
        "no amplitude met the error guard, so the best gap is reported instead.",
        "",
        "| penalty | problem | set | field | kappa | c | gap ratio | error ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for reg in args.penalties:
        for problem, domain in columns:
            for field in ("constant", "state"):
                e = picked.get((reg, problem, domain, field))
                if e is None:
                    continue
                flag = "" if e["clean"] else " (bias traded)"
                lines.append(
                    f"| {reg} | {PRETTY[problem]} | {SET[domain]} | "
                    f"{LABEL[field].replace('$', '`')}{flag} | {e['kappa']:g} | "
                    f"{e['tilt']:g} | {e['gap'][0]:.2f}x +/- {e['gap'][1]:.2f} | "
                    f"{e['err'][0]:.2f}x +/- {e['err'][1]:.2f} |"
                )
    text = "\n".join(lines) + "\n"
    open(os.path.join(RESULTS, f"anchored_regularizers_{args.n_iter}.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
