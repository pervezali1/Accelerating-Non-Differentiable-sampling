#!/usr/bin/env python3
"""Figures for the constrained ``d = 9`` experiment: PSGLD against SRNSGLD.

Three figures from the traces written by ``run_srnsgld.py``:

* ``figures/srnsgld_accuracy_loss.png`` -- test accuracy and predictive log-loss
  of the running posterior predictive, one column per dataset;
* ``figures/srnsgld_mean_error.png`` -- whitened distance from the running
  posterior mean to the exact constrained posterior's mean, which is the metric
  that actually separates the three fields;
* ``figures/srnsgld_rho.png`` -- the rotation-strength sweep, so that the
  amplitude in the other two figures can be seen to be a choice and not a peak.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

STYLE = {
    "zero": dict(color="#C0334D", lw=2.6, ls="-", zorder=4),
    "constant": dict(color="#2B5FD9", lw=2.0, ls="-", zorder=5),
    "state": dict(color="#0E9F5C", lw=2.2, ls=(0, (6, 3)), zorder=6),
}
LABEL = {
    "zero": r"$J = 0$  (PSGLD)",
    "constant": r"constant $J_a$  (SRNSGLD)",
    "state": r"state-dependent $J_s$  (SRNSGLD)",
}
INK, MUTED = "#1f2328", "#6b7280"


def frame(ax) -> None:
    ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


def load(datasets: list, tag: str = "srnsgld") -> list:
    runs = []
    for name in datasets:
        tr = os.path.join(RESULTS, f"traces_{name}_{tag}.npz")
        sm = os.path.join(RESULTS, f"summary_{name}_{tag}.json")
        if not (os.path.exists(tr) and os.path.exists(sm)):
            print(f"missing traces or summary for {name}; skipping")
            continue
        runs.append({"meta": json.load(open(sm)), "traces": dict(np.load(tr))})
    return runs


def pick(run: dict, key: str, rho: float | None, probe: float = 0.1) -> dict:
    """The run of field ``key`` at ``rho``; unset picks the fastest-converging.

    "Fastest" is read a tenth of the way into the run rather than at its end:
    the quantity of interest is the rate of convergence, and by the last
    iteration every field that converges has.
    """
    meta = run["meta"]
    cands = [r for r in meta["runs"] if r["key"] == key]
    if not cands:
        raise KeyError(key)
    if key == "zero":
        return cands[0]
    if rho is not None:
        exact = [r for r in cands if abs(r["rho"] - rho) < 1e-9]
        if exact:
            return exact[0]

    def early(r):
        curve = run["traces"][f"{r['key']}_rho{r['rho']:g}_mean_error"]
        return float(curve[max(int(round(probe * len(curve))) - 1, 0)])

    return min(cands, key=early)


def series(run: dict, chosen: dict, field: str) -> np.ndarray:
    return run["traces"][f"{chosen['key']}_rho{chosen['rho']:g}_{field}"]


def curves_figure(runs: list, out: str, rho, field: str, ylabel: str,
                  title: str, skip: int, log: bool = False) -> None:
    fig, axes = plt.subplots(
        1, len(runs), figsize=(5.7 * len(runs), 4.0), squeeze=False, layout="constrained"
    )
    for ax, run in zip(axes.ravel(), runs):
        meta = run["meta"]
        frame(ax)
        lo_hi = []
        for key in ("zero", "constant", "state"):
            chosen = pick(run, key, rho)
            y = series(run, chosen, field)
            x = np.arange(1, len(y) + 1)
            label = LABEL[key] + ("" if key == "zero" else f",  $\\rho$ = {chosen['rho']:g}")
            ax.plot(x, y, label=label, **STYLE[key])
            lo_hi.append(y[skip:])
        if field in ("accuracy", "loss"):
            ref = meta["reference"][field]
            ax.axhline(ref, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=7)
            ax.annotate(
                f"dashed: exact constrained posterior, {ref:.4f}",
                xy=(0.98, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8.5, color=MUTED, zorder=8,
            )
            lo_hi.append(np.array([ref]))
        if log:
            ax.set_yscale("log")
            ax.set_xscale("log")
        else:
            tail = np.concatenate(lo_hi)
            pad = 0.12 * (tail.max() - tail.min()) + 1e-4
            ax.set_ylim(tail.min() - pad, tail.max() + pad)
        ax.set_xlim(1, meta["n_iter"])
        ax.set_title(
            f"{meta['pretty']}  ($d$ = {meta['d']}, $n$ = {meta['n_train']} train)",
            color=INK, fontsize=11.5, pad=8,
        )
        ax.set_xlabel("iterations", color=INK, fontsize=10)
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=14, color=INK)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def accuracy_loss_figure(runs: list, out: str, rho, skip: int, max_iter: int | None = None) -> None:
    fig, axes = plt.subplots(
        2, len(runs), figsize=(5.7 * len(runs), 7.4), squeeze=False, layout="constrained"
    )
    for col, run in enumerate(runs):
        meta = run["meta"]
        for row, (field, ylabel) in enumerate(
            (("accuracy", "classification accuracy"), ("loss", "predictive log-loss (nats)"))
        ):
            ax = axes[row][col]
            frame(ax)
            tails = []
            for key in ("zero", "constant", "state"):
                chosen = pick(run, key, rho)
                y = series(run, chosen, field)
                if max_iter:
                    y = y[:max_iter]
                label = LABEL[key] + ("" if key == "zero" else f",  $\\rho$ = {chosen['rho']:g}")
                ax.plot(np.arange(1, len(y) + 1), y, label=label, **STYLE[key])
                tails.append(y[skip:])
            ref = meta["reference"][field]
            ax.axhline(ref, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=7)
            tails.append(np.array([ref]))
            tail = np.concatenate(tails)
            pad = 0.12 * (tail.max() - tail.min()) + 1e-4
            ax.set_ylim(tail.min() - pad, tail.max() + pad)
            ax.set_xlim(1, min(max_iter or meta["n_iter"], meta["n_iter"]))
            ax.annotate(
                f"dashed: exact constrained posterior, {ref:.4f}",
                xy=(0.98, 0.05 if row == 0 else 0.9), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color=MUTED, zorder=8,
            )
            ax.set_xlabel("iterations", color=INK, fontsize=10)
            ax.set_ylabel(ylabel, color=INK, fontsize=10)
            if row == 0:
                ax.set_title(
                    f"{meta['pretty']}  ($d$ = {meta['d']}, $n$ = {meta['n_train']} train)",
                    color=INK, fontsize=11.5, pad=8,
                )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle(
        "Constrained Bayesian logistic regression on $K_r$, from a uniform start on the unit ball",
        fontsize=13.5, color=INK,
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def rho_figure(runs: list, out: str, probe: float = 0.1) -> None:
    fig, axes = plt.subplots(
        1, len(runs), figsize=(5.3 * len(runs), 3.9), squeeze=False, layout="constrained"
    )
    for ax, run in zip(axes.ravel(), runs):
        meta = run["meta"]
        frame(ax)

        def early(r):
            curve = run["traces"][f"{r['key']}_rho{r['rho']:g}_mean_error"]
            return float(curve[max(int(round(probe * len(curve))) - 1, 0)])

        n_probe = max(int(round(probe * meta["n_iter"])), 1)
        base = early(pick(run, "zero", None))
        ax.axhline(base, color=STYLE["zero"]["color"], lw=2.2, ls="-", label=LABEL["zero"])
        rhos = set()
        for key in ("constant", "state"):
            rows = sorted((r for r in meta["runs"] if r["key"] == key), key=lambda r: r["rho"])
            rhos.update(r["rho"] for r in rows)
            ax.plot(
                [r["rho"] for r in rows], [early(r) for r in rows],
                marker="o", ms=5, label=LABEL[key],
                **{k: v for k, v in STYLE[key].items() if k != "ls"},
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = sorted(rhos)
        ax.set_xticks(ticks)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel(r"rotation strength $\rho = \|J\|$", color=INK, fontsize=10)
        ax.set_ylabel(f"whitened error at iteration {n_probe}", color=INK, fontsize=10)
        ax.set_title(meta["pretty"], color=INK, fontsize=11.5, pad=8)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=10)
    fig.suptitle("How much rotation, and where it stops helping", fontsize=13.5, color=INK)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def write_summary(runs: list, out: str) -> None:
    lines = [
        "# Constrained logistic regression in d = 9: PSGLD against SRNSGLD",
        "",
        "One step size per dataset from the curvature at the constrained MAP, shared by",
        "all three fields; one mini-batch stream per iteration, shared by all walkers;",
        "walkers started uniform on the centred unit ball.  `error` is the whitened",
        "distance from the running posterior mean to the exact constrained posterior's.",
        "",
    ]
    for run in runs:
        meta = run["meta"]
        ref = meta["reference"]
        lines += [
            f"## {meta['pretty']}",
            "",
            f"`d` = {meta['d']}, `n_train` = {meta['n_train']}, `n_test` = {meta['n_test']}, "
            f"`r` = {meta['radius']}, `h` = {meta['step_size']:.3e}, "
            f"batch = {meta['batch_size']}, {meta['n_iter']} iterations, "
            f"{meta['n_walkers']} walkers.",
            "",
            f"Exact constrained posterior: accuracy {ref['accuracy']:.4f} "
            f"(halves {ref['accuracy_halves'][0]:.4f} / {ref['accuracy_halves'][1]:.4f}), "
            f"loss {ref['loss']:.4f}, acceptance {ref['acceptance']:.2f}, "
            f"|mean| {ref['mean_radius']:.3f} of `r` = {meta['radius']}.",
            "",
            "| field | rho | a | accuracy | loss | error | boundary hits | missed reflections |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in meta["runs"]:
            lines.append(
                f"| {r['key']} | {r['rho']:g} | {r['amplitude']:.3f} | "
                f"{r['accuracy_final']:.4f} | {r['loss_final']:.4f} | "
                f"{r['mean_error_final']:.4f} | {r['boundary_rate']:.3f} | "
                f"{r['missed_reflections']} |"
            )
        lines.append("")
    text = "\n".join(lines) + "\n"
    open(out, "w").write(text)
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["titanic", "magic"])
    parser.add_argument("--tag", default="srnsgld")
    parser.add_argument("--rho", type=float, default=None,
                        help="rotation strength to draw; the default picks, per field, "
                             "the one with the smallest final error")
    parser.add_argument("--skip", type=int, default=5,
                        help="iterations to ignore when choosing the y range")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="iterations to show in the accuracy and loss panels; the "
                             "curves are flat long before the run ends, and the "
                             "transient is what the fields differ in")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    runs = load(args.datasets, args.tag)
    if not runs:
        raise SystemExit("no traces found; run experiments/run_srnsgld.py first")
    sfx = args.suffix
    accuracy_loss_figure(runs, os.path.join(FIGURES, f"srnsgld_accuracy_loss{sfx}.png"),
                         args.rho, args.skip, args.max_iter)
    curves_figure(runs, os.path.join(FIGURES, f"srnsgld_mean_error{sfx}.png"), args.rho,
                  "mean_error", "whitened error of the posterior mean",
                  "Distance from the exact constrained posterior mean", args.skip, log=True)
    rho_figure(runs, os.path.join(FIGURES, f"srnsgld_rho{sfx}.png"))
    write_summary(runs, os.path.join(RESULTS, f"summary_srnsgld{sfx}.md"))


if __name__ == "__main__":
    main()
