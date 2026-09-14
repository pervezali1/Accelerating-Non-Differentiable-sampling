#!/usr/bin/env python3
"""Figures for the lasso-regularised, anchored version of the paper's Section 3.3.

One figure per constraint set, three rows (the three problems) by two columns
(training and test accuracy), in the paper's style: each line is the mean over
walkers of the accuracy of that walker's own parameter, with a band at one
standard deviation, plus a dashed line at the exact constrained lasso posterior
computed by random-walk Metropolis.
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
    "zero": dict(color="#C0334D", lw=2.4, ls="-"),
    "constant": dict(color="#2B5FD9", lw=2.0, ls="-"),
    "state": dict(color="#0E9F5C", lw=2.2, ls=(0, (6, 3))),
    "sublevel": dict(color="#0E9F5C", lw=2.2, ls=(0, (6, 3))),
}
INK, MUTED = "#1f2328", "#6b7280"
DOMAIN_TITLE = {
    "ball": r"centred ball $K_r = \{x : \|x\|_2^2 \leq r\}$",
    "lp": r"smoothed $\ell_p$ sublevel set $\{x : g(x) \leq \lambda\}$",
}


def frame(ax) -> None:
    ax.grid(True, axis="y", color="#d8dade", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


def load(problems: list, domain: str, suffix: str) -> list:
    out = []
    for problem in problems:
        tag = f"{problem}_{domain}{suffix}"
        tr = os.path.join(RESULTS, f"traces_anchored_{tag}.npz")
        sm = os.path.join(RESULTS, f"summary_anchored_{tag}.json")
        if not (os.path.exists(tr) and os.path.exists(sm)):
            print(f"missing {tag}; skipping")
            continue
        out.append({"meta": json.load(open(sm)), "traces": dict(np.load(tr))})
    return out


GENERIC = {
    "zero": r"$J = 0$  (anchored PSGLD)",
    "constant": r"constant $J_a$  (anchored SRNSGLD)",
    "state": r"state-dependent $J_s(x)$  (anchored SRNSGLD)",
    "sublevel": r"state-dependent $J_g(x)$  (anchored SRNSGLD)",
    "tilted": r"tilted $J_\psi(x)$  (anchored SRNSGLD)",
}


def smooth(A: np.ndarray, window: int) -> np.ndarray:
    """Moving average down the iteration axis, so the panels stay readable.

    Per-iteration accuracy of a single walker is a step function of a threshold
    and jumps by a whole test point at a time; with a large step size on a
    boundary-hugging posterior it rattles. The average is over iterations only,
    never over walkers, so the band still shows the spread across walkers.
    """
    if window <= 1 or len(A) < window:
        return A
    kernel = np.ones(window) / window
    pad = np.concatenate([np.repeat(A[:1], window - 1, axis=0), A], axis=0)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 0, pad)


def band_iteration(A: np.ndarray, ref: float, tol: float = 0.005):
    hit = np.flatnonzero(np.abs(A - ref) < tol)
    return int(hit[0]) + 1 if len(hit) else None


def legend_label(meta: dict, key: str) -> str:
    if key == "state" and meta.get("tilt", 0.0):
        return GENERIC["tilted"]
    return GENERIC[key]


def panel(ax, run: dict, split: str, window: int, zoom: bool) -> None:
    meta, traces = run["meta"], run["traces"]
    frame(ax)
    ref = meta["reference"][f"accuracy_{split}"]
    at = traces.get("scored_at", None)
    curves, bands = {}, []
    for row in meta["runs"]:
        key = row["key"]
        A = smooth(traces[f"{key}_accuracy_{split}"], window)
        curves[key] = A
        b = band_iteration(A.mean(axis=1), ref)
        if b is not None and at is not None:
            b = int(at[min(b - 1, len(at) - 1)])
        bands.append(b if b is not None else meta["n_iter"])

    xmax = meta["n_iter"]
    if zoom:
        xmax = int(min(meta["n_iter"], max(3 * max(bands), 60)))
    lo_hi = []
    for row in meta["runs"]:
        key = row["key"]
        at = traces.get("scored_at", np.arange(1, len(curves[key]) + 1))
        keep = at <= xmax
        A = curves[key][keep]
        mean, sd = A.mean(axis=1), A.std(axis=1)
        x = at[keep]
        ax.plot(x, mean, label=legend_label(meta, key), zorder=4, **STYLE[key])
        ax.fill_between(x, mean - sd, mean + sd, color=STYLE[key]["color"], alpha=0.16, lw=0)
        lo_hi += [mean - sd, mean + sd]
    ax.axhline(ref, color=INK, ls=(0, (5, 3)), lw=1.2, zorder=6)
    tail = np.concatenate(lo_hi + [np.array([ref])])
    pad = 0.08 * (tail.max() - tail.min()) + 0.002
    ax.set_ylim(max(0.0, tail.min() - pad), min(1.0, tail.max() + pad))
    ax.set_xlim(1, xmax)
    n_shown = meta["n_train"] if split == "train" else meta["n_test"]
    ax.set_xlabel("iterations $k$", color=INK, fontsize=10)
    ax.set_ylabel(f"accuracy, {split}", color=INK, fontsize=10)
    ax.set_title(
        f"{meta['pretty']}  ({split} set, $d$ = {meta['d']}, $n$ = {n_shown})",
        color=INK, fontsize=10.5, pad=6,
    )
    amp = meta["amplitudes"]
    s_txt = ", ".join(f"{v:g}" for v in np.round(amp["s"], 3))
    tilt = meta.get("tilt", 0.0)
    tilt_txt = f",  $c$ = {tilt:g}" if tilt else ""
    ax.annotate(
        f"dashed: exact constrained lasso posterior, {ref:.4f}\n"
        f"$a$ = {amp['a']:.3g},  $s$ = [{s_txt}]{tilt_txt}",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8, color=MUTED, zorder=8,
    )


def figure(runs: list, domain: str, out: str, title: str, window: int = 1,
           zoom: bool = True) -> None:
    fig, axes = plt.subplots(
        len(runs), 2, figsize=(11.4, 3.5 * len(runs)), squeeze=False, layout="constrained"
    )
    for row, run in enumerate(runs):
        for col, split in enumerate(("train", "test")):
            panel(axes[row][col], run, split, window, zoom)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13.5, color=INK)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def gap_figure(runs: list, out: str, title: str, window: int) -> None:
    """The same runs on log axes: distance to the exact posterior, two ways.

    Top row: ``|mean test accuracy - exact|``, which is what the paper plots,
    read as a gap so that a factor of two is visible.  Bottom row: the whitened
    distance from the running mean to the exact posterior's mean, which keeps
    resolving after the accuracy has saturated.
    """
    fig, axes = plt.subplots(
        2, len(runs), figsize=(5.2 * len(runs), 7.0), squeeze=False, layout="constrained"
    )
    for col, run in enumerate(runs):
        meta, traces = run["meta"], run["traces"]
        at = traces.get("scored_at", None)
        ref = meta["reference"]["accuracy_test"]
        for row in range(2):
            ax = axes[row][col]
            frame(ax)
            for r in meta["runs"]:
                key = r["key"]
                if row == 0:
                    y = np.abs(smooth(traces[f"{key}_accuracy_test"], window).mean(axis=1) - ref)
                    x = at if at is not None else np.arange(1, len(y) + 1)
                else:
                    y = traces[f"{key}_running_error"]
                    x = at if at is not None else np.arange(1, len(y) + 1)
                ax.plot(x, np.maximum(y, 1e-6), label=legend_label(meta, key),
                        zorder=4, **STYLE[key])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("iterations $k$", color=INK, fontsize=10)
            ax.set_ylabel(
                "gap to exact test accuracy" if row == 0
                else "whitened error of the running mean",
                color=INK, fontsize=10,
            )
            if row == 0:
                ax.set_title(f"{meta['pretty']}", color=INK, fontsize=11, pad=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, color=INK)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def write_summary(all_runs: dict, out: str) -> None:
    lines = [
        "# Lasso-regularised constrained sampling, anchored Langevin",
        "",
        "The paper's Section 3.3 experiments with `lam |x|_1` added to the potential and",
        "the dynamics replaced by anchored Langevin, so the kinked target is still the",
        "invariant law.  `J = 0` is anchored PSGLD; the other two are non-reversible",
        "anchored Langevin with skew reflection.  Accuracy is the paper's: each walker's",
        "own parameter classifies the set, averaged over walkers, with the standard",
        "deviation across walkers.  `iters to band` is the first iteration whose mean test",
        "accuracy is within 0.005 of the exact constrained lasso posterior's.",
        "",
    ]
    for (domain, suffix), runs in all_runs.items():
        if not runs:
            continue
        which = {
            "_paper": "the paper's own amplitudes",
            "_tuned": "amplitudes tuned by the sweep",
            "_noclock": "no clock",
        }.get(suffix, "calibrated amplitudes")
        lines += [f"## {DOMAIN_TITLE[domain]}, {which}", ""]
        for run in runs:
            meta, traces = run["meta"], run["traces"]
            ref = meta["reference"]["accuracy_test"]
            lines += [
                f"### {meta['pretty']}",
                "",
                f"`d` = {meta['d']}, `n_train` = {meta['n_train']}, `n_test` = {meta['n_test']}, "
                f"{meta['domain_label']}, `lam` = {meta['lam']:.4g} "
                f"(= {meta['lam_scale']:.3g} `n`), `delta` = {meta['delta']:.4g}, "
                f"`eta` = {meta['step_size']:.0e}, batch = {meta['batch_size']}, "
                f"{meta['n_iter']} iterations, {meta['n_walkers']} walkers, "
                + (
                    f"amplitude scales {meta['amp_scale']['constant']:.4g} "
                    f"(constant) and {meta['amp_scale']['state']:.4g} (state), "
                    f"tilt {meta.get('tilt', 0.0):g}."
                    if isinstance(meta["amp_scale"], dict)
                    else f"amplitude scale {meta['amp_scale']:.4g}."
                ),
                "",
                f"Exact reference: train {meta['reference']['accuracy_train']:.4f}, "
                f"test {ref:.4f} (halves "
                f"{meta['reference']['accuracy_test_halves'][0]:.4f} / "
                f"{meta['reference']['accuracy_test_halves'][1]:.4f}), acceptance "
                f"{meta['reference']['acceptance']:.2f}, |mean| "
                f"{meta['reference']['mean_radius']:.3f}, "
                f"{meta['reference']['n_near_zero']}/{meta['d']} coordinates within 0.01 of zero.",
                "",
                "| field | train | test | sd | iters to band | mean clock | boundary | failed pushes |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for row in meta["runs"]:
                A = traces[f"{row['key']}_accuracy_test"].mean(axis=1)
                hit = np.flatnonzero(np.abs(A - ref) < 0.005)
                band = str(int(hit[0]) + 1) if len(hit) else "never"
                label = row["label"].replace("$", "`").replace("\\", "")
                lines.append(
                    f"| {label} | "
                    f"{row['train_final']:.4f} | {row['test_final']:.4f} | "
                    f"{row['test_final_sd']:.4f} | {band} | {row['clock_mean']:.3f} | "
                    f"{row['boundary_rate']:.3f} | {row['failed_retractions']} |"
                )
            lines.append("")
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["synthetic", "magic", "titanic"])
    parser.add_argument("--domains", nargs="+", default=["ball", "lp"])
    parser.add_argument("--suffixes", nargs="+", default=["", "_paper"])
    parser.add_argument("--summary", default="summary_anchored_srnsgld.md")
    parser.add_argument("--gap", action="store_true",
                        help="also draw the log-log distance-to-the-posterior figure")
    parser.add_argument("--smooth", type=int, default=15,
                        help="moving average over iterations, for readability")
    parser.add_argument("--no-zoom", action="store_true",
                        help="draw the whole run instead of the region where the "
                             "curves are still moving")
    args = parser.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    all_runs = {}
    for domain in args.domains:
        for suffix in args.suffixes:
            runs = load(args.problems, domain, suffix)
            all_runs[(domain, suffix)] = runs
            if not runs:
                continue
            which = {
                "_paper": "the paper's amplitudes",
                "_tuned": "amplitudes tuned by the sweep",
                "_noclock": "no clock",
            }.get(suffix, "calibrated amplitudes")
            figure(
                runs, domain,
                os.path.join(FIGURES, f"anchored_srnsgld_{domain}{suffix}.png"),
                f"Lasso-regularised constrained logistic regression, anchored Langevin\n"
                f"{DOMAIN_TITLE[domain]}, {which}",
                args.smooth, not args.no_zoom,
            )
            if args.gap:
                gap_figure(
                    runs, os.path.join(FIGURES, f"anchored_srnsgld_{domain}{suffix}_gap.png"),
                    f"Distance to the exact constrained lasso posterior\n"
                    f"{DOMAIN_TITLE[domain]}, {which}", args.smooth,
                )
    write_summary(all_runs, os.path.join(RESULTS, args.summary))


if __name__ == "__main__":
    main()
