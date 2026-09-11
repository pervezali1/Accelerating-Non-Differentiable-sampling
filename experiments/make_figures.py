#!/usr/bin/env python3
"""Render every figure from the JSON produced by the experiment scripts.

Missing inputs are skipped with a note, so this can be run at any point.
"""

import argparse
import glob
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402

from skewanchor import plotting, runner  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "results", "data")
FIGS = os.path.join(HERE, "results", "figures")


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"  skip: {name} not found")
        return None
    return runner.load_json(path)


# ------------------------------------------------------------------ fig 1


def fig_speedup_scaling(mode):
    res = _load("exp1_theory_sweeps.json")
    if res is None:
        return
    p = plotting.use_style(mode)
    rows = res["scaling"]
    dims = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    for i, d in enumerate(dims):
        sub = sorted([r for r in rows if r["d"] == d], key=lambda r: r["kappa"])
        k = [r["kappa"] for r in sub]
        color = p["categorical"][i % len(p["categorical"])]
        axes[0].plot(k, [r["speedup"] for r in sub], "-o", color=color, label=f"d = {d}")
        plotting.label_line(axes[0], k[-1], sub[-1]["speedup"], f"d={d}", color)
        axes[1].plot(k, [r["gap_ceiling_ratio"] for r in sub], "-o", color=color, label=f"d = {d}")
        plotting.label_line(axes[1], k[-1], sub[-1]["gap_ceiling_ratio"], f"d={d}", color)

    for ax, title, ylab in [
        (axes[0], "Realised speed-up at equal bias and equal cost",
         "iterations saved  (rate ratio)"),
        (axes[1], "What the continuous-time spectral gap alone would promise",
         "gap ceiling / gap at J = 0"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("condition number of $\\Sigma$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.axhline(1.0, color=p["reference"], lw=0.9, ls="--")
        ax.margins(x=0.12)
    axes[0].legend(loc="upper left", ncols=2)
    print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig1_speedup_scaling_{mode}.png")))


# ------------------------------------------------------------------ fig 2


def fig_tuning_curve(mode):
    res = _load("exp1_theory_sweeps.json")
    if res is None:
        return
    p = plotting.use_style(mode)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    keep = [s for s in res["sweeps"] if s["target"]["d"] in (2, 3, 5, 10)][:5]
    for i, sw in enumerate(keep):
        rows = [r for r in sw["rows"] if r["kind"] in ("none", "optimal_direction")]
        rows.sort(key=lambda r: r["delta"])
        x = [r["delta"] for r in rows]
        y = [r["speedup"] for r in rows]
        t = sw["target"]
        color = p["categorical"][i % len(p["categorical"])]
        lab = f"d={t['d']}, $\\nu$={t['nu']:g}, $\\kappa$={t['kappa']:g}"
        ax.plot(x, y, "-o", color=color, label=lab)
        best = max(rows, key=lambda r: r["speedup"])
        ax.plot([best["delta"]], [best["speedup"]], "o", color=color, ms=8,
                markerfacecolor="none", markeredgewidth=1.6)
    ax.axhline(1.0, color=p["reference"], lw=0.9, ls="--")
    ax.set_xlabel("$\\|J\\|_2$  (along the optimal direction)")
    ax.set_ylabel("speed-up at equal bias")
    ax.set_title("Too little $J$ does nothing; too much forces a smaller step",
                 loc="left")
    ax.set_yscale("log")
    ax.legend(loc="lower left", fontsize=8)
    print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig2_tuning_curve_{mode}.png")))


# ------------------------------------------------------------------ fig 3


def fig_w2_curves(mode):
    for path in sorted(glob.glob(os.path.join(DATA, "exp2_*.json"))):
        res = runner.load_json(path)
        tag = os.path.basename(path)[5:-5]
        p = plotting.use_style(mode)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
        runs = [r for r in res["equalbias"] if r.get("iters")]
        colors = plotting.ramp(len(runs), mode)
        for run, color in zip(runs, colors):
            it = np.asarray(run["iters"], dtype=float)
            it[0] = max(it[1] * 0.5, 0.5) if len(it) > 1 else 1.0
            for ax, key, in ((axes[0], "w2"), (axes[1], "slow")):
                m = np.asarray(run[key]["mean"])
                lo = np.asarray(run[key]["lo"])
                hi = np.asarray(run[key]["hi"])
                ax.plot(it, m, color=color, label=run["label"])
                ax.fill_between(it, lo, hi, color=color, alpha=0.13, linewidth=0)
        floor = res["w2_floor"]
        lo_lim = max(floor * 0.35, 1e-4)
        axes[0].set_ylim(bottom=lo_lim)
        axes[0].axhspan(lo_lim, floor + res["w2_floor_std"], color=p["reference"],
                        alpha=0.16, linewidth=0)
        axes[0].annotate("estimator floor (exact draws)", xy=(it[-1], floor),
                         xytext=(-4, 3), textcoords="offset points", ha="right",
                         color=p["text_secondary"], fontsize=8, va="bottom")
        axes[1].axhline(0.10, color=p["reference"], lw=0.9, ls="--")
        axes[0].set_ylabel("sliced 2-Wasserstein distance")
        axes[0].set_title("Distance to the exact target", loc="left")
        axes[1].set_ylabel("relative variance error, slow direction")
        axes[1].set_title("Slow direction of $\\Sigma$ (higher resolution)", loc="left")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("iteration")
        axes[0].legend(loc="lower left", ncols=1, fontsize=7.5)
        fig.suptitle(f"{tag}: equal discretisation bias, equal cost per iteration",
                     x=0.01, ha="left", fontsize=10.5, color=p["text"])
        print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig3_w2_{tag}_{mode}.png")))


# ------------------------------------------------------------------ fig 4


def fig_method_comparison(mode):
    for path in sorted(glob.glob(os.path.join(DATA, "exp2_*.json"))):
        res = runner.load_json(path)
        if not res.get("tuned"):
            continue
        tag = os.path.basename(path)[5:-5]
        p = plotting.use_style(mode)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for i, e in enumerate(res["tuned"]):
            if "curve_iters" not in e:
                continue
            it = np.asarray(e["curve_iters"], dtype=float)
            it[0] = max(it[1] * 0.5, 0.5)
            color = p["categorical"][i % len(p["categorical"])]
            ax.plot(it, e["curve_w2"], color=color,
                    label=f"{e['label']}  ($\\eta$={e['eta']:.1e})")
        # the tuned grid runs at its own (smaller) particle count, so it has its
        # own estimator floor -- using the main one would mis-place the band
        floor = res.get("grid_w2_floor", res["w2_floor"])
        fstd = res.get("grid_w2_floor_std", res["w2_floor_std"])
        gc = res.get("grid_config", {})
        lo_lim = max(floor * 0.35, 1e-4)
        ax.set_ylim(bottom=lo_lim)
        ax.axhspan(lo_lim, floor + fstd, color=p["reference"], alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("sliced 2-Wasserstein distance")
        n_txt = f", {gc['n']} particles" if gc.get("n") else ""
        ax.set_title(f"{tag}: every method at its own best stepsize{n_txt}", loc="left")
        ax.legend(loc="lower left")
        print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig4_methods_{tag}_{mode}.png")))


# ------------------------------------------------------------------ fig 5


def fig_isotropic_control(mode):
    res = _load("exp1_theory_sweeps.json")
    if res is None or not res.get("isotropic_control"):
        return
    p = plotting.use_style(mode)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ctrl = res["isotropic_control"]
    labels = [f"d={c['target']['d']}\n$\\iota$={c['target']['iota']:g}" for c in ctrl]
    vals = [c["speedup"] for c in ctrl]
    ax.bar(labels, vals, color=p["categorical"][0], width=0.55)
    for x, v in zip(labels, vals):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=p["text_secondary"])
    ax.axhline(1.0, color=p["reference"], lw=1.0, ls="--")
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("best speed-up over all skew $J$")
    ax.set_title("Negative control: on an isotropic heavy tail no $J$ can help",
                 loc="left")
    print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig5_isotropic_{mode}.png")))


# ------------------------------------------------------------------ fig 6


def fig_iact(mode):
    for path in sorted(glob.glob(os.path.join(DATA, "exp3_*.json"))):
        res = runner.load_json(path)
        if not res.get("rows"):
            continue
        p = plotting.use_style(mode)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        x = [r["J_norm"] for r in res["rows"]]
        y = [r["iact_mean"] for r in res["rows"]]
        lo = [r["iact_lo"] for r in res["rows"]]
        hi = [r["iact_hi"] for r in res["rows"]]
        c = p["categorical"][0]
        ax.plot(x, y, "-o", color=c)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.set_xlabel("$\\|J\\|_2$")
        ax.set_ylabel("integrated autocorrelation time (iterations)")
        ax.set_yscale("log")
        ax.set_title("Autocorrelation of the slow coordinate, single chain", loc="left")
        tag = os.path.basename(path)[:-5]
        print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig6_{tag}_{mode}.png")))


# ------------------------------------------------------------------ fig 7


def fig_nonsmooth(mode):
    """The composite target, with and without the warm-up ramp side by side."""
    plain = sorted(glob.glob(os.path.join(DATA, "exp4_nonsmooth_*[0-9].json")))
    ramped = sorted(glob.glob(os.path.join(DATA, "exp4_nonsmooth_*_ramp.json")))
    if not plain:
        print("  skip: no exp4 data")
        return
    p = plotting.use_style(mode)
    panels = [("no warm-up: the rotation flings the prior's stiff-direction\n"
               "excess along the soft axis", plain[0])]
    if ramped:
        panels.append(("with a warm-up ramp: the hump is gone and it converges\n"
                       "no later", ramped[0]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 4.1), squeeze=False)
    for ax, (title, path) in zip(axes[0], panels):
        res = runner.load_json(path)
        colours = plotting.ramp(len(res["rows"]), mode)
        for row, colour in zip(res["rows"], colours):
            it = np.asarray(row["iters"], dtype=float)
            it[0] = max(it[1] * 0.5, 0.5)
            ax.plot(it, row["w2"], color=colour, label=f"$\\|J\\|$ = {row['J_norm']:.2f}")
        for e in res.get("ula", [])[:1]:
            it = np.asarray(e["iters"], dtype=float)
            it[0] = max(it[1] * 0.5, 0.5)
            ax.plot(it, e["w2"], color=p["categorical"][1], ls=":", label="subgradient ULA")
        ax.axhspan(0, res["floor"] * 1.15, color=p["reference"], alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("sliced 2-Wasserstein distance")
        ax.set_title(title, loc="left", fontsize=9.5)
    ylims = [ax.get_ylim() for ax in axes[0]]
    lo, hi = min(y[0] for y in ylims), max(y[1] for y in ylims)
    for ax in axes[0]:
        ax.set_ylim(lo, hi)
    axes[0][0].legend(loc="lower left", ncols=2, fontsize=8)
    tag = os.path.basename(plain[0])[:-5]
    print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig7_{tag}_{mode}.png")))


# ------------------------------------------------------------------ fig 8


def fig_flow_field(mode):
    """Why tilting helps: the rotation should feed the stiff axis, not fight it."""
    from skewanchor import skewfield as sf
    from skewanchor.targets import anisotropic_student_t

    T = anisotropic_student_t(2, 5.0, 100.0)
    p = plotting.use_style(mode)
    sd = np.sqrt(np.diag(T.cov()))
    gx = np.linspace(-3.2 * sd[0], 3.2 * sd[0], 34)
    gy = np.linspace(-3.2 * sd[1], 3.2 * sd[1], 34)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    dens = np.exp(-T.U(pts)).reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, (a, title) in zip(axes, [(0.0, "constant field: uniform circulation"),
                                     (-2.0, "tilted field: strong on the soft axis, "
                                            "reversed on the stiff one")]):
        c = sf.StreamField2D.quadrupole(T, 4.95, a, 0.0).drift(pts)
        U = c[:, 0].reshape(X.shape)
        V = c[:, 1].reshape(X.shape)
        mag = np.hypot(U, V)
        ax.contour(X, Y, dens, levels=6, colors=p["reference"], linewidths=0.6, alpha=0.7)
        ax.streamplot(gx, gy, U, V, color=np.log10(mag + 1e-9), cmap="Blues",
                      density=1.1, linewidth=0.9, arrowsize=0.8)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("stiff direction")
        ax.set_ylabel("soft direction")
        ax.set_xlim(gx[0], gx[-1])
        ax.set_ylim(gy[0], gy[-1])
        ax.grid(False)
    print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig8_flow_field_{mode}.png")))


def fig_state_dependent(mode):
    res = _load("exp7_state_dependent_k100.json")
    if res is None or not res.get("measured"):
        return
    p = plotting.use_style(mode)
    rows = [r for r in res["measured"] if r.get("kind") == "stream"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    integs = sorted({r["integrator"] for r in rows})
    for i, integ in enumerate(integs):
        for j, jn in enumerate(sorted({r["J_norm"] for r in rows if r["integrator"] == integ})):
            sub = sorted([r for r in rows if r["integrator"] == integ and r["J_norm"] == jn],
                         key=lambda r: r["tilt"])
            colour = p["categorical"][(2 * i + j) % len(p["categorical"])]
            lab = f"{integ}, $\|J\|$={jn:g}"
            ax.plot([r["tilt"] for r in sub], [r["speedup"] for r in sub], "-o",
                    color=colour, label=lab)
    const = [r for r in res["measured"] if r.get("kind") == "constant"]
    if const:
        best = max(r["speedup"] for r in const)
        ax.axhline(best, color=p["reference"], lw=1.0, ls="--")
        ax.annotate(f"best constant field ({best:.1f}x)", xy=(ax.get_xlim()[0], best),
                    xytext=(3, 3), textcoords="offset points",
                    color=p["text_secondary"], fontsize=8)
    ax.set_xlabel("quadrupole tilt $a$   (negative = rotate harder on the soft axis)")
    ax.set_ylabel("speed-up at equal accuracy")
    ax.set_yscale("log")
    ax.set_title("State dependence beats any constant field", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig9_state_dependent_{mode}.png")))


# ----------------------------------------------------------------- fig 10


def fig_three_way(mode):
    """J = 0, J constant, J state dependent -- one curve each."""
    for path in sorted(glob.glob(os.path.join(DATA, "exp8_three_way_*.json"))):
        res = runner.load_json(path)
        rows = res.get("rows") or []
        if len(rows) < 2:
            continue
        p = plotting.use_style(mode)
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        floor = res["floor"]
        for i, row in enumerate(rows):
            colour = p["categorical"][i % len(p["categorical"])]
            it = np.asarray(row["iters"], dtype=float)
            it[0] = max(it[1] * 0.5, 0.5)
            w = np.asarray(row["w2"], dtype=float)
            ax.plot(it, w, color=colour, label=row["label"], zorder=3 - i * 0.1)
            hit = row.get("iters_to_2xfloor", -1)
            if hit and hit > 0:
                j = int(np.argmin(np.abs(it - hit)))
                ax.plot([it[j]], [w[j]], "o", color=colour, ms=6,
                        markeredgecolor=p["surface"], markeredgewidth=1.4, zorder=4)
                sp = row.get("speedup")
                txt = f"{hit:d} iters" + (f"  ({sp:.1f}x)" if sp and sp > 1.01 else "")
                ax.annotate(txt, xy=(it[j], w[j]), xytext=(0, 11),
                            textcoords="offset points", ha="center", va="bottom",
                            color=colour, fontsize=8.5, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.18", fc=p["surface"],
                                      ec="none", alpha=0.85))
        ax.axhspan(0, floor * 1.12, color=p["reference"], alpha=0.18, linewidth=0)
        ax.axhline(2 * floor, color=p["reference"], lw=0.9, ls="--")
        ax.annotate("twice the measurement floor", xy=(1.0, 2 * floor),
                    xytext=(0, 4), textcoords="offset points",
                    color=p["text_secondary"], fontsize=8, va="bottom")
        ax.annotate("measurement floor", xy=(ax.get_xlim()[0], floor * 0.78),
                    xytext=(6, 0), textcoords="offset points",
                    color=p["text_secondary"], fontsize=8, ha="left", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("sliced 2-Wasserstein distance")
        which = "heavy tailed and non-differentiable" if "composite" in path else "heavy tailed"
        ax.set_title(f"Anchored Langevin on a {which} target", loc="left")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
        tag = os.path.basename(path)[:-5]
        print(" ", plotting.finish(fig, os.path.join(FIGS, f"fig10_{tag}_{mode}.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["light", "dark"])
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    for mode in args.modes:
        print(f"[{mode}]")
        fig_speedup_scaling(mode)
        fig_tuning_curve(mode)
        fig_w2_curves(mode)
        fig_method_comparison(mode)
        fig_isotropic_control(mode)
        fig_iact(mode)
        fig_nonsmooth(mode)
        fig_flow_field(mode)
        fig_state_dependent(mode)
        fig_three_way(mode)


if __name__ == "__main__":
    main()
