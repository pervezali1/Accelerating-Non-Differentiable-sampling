#!/usr/bin/env python3
"""Both fields against ``J = 0``, and whether the rate criterion earns its keep.

Left: the accuracy gap each field leaves at the end of a fixed 250-iteration
budget, as a ratio over ``J = 0``, so right of the dashed line the field wins.
The state-dependent field is shown in whichever of the two frames is better --
the paper's column order or the searched signed permutation -- since which
frame to read ``J_s`` in is a free design choice like the amplitude, and the bar
is annotated with the one that won.

Right: does the criterion pick that frame?  The axes are the searched frame over
the column frame -- predicted rate ratio against measured gap ratio -- so every
point is right of 1 by construction and the question is only whether it is
*above* 1.  The sublevel sets say yes and the balls say no, which is the
limitation worth seeing rather than describing: on a ball ``J(x) x = 0`` holds
everywhere and survives conjugation, so no frame can change the radial motion
that is the whole of the journey there, yet the criterion scores the full
spectrum and credits it anyway.
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

# validated for CVD separation and contrast in both light and dark surfaces
CONSTANT, COLUMNS, SPECTRAL = "#2B5FD9", "#0E9F5C", "#C2410C"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#d8dade"
PRETTY = {"titanic": "Titanic", "magic": "MAGIC", "synthetic": "Synthetic"}
SET = {"ball": "ball", "lp": "sublevel"}
# the step size at which 250 iterations is a budget rather than a formality
ETA = {"titanic": 1e-4, "magic": 1.5625e-6, "synthetic": 6.25e-6}
SERIES = (
    ("constant", CONSTANT, r"constant $J_a$"),
    ("state", SPECTRAL, r"state-dependent $J_s$ / $J_g$, better frame"),
)
FRAME_MARK = {"columns": "column frame", "spectral": "searched frame"}


class _Args:
    """The handful of fields ``build`` reads, at their defaults."""

    def __init__(self, problem, domain, reg):
        self.problem, self.domain, self.regularizer = problem, domain, reg
        self.amp_scale = 1.0
        self.amp_scale_constant = self.amp_scale_state = None
        self.tilt, self.no_clock, self.step_size, self.seed = 0.0, False, None, 0
        self.n_iter = self.n_walkers = self.score_every = None
        self.start_radius, self.lam_scale, self.lam, self.delta = 1.0, None, None, None
        self.clock_budget, self.rotation_budget, self.split_seed = 0.5, 0.1, 0
        self.profile, self.block_order = "paper", "columns"


_SETUP: dict = {}


def setup(problem: str, domain_key: str, reg: str):
    """``(dcfg, domain, target, x_star, H)``, built once per cell and cached."""
    from run_anchored_srnsgld import build  # noqa: E402
    from nds.anchored_constrained import anchor_hessian, constrained_lasso_map

    key = (problem, domain_key, reg)
    if key not in _SETUP:
        _, dcfg, _, domain, target, _, _ = build(
            problem, domain_key, _Args(problem, domain_key, reg))
        x_star = constrained_lasso_map(target, domain)
        _SETUP[key] = (dcfg, domain, target, x_star, anchor_hessian(target, x_star))
    return _SETUP[key]


def predicted_rate(problem, domain_key, reg, order, kappa, tilt) -> float:
    """Slowest rate of ``(I + J) H`` for the state field, over the ``J = 0`` one.

    Recomputed rather than read off the sweep rows, so the figure does not
    depend on which sweeps happened to record it, and so both frames are scored
    the same way.
    """
    from sweep_anchored import make_field  # noqa: E402
    from nds.constrained import slowest_rate, spectral_frame
    from nds.anchored_constrained import outward_tilt_direction

    dcfg, domain, target, x_star, H = setup(problem, domain_key, reg)
    d = target.d
    direction = outward_tilt_direction(target, domain, seed=0)
    plain = make_field("state", d, dcfg, kappa, domain_key, "paper", None,
                       tilt=tilt, domain=domain, direction=direction)
    Q = (spectral_frame(plain, H, x_star, seed=0) if order == "spectral"
         else np.eye(d))
    J = Q @ plain.matrix(Q.T @ x_star.ravel()) @ Q.T
    return slowest_rate(J, H) / slowest_rate(np.zeros((d, d)), H)


def frame(ax) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)


def ratio(base: dict, pick: dict, stat: str) -> tuple:
    b, sb = base[stat], base.get(stat + "_se", 0.0)
    p, sp = max(pick[stat], 1e-12), pick.get(stat + "_se", 0.0)
    r = b / p
    return r, r * np.sqrt((sb / max(b, 1e-12)) ** 2 + (sp / p) ** 2)


def collect(n_iter: int, guard: str, max_fail: float) -> dict:
    from select_anchored import select  # noqa: E402

    groups: dict = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "sweep_anchored_*.json"))):
        meta = json.load(open(path))
        order = meta.get("block_order", "columns")
        if order not in ("columns", "spectral"):
            continue
        for r in meta["rows"]:
            if r["n_iter"] != n_iter or "gap_tail" not in r:
                continue
            if abs(r["eta"] - ETA.get(meta["problem"], -1)) > 1e-12:
                continue
            key = (meta["problem"], meta["domain"], meta.get("regularizer", "l1"),
                   tuple(meta["seeds"]))
            groups.setdefault(key, {}).setdefault(order, []).append(r)

    picked: dict = {}
    for (problem, domain, reg, seeds), per in groups.items():
        for order, rows in per.items():
            base = next((r for r in rows if r["field"] == "zero"), None)
            if base is None:
                continue
            for field in ("constant", "state"):
                pick, clean = select(rows, field, base, max_fail, "gap", guard)
                if pick is None:
                    continue
                cell = (problem, domain, reg, field, order)
                entry = {
                    "gap": ratio(base, pick, "gap_tail"),
                    "err": ratio(base, pick, guard),
                    "rate": (pick.get("slowest_rate") or 0.0)
                    / max(base.get("slowest_rate") or 0.0, 1e-12)
                    if base.get("slowest_rate") else None,
                    "clean": clean,
                    "seeds": len(seeds),
                    "kappa": pick["kappa"],
                    "tilt": pick.get("tilt", 0.0),
                }
                old = picked.get(cell)
                if old is None or entry["seeds"] > old["seeds"]:
                    picked[cell] = entry
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--guard", default="error_running",
                        choices=["error", "error_running"])
    parser.add_argument("--max-fail-rate", type=float, default=0.001)
    parser.add_argument("--out", default="anchored_frames.png")
    args = parser.parse_args()

    picked = collect(args.n_iter, args.guard, args.max_fail_rate)
    cells = [(p, d, r) for p in ("titanic", "magic", "synthetic")
             for d in ("ball", "lp") for r in ("l1", "group", "tv")]
    cells = [c for c in cells
             if (*c, "constant", "columns") in picked
             or (*c, "state", "columns") in picked
             or (*c, "state", "spectral") in picked]

    def better(cell):
        """The better of the two frames for the state field, and its name."""
        opts = [(o, picked[(*cell, "state", o)]) for o in ("columns", "spectral")
                if (*cell, "state", o) in picked]
        if not opts:
            return None, None
        order, entry = max(opts, key=lambda t: t[1]["gap"][0])
        return order, entry

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(15.0, 7.4), layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    # horizontal bars, so eighteen cell names read straight across with no
    # rotation and nothing to collide with
    y = np.arange(len(cells))[::-1]
    height = 0.38
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c7cad0")
    ax.tick_params(colors=MUTED, labelsize=9)

    for k, (field, colour, label) in enumerate(SERIES):
        vals, errs, flags, notes = [], [], [], []
        for cell in cells:
            if field == "constant":
                e = picked.get((*cell, "constant", "columns"))
                note = ""
            else:
                order, e = better(cell)
                note = FRAME_MARK.get(order, "")
            vals.append(e["gap"][0] if e else np.nan)
            errs.append(e["gap"][1] if e else 0.0)
            flags.append(bool(e and not e["clean"]))
            notes.append(note)
        bars = ax.barh(y + (0.5 - k) * height, vals, height * 0.9,
                       xerr=errs, capsize=2, color=colour, label=label,
                       error_kw=dict(lw=1.0, ecolor=MUTED))
        for bar, flag, note, v, er in zip(bars, flags, notes, vals, errs):
            if flag:  # no amplitude met the error guard; best gap shown
                bar.set_hatch("//")
                bar.set_edgecolor("#ffffff")
                bar.set_linewidth(0.0)
            if note and np.isfinite(v):
                ax.annotate(note, xy=(v + er, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7, color=MUTED, va="center")
    ax.axvline(1.0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 60)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{PRETTY[p]}  {SET[d]}  \u00b7 {r}" for p, d, r in cells],
                       fontsize=8.5, color=INK)
    ax.set_xlabel("accuracy gap left at 250:  $J = 0$ / field", color=INK, fontsize=10)
    ax.set_title(
        "Both fields reach or beat $J = 0$ in all eighteen cells\n"
        "(right of the dashed line the field wins; MAGIC sublevel group ties)",
        color=INK, fontsize=11.5, pad=8,
    )
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    frame(bx)
    bx.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.7)
    STYLE_BY_SET = {"lp": (SPECTRAL, "o", "sublevel set"),
                    "ball": (CONSTANT, "s", "ball")}
    points = []
    for cell in cells:
        problem, domain_key, reg = cell
        a = picked.get((*cell, "state", "columns"))
        b = picked.get((*cell, "state", "spectral"))
        if not a or not b:
            continue
        ra = predicted_rate(problem, domain_key, reg, "columns", a["kappa"], a["tilt"])
        rb = predicted_rate(problem, domain_key, reg, "spectral", b["kappa"], b["tilt"])
        points.append((domain_key, rb / ra, b["gap"][0] / a["gap"][0],
                       f"{PRETTY[problem]} {reg}"))
    for domain_key, (colour, marker, label) in STYLE_BY_SET.items():
        sel = [pt for pt in points if pt[0] == domain_key]
        bx.scatter([pt[1] for pt in sel], [pt[2] for pt in sel], s=70, color=colour,
                   marker=marker, label=label, edgecolor="#ffffff", linewidth=1.2,
                   zorder=4)
    # fan the labels vertically in one global order, so the two series cannot
    # collide with each other either
    for rank, (domain_key, xv, yv, tag) in enumerate(sorted(points, key=lambda t: t[1])):
        dy = (-15, 8, -6, 17, -25, 0)[rank % 6]
        bx.annotate(tag, xy=(xv, yv), xytext=(9, dy), textcoords="offset points",
                    fontsize=7.5, color=MUTED, va="center")
    if points:
        lo = min(pt[1] for pt in points)
        hi = max(pt[1] for pt in points)
        bx.set_xlim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))
    bx.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=3)
    bx.set_xlabel("predicted: slowest rate of $(I + J)H$,\nsearched frame / column frame",
                  color=INK, fontsize=10)
    bx.set_ylabel("measured: accuracy gap ratio,\nsearched frame / column frame",
                  color=INK, fontsize=10)
    bx.set_title("Does the rate criterion pick the better frame?\n"
                 "the search always raises the rate, so only height decides",
                 color=INK, fontsize=11.5, pad=8)
    bx.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        f"Anchored SRNSGLD against anchored PSGLD, {args.n_iter} iterations, "
        "three non-differentiable penalties",
        fontsize=12.5, color=INK,
    )
    fig.supxlabel(
        "Amplitude, tilt and frame selected by experiments/select_anchored.py on "
        "the runs shown; hatched: no amplitude met the error guard, best gap shown.",
        fontsize=8, color=MUTED,
    )
    out = os.path.join(FIGURES, args.out)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
