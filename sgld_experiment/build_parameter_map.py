"""Tabulate the parameter-range search behind ``nonreversible_beats_reversible.py``.

Reads the raw ``hunt_helper.py`` outputs saved under ``results_beat/search/`` and
writes ``results_beat/parameter_map.csv`` (one row per configuration of the
one-factor-at-a-time map around the winning configuration) and
``results_beat/parameter_map.md`` (the same as a readable table, plus the best
rows of the exploratory rounds).  Re-run the search itself with

    python parameter_sweep.py results_beat/search/<round>_configs.json out.jsonl
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

SEARCH = os.path.join("results_beat", "search")
OUT_CSV = os.path.join("results_beat", "parameter_map.csv")
OUT_MD = os.path.join("results_beat", "parameter_map.md")

FACTOR_NAMES = {
    "kappa_v2": ("v_fast (kappa = v_fast / v_slow, v_slow = 2)", "v_fast"),
    "s": ("block strength s", "s"),
    "eta": ("step size eta", "eta"),
    "lambda": ("lambda_lasso", "lambda"),
    "delta": ("anchor delta", "delta"),
    "n": ("n_total (train = 80%)", "n_total"),
    "init": ("initialisation", "init"),
    "eps": ("constraint smoothing epsilon", "epsilon"),
    "radius": ("l1 radius minus |beta|_1", "radius_offset"),
    "v1": ("v_axis (variance along the rotation axis)", "v_axis"),
    "b": ("slow-direction coefficient b", "b"),
}
BASE = "v_slow 2, v_axis 1, v_fast 64, b 1.5, e 0.25, eps 0.2, s 4, eta 7e-6, lambda 2, delta 0.02, n 2000, L1, radius |beta|_1 + 1, uniform init"


def load(name):
    rows = []
    with open(os.path.join(SEARCH, f"{name}_results.jsonl")) as handle:
        for line in handle:
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


def at(r, it):
    return min(r["trajectory"], key=lambda x: abs(x["it"] - it))


def row(r, factor="", value=""):
    p, a1, a4, fin = r["peak"], at(r, 100), at(r, 400), r["trajectory"][-1]
    d = {"factor": factor, "value": value, "tag": r["config"]["tag"],
         "peak_gap": p["diff"], "peak_iteration": p["it"], "peak_t": p["t"],
         "rev_at_peak": p["rev"], "nr_at_peak": p["nr"],
         "gap_at_100": a1["diff"], "t_at_100": a1["t"], "gap_at_400": a4["diff"],
         "gap_final": fin["diff"], "t_final": fin["t"], "final_iteration": fin["it"],
         "projection_rate_nr": r["projection_rate_nr"], "bayes_ceiling": r["bayes_ceiling"]}
    if "held_out_pooled_diff" in r:
        d.update(held_out_pooled_gap=r["held_out_pooled_diff"], held_out_pooled_t=r["held_out_pooled_t"],
                 held_out_all_positive=r["held_out_all_positive"])
    return d


def verdict(d):
    if d["projection_rate_nr"] >= 0.05:
        return "unstable (projection pinned)"
    if d["peak_gap"] >= 0.03 and d["peak_t"] >= 3:
        return "NR wins" + (" (late deficit)" if d["gap_final"] < -0.005 and d["t_final"] < -2 else "")
    if d["peak_gap"] >= 0.01 and d["peak_t"] >= 3:
        return "NR wins (small)"
    if d["gap_final"] < -0.005 and d["t_final"] < -2:
        return "NR loses"
    return "tie"


def main() -> None:
    records = []
    for r in load("map"):
        tag = r["config"]["tag"][4:]
        m = re.match(r"(kappa_v2|eta|lambda|delta|eps|radius|init|v1|s|n|b)(.*)$", tag)
        records.append(row(r, m.group(1), m.group(2)))
    df = pd.DataFrame(records)
    df["verdict"] = df.apply(verdict, axis=1)
    order = list(FACTOR_NAMES)
    df["factor_order"] = df["factor"].map(order.index)
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values(["factor_order", "value_num", "value"]).drop(columns=["factor_order", "value_num"])
    df.to_csv(OUT_CSV, index=False)

    lines = ["# Parameter-range map", "",
             "One factor at a time around the winning configuration (" + BASE + ").",
             "Paired NR-minus-REV single-iterate test accuracy over R = 60 replicates with shared "
             "initialisation and noise; `peak` is the largest gap over the 1500-iteration run, "
             "`final` the gap at iteration 1500.  Verdicts: NR wins = peak >= 0.03 with t >= 3; "
             "tie = |gap| below that everywhere; late deficit = significant negative gap at the end.", ""]
    for fac in order:
        sub = df[df["factor"] == fac]
        if sub.empty:
            continue
        lines += [f"## {FACTOR_NAMES[fac][0]}", "",
                  "| value | peak gap | at it. | t | REV / NR at peak | gap @100 | gap @400 | final gap | proj. rate | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, d in sub.iterrows():
            lines.append(f"| {d['value']} | {d['peak_gap']:+.3f} | {d['peak_iteration']} | {d['peak_t']:+.1f} | "
                         f"{d['rev_at_peak']:.3f} / {d['nr_at_peak']:.3f} | {d['gap_at_100']:+.3f} | {d['gap_at_400']:+.3f} | "
                         f"{d['gap_final']:+.4f} | {d['projection_rate_nr']:.3f} | {d['verdict']} |")
        lines.append("")

    lines += ["## Exploratory rounds (best valid configurations)", ""]
    for name, title in (("round2", "Round 2 - in-plane slow direction, L1 (384 configs)"),
                        ("round3_ball", "Round 3 - same design on the unit ball (30 configs)"),
                        ("round1", "Round 1 - slow direction on a coordinate axis, oblique to the L1 axis (288 configs)")):
        rows = [row(r) for r in load(name)]
        n_bad = sum(1 for d in rows if d["projection_rate_nr"] >= 0.05)
        rows = sorted([d for d in rows if d["projection_rate_nr"] < 0.05], key=lambda d: -d["peak_gap"])
        lines += [f"### {title}", "", f"{len(rows)} valid, {n_bad} pinned by the projection.  Top 8 by peak gap:", "",
                  "| tag | peak gap | at it. | t | REV / NR at peak | final gap | proj. rate |", "|---|---|---|---|---|---|---|"]
        for d in rows[:8]:
            lines.append(f"| {d['tag']} | {d['peak_gap']:+.3f} | {d['peak_iteration']} | {d['peak_t']:+.1f} | "
                         f"{d['rev_at_peak']:.3f} / {d['nr_at_peak']:.3f} | {d['gap_final']:+.4f} | {d['projection_rate_nr']:.3f} |")
        lines.append("")
    lines += ["## Held-out confirmations (R = 100 per seed, four extra sampler seeds)", "",
              "| tag | eval it. | search-seed gap (t) | held-out pooled gap (t) | all positive |", "|---|---|---|---|---|"]
    for name in ("confirm", "confirm_ball"):
        for r in load(name):
            d = row(r)
            lines.append(f"| {d['tag']} | {r['eval_iteration']} | {r['paired_diff']:+.3f} ({r['t']:+.1f}) | "
                         f"{d['held_out_pooled_gap']:+.3f} ({d['held_out_pooled_t']:+.1f}) | {d['held_out_all_positive']} |")
    with open(OUT_MD, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV} ({len(df)} rows) and {OUT_MD}")
    print(df.groupby("factor")["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
