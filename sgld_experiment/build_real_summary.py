"""Summarise the real-data runs: results_real/summary.md from the final JSONs and the search rounds."""
from __future__ import annotations

import glob
import json
import os

OUT = os.path.join("results_real", "summary.md")
FINALS = [("titanic_l1", "Titanic, L1-smooth ball"), ("titanic_ball", "Titanic, unit ball"),
          ("magic_l1", "MAGIC telescope, L1-smooth ball"), ("magic_ball", "MAGIC telescope, unit ball")]
GROUPS = [("standard scaling, natural order", lambda t: "_standard_" in t and "ball" not in t),
          ("standard scaling, importance order", lambda t: "_importance_" in t),
          ("equalised coefficients", lambda t: "_equalise_" in t),
          ("raw (unstandardised) scales", lambda t: "_raw_" in t),
          ("block reparametrisation, L1", lambda t: "_design_" in t and "_ball_" not in t),
          ("standard scaling, unit ball", lambda t: "_ball_standard_" in t),
          ("block reparametrisation, unit ball", lambda t: "_ball_design_" in t)]


def main() -> None:
    lines = ["# Real data: reversible vs non-reversible anchored Langevin", "",
             "Final runs: R = 100 shared-randomness replicates, held-out confirmation on four extra sampler seeds "
             "(R = 100 each) at the iteration where the search-run gap peaked.  Search rounds: R = 40, 1000 iterations "
             "(2000 for the raw-scale runs), MAGIC subsampled to 4000 training rows.", "",
             "## Final runs", "",
             "| run | n_train | reference acc. | REV / NR at the peak | peak gap (it.) | held-out pooled gap | t | gap at the end |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, title in FINALS:
        path = os.path.join("results_real", f"{tag}.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path)); c = s["curves"]; info = s["info"]
        gaps = [n - r for n, r in zip(c["nr_test"], c["rev_test"])]
        k = max(range(len(gaps)), key=lambda i: gaps[i])
        lines.append(f"| {title} | {info['n_train']} | {info['reference_test_accuracy']:.3f} | {c['rev_test'][k]:.3f} / {c['nr_test'][k]:.3f} | "
                     f"{gaps[k]:+.3f} ({c['iteration'][k]}) | {s['pooled_diff']:+.3f} | {s['pooled_t']:+.1f} | {s['trajectory'][-1]['paired_diff']:+.4f} |")
    lines += ["", "## Search: best valid configuration per preprocessing ladder (projection rate < 0.05)", "",
              "| data set | ladder | best peak gap (it.) | t | REV / NR | configurations |", "|---|---|---|---|---|---|"]
    rows = []
    for path in sorted(glob.glob(os.path.join("results_real", "search", "*_results.jsonl"))):
        rows += [json.loads(l) for l in open(path)]
    rows = [r for r in rows if "error" not in r]
    for ds in ("titanic", "magic"):
        for name, pred in GROUPS:
            rs = [r for r in rows if r["config"]["dataset"] == ds and pred(r["config"]["tag"]) and r["projection_rate_nr"] < 0.05]
            if not rs:
                continue
            b = max(rs, key=lambda r: r["peak"]["diff"]); p = b["peak"]
            lines.append(f"| {ds} | {name} | {p['diff']:+.3f} ({p['it']}) | {p['t']:+.1f} | {p['rev']:.3f} / {p['nr']:.3f} | {len(rs)} |")
    with open(OUT, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
