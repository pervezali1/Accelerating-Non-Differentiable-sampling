#!/usr/bin/env python3
"""Print a compact summary of every result file, using robust statistics.

The Wasserstein estimator's sampling distribution is heavy tailed on these
targets (``exp6``), so summaries here quote the **median** across replications
rather than the mean wherever both are available.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import runner  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "results", "data")


def _stat(entry, key, which):
    block = entry.get(key)
    if not isinstance(block, dict) or which not in block:
        return float("nan")
    return block[which][-1]


def summarize_exp0():
    path = os.path.join(DATA, "exp0_paper_replication.json")
    if not os.path.exists(path):
        return
    r = runner.load_json(path)
    print("== exp0: the paper's Section 6.4 target ==")
    print(f"  Setting A floor (mean) {r['settingA_floor']:.4f}, "
          f"midpoint estimator {r['settingA_floor_midpoint']:.4f}")
    for a in r["settingA"]:
        print(f"    {a['prior']:9s} {a['method']:9s} "
              f"iters_to_2xfloor={a.get('iters_to_2xfloor', -1):5d} "
              f"final W2 median={_stat(a, 'w2', 'median'):.4f} "
              f"mean={_stat(a, 'w2', 'mean'):.4f}")
    print(f"  Setting B (d=3, still radial) floor (mean) {r['settingB_floor']:.4f} "
          f"+- {r['settingB_floor_std']:.4f}")
    base = None
    for a in r["settingB"]:
        med = _stat(a, "w2", "median")
        if a.get("label", "").startswith("anchored"):
            base = med
        print(f"    {a.get('label', a['method']):32s} "
              f"iters_to_2xfloor={a.get('iters_to_2xfloor', -1):5d} "
              f"final W2 median={med:.4f} mean={_stat(a, 'w2', 'mean'):.4f} "
              f"diverged={a['n_diverged']}/{a['n_rep']}")
    if base is not None:
        # exclude variants that blew up: a mean-square-unstable scheme can stay
        # finite for the whole run yet reach absurd magnitudes
        ok = [a for a in r["settingB"] if a["method"] == "skew_anchored"
              and a["n_diverged"] < a["n_rep"]
              and _stat(a, "w2", "median") < 100 * r["settingB_floor"]]
        blew = [a for a in r["settingB"] if a["method"] == "skew_anchored" and a not in ok]
        if ok:
            devs = [abs(_stat(a, "w2", "median") - base) for a in ok]
            print(f"    largest median deviation of a stable skew variant from J=0: "
                  f"{max(devs):.4f}, against a floor spread of "
                  f"{r['settingB_floor_std']:.4f}")
        for a in blew:
            print(f"    {a.get('label', '')}: blew up (median W2 "
                  f"{_stat(a, 'w2', 'median'):.3g}) -- the exact analysis predicts "
                  f"mean-square instability at this stepsize")
    print()


def summarize_exp2():
    for path in sorted(glob.glob(os.path.join(DATA, "exp2_*.json"))):
        r = runner.load_json(path)
        print(f"== {os.path.basename(path)} ==")
        print(f"  floor {r['w2_floor']:.4f} +- {r['w2_floor_std']:.4f}; "
              f"|J_opt| = {r['J_optimal_norm']:.3f}")
        base_slow = None
        for a in r.get("equalbias", []):
            slow = a.get("iters_to_slow10", -1)
            if base_slow is None and slow > 0:
                base_slow = slow
            sp = f"{base_slow / slow:5.2f}x" if slow > 0 and base_slow else "    -"
            print(f"    {a['label']:28s} eta={a['eta']:.2e} "
                  f"predicted_rate={a['predicted_rate']:.5f} "
                  f"iters_to_slow10={slow:6d} speedup={sp} "
                  f"final W2 median={_stat(a, 'w2', 'median'):.4f}")
        for a in r.get("tuned", []):
            print(f"    tuned {a['label']:22s} eta={a['eta']:.2e} "
                  f"iters_to_2xfloor={a.get('iters_to_2xfloor', -1):6d} "
                  f"final W2={a.get('final_w2', float('nan')):.4f}")
        print()


def summarize_exp3():
    for path in sorted(glob.glob(os.path.join(DATA, "exp3_*.json"))):
        r = runner.load_json(path)
        print(f"== {os.path.basename(path)} ==")
        for row in r["rows"]:
            print(f"    |J|={row['J_norm']:7.2f} eta={row['eta']:.2e} "
                  f"IACT={row['iact_mean']:9.1f} [{row['iact_lo']:.1f}, {row['iact_hi']:.1f}] "
                  f"speedup={row.get('iact_speedup', float('nan')):5.2f}x")
        print()


def summarize_exp4():
    for path in sorted(glob.glob(os.path.join(DATA, "exp4_*.json"))):
        r = runner.load_json(path)
        print(f"== {os.path.basename(path)} ==")
        print(f"  two-sample floor {r['floor']:.4f}; |J_opt| = {r['J_optimal_norm']:.3f}")
        for row in r["rows"]:
            print(f"    |J|={row['J_norm']:7.2f} eta={row['eta']:.2e} "
                  f"iters_to_2xfloor={row['iters_to_2xfloor']:6d} "
                  f"iters_to_slow10={row['iters_to_slow10']:6d} "
                  f"speedup={row.get('speedup_slow', float('nan')):5.2f}x "
                  f"diverged={row['n_diverged']}")
        for e in r.get("ula", []):
            print(f"    subgradient ULA eta={e['eta']:.2e} "
                  f"iters_to_2xfloor={e['iters_to_2xfloor']:6d} "
                  f"final W2={e['final_w2']:.4f}")
        print()


def summarize_exp1_exp5():
    path = os.path.join(DATA, "exp1_theory_sweeps.json")
    if os.path.exists(path):
        r = runner.load_json(path)
        print("== exp1: exact equal-bias speed-up ==")
        dims = sorted({row["d"] for row in r["scaling"]})
        kaps = sorted({row["kappa"] for row in r["scaling"]})
        print("    kappa " + "".join(f"{'d=%d' % d:>10}" for d in dims))
        for k in kaps:
            cells = []
            for d in dims:
                m = [x for x in r["scaling"] if x["d"] == d and x["kappa"] == k]
                cells.append(f"{m[0]['speedup']:10.2f}" if m else f"{'-':>10}")
            print(f"    {k:5g} " + "".join(cells))
        for c in r.get("isotropic_control", []):
            print(f"    isotropic d={c['target']['d']:2d}: best speedup "
                  f"{c['speedup']:.4f}x")
        print()
    path = os.path.join(DATA, "exp5_rate_decomposition.json")
    if os.path.exists(path):
        r = runner.load_json(path)
        print("== exp5: SDE gain vs realisable gain ==")
        for t in r["targets"]:
            if t.get("target", {}).get("isotropic"):
                print(f"    isotropic d={t['target']['d']}: SDE rate "
                      f"{t['base_continuous_rate']:.4f} (J=0) vs "
                      f"{t['with_J_continuous_rate']:.4f} (|J|=4)")
                continue
            rows = t["rows"]
            bc = max(rows, key=lambda x: x["continuous_rate"])
            bd = max(rows, key=lambda x: x["discrete_rate"])
            tg = t["target"]
            print(f"    d={tg['d']:2d} kappa={tg['kappa']:6g}: SDE up to "
                  f"{bc['continuous_speedup']:6.1f}x, per iteration up to "
                  f"{bd['discrete_speedup']:5.2f}x at |J|={bd['J_norm']:.2f}")
        print()


if __name__ == "__main__":
    summarize_exp1_exp5()
    summarize_exp0()
    summarize_exp2()
    summarize_exp3()
    summarize_exp4()
