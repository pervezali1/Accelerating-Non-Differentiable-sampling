"""Summary table + figure for probe_overdispersed (reads only what the probe wrote)."""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("results", "probe_overdispersed")


def load(part):
    p = os.path.join(OUT, f"confirm_{part}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def main():
    res = load("titanic") + load("magic")
    if not res:
        print("nothing to report"); return 1
    rows = []
    for r in res:
        m, b = r["matched"]["train"], r["best_vs_best"]["train"]
        mt, bt = r["matched"]["test"], r["best_vs_best"]["test"]
        rows.append(dict(exp=r["exp"], init=r["init"], n_cfg=r["n_configs"], R=r["R_conf"],
                         eta_nrev=r["best_nrev_eta"], s=r["best_nrev_s"],
                         eta_rev_best=r["best_eta_rev"],
                         nrev_train=r["nrev_best"]["train"], rev_m_train=r["rev_matched"]["train"],
                         rev_b_train=r["rev_best"]["train"],
                         d_match=m["diff"], se_match=m["se"], t_match=m["t"],
                         d_bvb=b["diff"], se_bvb=b["se"], t_bvb=b["t"],
                         nrev_test=r["nrev_best"]["test"], rev_b_test=r["rev_best"]["test"],
                         d_bvb_test=bt["diff"], t_bvb_test=bt["t"],
                         d_match_test=mt["diff"], t_match_test=mt["t"],
                         survives=bool(b["diff"] > 0 and b["t"] > 2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # figure: paired train diff, matched-eta vs best-vs-best, with +-1 se
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(df))
    ax.errorbar(x - 0.12, df.d_match, yerr=df.se_match, fmt="o", color="#1a9850", capsize=3,
                label="matched $\\eta$ (paired, same $W_0$ & noise)")
    ax.errorbar(x + 0.12, df.d_bvb, yerr=df.se_bvb, fmt="s", color="#b2182b", capsize=3,
                label="best-vs-best (reversible at its OWN best $\\eta$)")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(df.exp, df.init)], fontsize=8)
    ax.set_ylabel("paired TRAIN accuracy, non-reversible $-$ reversible")
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(fontsize=8)
    ax.set_title("Overdispersed initialisation: does the transient win survive a fair comparison?",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overdispersed_summary.png"), dpi=200)
    print("\nwrote", os.path.join(OUT, "summary.csv"), "and overdispersed_summary.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
