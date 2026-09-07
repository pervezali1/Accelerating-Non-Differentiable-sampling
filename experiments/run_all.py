"""Populate ``results/`` with every run the notebook asks for.

The notebook calls exactly the same ``ands.experiments`` entry points, so running this
first turns notebook execution into a read from cache.  Runs are printed as they finish
so a long job stays legible.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)

from ands import diagnostics as G          # noqa: E402
from ands import experiments as E          # noqa: E402

CFG = E.CONFIG


def main(keys=("titanic", "magic")):
    for key in keys:
        t_start = time.time()
        ref = E.reference(key)
        tgt, R = ref["target"], E.DATASETS[key]["R"]

        jobs = [("none", 0.0, False, CFG["eta"], CFG["n_steps"])]
        for s in CFG["sweep"][1:]:
            jobs.append(("const", s, False, CFG["eta"], CFG["n_steps"]))
            jobs.append(("axial", s, False, CFG["eta"], CFG["n_steps"]))
            jobs.append(("axial", s, True, CFG["eta"], CFG["n_steps"]))
        T = CFG["eta"] * CFG["n_steps"]
        for eta in (CFG["eta"] / 2, CFG["eta"] / 4):
            n_steps = int(round(T / eta))
            for kind, s in [("none", 0.0), ("const", CFG["s_main"]), ("axial", CFG["s_main"])]:
                jobs.append((kind, s, False, eta, n_steps))

        for i, (kind, s, drop, eta, n_steps) in enumerate(jobs, 1):
            t0 = time.time()
            r = E.chain(key, kind, s, eta=eta, n_steps=n_steps, ref=ref["ref"],
                        drop_correction=drop)
            sc = E.score(ref["ref"], r["x"], R)
            print("[{}] {:2d}/{:2d} {:5s} s={:.0f} drop={:d} eta={:.1e} steps={:5d}"
                  "  W1={:.4f}  worst={:.4f}  KS={:.3f}  dK={:5.2f}%  ({:.0f}s)".format(
                      key, i, len(jobs), kind, s, drop, eta, n_steps, sc["W1_mean"],
                      float(sc["W1"].max()), sc["maxKS"], 100 * sc["boundary"],
                      time.time() - t0), flush=True)
        print("[{}] done in {:.0f}s   floor={:.4f}  target dK={:.2%}\n".format(
            key, time.time() - t_start, ref["floor"].mean(),
            G.boundary_mass(ref["ref"], R)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ("titanic", "magic"))
