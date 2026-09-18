"""Run many ``hunt_helper.py`` configurations in parallel and tabulate them.

    python3 parameter_sweep.py CONFIGS.json RESULTS.jsonl [--workers 3]

``CONFIGS.json`` is a list of hunt_helper config dicts (an optional ``"tag"``
key is carried through untouched).  Each finished run is appended to
``RESULTS.jsonl`` as it completes, and a one-line summary is printed:

    tag | eval it  rev  nr  diff  t | peak it diff t | proj  rising | held pooled diff t allpos | s

Every worker is a separate single-threaded process (BLAS threads pinned to 1),
so ``--workers`` should be the number of physical cores you want to use.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(cfg: dict) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    proc = subprocess.run([sys.executable, "hunt_helper.py", json.dumps(cfg)],
                          capture_output=True, text=True, env=env, cwd=HERE, timeout=3600)
    if proc.returncode != 0:
        return {"config": cfg, "error": proc.stderr[-3000:], "seconds": time.time() - t0}
    result = json.loads(proc.stdout)
    result["seconds"] = time.time() - t0
    return result


def summary_line(r: dict) -> str:
    tag = r["config"].get("tag", "")
    if "error" in r:
        return f"{tag} | ERROR {r['error'].strip().splitlines()[-1] if r['error'].strip() else '?'}"
    p = r.get("peak") or {}
    held = ""
    if "held_out_pooled_diff" in r:
        held = (f" | held {r['held_out_pooled_diff']:+.4f} t={r['held_out_pooled_t']:+.2f}"
                f" allpos={r['held_out_all_positive']}")
    return (f"{tag} | @{r['eval_iteration']} rev={r['rev_acc']:.4f} nr={r['nr_acc']:.4f}"
            f" diff={r['paired_diff']:+.4f} t={r['t']:+.2f}"
            f" | peak @{p.get('it')} {p.get('diff', float('nan')):+.4f} t={p.get('t', float('nan')):+.2f}"
            f" | proj={r['projection_rate_nr']:.3f} rising={r['rev_still_rising']}{held}"
            f" | {r['seconds']:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs")
    ap.add_argument("results")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    configs = json.load(open(args.configs))
    print(f"{len(configs)} configs, {args.workers} workers", flush=True)
    t0 = time.time()
    with open(args.results, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_one, c): c for c in configs}
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"[{n}/{len(configs)} {time.time() - t0:.0f}s] {summary_line(r)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
