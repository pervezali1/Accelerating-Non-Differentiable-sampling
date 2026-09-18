"""probe_overdispersed.py -- HYPOTHESIS "overdispersed".

Question
--------
The only place the non-reversible arm can win is the TRANSIENT (the continuous process has the
same invariant law for both arms -- established, not re-litigated here).  So: make the transient
HARD by a declared design choice on the INITIALISATION, not by crippling the step size.

Three initialisations, all legitimate design choices fixed in advance:
  * ball_uniform : the reference used by all earlier runs (uniform in the unit ball).
  * boundary     : uniform direction on the sphere, radially pushed onto the BOUNDARY of K.
                   Maximally overdispersed inside K, and where |J| is largest.
  * antimap      : ALL replicates start at one fixed adversarial point, the boundary point in the
                   direction OPPOSITE to the (training-data) MAP.  Replicates still differ,
                   because each replicate gets its own Gaussian noise stream.

Protocol (the four "defensibility" requirements)
------------------------------------------------
(a) PAIRED: for every comparison both arms get the SAME W0 array and the SAME seed, so replicate i
    of the reversible arm and replicate i of the non-reversible arm see the identical Gaussian
    noise stream.  All standard errors are standard errors of the per-replicate DIFFERENCE.
(b) SELECTION IS ON TRAINING ACCURACY ONLY.  Test labels are never looked at until the headline
    numbers are printed, and never enter any argmax.  Search runs use seed SEARCH_SEED; the
    headline comes from a CONFIRMATION run at CONFIRM_SEED, a seed not used in the search.
(c) BEST-vs-BEST: the reversible arm gets its OWN best step size from the SAME initialisation,
    searched over the SAME eta grid.  Both the matched-eta comparison (which is where the earlier
    transient "win" lived) and the best-vs-best comparison are reported.
(d) paired SE, paired t, and the number of configurations tried are all reported.

Headline metric (fixed in advance): LAST-ITERATE TRAINING accuracy at the fixed iteration budget
of the experiment.  Test accuracy at the same configurations is reported alongside, purely as a
read-out.

Nothing in this file modifies any existing module; everything is imported from exact_anchored.py
and nral.py.
"""
from __future__ import annotations

import argparse
import json
import zlib
import os
import time

import numpy as np
import pandas as pd

from nral import BallGeometry, build_magic_dataset, build_titanic_dataset
from exact_anchored import (D10, Exp10, Potential, accuracy10, geom_for, init_unit_ball10,
                            make_potential, run_exact)

OUTDIR = os.path.join("results", "probe_overdispersed")

SEARCH_SEED = 5100          # used for every selection run
CONFIRM_SEED = 6700         # never used in any argmax
ETAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
SHRINK = 1e-6               # pull the boundary init just inside K so feasible() passes


# ----------------------------------------------------------------- MAP direction (training only)
def map_point(pot: Potential, geom) -> np.ndarray:
    """Projected gradient descent on the SMOOTH anchor U_0 inside K.  Training data only."""
    best, best_u = None, np.inf
    for eta in (1e-3, 1e-4, 1e-5, 1e-6):
        w = np.zeros((1, D10))
        for _ in range(4000):
            w = geom.project(w - eta * pot.grad_U0(w))[0]
            if not np.isfinite(w).all():
                w = None
                break
        if w is None:
            continue
        u = float(pot.U0(w)[0])
        if u < best_u:
            best_u, best = u, w
    assert best is not None
    return best[0]


# ----------------------------------------------------------------- initialisations
def make_W0(kind: str, geom, pot: Potential, R: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "ball_uniform":
        W0 = init_unit_ball10(rng, R)
    elif kind == "boundary":
        Z = rng.standard_normal((R, D10))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        W0 = np.asarray(geom.to_boundary(Z)) * (1.0 - SHRINK)
    elif kind == "antimap":
        wm = map_point(pot, geom)
        d = -wm / np.linalg.norm(wm)
        b = np.asarray(geom.to_boundary(d[None, :])) * (1.0 - SHRINK)
        W0 = np.repeat(b, R, axis=0)
    else:
        raise ValueError(kind)
    assert np.all(geom.feasible(W0)), f"{kind} init left K"
    return np.ascontiguousarray(W0)


# ----------------------------------------------------------------- J strength patterns
def gradnorm_scales(pot: Potential, W0: np.ndarray, c: float) -> tuple:
    """s_l proportional to 1 / ||grad_{I_l} U_0||, normalised to geometric mean c.

    Uses the gradient at the INITIAL states only (training data, no test labels, no accuracy)."""
    G = np.abs(pot.grad_U0(W0)).mean(axis=0)
    nb = np.array([np.linalg.norm(G[1:4]), np.linalg.norm(G[4:7]), np.linalg.norm(G[7:10])])
    inv = 1.0 / np.maximum(nb, 1e-12)
    inv = inv / float(np.exp(np.log(inv).mean()))
    return tuple(float(c * v) for v in inv)


def s_patterns(pot: Potential, W0: np.ndarray) -> dict:
    pat = {f"u{c:g}": (c, c, c) for c in (0.25, 1.0, 2.0, 5.0)}
    pat["b2_5"] = (0.0, 5.0, 0.0)                       # block 2 only (prior finding on Titanic)
    for c in (1.0, 5.0):
        pat[f"gn{c:g}"] = gradnorm_scales(pot, W0, c)   # gradient-normalised per-block strengths
    return pat


# ----------------------------------------------------------------- one paired cell
def one_arm(ds, pot, geom, n_iter, alpha, scales, eta, W0, seed):
    r = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                  W0=W0, seed=seed, checkpoint_every=max(1, n_iter // 10))
    Wl = r["betas"][-1]
    return dict(train=accuracy10(ds.X_train, ds.y_train, Wl[None])[0],
                test=accuracy10(ds.X_test, ds.y_test, Wl[None])[0],
                U=pot.U(Wl), proj=r["projection_rate"], nbad=r["n_nonfinite"])


def paired_stats(a_nrev, a_rev, key="train"):
    d = a_nrev[key] - a_rev[key]
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) + 1e-300
    return dict(diff=float(d.mean()), se=se, t=float(d.mean() / se), win=float((d > 0).mean()), n=n)


# ----------------------------------------------------------------- the probe
def probe(exp: Exp10, ds, pot, init_kind: str, R_search: int, R_conf: int, rows: list,
          verbose=True, etas=None, lean=False):
    etas = ETAS if etas is None else etas
    geom = geom_for(exp)
    rng = np.random.default_rng(zlib.crc32(f"{exp.key}|{init_kind}".encode()))
    W0s = make_W0(init_kind, geom, pot, R_search, rng)
    pats = s_patterns(pot, W0s)
    if lean:
        pats = {k: v for k, v in pats.items() if k in ("u1", "u5", "gn5")}
    tag = f"{exp.key}/{init_kind}"

    # ---------- SEARCH (training accuracy only) ----------
    rev_search, nrev_search, n_cfg = {}, {}, 0
    for eta in etas:
        a_rev = one_arm(ds, pot, geom, exp.n_iter, 0, (0, 0, 0), eta, W0s, SEARCH_SEED)
        rev_search[eta] = a_rev
        n_cfg += 1
        rows.append(dict(stage="search", exp=exp.key, init=init_kind, arm="rev", eta=eta,
                         s="-", train=float(a_rev["train"].mean()), test=float(a_rev["test"].mean()),
                         proj=a_rev["proj"], nbad=a_rev["nbad"]))
        for name, sc in pats.items():
            a_nr = one_arm(ds, pot, geom, exp.n_iter, 1, sc, eta, W0s, SEARCH_SEED)
            nrev_search[(eta, name)] = a_nr
            n_cfg += 1
            st = paired_stats(a_nr, a_rev)
            rows.append(dict(stage="search", exp=exp.key, init=init_kind, arm="nrev", eta=eta,
                             s=name, train=float(a_nr["train"].mean()),
                             test=float(a_nr["test"].mean()), proj=a_nr["proj"],
                             nbad=a_nr["nbad"], d_train=st["diff"], t_train=st["t"]))
    best_eta_rev = max(etas, key=lambda e: rev_search[e]["train"].mean())
    best_nrev = max(nrev_search, key=lambda k: nrev_search[k]["train"].mean())
    if verbose:
        print(f"\n--- SEARCH {tag}  (R={R_search}, {n_cfg} configs, TRAIN accuracy only) ---")
        print("  rev  train by eta: " +
              "  ".join(f"{e:g}:{rev_search[e]['train'].mean():.4f}" for e in etas))
        print(f"  rev  BEST eta = {best_eta_rev:g}  train={rev_search[best_eta_rev]['train'].mean():.4f}")
        print(f"  nrev BEST (eta={best_nrev[0]:g}, s={best_nrev[1]}={pats[best_nrev[1]]}) "
              f"train={nrev_search[best_nrev]['train'].mean():.4f}")

    # ---------- CONFIRMATION on an independent seed ----------
    rngc = np.random.default_rng(zlib.crc32(f"{exp.key}|{init_kind}|c".encode()))
    W0c = make_W0(init_kind, geom, pot, R_conf, rngc)
    patc = s_patterns(pot, W0c)
    if lean:
        patc = {k: v for k, v in patc.items() if k in pats}
    eta_n, sname = best_nrev
    sc = patc[sname]
    c_nrev = one_arm(ds, pot, geom, exp.n_iter, 1, sc, eta_n, W0c, CONFIRM_SEED)
    c_rev_m = one_arm(ds, pot, geom, exp.n_iter, 0, (0, 0, 0), eta_n, W0c, CONFIRM_SEED)
    c_rev_b = (c_rev_m if best_eta_rev == eta_n else
               one_arm(ds, pot, geom, exp.n_iter, 0, (0, 0, 0), best_eta_rev, W0c, CONFIRM_SEED))

    out = dict(exp=exp.key, init=init_kind, n_iter=exp.n_iter, n_configs=n_cfg,
               R_search=R_search, R_conf=R_conf, search_seed=SEARCH_SEED,
               confirm_seed=CONFIRM_SEED, eta_grid=etas, s_patterns={k: list(v) for k, v in pats.items()},
               best_eta_rev=best_eta_rev, best_nrev_eta=eta_n, best_nrev_s=sname,
               best_nrev_scales=list(sc))
    for lab, arm in (("nrev_best", c_nrev), ("rev_matched", c_rev_m), ("rev_best", c_rev_b)):
        out[lab] = {k: float(arm[k].mean()) if isinstance(arm[k], np.ndarray) else float(arm[k])
                    for k in ("train", "test", "U", "proj")}
        out[lab]["train_sd"] = float(arm["train"].std(ddof=1))
        out[lab]["test_sd"] = float(arm["test"].std(ddof=1))
    out["matched"] = dict(train=paired_stats(c_nrev, c_rev_m, "train"),
                          test=paired_stats(c_nrev, c_rev_m, "test"))
    out["best_vs_best"] = dict(train=paired_stats(c_nrev, c_rev_b, "train"),
                               test=paired_stats(c_nrev, c_rev_b, "test"))
    out["rev_best_eta_equals_nrev_eta"] = bool(best_eta_rev == eta_n)
    if verbose:
        m, b = out["matched"], out["best_vs_best"]
        print(f"--- CONFIRM {tag} (seed {CONFIRM_SEED}, R={R_conf}) ---")
        print(f"  nrev  best  (eta={eta_n:g}, s={sname})  train={out['nrev_best']['train']:.4f} "
              f"test={out['nrev_best']['test']:.4f}  proj={out['nrev_best']['proj']:.3f}")
        print(f"  rev   matched eta={eta_n:g}            train={out['rev_matched']['train']:.4f} "
              f"test={out['rev_matched']['test']:.4f}")
        print(f"  rev   OWN best eta={best_eta_rev:g}       train={out['rev_best']['train']:.4f} "
              f"test={out['rev_best']['test']:.4f}")
        print(f"  MATCHED-eta   paired train {m['train']['diff']:+.4f} +- {m['train']['se']:.4f} "
              f"(t={m['train']['t']:+.2f}, wins {m['train']['win']:.0%})   "
              f"test {m['test']['diff']:+.4f} (t={m['test']['t']:+.2f})")
        print(f"  BEST-vs-BEST  paired train {b['train']['diff']:+.4f} +- {b['train']['se']:.4f} "
              f"(t={b['train']['t']:+.2f}, wins {b['train']['win']:.0%})   "
              f"test {b['test']['diff']:+.4f} (t={b['test']['t']:+.2f})")
        print(f"  --> best-vs-best survives: "
              f"{'YES' if b['train']['diff'] > 0 and b['train']['t'] > 2 else 'NO'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="titanic", choices=["titanic", "magic"])
    ap.add_argument("--r-search", type=int, default=32)
    ap.add_argument("--r-conf", type=int, default=150)
    ap.add_argument("--lean", action="store_true",
                    help="reduced eta grid / 3 J-strength patterns (used for the MAGIC check)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.perf_counter()
    rows, results = [], []

    if args.part == "titanic":
        ds = build_titanic_dataset()
        pot = make_potential(ds)
        jobs = [(Exp10("titanic_ball", "titanic", "ball", 1500), k)
                for k in ("boundary", "antimap", "ball_uniform")]
        jobs += [(Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18), "boundary")]
    else:
        ds = build_magic_dataset()
        pot = make_potential(ds)
        jobs = [(Exp10("magic_ball", "magic", "ball", 1000), "boundary"),
                (Exp10("magic_ball", "magic", "ball", 1000), "antimap")]
        args.r_search, args.r_conf = min(args.r_search, 20), min(args.r_conf, 80)

    print(f"[{args.part}] n_train={ds.n_train} n_test={ds.n_test} lam={pot.lam:.4g} "
          f"delta={pot.delta:.4g} a>={pot.a_lower_bound:.3f}")
    for exp, kind in jobs:
        etas = [1e-6, 1e-5, 3e-5, 1e-4] if args.lean else None
        results.append(probe(exp, ds, pot, kind, args.r_search, args.r_conf, rows,
                             etas=etas, lean=args.lean))
        print(f"  [{time.perf_counter()-t0:.0f}s elapsed]")

    df = pd.DataFrame(rows)
    p = os.path.join(OUTDIR, f"search_{args.part}.csv")
    df.to_csv(p, index=False)
    with open(os.path.join(OUTDIR, f"confirm_{args.part}.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {p} and confirm_{args.part}.json   total {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
