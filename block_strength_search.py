"""Per-block block strengths s = (s1, s2, s3): how hard can the non-reversible arm be pushed?

WHAT IS SETTLED.  J's three conditions involve only the constraint set, so every triple s leaves
both arms sharing pi_K.  No s beats the reversible arm on a stationary quantity, and accuracy is
one.  That is not re-litigated here; it is measured and reported alongside, with a multiplicity
bar, so the claim can be checked.

WHAT IS OPEN, AND WHY PER-BLOCK s SHOULD MATTER.  regularizer_study.py established that the
non-reversible speed-up is driven by the anisotropy INSIDE the triples J rotates, and is blind to
anisotropy across them.  Each block has its OWN within-block anisotropy, so the right strength for
block 1 need not be the right strength for block 3.  That is a mechanism-level reason to expect a
non-uniform optimum -- different from the earlier `s_l ~ 1/(r_l |grad_l U0|)` gradient-scale
heuristic in the README, and testable against it.

PROTOCOL (search and confirmation are on disjoint seeds, and the whole search is reported).

  Stage 0, GOLD.  A long reversible run at eta/10 fixes reference means and sds for thirteen
    test functions: the ten coordinates, |w|^2, U_0, and TRAINING accuracy (train labels only --
    test labels are never used for any selection here).  BIAS(cell) = max_f |mean_f(cell) -
    mean_f(gold)| / sd_f(gold), in gold sd units.  Training accuracy is in the set on purpose:
    a bias metric made only of parameter functionals can pass a cell that has visibly moved the
    thing the study is about.
  Stage A, SEARCH on seed 8100.  For each block l, scan s_l over a grid with the other two blocks
    held at ZERO.  This isolates block l's own response.  A uniform scan s = (v,v,v) runs in the
    same stage as the baseline to beat.
  Stage B.  Compose s* = (argmax_1, argmax_2, argmax_3) from the three 1-D curves, MAXIMISING the
    geometric-mean ESS ratio over that block's three coordinates SUBJECT TO a bias constraint.

    The constraint is not a free parameter.  The reversible arm ignores s entirely, so its bias at
    this eta is one fixed number per potential; we require the non-reversible arm's bias to be NO
    LARGER.  Any speed-up that survives is then free: it is bought at a discretisation error no
    worse than the arm it is being compared with.  Selections at fixed thresholds of 0.10, 0.25
    and 0.50 gold sds are reported alongside.

    This constraint is the whole point.  Without it the search is degenerate: the ESS ratio rises
    monotonically in s, so `argmax ESS` returns the largest s on the grid, where the chain is fast
    and WRONG -- at s = 20 the accuracy falls by 7 to 10 percentage points and split-Rhat reaches
    1.5.  Those rows are kept in the output as the cautionary case.
  Stage C, CONFIRM on seed 9200, disjoint from the search.  s*, the best uniform triple, the
    project default (5,5,5), the requested (4,2,7), and order-reversal controls with identical
    total strength are re-run paired at higher replication.

Two potentials are used.  `lasso` is the project's own regularizer.  `graded` is an elastic net
whose three blocks carry DIFFERENT within-block anisotropy (k = 1, 5, 25) -- built so that, if the
mechanism claim is right, its optimal triple must be non-uniform.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exact_nb as X                                                       # noqa: E402
from exact_nb import D10, design_with_intercept                            # noqa: E402
from nral import build_titanic_dataset                                     # noqa: E402
from regularizer_study import ElasticPotential, ess_per_chain              # noqa: E402
from probe_mixing import split_rhat                                        # noqa: E402

RES = os.path.join(HERE, "results", "blockstrength")
os.makedirs(RES, exist_ok=True)

QUICK = os.environ.get("BS_MODE", "full").lower() == "quick"
ETA = 1e-4
SEED_SEARCH, SEED_CONF = 8100, 9200            # disjoint, and used nowhere else in the repo
LAM1_FRAC, LAM2_FRAC = 0.01, 0.05
BLOCKS = X.BLOCKS10

R_S = 12 if QUICK else 48                      # search
NB_S, NM_S = 400 if QUICK else 3000, 1200 if QUICK else 12000
R_C = 16 if QUICK else 96                      # confirmation
NB_C, NM_C = 400 if QUICK else 4000, 1500 if QUICK else 16000
CK = 2

LEVELS = (1.0, 2.0, 5.0, 10.0, 20.0)
GOLD_DIV = 10.0                                # gold runs at ETA / GOLD_DIV
R_G = 8 if QUICK else 32
NB_G, NM_G = 400 if QUICK else 4000, 3000 if QUICK else 160000
BSTARS = (0.10, 0.25, 0.50)
GRADED_K = (1.0, 5.0, 25.0)                    # per-block within-block anisotropy, by design


def graded_c(ks=GRADED_K) -> np.ndarray:
    """Block l gets the spread (1/k_l, 1, k_l): a DIFFERENT anisotropy in each block."""
    return np.concatenate([[1.0 / k, 1.0, k] for k in ks])


THIN_BIAS = 8          # the bias functionals are evaluated on every THIN_BIAS-th stored state


def test_functions(Ws, pot, Xtr, ytr):
    """(T, R, 10) -> (13, n): the ten coordinates, |w|^2, U_0 and TRAINING accuracy."""
    F = Ws[::THIN_BIAS].reshape(-1, D10)
    acc = np.mean(((F @ Xtr.T) >= 0.0) == ytr.astype(bool)[None, :], axis=1)
    return np.vstack([F.T, np.sum(F ** 2, axis=1)[None, :], pot.U0(F)[None, :], acc[None, :]])


def gold_reference(pot, geom, ds, *, seed):
    """Long REVERSIBLE run at eta/GOLD_DIV. Fair to both arms: the continuous process has the
    same invariant law for both, so the eta -> 0 limit is common."""
    eta_g = ETA / GOLD_DIV
    W0 = X.init_unit_ball(np.random.default_rng(seed + 1), R_G)
    burn = X.run_chain(pot, geom, alpha=0, scales=(0., 0., 0.), eta=eta_g,
                       n_iter=int(NB_G * GOLD_DIV), W0=W0, seed=seed,
                       checkpoint_every=max(1, int(NB_G * GOLD_DIV) // 2))
    r = X.run_chain(pot, geom, alpha=0, scales=(0., 0., 0.), eta=eta_g, n_iter=NM_G,
                    W0=burn["W"][-1], seed=seed + 99, checkpoint_every=int(CK * GOLD_DIV))
    T = test_functions(r["W"], pot, pot.X, pot.y)
    return T.mean(axis=1), T.std(axis=1, ddof=1)


def measure(pot, geom, scales, *, R, n_burn, n_meas, seed, ds, gold=None):
    """Paired run of both arms from one common stationary state. Returns per-block ESS ratios."""
    W0 = X.init_unit_ball(np.random.default_rng(seed + 1), R)
    burn = X.run_chain(pot, geom, alpha=0, scales=(0., 0., 0.), eta=ETA, n_iter=n_burn,
                       W0=W0, seed=seed, checkpoint_every=max(1, n_burn // 2))
    Wstat = burn["W"][-1]
    res = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        res[tag] = X.run_chain(pot, geom, alpha=alpha, scales=scales, eta=ETA, n_iter=n_meas,
                               W0=Wstat, seed=seed + 99, checkpoint_every=CK)
        assert res[tag]["n_nonfinite"] == 0, f"non-finite states at s={scales}"

    per_block, rhat = [], 0.0
    logr_all = []
    for b, blk in enumerate(BLOCKS):
        lr = []
        for j in blk:
            e = {}
            for tag in ("rev", "nrev"):
                S = res[tag]["W"][:, :, j]
                e[tag] = ess_per_chain(S, CK)
                rhat = max(rhat, split_rhat(S))
            ok = np.isfinite(e["rev"]) & np.isfinite(e["nrev"]) & (e["rev"] > 0)
            lr.append(np.log(e["nrev"][ok] / e["rev"][ok]))
        L = np.concatenate(lr)
        logr_all.append(L)
        se = L.std(ddof=1) / np.sqrt(L.size)
        per_block.append((float(np.exp(L.mean())), float(se), float(L.mean() / se)))
    LA = np.concatenate(logr_all)
    seA = LA.std(ddof=1) / np.sqrt(LA.size)

    bias = {}
    if gold is not None:
        gm, gs = gold
        for tag in ("rev", "nrev"):
            m = test_functions(res[tag]["W"], pot, pot.X, pot.y).mean(axis=1)
            bias[tag] = float(np.max(np.abs(m - gm) / np.maximum(gs, 1e-12)))

    acc = {t: X.accuracy(ds.X_test, ds.y_test, res[t]["W"]).mean(axis=0) for t in ("rev", "nrev")}
    d_acc = X.paired(acc["nrev"], acc["rev"])
    return dict(per_block=per_block, overall=float(np.exp(LA.mean())), overall_se=float(seA),
                overall_t=float(LA.mean() / seA), rhat=float(rhat),
                bias_rev=bias.get("rev", np.nan), bias_nrev=bias.get("nrev", np.nan),
                acc_rev=float(acc["rev"].mean()), acc_nrev=float(acc["nrev"].mean()),
                d_acc_pts=d_acc[0] * 100.0, t_acc=d_acc[2],
                proj=res["nrev"]["projection_rate"])


def main() -> int:
    ds = build_titanic_dataset()
    Xd, y = design_with_intercept(ds.X_train), ds.y_train
    lam1 = LAM1_FRAC * ds.n_train
    lam2 = LAM2_FRAC * ds.n_train
    delta = -np.log(X.A_LOWER) / (9.0 * lam1)
    geom = X.make_geom("ball")
    POTS = {
        "lasso": ElasticPotential(Xd, y, X.SIGMA, lam1, 0.0, np.ones(9), delta),
        "graded": ElasticPotential(Xd, y, X.SIGMA, lam1, lam2, graded_c(), delta),
    }
    print(f"titanic n_train={ds.n_train}  eta={ETA:g}  search seed {SEED_SEARCH} "
          f"(R={R_S}, {NM_S} it)  confirm seed {SEED_CONF} (R={R_C}, {NM_C} it)")
    print(f"graded per-block anisotropy k = {GRADED_K}, c = "
          f"{np.round(graded_c(), 3).tolist()}\n")
    t0 = time.perf_counter()

    # ---------------------------------------------------------------- Stage 0: gold
    GOLD = {}
    for pname, pot in POTS.items():
        GOLD[pname] = gold_reference(pot, geom, ds, seed=SEED_SEARCH + 7)
        print(f"  [gold {pname:<6}] eta={ETA / GOLD_DIV:g}, R={R_G}, {NM_G} it  "
              f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---------------------------------------------------------------- Stage A: search
    rows = []
    for pname, pot in POTS.items():
        for b in range(3):
            for v in LEVELS:
                sc = [0.0, 0.0, 0.0]
                sc[b] = v
                m = measure(pot, geom, tuple(sc), R=R_S, n_burn=NB_S, n_meas=NM_S,
                            seed=SEED_SEARCH, ds=ds, gold=GOLD[pname])
                rows.append(dict(stage="search", potential=pname, mode=f"block{b+1}",
                                 s1=sc[0], s2=sc[1], s3=sc[2], target_block=b + 1,
                                 own_ratio=m["per_block"][b][0], own_t=m["per_block"][b][2],
                                 overall=m["overall"], overall_t=m["overall_t"],
                                 rhat=m["rhat"], bias_rev=m["bias_rev"],
                                 bias_nrev=m["bias_nrev"],
                                 d_acc_pts=m["d_acc_pts"], t_acc=m["t_acc"]))
                print(f"  [{pname:<6} block{b+1} s={v:>4g}] own-block ESS ratio "
                      f"{m['per_block'][b][0]:.3f} (t={m['per_block'][b][2]:+5.1f})   "
                      f"bias {m['bias_nrev']:.3f} (rev {m['bias_rev']:.3f})   "
                      f"Rhat<={m['rhat']:.3f}  "
                      f"[{time.perf_counter()-t0:.0f}s]", flush=True)
        for v in LEVELS:
            m = measure(pot, geom, (v, v, v), R=R_S, n_burn=NB_S, n_meas=NM_S,
                        seed=SEED_SEARCH, ds=ds, gold=GOLD[pname])
            rows.append(dict(stage="search", potential=pname, mode="uniform",
                             s1=v, s2=v, s3=v, target_block=0, own_ratio=np.nan, own_t=np.nan,
                             overall=m["overall"], overall_t=m["overall_t"], rhat=m["rhat"],
                             bias_rev=m["bias_rev"], bias_nrev=m["bias_nrev"],
                             d_acc_pts=m["d_acc_pts"], t_acc=m["t_acc"]))
            print(f"  [{pname:<6} uniform  s={v:>4g}] overall ESS ratio {m['overall']:.3f} "
                  f"(t={m['overall_t']:+5.1f})   bias {m['bias_nrev']:.3f} "
                  f"(rev {m['bias_rev']:.3f})   Rhat<={m['rhat']:.3f}  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    search = pd.DataFrame(rows)
    search.to_csv(os.path.join(RES, "search.csv"), index=False)

    # ---------------------------------------------------------------- Stage B: compose
    def pick(df, value_col, s_col, cap):
        ok = df[df["bias_nrev"] <= cap]
        if len(ok) == 0:                      # nothing meets the constraint -> the weakest cell
            return float(df.loc[df["bias_nrev"].idxmin(), s_col]), False
        return float(ok.loc[ok[value_col].idxmax(), s_col]), True

    composed, best_uni, selection = {}, {}, []
    for pname in POTS:
        sub = search[(search["potential"] == pname) & (search["mode"] != "uniform")]
        uni = search[(search["potential"] == pname) & (search["mode"] == "uniform")]
        cap_rev = float(sub["bias_rev"].median())      # the reversible arm ignores s
        star = []
        for b in range(3):
            sb = sub[sub["target_block"] == b + 1]
            v, feasible = pick(sb, "own_ratio", f"s{b+1}", cap_rev)
            star.append(v)
            selection.append(dict(potential=pname, block=b + 1, cap="rev-matched",
                                  cap_value=cap_rev, s=v, feasible=feasible))
            for B in BSTARS:                            # reported alongside, not used
                vb, fb = pick(sb, "own_ratio", f"s{b+1}", B)
                selection.append(dict(potential=pname, block=b + 1, cap=f"B*={B}",
                                      cap_value=B, s=vb, feasible=fb))
        composed[pname] = tuple(star)
        best_uni[pname] = pick(uni, "overall", "s1", cap_rev)[0]
        print(f"\n[{pname}] reversible-arm bias at eta = {cap_rev:.3f} gold sds  ->  "
              f"composed s* = {composed[pname]}   best uniform s = {best_uni[pname]:g}")
    pd.DataFrame(selection).to_csv(os.path.join(RES, "selection.csv"), index=False)

    # ---------------------------------------------------------------- Stage C: confirm
    crows = []
    for pname, pot in POTS.items():
        s_star, v_uni = composed[pname], best_uni[pname]
        cands = {
            "composed s*": s_star,
            "composed s* REVERSED": tuple(reversed(s_star)),
            "composed s* ROTATED": (s_star[1], s_star[2], s_star[0]),
            f"best uniform ({v_uni:g},{v_uni:g},{v_uni:g})": (v_uni, v_uni, v_uni),
            "project default (5,5,5)": (5.0, 5.0, 5.0),
            "requested (4,2,7)": (4.0, 2.0, 7.0),
            "requested REVERSED (7,2,4)": (7.0, 2.0, 4.0),
            "ascending (2,5,10)": (2.0, 5.0, 10.0),
            "descending (10,5,2)": (10.0, 5.0, 2.0),
        }
        seen, uniq = set(), {}
        for label, sc in cands.items():          # drop controls that collide with what they check
            key = tuple(float(z) for z in sc)
            if key in seen:
                print(f"  (skipping '{label}' -- identical triple to an earlier row)")
                continue
            seen.add(key)
            uniq[label] = sc
        print(f"\n=== confirm [{pname}], seed {SEED_CONF} ===", flush=True)
        for label, sc in uniq.items():
            m = measure(pot, geom, tuple(float(z) for z in sc), R=R_C, n_burn=NB_C,
                        n_meas=NM_C, seed=SEED_CONF, ds=ds, gold=GOLD[pname])
            crows.append(dict(stage="confirm", potential=pname, label=label,
                              s1=sc[0], s2=sc[1], s3=sc[2],
                              b1=m["per_block"][0][0], b2=m["per_block"][1][0],
                              b3=m["per_block"][2][0], overall=m["overall"],
                              overall_se=m["overall_se"], overall_t=m["overall_t"],
                              rhat=m["rhat"], bias_rev=m["bias_rev"],
                              bias_nrev=m["bias_nrev"],
                              acc_rev=m["acc_rev"], acc_nrev=m["acc_nrev"],
                              d_acc_pts=m["d_acc_pts"], t_acc=m["t_acc"], proj=m["proj"]))
            print(f"  {label:<32} s={tuple(float(z) for z in sc)}  ESS ratio "
                  f"{m['overall']:.3f} (t={m['overall_t']:+5.1f})  blocks "
                  f"{m['per_block'][0][0]:.2f}/{m['per_block'][1][0]:.2f}/"
                  f"{m['per_block'][2][0]:.2f}  bias {m['bias_nrev']:.3f}"
                  f"/{m['bias_rev']:.3f}  acc {m['d_acc_pts']:+.3f} pts "
                  f"(t={m['t_acc']:+5.2f})  Rhat<={m['rhat']:.3f}  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    conf = pd.DataFrame(crows)
    conf.to_csv(os.path.join(RES, "confirm.csv"), index=False)
    with open(os.path.join(RES, "meta.json"), "w") as f:
        json.dump(dict(eta=ETA, seed_search=SEED_SEARCH, seed_confirm=SEED_CONF,
                       R_search=R_S, R_confirm=R_C, n_meas_search=NM_S, n_meas_confirm=NM_C,
                       levels=LEVELS, graded_k=GRADED_K,
                       composed={k: list(v) for k, v in composed.items()},
                       best_uniform=best_uni), f, indent=2)

    print()
    for pname in POTS:
        print(f"=== {pname} ===")
        print(conf[conf["potential"] == pname][
            ["label", "s1", "s2", "s3", "b1", "b2", "b3", "overall", "overall_t",
             "bias_nrev", "bias_rev", "d_acc_pts", "t_acc"]].to_string(index=False,
                                              float_format=lambda z: f"{z:.3f}"))
        print()
    print(f"total {time.perf_counter()-t0:.0f}s -> {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
