"""probe_ergodic.py -- HYPOTHESIS `ergodic`.

Do POSTERIOR/ERGODIC AVERAGES separate the non-reversible arm from the reversible one?

Theory: for a reversible generator L, the non-reversible perturbation L + J.grad has asymptotic
variance for time averages that is never larger and generically strictly smaller
(Hwang-Hwang-Sheu; Duncan-Lelievre-Pavliotis).  The LAST ITERATE cannot show this (identical
invariant law), but an ERGODIC AVERAGE can, because its Monte-Carlo error is sigma_asym^2 / N.

Two consequences we test:
  (V) VARIANCE: across independent replicates, the ergodic average of a fixed observable
      (each coordinate w_i, and the averaged predictive probability) should have SMALLER spread
      for the non-reversible arm.  This is the direct signature.
  (J) JENSEN / NONLINEAR METRIC: log loss and Brier score of the AVERAGED PROBABILITY are convex
      in that probability, so E[metric] = metric(E p) + (1/2) f'' Var(p) + ...  A genuine
      variance reduction therefore makes the EXPECTED log loss / Brier score of the
      non-reversible arm STRICTLY SMALLER, with a gap proportional to the variance reduction.
      This is a *mean* effect, so it can be tested with an ordinary paired t statistic.

Protocol (all four "defensibility" requirements):
  (a) PAIRED: both arms get the SAME W0 and the SAME seed -> identical Gaussian noise stream.
  (b) Every configuration choice (block strength s, and the reversible arm's own step size) is
      made on TRAINING data only, on a SEARCH seed; the headline number comes from a
      CONFIRMATION run on an INDEPENDENT seed never used in the search.
  (c) The reversible arm is ALSO given its own best step size from its own training-only sweep,
      and best-vs-best is reported.
  (d) Paired SE, t statistic and the number of configurations tried are reported.

Creates only results/probe_ergodic/.  Imports exact_anchored / nral; modifies nothing.
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy.special import expit

from exact_anchored import (Potential, make_potential, run_exact, accuracy10, init_unit_ball10,
                            make_geom10, Exp10, geom_for, D10)
from nral import build_magic_dataset, build_titanic_dataset

RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_ergodic")
SEARCH_SEED = 7001          # used ONLY for configuration selection (training metrics)
CONFIRM_SEED = 9101         # independent, used ONLY for the headline confirmation
EPSP = 1e-12


# ---------------------------------------------------------------- metrics on ergodic averages
def design(X):
    return np.ascontiguousarray(np.column_stack([np.ones(len(X)), X]))


def ergodic_prob(Xi, Ws, burn_frac=0.5, thin=1):
    """Average the PREDICTIVE PROBABILITY sigma(x.w) over post-burn-in checkpoints (not w)."""
    K = Ws.shape[0]
    k0 = int(round(burn_frac * (K - 1)))
    P = np.zeros((Ws.shape[1], Xi.shape[0]))
    n = 0
    for k in range(k0, K, thin):
        P += expit(Ws[k] @ Xi.T)
        n += 1
    return P / n, n


def prob_metrics(P, y):
    yb = y.astype(bool)
    acc = np.mean((P >= 0.5) == yb[None, :], axis=1)
    ll = -np.mean(y[None, :] * np.log(P + EPSP) + (1 - y)[None, :] * np.log(1 - P + EPSP), axis=1)
    br = np.mean((P - y[None, :]) ** 2, axis=1)
    return dict(acc=acc, logloss=ll, brier=br)


def ergodic_w(Ws, burn_frac=0.5):
    K = Ws.shape[0]
    k0 = int(round(burn_frac * (K - 1)))
    return Ws[k0:].mean(axis=0)                         # (R, 10)


def paired_stats(a_nrev, a_rev):
    d = np.asarray(a_nrev) - np.asarray(a_rev)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    return dict(mean_nrev=float(np.mean(a_nrev)), mean_rev=float(np.mean(a_rev)),
                sd_nrev=float(np.std(a_nrev, ddof=1)), sd_rev=float(np.std(a_rev, ddof=1)),
                diff=float(d.mean()), se=float(se), t=float(d.mean() / se),
                win_frac=float((d > 0).mean()), n=int(n))


# ---------------------------------------------------------------- one paired cell
def run_pair(ds, pot, geom, *, eta, scales, n_iter, R, seed, burn_frac, thin_tr, thin_te,
             ck_every=1, alphas=(0, 1)):
    W0 = init_unit_ball10(np.random.default_rng(seed + 1), R)
    assert np.all(geom.feasible(W0))
    Xtr, Xte = design(ds.X_train), design(ds.X_test)
    out = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        if alpha not in alphas:
            continue
        r = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                      W0=W0, seed=seed, checkpoint_every=ck_every)   # SAME W0, SAME noise stream
        Ws = r["betas"]
        Ptr, ntr = ergodic_prob(Xtr, Ws, burn_frac, thin_tr)
        Pte, nte = ergodic_prob(Xte, Ws, burn_frac, thin_te)
        m = dict(train=prob_metrics(Ptr, ds.y_train), test=prob_metrics(Pte, ds.y_test))
        m["wbar"] = ergodic_w(Ws, burn_frac)
        m["Pte"] = Pte
        m["last_acc_train"] = accuracy10(ds.X_train, ds.y_train, Ws[-1:])[0]
        m["last_acc_test"] = accuracy10(ds.X_test, ds.y_test, Ws[-1:])[0]
        m["proj"] = r["projection_rate"]
        m["n_avg_train"], m["n_avg_test"] = ntr, nte
        m["runtime"] = r["runtime"]
        out[tag] = m
        del Ws, r
    return out


def var_ratio_report(arms):
    """Across-replicate variance of the ergodic average, per coordinate: Var_rev / Var_nrev."""
    vr = np.var(arms["rev"]["wbar"], axis=0, ddof=1)
    vn = np.var(arms["nrev"]["wbar"], axis=0, ddof=1)
    pr = np.var(arms["rev"]["Pte"], axis=0, ddof=1).mean()
    pn = np.var(arms["nrev"]["Pte"], axis=0, ddof=1).mean()
    return dict(coord_var_rev=vr.tolist(), coord_var_nrev=vn.tolist(),
                coord_ratio=(vr / (vn + 1e-300)).tolist(),
                pooled_coord_ratio=float(vr.sum() / (vn.sum() + 1e-300)),
                prob_var_rev=float(pr), prob_var_nrev=float(pn),
                prob_var_ratio=float(pr / (pn + 1e-300)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["timing", "search", "confirm", "magic", "all"])
    ap.add_argument("--search-R", type=int, default=40)
    ap.add_argument("--confirm-R", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)
    t00 = time.perf_counter()

    tit = build_titanic_dataset()
    pot_t = make_potential(tit)
    print(f"[titanic] n_train={tit.n_train} n_test={tit.n_test} lam={pot_t.lam:.4g} "
          f"delta={pot_t.delta:.4g} a>= {pot_t.a_lower_bound:.3f}")

    # geometry / iteration budget: exactly the converged settings named in the task brief
    TIT_BALL = Exp10("titanic_ball", "titanic", "ball", 1500)
    TIT_LP = Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18)
    ETA_TIT = 3e-5

    if args.stage == "timing":
        g = geom_for(TIT_BALL)
        t0 = time.perf_counter()
        a = run_pair(tit, pot_t, g, eta=ETA_TIT, scales=(2., 2., 2.), n_iter=TIT_BALL.n_iter,
                     R=20, seed=1, burn_frac=0.5, thin_tr=1, thin_te=1)
        print(f"timing R=20 both arms: {time.perf_counter()-t0:.1f}s  "
              f"sampler {a['rev']['runtime']:.1f}+{a['nrev']['runtime']:.1f}s")
        return 0

    rows = []
    n_cfg = 0

    # ---------------- STAGE 1: TRAINING-ONLY search over block strength s ------------------
    S_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
    best = {}
    if args.stage in ("search", "all"):
        print("\n=== SEARCH (seed %d, R=%d): select s by TRAIN log loss of the ergodic-average "
              "probability. TEST LABELS NOT USED. ===" % (SEARCH_SEED, args.search_R))
        for e in (TIT_BALL, TIT_LP):
            g = geom_for(e)
            recs = []
            for s in S_GRID:
                n_cfg += 1
                arms = run_pair(tit, pot_t, g, eta=ETA_TIT, scales=(s, s, s), n_iter=e.n_iter,
                                R=args.search_R, seed=SEARCH_SEED, burn_frac=0.5,
                                thin_tr=1, thin_te=1)
                st_ll = paired_stats(arms["nrev"]["train"]["logloss"], arms["rev"]["train"]["logloss"])
                st_br = paired_stats(arms["nrev"]["train"]["brier"], arms["rev"]["train"]["brier"])
                st_ac = paired_stats(arms["nrev"]["train"]["acc"], arms["rev"]["train"]["acc"])
                vr = var_ratio_report(arms)
                rec = dict(stage="search", exp=e.key, eta=ETA_TIT, s=s, R=args.search_R,
                           d_train_logloss=st_ll["diff"], se_train_logloss=st_ll["se"],
                           t_train_logloss=st_ll["t"],
                           d_train_brier=st_br["diff"], t_train_brier=st_br["t"],
                           d_train_acc=st_ac["diff"], t_train_acc=st_ac["t"],
                           pooled_coord_var_ratio=vr["pooled_coord_ratio"],
                           prob_var_ratio=vr["prob_var_ratio"],
                           proj_nrev=arms["nrev"]["proj"],
                           train_logloss_rev=st_ll["mean_rev"], train_logloss_nrev=st_ll["mean_nrev"])
                recs.append(rec); rows.append(rec)
                print(f"  [{e.key}] s={s:<5g} dTRAIN_ll={st_ll['diff']:+.3e} (t={st_ll['t']:+.2f}) "
                      f"dTRAIN_brier={st_br['diff']:+.3e} (t={st_br['t']:+.2f})  "
                      f"coordVarRatio={vr['pooled_coord_ratio']:.3f} "
                      f"probVarRatio={vr['prob_var_ratio']:.3f} proj={arms['nrev']['proj']:.3f} "
                      f"({time.perf_counter()-t00:.0f}s)")
            df = pd.DataFrame(recs)
            k = int(df.d_train_logloss.idxmin())          # most NEGATIVE = biggest LL improvement
            best[e.key] = float(df.loc[k, "s"])
            print(f"  -> selected s={best[e.key]:g} for {e.key} (training log loss only)")

        # reversible arm's OWN best step size (training only), to satisfy requirement (c)
        print("\n=== SEARCH: reversible arm's own best eta (TRAIN log loss, seed %d) ===" % SEARCH_SEED)
        for e in (TIT_BALL, TIT_LP):
            g = geom_for(e)
            for eta in (1e-5, 3e-5, 1e-4):
                n_cfg += 1
                arms = run_pair(tit, pot_t, g, eta=eta, scales=(0., 0., 0.), n_iter=e.n_iter,
                                R=args.search_R, seed=SEARCH_SEED, burn_frac=0.5,
                                thin_tr=1, thin_te=1, alphas=(0,))
                ll = float(arms["rev"]["train"]["logloss"].mean())
                ac = float(arms["rev"]["train"]["acc"].mean())
                rows.append(dict(stage="rev_eta", exp=e.key, eta=eta, s=0.0, R=args.search_R,
                                 train_logloss_rev=ll, train_acc_rev=ac))
                print(f"  [{e.key}] REV eta={eta:g}: train logloss={ll:.5f} train acc={ac:.4f}")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RESDIR, "search.csv"), index=False)

    # ---------------- STAGE 2: CONFIRMATION on an independent seed ------------------------
    conf = {}
    if args.stage in ("confirm", "all"):
        sel = json.load(open(os.path.join(RESDIR, "selected.json"))) if not best else None
        if sel:
            best = sel
        json.dump(best, open(os.path.join(RESDIR, "selected.json"), "w"), indent=2)
        print("\n=== CONFIRMATION (independent seed %d, R=%d) ===" % (CONFIRM_SEED, args.confirm_R))
        for e in (TIT_BALL, TIT_LP):
            s = best[e.key]
            g = geom_for(e)
            arms = run_pair(tit, pot_t, g, eta=ETA_TIT, scales=(s, s, s), n_iter=e.n_iter,
                            R=args.confirm_R, seed=CONFIRM_SEED, burn_frac=0.5,
                            thin_tr=1, thin_te=1)
            blk = {}
            for split in ("train", "test"):
                for m in ("acc", "logloss", "brier"):
                    blk[f"{split}_{m}"] = paired_stats(arms["nrev"][split][m], arms["rev"][split][m])
            blk["last_iterate_test_acc"] = paired_stats(arms["nrev"]["last_acc_test"],
                                                        arms["rev"]["last_acc_test"])
            blk["variance"] = var_ratio_report(arms)
            blk["s"] = s; blk["eta"] = ETA_TIT; blk["n_iter"] = e.n_iter
            blk["n_avg"] = arms["rev"]["n_avg_train"]
            blk["proj_nrev"] = arms["nrev"]["proj"]; blk["proj_rev"] = arms["rev"]["proj"]
            # robustness: longer burn-in (75%)
            conf[e.key] = blk
            print(f"\n[{e.key}] eta={ETA_TIT:g} s={s:g} R={args.confirm_R} "
                  f"n_avg={blk['n_avg']} checkpoints")
            for kk in ("train_logloss", "test_logloss", "train_brier", "test_brier",
                       "train_acc", "test_acc", "last_iterate_test_acc"):
                v = blk[kk]
                print(f"   {kk:<22} rev={v['mean_rev']:.6f} nrev={v['mean_nrev']:.6f} "
                      f"diff={v['diff']:+.3e} se={v['se']:.2e} t={v['t']:+.2f} "
                      f"win={v['win_frac']:.0%}")
            v = blk["variance"]
            print(f"   across-replicate Var(ergodic w) ratio rev/nrev pooled = "
                  f"{v['pooled_coord_ratio']:.4f}; per-coord = "
                  f"{', '.join(f'{x:.2f}' for x in v['coord_ratio'])}")
            print(f"   Var(ergodic test prob) rev={v['prob_var_rev']:.3e} "
                  f"nrev={v['prob_var_nrev']:.3e} ratio={v['prob_var_ratio']:.4f}")

        # best-vs-best: reversible at its own best eta vs non-reversible at the selected config
        print("\n=== BEST-vs-BEST (requirement c), independent seed %d ===" % CONFIRM_SEED)
        sdf = pd.read_csv(os.path.join(RESDIR, "search.csv"))
        for e in (TIT_BALL, TIT_LP):
            sub = sdf[(sdf.stage == "rev_eta") & (sdf.exp == e.key)]
            eta_rev = float(sub.loc[sub.train_logloss_rev.idxmin(), "eta"])
            g = geom_for(e)
            a = run_pair(tit, pot_t, g, eta=eta_rev, scales=(0., 0., 0.), n_iter=e.n_iter,
                         R=args.confirm_R, seed=CONFIRM_SEED, burn_frac=0.5, thin_tr=1,
                         thin_te=1, alphas=(0,))
            conf[e.key]["rev_best_eta"] = eta_rev
            conf[e.key]["rev_best_test_logloss"] = float(a["rev"]["test"]["logloss"].mean())
            conf[e.key]["rev_best_test_acc"] = float(a["rev"]["test"]["acc"].mean())
            conf[e.key]["rev_best_train_logloss"] = float(a["rev"]["train"]["logloss"].mean())
            print(f"  [{e.key}] reversible own-best eta={eta_rev:g}: "
                  f"train ll={conf[e.key]['rev_best_train_logloss']:.6f} "
                  f"test ll={conf[e.key]['rev_best_test_logloss']:.6f} "
                  f"test acc={conf[e.key]['rev_best_test_acc']:.4f}  |  "
                  f"nrev at selected: test ll={conf[e.key]['test_logloss']['mean_nrev']:.6f} "
                  f"test acc={conf[e.key]['test_acc']['mean_nrev']:.4f}")
        conf["_n_configs"] = n_cfg
        json.dump(conf, open(os.path.join(RESDIR, "confirmation.json"), "w"), indent=2, default=float)

    # ---------------- STAGE 3: MAGIC ------------------------------------------------------
    if args.stage in ("magic",):
        mag = build_magic_dataset()
        pot_m = make_potential(mag)
        e = Exp10("magic_ball", "magic", "ball", 1000)
        g = geom_for(e)
        out = {}
        for s in (1.0, 2.0):
            arms = run_pair(mag, pot_m, g, eta=1e-5, scales=(s, s, s), n_iter=e.n_iter,
                            R=60, seed=CONFIRM_SEED, burn_frac=0.5, thin_tr=5, thin_te=1,
                            ck_every=2)
            blk = {}
            for split in ("train", "test"):
                for m in ("acc", "logloss", "brier"):
                    blk[f"{split}_{m}"] = paired_stats(arms["nrev"][split][m], arms["rev"][split][m])
            blk["variance"] = var_ratio_report(arms)
            out[f"s={s}"] = blk
            print(f"\n[magic_ball s={s}] eta=1e-5 R=60")
            for kk in ("train_logloss", "test_logloss", "test_acc"):
                v = blk[kk]
                print(f"   {kk:<16} rev={v['mean_rev']:.6f} nrev={v['mean_nrev']:.6f} "
                      f"diff={v['diff']:+.3e} se={v['se']:.2e} t={v['t']:+.2f}")
            v = blk["variance"]
            print(f"   pooled coord var ratio rev/nrev = {v['pooled_coord_ratio']:.4f}  "
                  f"prob var ratio = {v['prob_var_ratio']:.4f}")
        json.dump(out, open(os.path.join(RESDIR, "magic.json"), "w"), indent=2, default=float)

    print(f"\ntotal {time.perf_counter()-t00:.1f}s")
    return 0




# =====================================================================================
# STAGE 4 ("best") : give BOTH arms their own best step size (training-only selection),
#                    then compare best-vs-best on the independent confirmation seed.
# STAGE 5 ("mode") : start EVERY replicate from the SAME point (the projected-gradient mode
#                    of U_0 inside K).  Across-replicate variance of the ergodic average is
#                    then a pure Monte-Carlo variance -- no initial-condition spread -- so the
#                    ratio Var_rev/Var_nrev estimates the asymptotic-variance ratio directly.
# =====================================================================================
def find_mode(pot, geom, *, h=3e-5, n=40000, seed=0):
    w = np.zeros((1, D10))
    for _ in range(n):
        w, _ = geom.project(w - h * pot.grad_U0(w))
    return w


def stage_best(tit, pot_t, exps, eta_grid, s_grid, searchR, confR, resdir):
    t0 = time.perf_counter()
    rows, n_cfg = [], 0
    print("\n=== STAGE 'best': TRAINING-ONLY eta search for EACH arm separately (seed %d, R=%d) ==="
          % (SEARCH_SEED, searchR))
    pick = {}
    for e in exps:
        g = geom_for(e)
        recs = []
        for eta in eta_grid:
            n_cfg += 1
            a = run_pair(tit, pot_t, g, eta=eta, scales=(0., 0., 0.), n_iter=e.n_iter, R=searchR,
                         seed=SEARCH_SEED, burn_frac=0.5, thin_tr=1, thin_te=1, alphas=(0,))
            ll = float(a["rev"]["train"]["logloss"].mean())
            recs.append(dict(exp=e.key, arm="rev", eta=eta, s=0.0, train_logloss=ll,
                             train_acc=float(a["rev"]["train"]["acc"].mean())))
            print(f"  [{e.key}] rev  eta={eta:<8g} train_ll={ll:.6f}")
            for s in s_grid:
                n_cfg += 1
                a = run_pair(tit, pot_t, g, eta=eta, scales=(s, s, s), n_iter=e.n_iter, R=searchR,
                             seed=SEARCH_SEED, burn_frac=0.5, thin_tr=1, thin_te=1, alphas=(1,))
                ll = float(a["nrev"]["train"]["logloss"].mean())
                recs.append(dict(exp=e.key, arm="nrev", eta=eta, s=s, train_logloss=ll,
                                 train_acc=float(a["nrev"]["train"]["acc"].mean())))
                print(f"  [{e.key}] nrev eta={eta:<8g} s={s:<5g} train_ll={ll:.6f} "
                      f"({time.perf_counter()-t0:.0f}s)")
        df = pd.DataFrame(recs); rows += recs
        br = df[df.arm == "rev"]; bn = df[df.arm == "nrev"]
        pick[e.key] = dict(rev_eta=float(br.loc[br.train_logloss.idxmin(), "eta"]),
                           nrev_eta=float(bn.loc[bn.train_logloss.idxmin(), "eta"]),
                           nrev_s=float(bn.loc[bn.train_logloss.idxmin(), "s"]))
        print(f"  -> {e.key}: rev best eta={pick[e.key]['rev_eta']:g}; "
              f"nrev best eta={pick[e.key]['nrev_eta']:g}, s={pick[e.key]['nrev_s']:g}")
    pd.DataFrame(rows).to_csv(os.path.join(resdir, "best_search.csv"), index=False)

    print("\n=== STAGE 'best': CONFIRMATION best-vs-best (independent seed %d, R=%d) ===" %
          (CONFIRM_SEED, confR))
    out = {"_n_configs": n_cfg, "eta_grid": list(eta_grid), "s_grid": list(s_grid)}
    for e in exps:
        p = pick[e.key]; g = geom_for(e)
        ar = run_pair(tit, pot_t, g, eta=p["rev_eta"], scales=(0., 0., 0.), n_iter=e.n_iter,
                      R=confR, seed=CONFIRM_SEED, burn_frac=0.5, thin_tr=1, thin_te=1, alphas=(0,))
        an = run_pair(tit, pot_t, g, eta=p["nrev_eta"], scales=(p["nrev_s"],) * 3,
                      n_iter=e.n_iter, R=confR, seed=CONFIRM_SEED, burn_frac=0.5,
                      thin_tr=1, thin_te=1, alphas=(1,))
        blk = dict(pick=p)
        for split in ("train", "test"):
            for m in ("acc", "logloss", "brier"):
                blk[f"{split}_{m}"] = paired_stats(an["nrev"][split][m], ar["rev"][split][m])
        blk["last_iterate_test_acc"] = paired_stats(an["nrev"]["last_acc_test"],
                                                    ar["rev"]["last_acc_test"])
        vr = np.var(ar["rev"]["wbar"], axis=0, ddof=1); vn = np.var(an["nrev"]["wbar"], axis=0, ddof=1)
        blk["pooled_coord_var_ratio"] = float(vr.sum() / vn.sum())
        blk["prob_var_ratio"] = float(np.var(ar["rev"]["Pte"], axis=0, ddof=1).mean() /
                                      np.var(an["nrev"]["Pte"], axis=0, ddof=1).mean())
        out[e.key] = blk
        print(f"\n[{e.key}] BEST-vs-BEST  rev(eta={p['rev_eta']:g})  vs  "
              f"nrev(eta={p['nrev_eta']:g}, s={p['nrev_s']:g})   R={confR}")
        for kk in ("train_logloss", "test_logloss", "train_brier", "test_brier",
                   "train_acc", "test_acc", "last_iterate_test_acc"):
            v = blk[kk]
            print(f"   {kk:<22} rev={v['mean_rev']:.6f} nrev={v['mean_nrev']:.6f} "
                  f"diff={v['diff']:+.3e} se={v['se']:.2e} t={v['t']:+.2f} win={v['win_frac']:.0%}")
        print(f"   pooled coord var ratio rev/nrev = {blk['pooled_coord_var_ratio']:.3f}  "
              f"prob var ratio = {blk['prob_var_ratio']:.3f}")
    json.dump(out, open(os.path.join(resdir, "best_vs_best.json"), "w"), indent=2, default=float)
    return out


def stage_mode(tit, pot_t, exps, eta, s_map, confR, resdir):
    print("\n=== STAGE 'mode': all replicates start at the SAME projected-gradient mode of U_0 ===")
    out = {}
    for e in exps:
        g = geom_for(e)
        wst = find_mode(pot_t, g)
        print(f"  [{e.key}] mode w* = {np.round(wst[0], 4)}  U0={pot_t.U0(wst)[0]:.3f} "
              f"feasible={bool(g.feasible(wst)[0] if np.ndim(g.feasible(wst)) else g.feasible(wst))}")
        W0 = np.repeat(wst, confR, axis=0)
        s = s_map[e.key]
        Xtr, Xte = design(tit.X_train), design(tit.X_test)
        arms = {}
        for tag, alpha in (("rev", 0), ("nrev", 1)):
            r = run_exact(pot_t, g, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=e.n_iter,
                          W0=W0, seed=CONFIRM_SEED, checkpoint_every=1)
            Ws = r["betas"]
            Ptr, _ = ergodic_prob(Xtr, Ws, 0.5, 1); Pte, _ = ergodic_prob(Xte, Ws, 0.5, 1)
            arms[tag] = dict(train=prob_metrics(Ptr, tit.y_train), test=prob_metrics(Pte, tit.y_test),
                             wbar=ergodic_w(Ws, 0.5), Pte=Pte,
                             last_acc_test=accuracy10(tit.X_test, tit.y_test, Ws[-1:])[0])
            del Ws, r
        blk = {}
        for split in ("train", "test"):
            for m in ("acc", "logloss", "brier"):
                blk[f"{split}_{m}"] = paired_stats(arms["nrev"][split][m], arms["rev"][split][m])
        blk["last_iterate_test_acc"] = paired_stats(arms["nrev"]["last_acc_test"],
                                                    arms["rev"]["last_acc_test"])
        blk["variance"] = var_ratio_report(arms)
        blk["s"] = s; blk["eta"] = eta
        out[e.key] = blk
        print(f"[{e.key}] common-start, eta={eta:g}, s={s:g}, R={confR}")
        for kk in ("train_logloss", "test_logloss", "test_brier", "test_acc",
                   "last_iterate_test_acc"):
            v = blk[kk]
            print(f"   {kk:<22} rev={v['mean_rev']:.6f} nrev={v['mean_nrev']:.6f} "
                  f"diff={v['diff']:+.3e} se={v['se']:.2e} t={v['t']:+.2f} win={v['win_frac']:.0%}")
        v = blk["variance"]
        print(f"   PURE MC Var(ergodic w) ratio rev/nrev pooled = {v['pooled_coord_ratio']:.3f}; "
              f"per-coord = {', '.join(f'{x:.2f}' for x in v['coord_ratio'])}")
        print(f"   PURE MC Var(ergodic test prob) ratio = {v['prob_var_ratio']:.3f}")
    json.dump(out, open(os.path.join(resdir, "mode_start.json"), "w"), indent=2, default=float)
    return out


def main2():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["best", "mode"])
    ap.add_argument("--search-R", type=int, default=60)
    ap.add_argument("--confirm-R", type=int, default=300)
    ap.add_argument("--eta", type=float, default=3e-5)
    ap.add_argument("--eta-grid", default="3e-5,1e-4,3e-4,1e-3")
    ap.add_argument("--s-grid", default="2,5,10")
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode-s", type=float, default=-1.0)
    args = ap.parse_args()
    os.makedirs(RESDIR, exist_ok=True)
    t0 = time.perf_counter()
    tit = build_titanic_dataset(); pot_t = make_potential(tit)
    exps = (Exp10("titanic_ball", "titanic", "ball", 1500),
            Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18))
    if args.stage == "best":
        eg = tuple(float(x) for x in args.eta_grid.split(","))
        sg = tuple(float(x) for x in args.s_grid.split(","))
        d = os.path.join(RESDIR, args.tag) if args.tag else RESDIR
        os.makedirs(d, exist_ok=True)
        stage_best(tit, pot_t, exps, eg, sg, args.search_R, args.confirm_R, d)
    else:
        sel = json.load(open(os.path.join(RESDIR, "selected.json")))
        if args.mode_s >= 0:
            sel = {k: args.mode_s for k in sel}
        d = os.path.join(RESDIR, args.tag) if args.tag else RESDIR
        os.makedirs(d, exist_ok=True)
        stage_mode(tit, pot_t, exps, args.eta, sel, args.confirm_R, d)
    print(f"\ntotal {time.perf_counter()-t0:.1f}s")
    return 0


import sys
if __name__ == "__main__":
    _a = sys.argv
    if "--stage" in _a and _a[_a.index("--stage") + 1] in ("best", "mode"):
        raise SystemExit(main2())
    raise SystemExit(main())
