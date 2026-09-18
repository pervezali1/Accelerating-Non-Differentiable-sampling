"""probe_ergodic_refute.py -- ADVERSARIAL re-check of hypothesis `ergodic`.

Attacks, in order:
  A. s=0 CONTROL. With scales=(0,0,0) the J term must vanish, so alpha=1 and alpha=0 must be
     BIT-IDENTICAL given the same W0 and seed.  If not, the pairing is broken.
  B. INDEPENDENT RE-RUN of the headline cell on seed 5501 (used by neither the search seed 7001
     nor the confirmation seed 9101), R=300.
  C. THE DECISIVE TEST -- "is this the transient in disguise?"  Common start (same PGD mode of
     U_0 for every replicate, so across-replicate spread is pure Monte-Carlo error), eta held at
     3e-5, and n_iter swept over 1500 -> 6000 -> 24000 (burn-in always the first 50%,
     checkpoint_every=1, so the number of averaged samples N grows with n_iter).
       * If the claim is a genuine ASYMPTOTIC-VARIANCE reduction, Var(ergodic avg) must decay
         like sigma_asym^2/N for BOTH arms and the RATIO Var_rev/Var_nrev must PERSIST near 2.3.
       * If it is a transient/burn-in artefact, the ratio must collapse towards 1.
     The mean (Jensen) gaps must decay like 1/N under the variance explanation; if they persist
     they are a BIAS difference, which would contradict the identical-invariant-law argument.
  D. BUDGET FAIRNESS at the same eta: nrev costs ~8% more wallclock per step, but give the
     reversible arm 2x and 4x the iterations at the same eta=3e-5 and see whether the ergodic
     win survives a budget-matched (indeed budget-generous) comparison.
  E. SELECTION-BIAS sanity: re-read the claim's own search grids and count.

Creates only results/probe_ergodic_refute/.  Imports exact_anchored / nral; modifies nothing.
"""
from __future__ import annotations
import json, os, time, sys
import numpy as np
from scipy.special import expit

from exact_anchored import (make_potential, run_exact, accuracy10, init_unit_ball10,
                            Exp10, geom_for, D10, apply_J10)
from nral import build_titanic_dataset

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_ergodic_refute")
EPSP = 1e-12
INDEP_SEED = 5501          # used by NEITHER the original search (7001) nor its confirmation (9101)


def design(X):
    return np.ascontiguousarray(np.column_stack([np.ones(len(X)), X]))


def ergodic_prob(Xi, Ws, burn_frac=0.5):
    K = Ws.shape[0]
    k0 = int(round(burn_frac * (K - 1)))
    P = np.zeros((Ws.shape[1], Xi.shape[0]))
    n = 0
    for k in range(k0, K):
        P += expit(Ws[k] @ Xi.T)
        n += 1
    return P / n, n


def prob_metrics(P, y):
    yb = y.astype(bool)
    return dict(acc=np.mean((P >= 0.5) == yb[None, :], axis=1),
                logloss=-np.mean(y[None, :] * np.log(P + EPSP) + (1 - y)[None, :] * np.log(1 - P + EPSP), axis=1),
                brier=np.mean((P - y[None, :]) ** 2, axis=1))


def paired(a_n, a_r):
    d = np.asarray(a_n) - np.asarray(a_r)
    se = d.std(ddof=1) / np.sqrt(len(d)) + 1e-300
    return dict(mean_rev=float(np.mean(a_r)), mean_nrev=float(np.mean(a_n)), diff=float(d.mean()),
                se=float(se), t=float(d.mean() / se), win=float((d > 0).mean()), n=int(len(d)))


def one_arm(tit, pot, geom, *, alpha, s, eta, n_iter, W0, seed, burn=0.5):
    r = run_exact(pot, geom, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=n_iter,
                  W0=W0, seed=seed, checkpoint_every=1)
    Ws = r["betas"]
    Ptr, n = ergodic_prob(design(tit.X_train), Ws, burn)
    Pte, _ = ergodic_prob(design(tit.X_test), Ws, burn)
    k0 = int(round(burn * (Ws.shape[0] - 1)))
    out = dict(train=prob_metrics(Ptr, tit.y_train), test=prob_metrics(Pte, tit.y_test),
               wbar=Ws[k0:].mean(axis=0), Pte=Pte,
               last_acc_test=accuracy10(tit.X_test, tit.y_test, Ws[-1:])[0],
               proj=r["projection_rate"], runtime=r["runtime"], n_avg=n, Wlast=Ws[-1].copy())
    del Ws, r
    return out


def blockstats(an, ar):
    b = {}
    for sp in ("train", "test"):
        for m in ("acc", "logloss", "brier"):
            b[f"{sp}_{m}"] = paired(an[sp][m], ar[sp][m])
    b["last_iterate_test_acc"] = paired(an["last_acc_test"], ar["last_acc_test"])
    vr = np.var(ar["wbar"], axis=0, ddof=1); vn = np.var(an["wbar"], axis=0, ddof=1)
    b["pooled_coord_var_ratio"] = float(vr.sum() / (vn.sum() + 1e-300))
    pr = float(np.var(ar["Pte"], axis=0, ddof=1).mean()); pn = float(np.var(an["Pte"], axis=0, ddof=1).mean())
    b["prob_var_rev"], b["prob_var_nrev"], b["prob_var_ratio"] = pr, pn, float(pr / (pn + 1e-300))
    b["mean_test_prob_rev"] = float(ar["Pte"].mean()); b["mean_test_prob_nrev"] = float(an["Pte"].mean())
    b["wbar_mean_rev"] = ar["wbar"].mean(axis=0).tolist(); b["wbar_mean_nrev"] = an["wbar"].mean(axis=0).tolist()
    b["n_avg"] = int(ar["n_avg"]); b["proj_rev"] = ar["proj"]; b["proj_nrev"] = an["proj"]
    b["runtime_rev"] = ar["runtime"]; b["runtime_nrev"] = an["runtime"]
    return b


def show(tag, b, keys=("test_acc", "test_logloss", "test_brier", "train_logloss", "last_iterate_test_acc")):
    print(f"  [{tag}] N_avg={b['n_avg']}  proj rev/nrev={b['proj_rev']:.3f}/{b['proj_nrev']:.3f}")
    for k in keys:
        v = b[k]
        print(f"    {k:<22} rev={v['mean_rev']:.6f} nrev={v['mean_nrev']:.6f} "
              f"diff={v['diff']:+.4e} se={v['se']:.2e} t={v['t']:+.2f} win={v['win']:.0%}")
    print(f"    VarRatio rev/nrev: w pooled={b['pooled_coord_var_ratio']:.3f}  "
          f"testprob={b['prob_var_ratio']:.3f} (rev={b['prob_var_rev']:.3e} nrev={b['prob_var_nrev']:.3e})")


def find_mode(pot, geom, h=3e-5, n=40000):
    w = np.zeros((1, D10))
    for _ in range(n):
        w, _ = geom.project(w - h * pot.grad_U0(w))
    return w


def main():
    os.makedirs(RES, exist_ok=True)
    t00 = time.perf_counter()
    tit = build_titanic_dataset(); pot = make_potential(tit)
    BALL = Exp10("titanic_ball", "titanic", "ball", 1500)
    LP = Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18)
    ETA = 3e-5
    out = {}

    # ---------------- A. s = 0 control -------------------------------------------------
    print("=== A. s=0 CONTROL: with scales=0 the two arms must be BIT-IDENTICAL ===")
    ctl = {}
    for e, sname in ((BALL, "titanic_ball"), (LP, "titanic_lp")):
        g = geom_for(e)
        W0 = init_unit_ball10(np.random.default_rng(INDEP_SEED + 1), 60)
        # direct algebraic check on J itself
        G = pot.grad_U0(W0)
        J0 = apply_J10(W0, G, g, (0., 0., 0.))
        r0 = run_exact(pot, g, alpha=0, scales=(0., 0., 0.), eta=ETA, n_iter=e.n_iter, W0=W0,
                       seed=INDEP_SEED, checkpoint_every=e.n_iter)
        r1 = run_exact(pot, g, alpha=1, scales=(0., 0., 0.), eta=ETA, n_iter=e.n_iter, W0=W0,
                       seed=INDEP_SEED, checkpoint_every=e.n_iter)
        same = bool(np.array_equal(r0["betas"], r1["betas"]))
        maxJ = float(np.abs(J0).max())
        ctl[sname] = dict(bit_identical=same, max_abs_J_at_s0=maxJ,
                          max_abs_traj_diff=float(np.abs(r0["betas"] - r1["betas"]).max()))
        print(f"  [{sname}] max|J(w,g)| at s=0 = {maxJ:.3e};  trajectories bit-identical = {same}; "
              f"max|diff| = {ctl[sname]['max_abs_traj_diff']:.3e}")
        # also: same-seed different-alpha noise stream check -- first step must differ ONLY by J
    out["A_s0_control"] = ctl

    # ---------------- B. independent re-run of the headline ----------------------------
    print(f"\n=== B. INDEPENDENT RE-RUN, seed {INDEP_SEED} (never used in search or confirmation), R=300 ===")
    R = 300
    rep = {}
    for e, s in ((BALL, 10.0), (LP, 5.0)):
        g = geom_for(e)
        W0 = init_unit_ball10(np.random.default_rng(INDEP_SEED + 1), R)
        ar = one_arm(tit, pot, g, alpha=0, s=s, eta=ETA, n_iter=e.n_iter, W0=W0, seed=INDEP_SEED)
        an = one_arm(tit, pot, g, alpha=1, s=s, eta=ETA, n_iter=e.n_iter, W0=W0, seed=INDEP_SEED)
        b = blockstats(an, ar); b["s"] = s; b["eta"] = ETA; b["n_iter"] = e.n_iter
        rep[e.key] = b
        print(f"\n {e.key}  eta={ETA:g} s={s:g} R={R}")
        show(e.key, b)
        del ar, an
    out["B_independent_rerun"] = rep
    print(f"\n[{time.perf_counter()-t00:.0f}s]")

    # ---------------- C. n_iter scaling from a COMMON START ----------------------------
    print(f"\n=== C. DECISIVE: does the variance ratio PERSIST as the averaging window grows? ===")
    print("    common start (PGD mode of U_0), eta=3e-5 fixed, burn-in 50%, checkpoint_every=1")
    Rc = 100
    scal = {}
    g = geom_for(BALL)
    wst = find_mode(pot, g)
    print(f"  ball mode w* = {np.round(wst[0],4)}")
    W0 = np.repeat(wst, Rc, axis=0)
    for ni in (1500, 6000, 24000):
        ar = one_arm(tit, pot, g, alpha=0, s=10.0, eta=ETA, n_iter=ni, W0=W0, seed=INDEP_SEED)
        an = one_arm(tit, pot, g, alpha=1, s=10.0, eta=ETA, n_iter=ni, W0=W0, seed=INDEP_SEED)
        b = blockstats(an, ar); b["n_iter"] = ni; b["eta"] = ETA; b["s"] = 10.0; b["R"] = Rc
        scal[f"ball_n{ni}"] = b
        print(f"\n ball common-start n_iter={ni} (T=eta*n_iter={ETA*ni:.3g})  R={Rc}")
        show(f"n={ni}", b)
        print(f"    [{time.perf_counter()-t00:.0f}s]")
        del ar, an
    out["C_niter_scaling_ball"] = scal

    # ---------------- D. budget fairness at the same eta -------------------------------
    print(f"\n=== D. BUDGET FAIRNESS at eta=3e-5: give the REVERSIBLE arm 2x and 4x the iterations ===")
    Rd = 200
    W0d = init_unit_ball10(np.random.default_rng(INDEP_SEED + 1), Rd)
    an = one_arm(tit, pot, g, alpha=1, s=10.0, eta=ETA, n_iter=1500, W0=W0d, seed=INDEP_SEED)
    bud = dict(nrev_1500=dict(test_acc=float(an["test"]["acc"].mean()),
                              test_logloss=float(an["test"]["logloss"].mean()),
                              test_brier=float(an["test"]["brier"].mean()),
                              runtime=an["runtime"]))
    print(f"  nrev n_iter=1500 s=10 : test acc={bud['nrev_1500']['test_acc']:.6f} "
          f"ll={bud['nrev_1500']['test_logloss']:.6f} brier={bud['nrev_1500']['test_brier']:.6f} "
          f"({an['runtime']:.1f}s)")
    for mult in (2, 4):
        ar = one_arm(tit, pot, g, alpha=0, s=0.0, eta=ETA, n_iter=1500 * mult, W0=W0d, seed=INDEP_SEED)
        st = dict(test_acc=paired(an["test"]["acc"], ar["test"]["acc"]),
                  test_logloss=paired(an["test"]["logloss"], ar["test"]["logloss"]),
                  test_brier=paired(an["test"]["brier"], ar["test"]["brier"]),
                  rev_runtime=ar["runtime"])
        bud[f"rev_{mult}x"] = st
        print(f"  rev n_iter={1500*mult} ({ar['runtime']:.1f}s, {ar['runtime']/an['runtime']:.2f}x nrev cost):")
        for k in ("test_acc", "test_logloss", "test_brier"):
            v = st[k]
            print(f"     {k:<14} rev={v['mean_rev']:.6f} nrev={v['mean_nrev']:.6f} "
                  f"diff(nrev-rev)={v['diff']:+.4e} se={v['se']:.2e} t={v['t']:+.2f}")
        del ar
    out["D_budget_fairness"] = bud
    del an

    json.dump(out, open(os.path.join(RES, "refute.json"), "w"), indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t00:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
