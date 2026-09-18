"""probe_calibration.py -- HYPOTHESIS: "metrics with headroom".

Accuracy saturates, and (established) the CONTINUOUS reversible / non-reversible processes share
the same invariant law, so the LAST-ITERATE distribution of any functional is identical and cannot
separate the arms.  The one place classical theory *does* predict a gap is the ASYMPTOTIC VARIANCE
OF ERGODIC AVERAGES (Hwang-Hwang-Sheu; Duncan-Lelievre-Pavliotis): adding a divergence-free skew
drift can only lower the asymptotic variance of a time average.  So we measure, per replicate:

    hat_F_r = (1/|T|) sum_{t in T} F(w_t^{(r)})          F in {w, ||w||^2, sigma(x_j . w)}

against a long high-accuracy REFERENCE chain, and decompose the error into bias^2 + variance.
We also report the calibration metrics with headroom (log loss, Brier, ROC-AUC) of the ergodic
posterior-predictive mean, and -- as a control -- of the LAST ITERATE.

All selection is done on a SEARCH seed using TRAIN features / the reference only (never test
labels); the headline comes from a CONFIRMATION run on an independent seed.

Outputs -> results/probe_calibration/
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exact_anchored as EA
from nral import build_titanic_dataset, build_magic_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_calibration")
os.makedirs(OUT, exist_ok=True)

BURN_FRAC = 0.34          # fixed a priori, never tuned
THIN = 2

# ----------------------------------------------------------------- functionals
def ergodic_functionals(Ws, ck, Xi_map, burn_frac=BURN_FRAC, thin=THIN):
    """Per-replicate time averages of w, ||w||^2 and sigma(X w) over the post-burn-in window."""
    ck = np.asarray(ck, dtype=float)
    keep = np.nonzero(ck >= burn_frac * ck[-1])[0][::thin]
    W = Ws[keep]                                   # (T, R, 10)
    T, R, _ = W.shape
    m_w = W.mean(axis=0)                           # (R, 10)
    n2 = (W * W).sum(-1).mean(axis=0)              # (R,)
    preds = {}
    for name, Xi in Xi_map.items():
        acc = np.zeros((R, Xi.shape[0]))
        for c0 in range(0, T, 128):
            blk = W[c0:c0 + 128]
            t_ = blk.shape[0]
            z = blk.reshape(t_ * R, 10) @ Xi.T
            acc += expit(z).reshape(t_, R, -1).sum(axis=0)
        preds[name] = acc / T
    return dict(m_w=m_w, n2=n2, preds=preds, n_used=T)


def last_iterate_preds(Ws, Xi_map):
    W = Ws[-1]
    return {name: expit(W @ Xi.T) for name, Xi in Xi_map.items()}


def with_intercept(X):
    return np.ascontiguousarray(np.column_stack([np.ones(len(X)), X]))


# ----------------------------------------------------------------- calibration metrics
def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y[None, :] * np.log(p) + (1 - y)[None, :] * np.log(1 - p)).mean(axis=1)


def brier(p, y):
    return ((p - y[None, :]) ** 2).mean(axis=1)


def auc_rows(p, y):
    """ROC-AUC per row via the rank / Mann-Whitney identity."""
    y = y.astype(bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    out = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        order = np.argsort(p[i], kind="mergesort")
        s = p[i][order]
        ranks = np.empty(len(s))
        j = 0
        while j < len(s):
            k = j
            while k + 1 < len(s) and s[k + 1] == s[j]:
                k += 1
            ranks[j:k + 1] = 0.5 * (j + k) + 1.0
            j = k + 1
        r = np.empty(len(s)); r[order] = ranks
        out[i] = (r[y].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return out


def acc_rows(p, y):
    return ((p >= 0.5) == y.astype(bool)[None, :]).mean(axis=1)


# ----------------------------------------------------------------- paired stats
def paired(a_nrev, a_rev, lower_is_better):
    """diff is signed so that POSITIVE always means 'non-reversible is better'."""
    d = (a_rev - a_nrev) if lower_is_better else (a_nrev - a_rev)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    return dict(mean_nrev=float(a_nrev.mean()), mean_rev=float(a_rev.mean()),
                diff=float(d.mean()), se=float(se), t=float(d.mean() / se),
                win_frac=float((d > 0).mean()), n=int(n))


def mse_decomp(est, ref):
    """est: (R,) or (R,k).  Returns per-replicate squared error plus bias^2 / variance split."""
    est = np.atleast_2d(est.T).T if est.ndim > 1 else est[:, None]
    ref = np.atleast_1d(ref)
    err = est - ref[None, :]
    per_rep = (err ** 2).mean(axis=1)                     # (R,)
    bias = err.mean(axis=0)
    var = err.var(axis=0, ddof=1)
    return per_rep, float((bias ** 2).mean()), float(var.mean())


# ----------------------------------------------------------------- driver
def run(pot, geom, alpha, scales, eta, n_iter, R, seed, ck_every):
    W0 = EA.init_unit_ball10(np.random.default_rng(seed + 1), R)
    assert np.all(geom.feasible(W0))
    return EA.run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                        W0=W0, seed=seed, checkpoint_every=ck_every)


def main(geom_kind="ball", etas=None, s_grid=None, tag_out=""):
    t_start = time.time()
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    ds = build_titanic_dataset()
    pot = EA.make_potential(ds)
    geom = EA.make_geom10(geom_kind, p=2.4, eps=0.18, Lam=4.0)
    P(f"# Titanic {geom_kind}  n_train={ds.n_train} n_test={ds.n_test} lam={pot.lam:.3f} "
      f"delta={pot.delta:.5f} a_lower={pot.a_lower_bound:.3f}")

    rng_sub = np.random.default_rng(11)
    sub = rng_sub.choice(ds.n_train, size=256, replace=False)
    Xi_map = {"trsub": with_intercept(ds.X_train[sub]),
              "test": with_intercept(ds.X_test)}

    # ---------------------------------------------------------------- 1. REFERENCE
    ETA_REF, N_REF, R_REF, CK_REF = 1e-5, 20000, 160, 4
    P(f"\n## reference: eta={ETA_REF:g} n_iter={N_REF} R={R_REF} burn={BURN_FRAC}")
    refs = {}
    cache = os.path.join(OUT, f"reference_cache{tag_out}.npz")
    cached = dict(np.load(cache, allow_pickle=True)) if os.path.exists(cache) else None
    store = {}
    for tag, alpha, sc in (("rev", 0, (0., 0., 0.)), ("nrev_s2", 1, (2., 2., 2.))):
        t0 = time.time()
        if cached is not None:
            refs[tag] = dict(m_w=cached[f"{tag}_m_w"], n2=float(cached[f"{tag}_n2"]),
                             preds={k: cached[f"{tag}_p_{k}"] for k in Xi_map},
                             se_m_w=cached[f"{tag}_se_m_w"], se_n2=float(cached[f"{tag}_se_n2"]),
                             se_pred={k: cached[f"{tag}_sep_{k}"] for k in Xi_map},
                             proj=float(cached[f"{tag}_proj"]))
            P(f"  {tag:8s} [loaded from cache]  E||w||^2={refs[tag]['n2']:.5f}")
            continue
        r = run(pot, geom, alpha, sc, ETA_REF, N_REF, R_REF, 9001, CK_REF)
        f = ergodic_functionals(r["betas"], r["checkpoints"], Xi_map, thin=3)
        refs[tag] = dict(m_w=f["m_w"].mean(0), n2=float(f["n2"].mean()),
                         preds={k: v.mean(0) for k, v in f["preds"].items()},
                         se_m_w=f["m_w"].std(0, ddof=1) / np.sqrt(R_REF),
                         se_n2=float(f["n2"].std(ddof=1) / np.sqrt(R_REF)),
                         se_pred={k: v.std(0, ddof=1) / np.sqrt(R_REF) for k, v in f["preds"].items()},
                         proj=r["projection_rate"])
        P(f"  {tag:8s} {time.time()-t0:6.1f}s  proj={r['projection_rate']:.4f}  "
          f"E||w||^2={refs[tag]['n2']:.5f}+-{refs[tag]['se_n2']:.5f}  "
          f"E[w]={np.array2string(refs[tag]['m_w'], precision=4)}")
        store[f"{tag}_m_w"] = refs[tag]["m_w"]; store[f"{tag}_n2"] = refs[tag]["n2"]
        store[f"{tag}_se_m_w"] = refs[tag]["se_m_w"]; store[f"{tag}_se_n2"] = refs[tag]["se_n2"]
        store[f"{tag}_proj"] = refs[tag]["proj"]
        for k in Xi_map:
            store[f"{tag}_p_{k}"] = refs[tag]["preds"][k]
            store[f"{tag}_sep_{k}"] = refs[tag]["se_pred"][k]
        del r
    if cached is None:
        np.savez(cache, **store)

    REF = refs["rev"]
    dmw = refs["nrev_s2"]["m_w"] - REF["m_w"]
    smw = np.sqrt(refs["nrev_s2"]["se_m_w"] ** 2 + REF["se_m_w"] ** 2)
    dp = refs["nrev_s2"]["preds"]["test"] - REF["preds"]["test"]
    sp = np.sqrt(refs["nrev_s2"]["se_pred"]["test"] ** 2 + REF["se_pred"]["test"] ** 2)
    P("  invariant-law check (rev vs nrev reference, both eta=1e-5):")
    P(f"    max |dE[w_i]| / se = {np.max(np.abs(dmw / smw)):.2f}   "
      f"max |dE[sigma]| / se = {np.max(np.abs(dp / sp)):.2f}")
    P(f"    dE||w||^2 = {refs['nrev_s2']['n2'] - REF['n2']:+.5f} "
      f"(se {np.hypot(refs['nrev_s2']['se_n2'], REF['se_n2']):.5f})")
    ll_ref_test = float(logloss(REF["preds"]["test"][None, :], ds.y_test)[0])
    br_ref_test = float(brier(REF["preds"]["test"][None, :], ds.y_test)[0])
    auc_ref_test = float(auc_rows(REF["preds"]["test"][None, :], ds.y_test)[0])
    P(f"  REFERENCE posterior-mean predictive on test: logloss={ll_ref_test:.6f} "
      f"brier={br_ref_test:.6f} auc={auc_ref_test:.6f}")
    ref_check = dict(ll_ref_test=ll_ref_test, br_ref_test=br_ref_test, auc_ref_test=auc_ref_test,max_z_Ew=float(np.max(np.abs(dmw / smw))),
                     max_z_Esig=float(np.max(np.abs(dp / sp))),
                     d_n2=float(refs["nrev_s2"]["n2"] - REF["n2"]),
                     se_d_n2=float(np.hypot(refs["nrev_s2"]["se_n2"], REF["se_n2"])))

    # ---------------------------------------------------------------- 2. SEARCH (train-only)
    ETAS = etas if etas else [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    S_GRID = s_grid if s_grid else [0.3, 1.0, 2.0, 5.0, 7.0, 10.0]
    R_S, N_S, CK_S, SEED_S = 64, 3000, 2, 5101
    P(f"\n## search  seed={SEED_S} R={R_S} n_iter={N_S}   criterion = MSE of ergodic "
      f"E[sigma(x_train_sub . w)] vs reference  (NO test labels)")
    rows = []
    n_cfg = 0
    for eta in ETAS:
        for sval in [0.0] + S_GRID:
            alpha = 0 if sval == 0.0 else 1
            sc = (sval, sval, sval)
            r = run(pot, geom, alpha, sc, eta, N_S, R_S, SEED_S, CK_S)
            f = ergodic_functionals(r["betas"], r["checkpoints"], {"trsub": Xi_map["trsub"]})
            per_rep, b2, vv = mse_decomp(f["preds"]["trsub"], REF["preds"]["trsub"])
            pr_w, b2w, vvw = mse_decomp(f["m_w"], REF["m_w"])
            rows.append(dict(eta=eta, s=sval, arm=("rev" if alpha == 0 else "nrev"),
                             mse_sig=float(per_rep.mean()), se_mse_sig=float(per_rep.std(ddof=1) / np.sqrt(R_S)),
                             bias2_sig=b2, var_sig=vv,
                             mse_w=float(pr_w.mean()), bias2_w=b2w, var_w=vvw,
                             proj=r["projection_rate"], nonfinite=r["n_nonfinite"]))
            n_cfg += 1
            P(f"  eta={eta:8.1e} s={sval:5.2f} {rows[-1]['arm']:4s} "
              f"MSE_sig={rows[-1]['mse_sig']:.3e} (bias2={b2:.3e} var={vv:.3e}) "
              f"MSE_w={rows[-1]['mse_w']:.3e} proj={r['projection_rate']:.4f}")
            del r

    import csv
    with open(os.path.join(OUT, f"search{tag_out}.csv"), "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        [wtr.writerow(x) for x in rows]

    best_nrev = min([x for x in rows if x["arm"] == "nrev"], key=lambda x: x["mse_sig"])
    best_rev = min([x for x in rows if x["arm"] == "rev"], key=lambda x: x["mse_sig"])
    P(f"\n  best NREV (train-only): eta={best_nrev['eta']:g} s={best_nrev['s']:g} "
      f"MSE_sig={best_nrev['mse_sig']:.3e}")
    P(f"  best REV  (train-only): eta={best_rev['eta']:g}  MSE_sig={best_rev['mse_sig']:.3e}")

    # ---------------------------------------------------------------- 3. CONFIRMATION
    R_C, N_C, CK_C, SEED_C = 240, 3000, 2, 7717
    eta_star, s_star = best_nrev["eta"], best_nrev["s"]
    P(f"\n## confirmation  seed={SEED_C} (independent) R={R_C} n_iter={N_C}")
    P(f"   matched-eta pair at eta={eta_star:g}: rev(alpha=0) vs nrev(alpha=1,s={s_star:g}), "
      f"same W0 and same noise stream")

    def full_eval(alpha, sval, eta):
        sc = (sval, sval, sval)
        r = run(pot, geom, alpha, sc, eta, N_C, R_C, SEED_C, CK_C)
        f = ergodic_functionals(r["betas"], r["checkpoints"], Xi_map)
        li = last_iterate_preds(r["betas"], Xi_map)
        out = dict(proj=r["projection_rate"], nonfinite=r["n_nonfinite"])
        out["se_sig_test"], out["bias2_test"], out["var_test"] = mse_decomp(f["preds"]["test"], REF["preds"]["test"])
        out["se_sig_tr"], out["bias2_tr"], out["var_tr"] = mse_decomp(f["preds"]["trsub"], REF["preds"]["trsub"])
        out["se_w"], out["bias2_w"], out["var_w"] = mse_decomp(f["m_w"], REF["m_w"])
        out["se_n2"], out["bias2_n2"], out["var_n2"] = mse_decomp(f["n2"], np.array([REF["n2"]]))
        p = f["preds"]["test"]
        out["ll"] = logloss(p, ds.y_test); out["br"] = brier(p, ds.y_test)
        out["auc"] = auc_rows(p, ds.y_test); out["acc"] = acc_rows(p, ds.y_test)
        ptr = f["preds"]["trsub"]
        out["ll_tr"] = logloss(ptr, ds.y_train[sub])
        q = li["test"]
        out["ll_last"] = logloss(q, ds.y_test); out["br_last"] = brier(q, ds.y_test)
        out["auc_last"] = auc_rows(q, ds.y_test); out["acc_last"] = acc_rows(q, ds.y_test)
        out["pbar_test"] = f["preds"]["test"].mean(axis=0)
        del r
        return out

    A = full_eval(1, s_star, eta_star)
    B = full_eval(0, 0.0, eta_star)

    conf = {}
    for key, lower in (("se_sig_test", True), ("se_sig_tr", True), ("se_w", True), ("se_n2", True),
                       ("ll", True), ("br", True), ("auc", False), ("acc", False), ("ll_tr", True),
                       ("ll_last", True), ("br_last", True), ("auc_last", False), ("acc_last", False)):
        conf[key] = paired(A[key], B[key], lower)
    for k in ("bias2_test", "var_test", "bias2_w", "var_w", "proj", "nonfinite"):
        conf[k] = dict(nrev=A[k], rev=B[k])

    P("\n  PAIRED (positive diff = non-reversible better):")
    for k in ("se_sig_test", "se_sig_tr", "se_w", "se_n2", "ll", "br", "auc", "acc",
              "ll_last", "br_last", "auc_last", "acc_last"):
        c = conf[k]
        P(f"    {k:12s} nrev={c['mean_nrev']:.6g}  rev={c['mean_rev']:.6g}  "
          f"diff={c['diff']:+.4g} se={c['se']:.3g} t={c['t']:+6.2f} win={c['win_frac']:.2f}")
    P(f"\n    bias^2/var split, E[sigma] on test: nrev bias2={A['bias2_test']:.3e} var={A['var_test']:.3e}"
      f" | rev bias2={B['bias2_test']:.3e} var={B['var_test']:.3e}")
    P(f"    bias^2/var split, E[w]:              nrev bias2={A['bias2_w']:.3e} var={A['var_w']:.3e}"
      f" | rev bias2={B['bias2_w']:.3e} var={B['var_w']:.3e}")

    diag = {}
    for tag, O in (("nrev", A), ("rev", B)):
        pm = O["pbar_test"][None, :]
        diag[tag] = dict(ll_of_mean=float(logloss(pm, ds.y_test)[0]),
                         br_of_mean=float(brier(pm, ds.y_test)[0]),
                         auc_of_mean=float(auc_rows(pm, ds.y_test)[0]),
                         mean_abs_dev_from_ref=float(np.abs(O["pbar_test"] - REF["preds"]["test"]).mean()))
    diag["ll_ref_test"] = ll_ref_test
    P("\n  WHY log loss moves the other way -- log loss of the REPLICATE-AVERAGED predictive:")
    P(f"    reference (long chain)      logloss={ll_ref_test:.6f}")
    for tag in ("nrev", "rev"):
        P(f"    {tag:4s} averaged predictive   logloss={diag[tag]['ll_of_mean']:.6f} "
          f"auc={diag[tag]['auc_of_mean']:.6f} mean|p-p_ref|={diag[tag]['mean_abs_dev_from_ref']:.2e}")
    P("    -> if the reference's own log loss is the WORST of the three, estimation noise is")
    P("       moving predictions AWAY from the posterior mean in a direction that happens to")
    P("       lower test log loss, so 'less sampling error' shows up as 'slightly worse log loss'.")

    # -------------------------------------------------- 4. BEST vs BEST (rev at its own best eta)
    bvb = None
    if best_rev["eta"] != eta_star:
        P(f"\n## best-vs-best: reversible at its OWN best eta={best_rev['eta']:g} (unpaired vs nrev above)")
        C = full_eval(0, 0.0, best_rev["eta"])
        bvb = {}
        for key, lower in (("se_sig_test", True), ("ll", True), ("br", True), ("auc", False), ("acc", False)):
            a, c = A[key], C[key]
            d = (c.mean() - a.mean()) if lower else (a.mean() - c.mean())
            se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
            bvb[key] = dict(mean_nrev=float(a.mean()), mean_rev_bestEta=float(c.mean()),
                            diff=float(d), se=float(se), t=float(d / se))
            P(f"    {key:12s} nrev={a.mean():.6g}  rev*={c.mean():.6g}  diff={d:+.4g} "
              f"se={se:.3g} t={d/se:+6.2f}")
        bvb["rev_best_eta"] = best_rev["eta"]
    else:
        P(f"\n## best-vs-best: reversible's own best eta EQUALS eta*={eta_star:g}; "
          f"the matched-eta paired comparison already IS best-vs-best.")

    payload = dict(dataset="titanic", geometry=geom_kind, lasso_frac=EA.LASSO_FRAC, sigma=EA.SIGMA,
                   lam=float(pot.lam), delta=float(pot.delta), burn_frac=BURN_FRAC, thin=THIN,
                   reference=dict(eta=ETA_REF, n_iter=N_REF, R=R_REF,
                                  E_w=REF["m_w"].tolist(), se_E_w=REF["se_m_w"].tolist(),
                                  E_n2=REF["n2"], se_E_n2=REF["se_n2"]),
                   invariant_law_check=ref_check,
                   search=dict(etas=ETAS, s_grid=S_GRID, R=R_S, n_iter=N_S, seed=SEED_S,
                               n_configs=n_cfg, best_nrev=best_nrev, best_rev=best_rev),
                   confirmation=dict(seed=SEED_C, R=R_C, n_iter=N_C, eta=eta_star, s=s_star, **conf),
                   logloss_diagnostic=diag,
                   best_vs_best=bvb,
                   wallclock_s=time.time() - t_start)
    with open(os.path.join(OUT, f"calibration{tag_out}.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    P(f"\n# total wallclock {time.time()-t_start:.1f}s")
    with open(os.path.join(OUT, f"run_log{tag_out}.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "lp":
        main(geom_kind="lp", etas=[3e-4, 1e-3, 3e-3, 1e-2],
             s_grid=[0.3, 1.0, 2.0, 5.0, 10.0], tag_out="_lp")
    else:
        main()
