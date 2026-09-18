"""probe_mixing_refute.py -- ADVERSARIAL re-check of hypothesis `mixing`.

Claim under attack: non-reversible anchored Langevin (alpha=1, s=(5,5,5)) beats the reversible
arm (alpha=0) on ESS per second of wall clock for the test function w7, on Titanic / ball,
both arms at eta=1e-4, paired over 96 coupled chains.  Reported x1.31, t=7.27.

Attacks implemented:
  A  PAIRING CONTROL.  alpha=1 with s=(0,0,0) must be BIT-IDENTICAL to alpha=0.  If not, the
     "same noise stream" pairing is broken and every paired SE in the claim is wrong.
  B  INDEPENDENT-SEED REPLICATION.  Exact headline protocol on a 4th seed (5300), R=96,
     never used by the claimant for search, confirmation or robustness.
  C  STATIONARY-START / GEYER-FREE RE-MEASUREMENT.  Both arms started from a COMMON,
     pre-equilibrated ensemble (long reversible pre-burn), so the retained segment cannot
     contain burn-in drift; ESS measured BOTH with the claimant's Geyer estimator and with a
     truncation-free direct estimator  ESS_direct = sigma^2 / Var_chains(chain mean),
     which is exactly the Monte Carlo efficiency a user experiences.
  D  BEST-VS-BEST WITHOUT THE ARBITRARY BIAS BUDGET.  For each arm and each (eta, s) cell,
     total RMSE of the posterior-mean estimate after a wall-clock budget T:
        RMSE_f(T)^2 = bias_f^2 + sd_f^2 / (ESS_rate_f * T),
     bias measured against an eta -> 0 extrapolation of the ensemble mean (not against a
     single eta=3e-5 run).  Each arm picks its own RMSE-minimising cell at each budget.

Selection here uses TRAIN-ONLY / parameter-only quantities; no test label is ever read.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "probe_mixing_refute")
os.makedirs(OUT, exist_ok=True)

from exact_anchored import make_potential, make_geom10, run_exact, init_unit_ball10, D10
from nral import build_titanic_dataset
# use the CLAIMANT'S OWN estimator code, so no difference can be blamed on methodology
from probe_mixing import iact_geyer, split_rhat, test_functions, TRAIN_ONLY, CK

SEED_NEW   = 5300      # 4th, independent seed (claimant used 3000 search / 4100 / 4200)
SEED_BURN  = 5301
SEED_PROD  = 5302
R          = 96
NIT        = 16000
BURN_FRAC  = 0.40
ETA_H, S_H = 1e-4, 5.0

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

def dump(name):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


def go(pot, geom, *, alpha, s, eta, n_iter, seed, W0, ck=CK):
    t0 = time.perf_counter()
    r = run_exact(pot, geom, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=n_iter,
                  W0=W0, seed=seed, checkpoint_every=ck)
    return r, time.perf_counter() - t0


def series_of(B, pot, X_hold):
    tf, thin = test_functions(B, pot, X_hold)
    return tf, thin


def geyer_ess(S, ck, thin):
    """Claimant's estimator: per-chain tau (iterations) and per-chain ESS."""
    T, Rc = S.shape
    taus = np.array([iact_geyer(S[:, i]) for i in range(Rc)])
    return taus * ck * thin, (T - 1) / (2.0 * taus)


def direct_ess(S):
    """Truncation-free: ESS_per_chain = sigma^2 / Var_across_chains(chain time-average).
    Valid when the chains start from the stationary law (attack C)."""
    cm = S.mean(axis=0)                       # (R,) per-chain time averages
    sig2 = float(S.var(ddof=1))               # pooled marginal variance
    return cm, sig2


def paired_stats(d):
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    return float(d.mean()), float(se), float(d.mean() / se if se > 0 else np.nan), float((d > 0).mean())


# ======================================================================== A + B + C
def main():
    t_all = time.perf_counter()
    ds = build_titanic_dataset(); pot = make_potential(ds)
    X_hold = ds.X_test[:3]
    geom = make_geom10("ball")
    log(f"# titanic n_train={ds.n_train} lam={pot.lam:g} delta={pot.delta:.5g} "
        f"a_lo={pot.a_lower_bound:.3f}  eta={ETA_H:g} s={S_H:g} R={R} n_iter={NIT}")

    # ---------------------------------------------------------------- A: pairing control
    log("\n=== A. PAIRING CONTROL: alpha=1,s=0 must be bit-identical to alpha=0 ===")
    W0t = init_unit_ball10(np.random.default_rng(7), 16)
    r0, _ = go(pot, geom, alpha=0, s=0.0, eta=ETA_H, n_iter=400, seed=77, W0=W0t, ck=1)
    r1, _ = go(pot, geom, alpha=1, s=0.0, eta=ETA_H, n_iter=400, seed=77, W0=W0t, ck=1)
    same = np.array_equal(r0["betas"], r1["betas"])
    log(f"bit-identical (alpha=0) vs (alpha=1,s=0): {same}   "
        f"max|diff|={np.abs(r0['betas']-r1['betas']).max():.3e}")
    # and the same noise stream really is shared across DIFFERENT s
    r2, _ = go(pot, geom, alpha=1, s=S_H, eta=ETA_H, n_iter=400, seed=77, W0=W0t, ck=1)
    log(f"s=5 differs from s=0 as it must: max|diff|={np.abs(r2['betas']-r0['betas']).max():.3e}")
    del r0, r1, r2

    # ---------------------------------------------------------------- B: 4th-seed replication
    log(f"\n=== B. INDEPENDENT-SEED REPLICATION (seed {SEED_NEW}, R={R}, n_iter={NIT}, "
        f"burn {BURN_FRAC:.0%}) -- claimant's exact protocol ===")
    W0 = init_unit_ball10(np.random.default_rng(SEED_NEW + 1), R)
    repl = {}
    for tag, alpha, s in (("rev", 0, 0.0), ("nrev", 1, S_H)):
        r, wall = go(pot, geom, alpha=alpha, s=s, eta=ETA_H, n_iter=NIT, seed=SEED_NEW, W0=W0)
        B = r["betas"]; i0 = int(round(BURN_FRAC * (B.shape[0] - 1)))
        tf, thin = series_of(B[i0:], pot, X_hold)
        repl[tag] = dict(wall=wall, proj=r["projection_rate"], nonf=r["n_nonfinite"],
                         tf={k: geyer_ess(v, CK, thin[k]) for k, v in tf.items()},
                         mean={k: float(v.mean()) for k, v in tf.items()},
                         rhat={k: split_rhat(v) for k, v in tf.items()})
        log(f"  {tag}: wall={wall:.2f}s proj={r['projection_rate']:.4f} nonfinite={r['n_nonfinite']} "
            f"maxRhat={max(repl[tag]['rhat'].values()):.4f}")
        del r, B, tf
    log(f"{'fn':>13} {'tau_rev':>9} {'tau_nrev':>9} {'ratio':>7} | {'ESSps_rev':>10} "
        f"{'ESSps_nrev':>11} {'d':>9} {'se':>8} {'t':>8} {'x':>6} {'win%':>6}")
    tabB = {}
    for k in TRAIN_ONLY:
        tr, er = repl["rev"]["tf"][k];  tn, en = repl["nrev"]["tf"][k]
        er = er / repl["rev"]["wall"];  en = en / repl["nrev"]["wall"]
        d, se, t, w = paired_stats(en - er)
        tabB[k] = dict(tau_rev=float(np.nanmean(tr)), tau_nrev=float(np.nanmean(tn)),
                       essps_rev=float(er.mean()), essps_nrev=float(en.mean()),
                       d=d, se=se, t=t, win=w, x=float(en.mean() / er.mean()))
        e = tabB[k]
        log(f"{k:>13} {e['tau_rev']:9.1f} {e['tau_nrev']:9.1f} "
            f"{e['tau_nrev']/e['tau_rev']:7.3f} | {e['essps_rev']:10.3f} {e['essps_nrev']:11.3f} "
            f"{e['d']:9.3f} {e['se']:8.3f} {e['t']:8.2f} {e['x']:6.2f} {100*e['win']:6.1f}")
    p = tabB["w7"]
    log(f"\nB HEADLINE (w7, seed {SEED_NEW}): rev {p['essps_rev']:.3f} -> nrev {p['essps_nrev']:.3f} "
        f"ESS/s  x{p['x']:.2f}  d={p['d']:+.3f} se={p['se']:.3f} t={p['t']:.2f} win={100*p['win']:.1f}%")
    log(f"  claimant seed 4100: x1.31 (d=+0.580, t=7.27);  seed 4200: x1.26;  search seed 3000: x1.08")
    del repl

    # ---------------------------------------------------------------- C: stationary start
    log(f"\n=== C. STATIONARY-START + GEYER-FREE ESS ===")
    NB = 40000
    W0b = init_unit_ball10(np.random.default_rng(SEED_BURN + 1), R)
    t0 = time.perf_counter()
    rb, _ = go(pot, geom, alpha=0, s=0.0, eta=ETA_H, n_iter=NB, seed=SEED_BURN, W0=W0b, ck=NB)
    W0s = rb["betas"][-1].copy()
    log(f"  common pre-burn: reversible eta={ETA_H:g}, {NB} iterations ({time.perf_counter()-t0:.1f}s), "
        f"~{NB/204:.0f} IACT.  mean w7 after burn = {W0s[:,7].mean():+.4f}")
    del rb
    stat = {}
    for tag, alpha, s in (("rev", 0, 0.0), ("nrev", 1, S_H)):
        r, wall = go(pot, geom, alpha=alpha, s=s, eta=ETA_H, n_iter=NIT, seed=SEED_PROD, W0=W0s)
        B = r["betas"]                                     # NO burn-in discarded
        tf, thin = series_of(B, pot, X_hold)
        T = B.shape[0]
        h = T // 4
        stat[tag] = dict(wall=wall, proj=r["projection_rate"],
                         tf={k: v.copy() for k, v in tf.items()}, thin=thin,
                         rhat={k: split_rhat(v) for k, v in tf.items()},
                         drift={k: (float(tf[k][:h].mean()), float(tf[k][-h:].mean()),
                                    float(tf[k].std(ddof=1))) for k in TRAIN_ONLY})
        log(f"  {tag}: wall={wall:.2f}s proj={r['projection_rate']:.4f} "
            f"maxRhat={max(stat[tag]['rhat'].values()):.4f}  Rhat(w7)={stat[tag]['rhat']['w7']:.4f}")
        del r, B, tf
    log("  stationarity check (first-quarter vs last-quarter ensemble mean, in sd units):")
    for k in ("w7", "w4", "norm2", "U_train"):
        a1, a2, sdv = stat["rev"]["drift"][k]; b1, b2, sdn = stat["nrev"]["drift"][k]
        log(f"    {k:>9}  rev {(a2-a1)/sdv:+.4f}   nrev {(b2-b1)/sdn:+.4f}")

    rng = np.random.default_rng(12345)
    bidx = rng.integers(0, R, size=(2000, R))
    log(f"\n{'fn':>13} | {'GEYER tau_r':>11} {'tau_n':>8} {'ESSps_r':>8} {'ESSps_n':>8} {'x':>5} "
        f"{'t':>7} | {'DIRECT tau_r':>12} {'tau_n':>8} {'ESSps_r':>8} {'ESSps_n':>8} {'x':>5} "
        f"{'boot95':>15}")
    tabC = {}
    for k in TRAIN_ONLY:
        Sr, Sn = stat["rev"]["tf"][k], stat["nrev"]["tf"][k]
        th = stat["rev"]["thin"][k]
        tr, er = geyer_ess(Sr, CK, th); tn, en = geyer_ess(Sn, CK, th)
        erps = er / stat["rev"]["wall"]; enps = en / stat["nrev"]["wall"]
        dg, seg, tg, wg = paired_stats(enps - erps)
        cmr, s2r = direct_ess(Sr); cmn, s2n = direct_ess(Sn)
        Tn = Sr.shape[0]; dt = CK * th
        def d_ess(cm, s2):
            v = cm.var(ddof=1)
            return s2 / max(v, 1e-300)
        Er, En = d_ess(cmr, s2r), d_ess(cmn, s2n)
        taur_d = (Tn - 1) / (2 * Er) * dt; taun_d = (Tn - 1) / (2 * En) * dt
        Erps, Enps = Er / stat["rev"]["wall"], En / stat["nrev"]["wall"]
        # paired bootstrap over chain indices for the DIRECT ratio
        br = np.array([s2r / max(cmr[ix].var(ddof=1), 1e-300) / stat["rev"]["wall"] for ix in bidx])
        bn = np.array([s2n / max(cmn[ix].var(ddof=1), 1e-300) / stat["nrev"]["wall"] for ix in bidx])
        ratio = bn / br
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        pgt = float((ratio > 1).mean())
        tabC[k] = dict(geyer=dict(tau_rev=float(np.nanmean(tr)), tau_nrev=float(np.nanmean(tn)),
                                  essps_rev=float(erps.mean()), essps_nrev=float(enps.mean()),
                                  x=float(enps.mean()/erps.mean()), d=dg, se=seg, t=tg, win=wg),
                       direct=dict(tau_rev=float(taur_d), tau_nrev=float(taun_d),
                                   essps_rev=float(Erps), essps_nrev=float(Enps),
                                   x=float(Enps/Erps), lo=float(lo), hi=float(hi), p_gt1=pgt))
        g, dd = tabC[k]["geyer"], tabC[k]["direct"]
        ci = "[%.2f,%.2f]" % (dd["lo"], dd["hi"])
        log(f"{k:>13} | {g['tau_rev']:11.1f} {g['tau_nrev']:8.1f} {g['essps_rev']:8.3f} "
            f"{g['essps_nrev']:8.3f} {g['x']:5.2f} {g['t']:7.2f} | {dd['tau_rev']:12.1f} "
            f"{dd['tau_nrev']:8.1f} {dd['essps_rev']:8.3f} {dd['essps_nrev']:8.3f} "
            f"{dd['x']:5.2f} {ci:>15}")
    p = tabC["w7"]
    log(f"\nC HEADLINE (w7, stationary start, seed {SEED_PROD}):")
    log(f"  Geyer : rev {p['geyer']['essps_rev']:.3f} -> nrev {p['geyer']['essps_nrev']:.3f} ESS/s "
        f"(x{p['geyer']['x']:.2f}, t={p['geyer']['t']:.2f})")
    log(f"  DIRECT: rev {p['direct']['essps_rev']:.3f} -> nrev {p['direct']['essps_nrev']:.3f} ESS/s "
        f"(x{p['direct']['x']:.2f}, boot 95% [{p['direct']['lo']:.2f},{p['direct']['hi']:.2f}], "
        f"P(x>1)={p['direct']['p_gt1']:.3f})")
    wr = min(tabC[k]["direct"]["essps_rev"] for k in TRAIN_ONLY)
    wn = min(tabC[k]["direct"]["essps_nrev"] for k in TRAIN_ONLY)
    log(f"  worst-case train-only fn (direct): {wr:.3f} -> {wn:.3f} (x{wn/wr:.2f})")

    with open(os.path.join(OUT, "refute_main.json"), "w") as fh:
        json.dump(dict(bit_identical=bool(same), seed=SEED_NEW, R=R, n_iter=NIT,
                       eta=ETA_H, s=S_H, replication=tabB, stationary=tabC,
                       n_burn=NB, burn_frac=BURN_FRAC), fh, indent=1, default=float)
    log(f"\ntotal wall {time.perf_counter()-t_all:.1f}s")
    dump("refute_main_log.txt")


# ======================================================================== D: best vs best
def bvb():
    t_all = time.perf_counter()
    ds = build_titanic_dataset(); pot = make_potential(ds)
    X_hold = ds.X_test[:3]; geom = make_geom10("ball")
    SEED = 5400
    NITD = 16000
    CELLS = [("rev", 1e-4, 0.0), ("rev", 3e-4, 0.0), ("rev", 1e-3, 0.0),
             ("nrev", 1e-4, 5.0), ("nrev", 3e-4, 5.0), ("nrev", 3e-4, 1.0),
             ("nrev", 1e-3, 2.0)]
    log(f"\n=== D. BEST-VS-BEST, threshold-free (seed {SEED}, R={R}, n_iter={NITD}) ===")
    W0 = init_unit_ball10(np.random.default_rng(SEED + 1), R)
    cells = {}
    for arm, eta, s in CELLS:
        r, wall = go(pot, geom, alpha=(0 if arm == "rev" else 1), s=s, eta=eta,
                     n_iter=NITD, seed=SEED, W0=W0)
        B = r["betas"]; i0 = int(round(BURN_FRAC * (B.shape[0] - 1)))
        tf, thin = series_of(B[i0:], pot, X_hold)
        ent = {}
        for k in TRAIN_ONLY:
            tau, ess = geyer_ess(tf[k], CK, thin[k])
            ent[k] = dict(mean=float(tf[k].mean()), sd=float(tf[k].std(ddof=1)),
                          tau=float(np.nanmean(tau)), ess_total=float(ess.sum()),
                          ess_rate=float(ess.sum() / wall))
        cells[(arm, eta, s)] = dict(wall=wall, proj=r["projection_rate"],
                                    nonf=r["n_nonfinite"], f=ent)
        log(f"  {arm:>4} eta={eta:8.1e} s={s:4g} wall={wall:5.2f}s proj={r['projection_rate']:.4f} "
            f"nonf={r['n_nonfinite']}  mean(w7)={ent['w7']['mean']:+.5f} "
            f"tau(w7)={ent['w7']['tau']:7.1f} ESSrate(w7)={ent['w7']['ess_rate']:8.2f}/s")
        del r, B, tf

    # ---- eta -> 0 extrapolation of the ensemble mean, per test function, from the REVERSIBLE
    #      cells only (the invariant law is common to both arms, so this is the right yardstick)
    prev = json.load(open(os.path.join(HERE, "results", "probe_mixing", "mixing_results.json")))
    gold_mean, gold_sd = prev["gold"]["mean"], prev["gold"]["sd"]     # reversible eta=3e-5
    etas = np.array([3e-5, 1e-4, 3e-4, 1e-3])
    log("\n  eta -> 0 extrapolation of the reversible ensemble mean (mean vs eta, in gold-sd units):")
    mu0, fitinfo = {}, {}
    for k in TRAIN_ONLY:
        ys = np.array([gold_mean[k], cells[("rev", 1e-4, 0.0)]["f"][k]["mean"],
                       cells[("rev", 3e-4, 0.0)]["f"][k]["mean"],
                       cells[("rev", 1e-3, 0.0)]["f"][k]["mean"]])
        # fit mu(eta) = mu0 + c*eta  (weighted to the small-eta end) and mu0 + c*sqrt(eta)
        A1 = np.vstack([np.ones(4), etas]).T
        A2 = np.vstack([np.ones(4), np.sqrt(etas)]).T
        c1, r1, *_ = np.linalg.lstsq(A1, ys, rcond=None)
        c2, r2, *_ = np.linalg.lstsq(A2, ys, rcond=None)
        res1 = float(np.sum((A1 @ c1 - ys) ** 2)); res2 = float(np.sum((A2 @ c2 - ys) ** 2))
        use = c1 if res1 <= res2 else c2
        mu0[k] = float(use[0])
        fitinfo[k] = dict(form=("linear" if res1 <= res2 else "sqrt"),
                          mu0=float(use[0]), gold=gold_mean[k],
                          gold_offset_sd=float((gold_mean[k] - use[0]) / gold_sd[k]))
        if k in ("w7", "w4", "w9", "norm2", "U_train"):
            log(f"    {k:>10} form={fitinfo[k]['form']:>6}  mu0={mu0[k]:+.5f}  gold(3e-5)="
                f"{gold_mean[k]:+.5f}  gold's own bias = {fitinfo[k]['gold_offset_sd']:+.3f} sd")

    def bias_sd(key, k):
        return abs(cells[key]["f"][k]["mean"] - mu0[k]) / max(gold_sd[k], 1e-12)

    log("\n  bias (max over 12 train-only fns, in sd units) vs eta->0 EXTRAPOLATION, and vs the "
        "claimant's eta=3e-5 gold:")
    log(f"  {'cell':>22} {'bias_vs_mu0':>12} {'bias_vs_gold':>13} {'ESSrate(w7)':>12} "
        f"{'|bias_w7|':>10}")
    for key in cells:
        bx = max(bias_sd(key, k) for k in TRAIN_ONLY)
        bg = max(abs(cells[key]["f"][k]["mean"] - gold_mean[k]) / max(gold_sd[k], 1e-12)
                 for k in TRAIN_ONLY)
        cells[key]["bias_mu0"] = bx; cells[key]["bias_gold"] = bg
        log(f"  {str(key):>22} {bx:12.4f} {bg:13.4f} {cells[key]['f']['w7']['ess_rate']:12.2f} "
            f"{bias_sd(key,'w7'):10.4f}")

    # ---- RMSE of the estimate of E[w7] after a wall-clock budget T (no arbitrary threshold)
    log("\n  RMSE (in sd units) of the E[w7] estimate after a wall-clock budget T, each arm "
        "picking its OWN best cell:")
    log(f"  {'T (s)':>9} | {'rev best cell':>22} {'RMSE_rev':>9} | {'nrev best cell':>22} "
        f"{'RMSE_nrev':>10} | {'winner':>6} {'ratio':>6}")
    budget_rows = []
    for T in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 1e4, 1e5):
        best = {}
        for arm in ("rev", "nrev"):
            cand = [(np.sqrt(bias_sd(key, "w7") ** 2
                             + 1.0 / max(cells[key]["f"]["w7"]["ess_rate"] * T, 1e-300)), key)
                    for key in cells if key[0] == arm and cells[key]["nonf"] == 0]
            best[arm] = min(cand)
        rr, kr = best["rev"]; rn, kn = best["nrev"]
        win = "nrev" if rn < rr else "rev"
        log(f"  {T:9.0f} | {str(kr):>22} {rr:9.4f} | {str(kn):>22} {rn:10.4f} | {win:>6} "
            f"{rr/rn:6.3f}")
        budget_rows.append(dict(T=T, rev=list(kr), rmse_rev=float(rr), nrev=list(kn),
                                rmse_nrev=float(rn), winner=win, ratio=float(rr / rn)))

    # ---- matched-bias head-to-head at each eta
    log("\n  MATCHED-BIAS head-to-head at each eta (ESS rate of w7, whole ensemble, per second):")
    for eta in (1e-4, 3e-4, 1e-3):
        rk = ("rev", eta, 0.0)
        nk = [k for k in cells if k[0] == "nrev" and k[1] == eta]
        for k in nk:
            log(f"    eta={eta:8.1e}  rev(bias {cells[rk]['bias_mu0']:.3f}) "
                f"{cells[rk]['f']['w7']['ess_rate']:8.2f}/s   vs  nrev s={k[2]:g} "
                f"(bias {cells[k]['bias_mu0']:.3f}) {cells[k]['f']['w7']['ess_rate']:8.2f}/s   "
                f"x{cells[k]['f']['w7']['ess_rate']/cells[rk]['f']['w7']['ess_rate']:.2f}")
    log(f"\n  KEY CROSS-ETA: rev@3e-4 {cells[('rev',3e-4,0.0)]['f']['w7']['ess_rate']:.2f}/s "
        f"vs nrev@1e-4 s=5 {cells[('nrev',1e-4,5.0)]['f']['w7']['ess_rate']:.2f}/s  -> "
        f"x{cells[('rev',3e-4,0.0)]['f']['w7']['ess_rate']/cells[('nrev',1e-4,5.0)]['f']['w7']['ess_rate']:.2f} "
        f"in favour of the REVERSIBLE arm at ITS larger step size")

    with open(os.path.join(OUT, "refute_bvb.json"), "w") as fh:
        json.dump(dict(seed=SEED, R=R, n_iter=NITD,
                       cells={f"{k[0]}|eta={k[1]:g}|s={k[2]:g}": dict(
                           wall=v["wall"], proj=v["proj"], bias_mu0=v["bias_mu0"],
                           bias_gold=v["bias_gold"],
                           f={kk: vv for kk, vv in v["f"].items()}) for k, v in cells.items()},
                       mu0=mu0, fit=fitinfo, budgets=budget_rows), fh, indent=1, default=float)
    log(f"\ntotal wall {time.perf_counter()-t_all:.1f}s")
    dump("refute_bvb_log.txt")




# ======================================================================== E/F/G: post-hoc
def posthoc():
    """Re-analysis of the runs already stored by main()/bvb(): no new sampling."""
    import json, numpy as np, os
    H="/home/user/Accelerating-Non-Differentiable-sampling"
    O=os.path.join(H,"results","probe_mixing_refute")
    d=json.load(open(os.path.join(O,"refute_bvb.json")))
    g=json.load(open(os.path.join(H,"results","probe_mixing","mixing_results.json")))
    gsd=g["gold"]["sd"]; mu0=d["mu0"]
    TR=[f"w{j}" for j in range(1,10)]+["norm2","loglik_train","U_train"]
    L=[]
    def p(*a):
        s=" ".join(str(x) for x in a); print(s); L.append(s)

    p("=== E. IS THE BIAS THAT DRIVES 'BEST-VS-BEST' RESOLVABLE? (seed 5400, R=96) ===")
    p("MC standard error of each cell's estimate of E[f], in gold-sd units: se = sd_f/sqrt(ESS_total_f)")
    p(f"{'cell':>24} {'|bias_w7|':>10} {'se_w7':>8} {'bias/se':>8} | {'maxbias12':>10} {'se@argmax':>10} {'bias/se':>8} {'argmax':>12}")
    for k,v in d["cells"].items():
        f=v["f"]
        se7=f["w7"]["sd"]/np.sqrt(f["w7"]["ess_total"])/gsd["w7"]
        b7=abs(f["w7"]["mean"]-mu0["w7"])/gsd["w7"]
        bs={n:abs(f[n]["mean"]-mu0[n])/gsd[n] for n in TR}
        am=max(bs,key=bs.get)
        sem=f[am]["sd"]/np.sqrt(f[am]["ess_total"])/gsd[am]
        p(f"{k:>24} {b7:10.4f} {se7:8.4f} {b7/se7:8.2f} | {bs[am]:10.4f} {sem:10.4f} {bs[am]/sem:8.2f} {am:>12}")

    p("")
    p("=== F. SELF-CONSISTENT MULTI-FUNCTIONAL CRITERION ===")
    p("Headline metric should match the selection metric. WORST-CASE ESS rate over the 12 train-only")
    p("test functions (whole ensemble, per second of wall clock), with the bias ceiling applied:")
    p(f"{'cell':>24} {'maxbias(mu0)':>13} {'worstESS/s':>11} {'argmin':>12} {'ESS/s(w7)':>11}")
    rows={}
    for k,v in d["cells"].items():
        f=v["f"]
        rates={n:f[n]["ess_rate"] for n in TR}
        am=min(rates,key=rates.get)
        rows[k]=(v["bias_mu0"],rates[am],am,f["w7"]["ess_rate"])
        p(f"{k:>24} {v['bias_mu0']:13.4f} {rates[am]:11.2f} {am:>12} {f['w7']['ess_rate']:11.2f}")
    p("")
    for B in (0.21,0.25,0.40,0.65):
        ok={a:[k for k in rows if k.startswith(a) and rows[k][0]<=B] for a in ("rev","nrev")}
        if not ok["rev"] or not ok["nrev"]:
            p(f"  ceiling {B:.2f}: rev admissible={ok['rev']} nrev={ok['nrev']}  -> incomparable"); continue
        br=max(ok["rev"],key=lambda k:rows[k][1]); bn=max(ok["nrev"],key=lambda k:rows[k][1])
        p(f"  ceiling {B:.2f} (vs mu0): rev-best {br} worstESS/s={rows[br][1]:.2f} | "
          f"nrev-best {bn} worstESS/s={rows[bn][1]:.2f}  -> x{rows[bn][1]/rows[br][1]:.2f}")
    p("")
    p("  same, but scoring on ESS/s of w7 only (the claimant's headline metric):")
    for B in (0.21,0.25,0.40,0.65):
        ok={a:[k for k in rows if k.startswith(a) and rows[k][0]<=B] for a in ("rev","nrev")}
        if not ok["rev"] or not ok["nrev"]: continue
        br=max(ok["rev"],key=lambda k:rows[k][3]); bn=max(ok["nrev"],key=lambda k:rows[k][3])
        p(f"  ceiling {B:.2f}: rev-best {br} {rows[br][3]:.2f}/s | nrev-best {bn} {rows[bn][3]:.2f}/s "
          f" -> x{rows[bn][3]/rows[br][3]:.2f}")

    m=json.load(open(os.path.join(O,"refute_main.json")))
    p("")
    p("=== G. GEYER vs DIRECT on IDENTICAL stationary-start data (attack 5) ===")
    p(f"{'fn':>13} {'tauG_rev':>9} {'tauD_rev':>9} {'G/D_rev':>8} {'tauG_nrev':>10} {'tauD_nrev':>10} {'G/D_nrev':>9} {'xG':>6} {'xD':>6}")
    for n in TR:
        G=m["stationary"][n]["geyer"]; D=m["stationary"][n]["direct"]
        p(f"{n:>13} {G['tau_rev']:9.1f} {D['tau_rev']:9.1f} {G['tau_rev']/D['tau_rev']:8.3f} "
          f"{G['tau_nrev']:10.1f} {D['tau_nrev']:10.1f} {G['tau_nrev']/D['tau_nrev']:9.3f} "
          f"{G['x']:6.2f} {D['x']:6.2f}")
    p("  -> if G/D is SMALLER for the non-reversible arm, Geyer truncation FLATTERS it.")
    nn=[n for n in TR]
    gr=np.mean([m["stationary"][n]["geyer"]["tau_rev"]/m["stationary"][n]["direct"]["tau_rev"] for n in nn])
    gn=np.mean([m["stationary"][n]["geyer"]["tau_nrev"]/m["stationary"][n]["direct"]["tau_nrev"] for n in nn])
    p(f"  mean G/D: rev {gr:.3f}  nrev {gn:.3f}")
    open(os.path.join(O,"refute_posthoc_log.txt"),"w").write("\n".join(L)+"\n")

if __name__ == "__main__":
    {"bvb": bvb, "posthoc": posthoc}.get(sys.argv[1] if len(sys.argv) > 1 else "main", main)()
