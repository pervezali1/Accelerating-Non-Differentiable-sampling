"""probe_wallclock.py -- Fair-cost accounting for the claimed TRANSIENT non-reversible win.

Question
--------
The non-reversible arm pays extra arithmetic per step (J_s(w) grad U_0, three 3-d cross
products per replicate).  The established transient wins (Titanic ball eta=1e-5 s=5;
Titanic smoothed-l_p eta=1e-5 s=1 or s=5) are ITERATION-matched.  Are they still wins when
both arms are given the same WALL-CLOCK budget, i.e. when the reversible arm is allowed
proportionally more iterations?  And do they survive letting the reversible arm choose its
own best step size?

Protocol (all decisions made on TRAIN accuracy or on timing only; test labels are never used
to choose anything)
-------------------------------------------------------------------------------------------
Phase 0  Time one iteration of each arm, per dataset x geometry, with the same R and the same
         iteration count.  c = t_nrev_per_iter / t_rev_per_iter >= 1.  The reversible arm's
         compute-matched budget is M = round_to_10(c * N).
Phase 1  SEARCH seed.  For each geometry run BOTH arms on the full eta grid for M iterations
         with checkpoints every 10, so that accuracy at iteration N and at iteration M can be
         read off the same trajectory.  Pick each arm's own best eta by TRAIN accuracy at its
         own honest budget (nrev at N, rev at M).
Phase 2  CONFIRM seed (independent, never used in phase 1).  Re-run the selected configs and
         report the paired test-accuracy differences:
           B  iteration-matched, same eta      (the published claim)
           C  compute-matched,   same eta      (the fair-cost version of the claim)
           D  rev at its own best eta, N iters
           E  rev at its own best eta, M iters (best-vs-best at matched wall-clock)
Everything is paired: identical W0 and identical seed (hence identical Gaussian stream on the
shared prefix) for every arm.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exact_anchored import (Exp10, accuracy10, geom_for, init_unit_ball10, make_potential,
                            run_exact, ETA_GRID)
from nral import build_titanic_dataset, build_magic_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe_wallclock")
os.makedirs(OUT, exist_ok=True)

R = 120                      # replicates (>= 60 required; effect sizes here are ~0.02)
SEARCH_SEED = 3000
CONFIRM_SEED = 4100
TIMING_SEED = 777
CK = 10                      # checkpoint stride; N and M are both multiples of 10

LOG_LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG_LINES.append(s)


def round10(x):
    return int(max(10, 10 * round(float(x) / 10.0)))


# --------------------------------------------------------------------------- phase 0: timing
def time_arm(pot, geom, *, alpha, scales, eta, n_iter, W0, reps=3):
    """Median per-iteration wall-clock of run_exact's inner loop (run_exact reports its own
    loop runtime, which excludes potential construction and accuracy evaluation)."""
    ts = []
    for r in range(reps):
        out = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                        W0=W0, seed=TIMING_SEED + r, checkpoint_every=10 ** 9)
        ts.append(out["runtime"] / n_iter)
    return float(np.median(ts)), ts


# --------------------------------------------------------------------------- run helpers
def traj(pot, geom, ds, *, alpha, scales, eta, n_iter, W0, seed, want_iters):
    """One run; return train/test accuracy (per replicate) at each requested iteration."""
    out = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                    W0=W0, seed=seed, checkpoint_every=CK)
    ck = list(out["checkpoints"])
    res = {}
    for it in want_iters:
        i = ck.index(it)
        Wi = out["betas"][i:i + 1]
        res[it] = dict(train=accuracy10(ds.X_train, ds.y_train, Wi)[0],
                       test=accuracy10(ds.X_test, ds.y_test, Wi)[0])
    res["_proj"] = out["projection_rate"]
    res["_runtime"] = out["runtime"]
    return res


def paired_stats(a_nrev, a_rev):
    d = np.asarray(a_nrev) - np.asarray(a_rev)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) + 1e-300
    return dict(mean_nrev=float(np.mean(a_nrev)), mean_rev=float(np.mean(a_rev)),
                diff=float(d.mean()), se=se, t=float(d.mean() / se),
                win_frac=float((d > 0).mean()), n=n)


def fmt(tag, st):
    return (f"  {tag:<34s} nrev={st['mean_nrev']:.4f}  rev={st['mean_rev']:.4f}  "
            f"d={st['diff']:+.4f}  se={st['se']:.4f}  t={st['t']:+.2f}  "
            f"win={st['win_frac']:.2f}")


# --------------------------------------------------------------------------- main
def main():
    t_start = time.perf_counter()
    n_configs = 0

    ds_t = build_titanic_dataset()
    pot_t = make_potential(ds_t)
    log(f"Titanic: n_train={ds_t.n_train} n_test={len(ds_t.y_test)} lam={pot_t.lam:g} "
        f"delta={pot_t.delta:.3e} a_lower={pot_t.a_lower_bound:.3f}")

    # the three established transient-win configurations
    CASES = [
        dict(name="titanic_ball_s5", exp=Exp10("titanic_ball", "titanic", "ball", 1500),
             N=1500, s=5.0),
        dict(name="titanic_lp_s5", exp=Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18),
             N=2000, s=5.0),
        dict(name="titanic_lp_s1", exp=Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18),
             N=2000, s=1.0),
    ]
    ETA_T = 1e-5   # the transient step size at which the win was reported

    rng0 = np.random.default_rng(SEARCH_SEED + 1)
    W0_search = init_unit_ball10(rng0, R)
    W0_confirm = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R)

    # ---------------------------------------------------------------- PHASE 0: timing
    log("\n=== PHASE 0: per-iteration wall-clock (R=%d, OMP_NUM_THREADS=%s) ===" % (
        R, os.environ.get("OMP_NUM_THREADS", "unset")))
    timing = {}
    for geo_kind, exp in (("ball", CASES[0]["exp"]), ("lp", CASES[1]["exp"])):
        geom = geom_for(exp)
        W0 = W0_search
        tr, raw_r = time_arm(pot_t, geom, alpha=0, scales=(0.0, 0.0, 0.0), eta=ETA_T,
                             n_iter=300, W0=W0)
        tn, raw_n = time_arm(pot_t, geom, alpha=1, scales=(5.0, 5.0, 5.0), eta=ETA_T,
                             n_iter=300, W0=W0)
        c = tn / tr
        timing[f"titanic_{geo_kind}"] = dict(t_rev_us=tr * 1e6, t_nrev_us=tn * 1e6, ratio=c,
                                             raw_rev_us=[x * 1e6 for x in raw_r],
                                             raw_nrev_us=[x * 1e6 for x in raw_n])
        log(f"  titanic/{geo_kind:<4s}  rev={tr*1e6:8.1f} us/iter  nrev={tn*1e6:8.1f} us/iter  "
            f"ratio c={c:.4f}")

    # how does the overhead scale with dataset size?  (MAGIC has 21x more rows)
    try:
        ds_m = build_magic_dataset()
        pot_m = make_potential(ds_m)
        geom_m = geom_for(Exp10("magic_ball", "magic", "ball", 1000))
        W0m = init_unit_ball10(np.random.default_rng(SEARCH_SEED + 1), R)
        tr, _ = time_arm(pot_m, geom_m, alpha=0, scales=(0.0, 0.0, 0.0), eta=1e-5,
                         n_iter=60, W0=W0m, reps=2)
        tn, _ = time_arm(pot_m, geom_m, alpha=1, scales=(5.0, 5.0, 5.0), eta=1e-5,
                         n_iter=60, W0=W0m, reps=2)
        timing["magic_ball"] = dict(t_rev_us=tr * 1e6, t_nrev_us=tn * 1e6, ratio=tn / tr)
        log(f"  magic/ball    rev={tr*1e6:8.1f} us/iter  nrev={tn*1e6:8.1f} us/iter  "
            f"ratio c={tn/tr:.4f}   (n_train={ds_m.n_train})")
    except Exception as exc:                                     # pragma: no cover
        log(f"  magic timing skipped: {exc}")

    # ---------------------------------------------------------------- PHASE 1: search
    log("\n=== PHASE 1: eta search on TRAIN accuracy, seed=%d ===" % SEARCH_SEED)
    search = {}
    for case in CASES:
        name, exp, N, s = case["name"], case["exp"], case["N"], case["s"]
        geo_kind = "ball" if exp.geometry == "ball" else "lp"
        c = timing[f"titanic_{geo_kind}"]["ratio"]
        M = round10(c * N)
        M = max(M, N)                       # the extra-work arm can never get FEWER iterations
        case["M"], case["c"] = M, c
        geom = geom_for(exp)
        want = sorted({N, M})
        log(f"\n  [{name}] N={N} (nrev budget)  M={M} (rev compute-matched budget, c={c:.4f})")
        rows = {}
        for eta in ETA_GRID:
            rv = traj(pot_t, geom, ds_t, alpha=0, scales=(0.0, 0.0, 0.0), eta=eta,
                      n_iter=M, W0=W0_search, seed=SEARCH_SEED, want_iters=want)
            nv = traj(pot_t, geom, ds_t, alpha=1, scales=(s, s, s), eta=eta,
                      n_iter=M, W0=W0_search, seed=SEARCH_SEED, want_iters=want)
            n_configs += 2
            rows[eta] = dict(
                rev_tr_N=float(rv[N]["train"].mean()), rev_tr_M=float(rv[M]["train"].mean()),
                nrev_tr_N=float(nv[N]["train"].mean()), nrev_tr_M=float(nv[M]["train"].mean()),
                rev_te_N=float(rv[N]["test"].mean()), rev_te_M=float(rv[M]["test"].mean()),
                nrev_te_N=float(nv[N]["test"].mean()), nrev_te_M=float(nv[M]["test"].mean()),
                proj_rev=rv["_proj"], proj_nrev=nv["_proj"])
            log(f"    eta={eta:8.1e}  TRAIN rev@N={rows[eta]['rev_tr_N']:.4f} "
                f"rev@M={rows[eta]['rev_tr_M']:.4f}  nrev@N={rows[eta]['nrev_tr_N']:.4f}")
        # selection is on TRAIN accuracy only, each arm at its own honest budget
        eta_rev_N = max(ETA_GRID, key=lambda e: rows[e]["rev_tr_N"])
        eta_rev_M = max(ETA_GRID, key=lambda e: rows[e]["rev_tr_M"])
        eta_nrev = max(ETA_GRID, key=lambda e: rows[e]["nrev_tr_N"])
        case["eta_rev_N"], case["eta_rev_M"], case["eta_nrev"] = eta_rev_N, eta_rev_M, eta_nrev
        log(f"    -> best-by-TRAIN: rev@N eta={eta_rev_N:g}, rev@M eta={eta_rev_M:g}, "
            f"nrev@N eta={eta_nrev:g}")
        search[name] = dict(rows={f"{e:g}": rows[e] for e in ETA_GRID}, N=N, M=M, c=c, s=s,
                            eta_rev_N=eta_rev_N, eta_rev_M=eta_rev_M, eta_nrev=eta_nrev)

    # ---------------------------------------------------------------- PHASE 2: confirmation
    log("\n=== PHASE 2: CONFIRMATION on independent seed %d (R=%d, paired) ===" % (
        CONFIRM_SEED, R))
    confirm = {}
    for case in CASES:
        name, exp, N, M, s = case["name"], case["exp"], case["N"], case["M"], case["s"]
        geom = geom_for(exp)
        log(f"\n  [{name}]  s={s:g}  eta_transient={ETA_T:g}  N={N}  M={M}  c={case['c']:.4f}")

        # A: the non-reversible arm at the transient config (the claimed winner)
        A = traj(pot_t, geom, ds_t, alpha=1, scales=(s, s, s), eta=ETA_T, n_iter=N,
                 W0=W0_confirm, seed=CONFIRM_SEED, want_iters=[N])[N]
        # B/C: reversible arm, SAME eta, at N (iteration-matched) and M (compute-matched)
        BC = traj(pot_t, geom, ds_t, alpha=0, scales=(0.0, 0.0, 0.0), eta=ETA_T, n_iter=M,
                  W0=W0_confirm, seed=CONFIRM_SEED, want_iters=sorted({N, M}))
        # D/E: reversible arm at ITS OWN best train-selected eta
        eN, eM = case["eta_rev_N"], case["eta_rev_M"]
        DE = traj(pot_t, geom, ds_t, alpha=0, scales=(0.0, 0.0, 0.0), eta=eN, n_iter=M,
                  W0=W0_confirm, seed=CONFIRM_SEED, want_iters=sorted({N, M}))
        if eM != eN:
            DE2 = traj(pot_t, geom, ds_t, alpha=0, scales=(0.0, 0.0, 0.0), eta=eM, n_iter=M,
                       W0=W0_confirm, seed=CONFIRM_SEED, want_iters=[M])
            revE = DE2[M]
        else:
            revE = DE[M]
        # F: non-reversible arm at ITS OWN best train-selected eta, N iters (best vs best)
        enr = case["eta_nrev"]
        if enr != ETA_T:
            F = traj(pot_t, geom, ds_t, alpha=1, scales=(s, s, s), eta=enr, n_iter=N,
                     W0=W0_confirm, seed=CONFIRM_SEED, want_iters=[N])[N]
        else:
            F = A
        n_configs += 5

        st = {}
        st["B_iter_matched_same_eta"] = paired_stats(A["test"], BC[N]["test"])
        st["C_compute_matched_same_eta"] = paired_stats(A["test"], BC[M]["test"])
        st["D_rev_own_best_eta_N"] = paired_stats(A["test"], DE[N]["test"])
        st["E_rev_own_best_eta_M"] = paired_stats(A["test"], revE["test"])
        st["F_bestVbest_compute_matched"] = paired_stats(F["test"], revE["test"])
        st_train = {
            "B_iter_matched_same_eta": paired_stats(A["train"], BC[N]["train"]),
            "C_compute_matched_same_eta": paired_stats(A["train"], BC[M]["train"]),
            "E_rev_own_best_eta_M": paired_stats(A["train"], revE["train"]),
            "F_bestVbest_compute_matched": paired_stats(F["train"], revE["train"]),
        }
        for k, v in st.items():
            log(fmt(k + " [TEST]", v))
        log(f"    rev best-by-train eta: @N={eN:g}  @M={eM:g};  nrev best-by-train eta={enr:g}")
        confirm[name] = dict(test=st, train=st_train, N=N, M=M, c=case["c"], s=s,
                             eta_transient=ETA_T, eta_rev_N=eN, eta_rev_M=eM, eta_nrev=enr)

    elapsed = time.perf_counter() - t_start
    log(f"\nTotal wall-clock: {elapsed:.1f} s;  configurations run: {n_configs}")

    with open(os.path.join(OUT, "wallclock_results.json"), "w") as f:
        json.dump(dict(timing=timing, search=search, confirm=confirm, R=R,
                       search_seed=SEARCH_SEED, confirm_seed=CONFIRM_SEED,
                       n_configs=n_configs, elapsed_s=elapsed), f, indent=2)
    with open(os.path.join(OUT, "wallclock_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")




# =========================================================================== ADDENDUM
def addendum():
    """Two things the headline number depends on, measured rather than assumed.

    (1) The cost ratio c is implementation-specific.  Sweep R (the vectorisation width) to
        bound how large c can get.
    (2) The break-even overhead: how many iterations does the REVERSIBLE arm need at the SAME
        eta to reach the non-reversible arm's accuracy at N?  Call it N_eq.  The transient win
        survives compute-matching for any cost ratio c < N_eq/N, so N_eq/N is the overhead
        budget the acceleration actually buys.  N_eq is located on TRAIN accuracy (no test
        labels), then the test accuracy at that same iteration is reported.
    """
    import numpy as np
    t0 = time.perf_counter()
    ds_t = build_titanic_dataset()
    pot_t = make_potential(ds_t)
    W0 = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R)
    out = {}

    log("\n=== ADDENDUM 1: cost ratio c vs vectorisation width R (titanic, ball) ===")
    geom = geom_for(Exp10("titanic_ball", "titanic", "ball", 1500))
    cR = {}
    for Rw in (1, 10, 120, 600):
        W = init_unit_ball10(np.random.default_rng(99), Rw)
        ni = 2000 if Rw <= 10 else (300 if Rw == 120 else 120)
        tr, _ = time_arm(pot_t, geom, alpha=0, scales=(0.0, 0.0, 0.0), eta=1e-5,
                         n_iter=ni, W0=W, reps=3)
        tn, _ = time_arm(pot_t, geom, alpha=1, scales=(5.0, 5.0, 5.0), eta=1e-5,
                         n_iter=ni, W0=W, reps=3)
        cR[Rw] = dict(t_rev_us=tr * 1e6, t_nrev_us=tn * 1e6, ratio=tn / tr)
        log(f"  R={Rw:<5d} rev={tr*1e6:9.1f} us/iter  nrev={tn*1e6:9.1f} us/iter  c={tn/tr:.4f}")
    out["cost_ratio_vs_R"] = {str(k): v for k, v in cR.items()}

    log("\n=== ADDENDUM 2: break-even overhead budget N_eq/N (eta=1e-5, seed=%d) ===" % CONFIRM_SEED)
    for name, exp, N, s in (("titanic_ball_s5", Exp10("titanic_ball", "titanic", "ball", 1500), 1500, 5.0),
                            ("titanic_lp_s5", Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18), 2000, 5.0),
                            ("titanic_lp_s1", Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18), 2000, 1.0)):
        g = geom_for(exp)
        LONG = 6 * N
        nv = traj(pot_t, g, ds_t, alpha=1, scales=(s, s, s), eta=1e-5, n_iter=N,
                  W0=W0, seed=CONFIRM_SEED, want_iters=[N])[N]
        target_tr = float(nv["train"].mean())
        r = run_exact(pot_t, g, alpha=0, scales=(0.0, 0.0, 0.0), eta=1e-5, n_iter=LONG,
                      W0=W0, seed=CONFIRM_SEED, checkpoint_every=CK)
        ck = np.asarray(r["checkpoints"])
        acc_tr = accuracy10(ds_t.X_train, ds_t.y_train, r["betas"]).mean(axis=1)
        hit = np.nonzero(acc_tr >= target_tr)[0]
        if len(hit) == 0:
            log(f"  [{name}] rev never reaches nrev's TRAIN accuracy {target_tr:.4f} within "
                f"{LONG} iters (>= {LONG/N:.1f}x budget)")
            out[name] = dict(N=N, N_eq=None, ratio=None, target_train=target_tr, LONG=LONG)
            continue
        N_eq = int(ck[hit[0]])
        acc_te_eq = accuracy10(ds_t.X_test, ds_t.y_test, r["betas"][hit[0]:hit[0] + 1])[0]
        st = paired_stats(nv["test"], acc_te_eq)
        log(f"  [{name}] nrev TRAIN@N={target_tr:.4f}; rev reaches it at N_eq={N_eq} "
            f"({N_eq/N:.2f}x) -> overhead budget c < {N_eq/N:.2f} (measured c ~ 1.13)")
        log(fmt("    TEST at N_eq (rev) vs N (nrev)", st))
        out[name] = dict(N=N, N_eq=N_eq, ratio=N_eq / N, target_train=target_tr,
                         test_at_Neq=st, LONG=LONG)

    log(f"\nAddendum wall-clock: {time.perf_counter()-t0:.1f} s")
    with open(os.path.join(OUT, "wallclock_addendum.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT, "wallclock_addendum_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")


# =========================================================================== ADDENDUM 3
def addendum3():
    """The decisive test.  ADDENDUM 1 showed the cost ratio c depends strongly on the
    vectorisation width R: c = 2.77 at R=1 (one chain, the natural MCMC setting) but 1.01 at
    R=600.  ADDENDUM 2 showed the non-reversible acceleration is worth only 1.15x-1.28x in
    ITERATIONS.  So compute-matching at R=1 should REVERSE the win.  Measure it: give the
    reversible arm c_R1 * N iterations at the same eta and compare, paired, to the
    non-reversible arm at N.  (The accuracy comparison itself is still run with R=120
    replicates -- R here only sets the COST MODEL, not the statistics.)"""
    import numpy as np
    t0 = time.perf_counter()
    ds_t = build_titanic_dataset()
    pot_t = make_potential(ds_t)
    W0 = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R)
    C_BY_R = {1: 2.7673, 10: 1.8107, 120: 1.1168, 600: 1.0124}   # measured in ADDENDUM 1
    out = {}
    log("\n=== ADDENDUM 3: compute-matched at each cost model c(R), eta=1e-5, seed=%d ===" % CONFIRM_SEED)
    for name, exp, N, s_ in (("titanic_ball_s5", Exp10("titanic_ball", "titanic", "ball", 1500), 1500, 5.0),
                             ("titanic_lp_s5", Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18), 2000, 5.0),
                             ("titanic_lp_s1", Exp10("titanic_lp", "titanic", "lp", 2000, eps=0.18), 2000, 1.0)):
        g = geom_for(exp)
        nv = traj(pot_t, g, ds_t, alpha=1, scales=(s_, s_, s_), eta=1e-5, n_iter=N,
                  W0=W0, seed=CONFIRM_SEED, want_iters=[N])[N]
        want = sorted({round10(c * N) for c in C_BY_R.values()})
        rv = traj(pot_t, g, ds_t, alpha=0, scales=(0.0, 0.0, 0.0), eta=1e-5, n_iter=max(want),
                  W0=W0, seed=CONFIRM_SEED, want_iters=want)
        log(f"  [{name}] N={N}")
        rec = {}
        for Rw, c in sorted(C_BY_R.items()):
            M = round10(c * N)
            st = paired_stats(nv["test"], rv[M]["test"])
            sttr = paired_stats(nv["train"], rv[M]["train"])
            rec[str(Rw)] = dict(c=c, M=M, test=st, train=sttr)
            log(f"    R={Rw:<4d} c={c:.3f}  rev gets M={M:<5d}  "
                f"TEST nrev={st['mean_nrev']:.4f} rev={st['mean_rev']:.4f} "
                f"d={st['diff']:+.4f} se={st['se']:.4f} t={st['t']:+.2f}   "
                f"TRAIN d={sttr['diff']:+.4f} t={sttr['t']:+.2f}")
        out[name] = rec
    log(f"\nAddendum3 wall-clock: {time.perf_counter()-t0:.1f} s")
    with open(os.path.join(OUT, "wallclock_addendum3.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT, "wallclock_addendum3_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    if "--addendum3" in sys.argv:
        addendum3()
    elif "--addendum" in sys.argv:
        addendum()
    else:
        main()
