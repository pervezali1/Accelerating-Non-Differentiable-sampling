"""probe_mixing.py -- HYPOTHESIS `mixing`: does the NON-REVERSIBLE anchored Langevin sampler
mix faster than the REVERSIBLE one AT STATIONARITY, measured by integrated autocorrelation
time (IACT) and effective sample size (ESS)?

Design (all selection is on TRAIN-only / parameter-only quantities; test LABELS are never used):

  Stage 0  GOLD reference.  A long reversible run at a very small step size gives reference
           means and sds for every test function.  This is the yardstick for DISCRETIZATION
           BIAS.  It is a reversible run, which is fair: the continuous process has the same
           invariant law for both arms, so the eta -> 0 limit is common to both.

  Stage 1  SEARCH (seed SEARCH_SEED).  A grid over (eta, s).  For every cell measure
             bias(cell)  = max_f | mean_f(cell) - mean_f(gold) | / sd_f(gold)
             ESS/sec(cell) for every test function f
           Each ARM (reversible / non-reversible) then gets its OWN best configuration:
             argmax ESS/sec  subject to  bias <= B*        (B* fixed a priori, several shown)
           This is the "best vs best" fairness requirement: the reversible arm is free to use
           a larger step size if that is what maximises its ESS/sec at the same bias level.

  Stage 2  CONFIRM (seed CONFIRM_SEED, independent of the search).  The two selected
           configurations are re-run with more chains and more iterations, PAIRED: identical
           W0 and identical Gaussian noise stream.  Per-chain IACT -> per-chain ESS -> paired
           differences, paired SE, t statistic.

  Stage 3  Cross-geometry replication on the smoothed l_p set.

Conventions.  IACT is the Sokal integrated autocorrelation time
    tau_int = 1/2 + sum_{k>=1} rho_k,     ESS = N / (2 tau_int),
estimated by Geyer's initial-positive / initial-monotone sequence on each chain separately.
Series are stored every CK iterations; tau_int is reported in ITERATIONS (tau_series * CK).
ESS is invariant to CK as long as CK << IACT, which is checked and reported.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "probe_mixing")
os.makedirs(OUT, exist_ok=True)

from exact_anchored import (make_potential, make_geom10, run_exact, init_unit_ball10,  # noqa: E402
                            accuracy10, D10)
from nral import build_titanic_dataset                                                # noqa: E402

# ----------------------------------------------------------------------------- knobs
SEARCH_SEED = 3000
CONFIRM_SEED = 4100          # independent of SEARCH_SEED, used only for the headline
GOLD_SEED = 2600

CK = 2                       # store the state every CK iterations
BURN_FRAC = 0.40             # discard this fraction of the run as burn-in
THIN_EXPENSIVE = 4           # extra thinning for the likelihood-based test functions

R_GOLD, NIT_GOLD, CK_GOLD = 128, 40000, 20
R_SEARCH, NIT_SEARCH = 48, 10000
R_CONF, NIT_CONF = 96, 16000
R_LP, NIT_LP = 32, 6000

BSTARS = (0.10, 0.25, 0.50)  # bias thresholds (in gold sd units); 0.25 is the headline one
B_HEADLINE = 0.25

ETAS_SEARCH = (1e-4, 3e-4, 1e-3)
S_SEARCH = (1.0, 2.0, 5.0)

log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_lines.append(s)


# ----------------------------------------------------------------------------- IACT
def iact_geyer(x: np.ndarray) -> float:
    """Sokal IACT tau_int = 1/2 + sum_{k>=1} rho_k, via Geyer's initial positive+monotone
    sequence.  Returns tau_int in units of the sampling interval of `x`."""
    x = np.asarray(x, dtype=float)
    N = x.size
    x = x - x.mean()
    if not np.isfinite(x).all():
        return np.nan
    n2 = 1
    while n2 < 2 * N:
        n2 *= 2
    f = np.fft.rfft(x, n2)
    ac = np.fft.irfft(f * np.conj(f), n2)[:N].real
    if ac[0] <= 0:
        return np.nan
    ac = ac / ac[0]
    M = N // 2
    G = ac[0:2 * M:2] + ac[1:2 * M:2]          # Gamma_m = rho_{2m} + rho_{2m+1} >= 0
    k = 0
    while k < G.size and G[k] > 0:
        k += 1
    G = G[:k]
    if G.size == 0:
        return 0.5
    G = np.minimum.accumulate(G)               # initial monotone sequence
    tau2 = -1.0 + 2.0 * float(G.sum())         # = 1 + 2 sum_{k>=1} rho_k = 2 tau_int
    return max(0.5 * tau2, 0.5)


def split_rhat(S: np.ndarray) -> float:
    """Standard split-Rhat on S of shape (T, R)."""
    T, R = S.shape
    h = T // 2
    C = np.concatenate([S[:h], S[h:2 * h]], axis=1)     # (h, 2R)
    m = C.mean(axis=0)
    v = C.var(axis=0, ddof=1)
    W = v.mean()
    B = h * m.var(ddof=1)
    if W <= 0:
        return np.nan
    var_hat = (h - 1) / h * W + B / h
    return float(np.sqrt(var_hat / W))


# ----------------------------------------------------------------------------- test functions
def test_functions(B: np.ndarray, pot, X_hold: np.ndarray):
    """B: (T, R, 10) retained states.  Returns {name: (T, R)} plus {name: thinning factor}.

    All of w1..w9, ||w||^2, loglik_train, U_train use TRAINING data / parameters only.
    The `zhold*` functions use held-out FEATURE rows; test LABELS are never touched."""
    T, R, _ = B.shape
    tf, thin = {}, {}
    for j in range(1, D10):
        tf[f"w{j}"] = B[:, :, j].copy()
        thin[f"w{j}"] = 1
    tf["norm2"] = np.sum(B * B, axis=2)
    thin["norm2"] = 1
    for i, row in enumerate(X_hold):
        xi = np.concatenate([[1.0], row])
        tf[f"zhold{i}"] = B @ xi
        thin[f"zhold{i}"] = 1
    # likelihood-based: thinned, chunked (the (T*R, n_train) product is the expensive part)
    Bt = B[::THIN_EXPENSIVE]
    Tt = Bt.shape[0]
    ll = np.empty((Tt, R))
    uu = np.empty((Tt, R))
    step = max(1, 40000 // max(R, 1))
    for a in range(0, Tt, step):
        blk = Bt[a:a + step]
        flat = blk.reshape(-1, D10)
        ll[a:a + step] = pot.loglik_part(flat).reshape(blk.shape[0], R)
        uu[a:a + step] = pot.U(flat).reshape(blk.shape[0], R)
    tf["loglik_train"] = ll
    thin["loglik_train"] = THIN_EXPENSIVE
    tf["U_train"] = uu
    thin["U_train"] = THIN_EXPENSIVE
    return tf, thin


TRAIN_ONLY = [f"w{j}" for j in range(1, D10)] + ["norm2", "loglik_train", "U_train"]


def summarise(B: np.ndarray, pot, X_hold, ck: int, per_chain: bool = True):
    """Per-test-function: mean, sd, per-chain IACT (iterations), ESS, split-Rhat."""
    tf, thin = test_functions(B, pot, X_hold)
    n_kept_iter = (B.shape[0] - 1) * ck
    res = {}
    for name, S in tf.items():
        dt = ck * thin[name]                       # iterations between stored samples
        T, R = S.shape
        taus = np.array([iact_geyer(S[:, i]) for i in range(R)])
        ess = (T - 1) / (2.0 * taus)               # ESS of THIS series (dimensionless)
        res[name] = dict(
            mean=float(S.mean()), sd=float(S.std(ddof=1)),
            tau_iter=taus * dt,                    # per-chain IACT in iterations
            ess=ess,                               # per-chain ESS over the retained segment
            rhat=split_rhat(S), dt=dt, n_kept_iter=n_kept_iter,
        )
    return res


def run_arm(pot, geom, *, alpha, s, eta, n_iter, R, seed, ck, W0):
    t0 = time.perf_counter()
    r = run_exact(pot, geom, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=n_iter,
                  W0=W0, seed=seed, checkpoint_every=ck)
    wall = time.perf_counter() - t0
    B = r["betas"]
    i0 = int(round(BURN_FRAC * (B.shape[0] - 1)))
    return dict(B=B[i0:], wall=wall, sampler_time=r["runtime"], proj=r["projection_rate"],
                n_nonfinite=r["n_nonfinite"], kept_frac=1.0 - BURN_FRAC, n_iter=n_iter, ck=ck)


# ----------------------------------------------------------------------------- main
def main():
    t_start = time.perf_counter()
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    X_hold = ds.X_test[:3]                      # held-out FEATURES only (no labels)
    log(f"# dataset titanic n_train={ds.n_train} n_test={ds.n_test} "
        f"lam={pot.lam:g} delta={pot.delta:.5g} a_lower={pot.a_lower_bound:.3f}")

    geom_ball = make_geom10("ball")
    n_cfg = 0

    # ---------------------------------------------------------------- Stage 0: GOLD
    log("\n=== stage 0: gold reference (reversible, eta=3e-5, long) ===")
    W0g = init_unit_ball10(np.random.default_rng(GOLD_SEED + 1), R_GOLD)
    g = run_arm(pot, geom_ball, alpha=0, s=0.0, eta=3e-5, n_iter=NIT_GOLD, R=R_GOLD,
                seed=GOLD_SEED, ck=CK_GOLD, W0=W0g)
    n_cfg += 1
    gold = summarise(g["B"], pot, X_hold, CK_GOLD)
    gold_mean = {k: v["mean"] for k, v in gold.items()}
    gold_sd = {k: v["sd"] for k, v in gold.items()}
    log(f"gold wall={g['wall']:.1f}s proj={g['proj']:.3f} "
        f"maxRhat={max(v['rhat'] for v in gold.values()):.4f}")
    log("gold means : " + " ".join(f"{k}={gold_mean[k]:+.4f}" for k in TRAIN_ONLY[:10]))
    log("gold sds   : " + " ".join(f"{k}={gold_sd[k]:.4f}" for k in TRAIN_ONLY[:10]))
    log(f"gold ESS(total, w5) = {gold['w5']['ess'].sum():.0f}  "
        f"-> MC se of reference mean ~ {1.0/np.sqrt(gold['w5']['ess'].sum()):.4f} sd")

    def bias_of(res):
        return max(abs(res[k]["mean"] - gold_mean[k]) / max(gold_sd[k], 1e-12)
                   for k in TRAIN_ONLY)

    # ---------------------------------------------------------------- Stage 1: SEARCH
    log(f"\n=== stage 1: search grid, seed={SEARCH_SEED}, R={R_SEARCH}, "
        f"n_iter={NIT_SEARCH} ===")
    W0s = init_unit_ball10(np.random.default_rng(SEARCH_SEED + 1), R_SEARCH)
    cells = [("rev", eta, 0.0) for eta in ETAS_SEARCH] + \
            [("nrev", eta, s) for eta in ETAS_SEARCH for s in S_SEARCH]
    grid = {}
    log(f"{'arm':>5} {'eta':>8} {'s':>5} {'wall':>6} {'proj':>6} {'bias':>6} "
        f"{'maxRhat':>8} {'ESS/s(w5)':>10} {'ESS/s(slow)':>12} {'tau_slow':>9}")
    for arm, eta, s in cells:
        a = run_arm(pot, geom_ball, alpha=(0 if arm == "rev" else 1), s=s, eta=eta,
                    n_iter=NIT_SEARCH, R=R_SEARCH, seed=SEARCH_SEED, ck=CK, W0=W0s)
        n_cfg += 1
        res = summarise(a["B"], pot, X_hold, CK)
        b = bias_of(res)
        ess_ps = {k: float(v["ess"].sum()) / a["wall"] for k, v in res.items()}
        slow = min(TRAIN_ONLY, key=lambda k: ess_ps[k])        # bottleneck test function
        grid[(arm, eta, s)] = dict(bias=b, ess_ps=ess_ps, res=res, wall=a["wall"],
                                   proj=a["proj"], nonfinite=a["n_nonfinite"],
                                   maxrhat=max(v["rhat"] for v in res.values()), slow=slow)
        log(f"{arm:>5} {eta:8.1e} {s:5g} {a['wall']:6.1f} {a['proj']:6.3f} {b:6.3f} "
            f"{grid[(arm, eta, s)]['maxrhat']:8.4f} {ess_ps['w5']:10.1f} "
            f"{ess_ps[slow]:12.1f} {slow:>9}")
        del a

    # the a-priori primary test function: the bottleneck of the REVERSIBLE arm
    rev_cells = [(k, v) for k, v in grid.items() if k[0] == "rev"]
    bottleneck_votes = {}
    for k, v in rev_cells:
        bottleneck_votes[v["slow"]] = bottleneck_votes.get(v["slow"], 0) + 1
    PRIMARY = max(bottleneck_votes, key=bottleneck_votes.get)
    log(f"\nprimary (pre-registered = reversible-arm bottleneck on the search seed): "
        f"{PRIMARY}   votes={bottleneck_votes}")

    # best config per arm per bias threshold, scored on ESS/sec of the PRIMARY function
    picks = {}
    for B_ in BSTARS:
        row = {}
        for arm in ("rev", "nrev"):
            ok = [k for k, v in grid.items()
                  if k[0] == arm and v["bias"] <= B_ and v["nonfinite"] == 0
                  and v["maxrhat"] < 1.08]
            row[arm] = max(ok, key=lambda k: grid[k]["ess_ps"][PRIMARY]) if ok else None
        picks[B_] = row
        log(f"B*={B_:.2f}  rev-best={row['rev']}  nrev-best={row['nrev']}")
    sel_rev, sel_nrev = picks[B_HEADLINE]["rev"], picks[B_HEADLINE]["nrev"]
    if sel_rev is None or sel_nrev is None:
        log("!! no admissible configuration at the headline threshold; aborting")
        return
    log(f"\nheadline selection (B*={B_HEADLINE}): rev {sel_rev} vs nrev {sel_nrev}")

    # ---------------------------------------------------------------- Stage 2: CONFIRM
    log(f"\n=== stage 2: confirmation, INDEPENDENT seed={CONFIRM_SEED}, R={R_CONF}, "
        f"n_iter={NIT_CONF} ===")
    W0c = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R_CONF)
    conf = {}
    for tag, key in (("rev", sel_rev), ("nrev", sel_nrev)):
        _, eta, s = key
        a = run_arm(pot, geom_ball, alpha=(0 if tag == "rev" else 1), s=s, eta=eta,
                    n_iter=NIT_CONF, R=R_CONF, seed=CONFIRM_SEED, ck=CK, W0=W0c)
        n_cfg += 1
        res = summarise(a["B"], pot, X_hold, CK)
        acc_tr = accuracy10(ds.X_train, ds.y_train, a["B"][::50]).mean()
        conf[tag] = dict(res=res, wall=a["wall"], proj=a["proj"], eta=eta, s=s,
                         bias=bias_of(res), acc_train=float(acc_tr),
                         maxrhat=max(v["rhat"] for v in res.values()),
                         n_kept_iter=(a["B"].shape[0] - 1) * CK)
        log(f"{tag}: eta={eta:g} s={s:g} wall={a['wall']:.1f}s proj={a['proj']:.3f} "
            f"bias={conf[tag]['bias']:.3f} maxRhat={conf[tag]['maxrhat']:.4f} "
            f"acc_train={acc_tr:.4f}")
        del a

    # clean per-iteration cost of each arm, measured WITHOUT checkpoint storage
    cost = {}
    for tag, key in (("rev", sel_rev), ("nrev", sel_nrev)):
        _, eta, s = key
        t0 = time.perf_counter()
        run_exact(pot, geom_ball, alpha=(0 if tag == "rev" else 1), scales=(s, s, s),
                  eta=eta, n_iter=3000, W0=W0c, seed=1, checkpoint_every=3000)
        cost[tag] = (time.perf_counter() - t0) / 3000.0
    log(f"clean per-iteration cost (R={R_CONF}): rev={cost['rev']*1e3:.3f} ms  "
        f"nrev={cost['nrev']*1e3:.3f} ms   ratio nrev/rev = {cost['nrev']/cost['rev']:.3f}")

    # ---- paired statistics, per test function
    log(f"\n--- paired per-chain results (R={R_CONF} coupled chains, same W0, same noise) ---")
    log(f"{'fn':>13} {'tau_rev':>9} {'tau_nrev':>9} {'ratio':>7} | "
        f"{'ESSps_rev':>10} {'ESSps_nrev':>11} {'d':>9} {'se':>8} {'t':>7} {'win%':>6}")
    table = {}
    for name in list(conf["rev"]["res"].keys()):
        tr, tn = conf["rev"]["res"][name], conf["nrev"]["res"][name]
        # ESS per second, per chain (same wall clock constant within an arm)
        er = tr["ess"] / conf["rev"]["wall"]
        en = tn["ess"] / conf["nrev"]["wall"]
        d = en - er
        n = d.size
        se = d.std(ddof=1) / np.sqrt(n)
        t = d.mean() / se if se > 0 else np.nan
        table[name] = dict(
            tau_rev=float(np.nanmean(tr["tau_iter"])), tau_nrev=float(np.nanmean(tn["tau_iter"])),
            essps_rev=float(er.mean()), essps_nrev=float(en.mean()),
            d=float(d.mean()), se=float(se), t=float(t), win=float((d > 0).mean()),
            ess_rev=float(tr["ess"].mean()), ess_nrev=float(tn["ess"].mean()),
            mean_rev=tr["mean"], mean_nrev=tn["mean"], dt_rev=tr["dt"], dt_nrev=tn["dt"])
        e = table[name]
        log(f"{name:>13} {e['tau_rev']:9.1f} {e['tau_nrev']:9.1f} "
            f"{e['tau_nrev']/max(e['tau_rev'],1e-9):7.3f} | {e['essps_rev']:10.2f} "
            f"{e['essps_nrev']:11.2f} {e['d']:9.2f} {e['se']:8.2f} {e['t']:7.2f} "
            f"{100*e['win']:6.1f}")

    p = table[PRIMARY]
    log(f"\nHEADLINE  metric = ESS per second of wall clock, test function '{PRIMARY}', "
        f"confirmation seed {CONFIRM_SEED}")
    log(f"  reversible  {sel_rev}: {p['essps_rev']:.3f} ESS/s   (IACT {p['tau_rev']:.1f} it)")
    log(f"  non-rev     {sel_nrev}: {p['essps_nrev']:.3f} ESS/s   (IACT {p['tau_nrev']:.1f} it)")
    log(f"  paired d={p['d']:+.3f}  se={p['se']:.3f}  t={p['t']:.2f}  "
        f"win rate {100*p['win']:.1f}%  speed-up x{p['essps_nrev']/p['essps_rev']:.2f}")

    # aggregate over the train-only functions
    tot_r = sum(table[k]["essps_rev"] for k in TRAIN_ONLY)
    tot_n = sum(table[k]["essps_nrev"] for k in TRAIN_ONLY)
    worst_r = min(table[k]["essps_rev"] for k in TRAIN_ONLY)
    worst_n = min(table[k]["essps_nrev"] for k in TRAIN_ONLY)
    log(f"  sum over 12 train-only fns: rev {tot_r:.1f} -> nrev {tot_n:.1f} "
        f"(x{tot_n/tot_r:.2f});  worst-case fn: {worst_r:.2f} -> {worst_n:.2f} "
        f"(x{worst_n/worst_r:.2f})")
    log(f"  CK={CK} vs smallest IACT seen = "
        f"{min(min(table[k]['tau_rev'], table[k]['tau_nrev']) for k in TRAIN_ONLY):.1f} "
        f"iterations (CK must be << IACT)")

    # ---------------------------------------------------------------- Stage 3: l_p geometry
    log(f"\n=== stage 3: cross-geometry replication, smoothed l_p, seed={CONFIRM_SEED} ===")
    geom_lp = make_geom10("lp", eps=0.18)
    W0l = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R_LP)
    lp = {}
    for tag, key in (("rev", sel_rev), ("nrev", sel_nrev)):
        _, eta, s = key
        a = run_arm(pot, geom_lp, alpha=(0 if tag == "rev" else 1), s=s, eta=eta,
                    n_iter=NIT_LP, R=R_LP, seed=CONFIRM_SEED, ck=CK, W0=W0l)
        n_cfg += 1
        lp[tag] = dict(res=summarise(a["B"], pot, X_hold, CK), wall=a["wall"], proj=a["proj"],
                       eta=eta, s=s)
        del a
    lp_tab = {}
    for name in TRAIN_ONLY:
        er = lp["rev"]["res"][name]["ess"] / lp["rev"]["wall"]
        en = lp["nrev"]["res"][name]["ess"] / lp["nrev"]["wall"]
        d = en - er
        se = d.std(ddof=1) / np.sqrt(d.size)
        lp_tab[name] = dict(tau_rev=float(np.nanmean(lp["rev"]["res"][name]["tau_iter"])),
                            tau_nrev=float(np.nanmean(lp["nrev"]["res"][name]["tau_iter"])),
                            essps_rev=float(er.mean()), essps_nrev=float(en.mean()),
                            d=float(d.mean()), se=float(se),
                            t=float(d.mean() / se) if se > 0 else np.nan)
    log(f"{'fn':>13} {'tau_rev':>9} {'tau_nrev':>9} {'ESSps_rev':>10} {'ESSps_nrev':>11} "
        f"{'t':>7}")
    for name in TRAIN_ONLY:
        e = lp_tab[name]
        log(f"{name:>13} {e['tau_rev']:9.1f} {e['tau_nrev']:9.1f} {e['essps_rev']:10.2f} "
            f"{e['essps_nrev']:11.2f} {e['t']:7.2f}")
    e = lp_tab[PRIMARY]
    log(f"l_p, primary '{PRIMARY}': x{e['essps_nrev']/max(e['essps_rev'],1e-12):.2f} ESS/s, "
        f"t={e['t']:.2f}")

    # ---------------------------------------------------------------- dump
    log(f"\nconfigurations run by this script: {n_cfg}")
    log(f"total wall clock: {time.perf_counter()-t_start:.1f}s")
    out = dict(
        primary=PRIMARY, selection=dict(rev=list(sel_rev), nrev=list(sel_nrev)),
        picks={str(k): {a: (list(v) if v else None) for a, v in row.items()}
               for k, row in picks.items()},
        grid={f"{k[0]}|eta={k[1]:g}|s={k[2]:g}": dict(
            bias=v["bias"], proj=v["proj"], wall=v["wall"], maxrhat=v["maxrhat"],
            slow=v["slow"], ess_ps={kk: vv for kk, vv in v["ess_ps"].items()})
            for k, v in grid.items()},
        gold=dict(mean=gold_mean, sd=gold_sd, eta=3e-5, R=R_GOLD, n_iter=NIT_GOLD),
        confirm=dict(table=table,
                     rev=dict(eta=conf["rev"]["eta"], s=conf["rev"]["s"],
                              wall=conf["rev"]["wall"], bias=conf["rev"]["bias"],
                              proj=conf["rev"]["proj"], acc_train=conf["rev"]["acc_train"],
                              maxrhat=conf["rev"]["maxrhat"]),
                     nrev=dict(eta=conf["nrev"]["eta"], s=conf["nrev"]["s"],
                               wall=conf["nrev"]["wall"], bias=conf["nrev"]["bias"],
                               proj=conf["nrev"]["proj"], acc_train=conf["nrev"]["acc_train"],
                               maxrhat=conf["nrev"]["maxrhat"]),
                     cost_per_iter=cost, R=R_CONF, n_iter=NIT_CONF, seed=CONFIRM_SEED),
        lp=dict(table=lp_tab, R=R_LP, n_iter=NIT_LP),
        n_configs_this_script=n_cfg,
    )
    with open(os.path.join(OUT, "mixing_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(OUT, "mixing_log.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    log(f"wrote {OUT}/mixing_results.json and mixing_log.txt")




# =============================================================================== addendum
def addendum():
    """POST-HOC extension of the search grid in s only, at the step size the main protocol
    already selected (eta = 1e-4).  The main grid stopped at s = 5; theory says the usable
    strength ceiling scales roughly like 1/eta, so at the SMALLEST admissible eta the ceiling
    should be well above 5.  Selection is still done on the SEARCH seed and on the same
    train-only bias criterion; the number quoted still comes from the CONFIRM seed.  This is
    reported as EXPLORATORY because the grid was widened after seeing the first grid.
    """
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    X_hold = ds.X_test[:3]
    geom = make_geom10("ball")
    with open(os.path.join(OUT, "mixing_results.json")) as fh:
        prev = json.load(fh)
    gold_mean, gold_sd = prev["gold"]["mean"], prev["gold"]["sd"]
    PRIMARY = prev["primary"]
    eta = prev["selection"]["rev"][1]

    def bias_of(res):
        return max(abs(res[k]["mean"] - gold_mean[k]) / max(gold_sd[k], 1e-12)
                   for k in TRAIN_ONLY)

    log("\n=== ADDENDUM (exploratory): wider s grid at the selected eta ===")
    W0s = init_unit_ball10(np.random.default_rng(SEARCH_SEED + 1), R_SEARCH)
    S_EXT = (8.0, 12.0, 20.0, 35.0)
    log(f"{'arm':>5} {'eta':>8} {'s':>5} {'wall':>6} {'proj':>6} {'bias':>6} {'maxRhat':>8} "
        f"{'ESS/s(prim)':>12}")
    best, best_val = None, -1.0
    for s in S_EXT:
        a = run_arm(pot, geom, alpha=1, s=s, eta=eta, n_iter=NIT_SEARCH, R=R_SEARCH,
                    seed=SEARCH_SEED, ck=CK, W0=W0s)
        res = summarise(a["B"], pot, X_hold, CK)
        b = bias_of(res)
        v = float(res[PRIMARY]["ess"].sum()) / a["wall"]
        mr = max(x["rhat"] for x in res.values())
        log(f"{'nrev':>5} {eta:8.1e} {s:5g} {a['wall']:6.1f} {a['proj']:6.3f} {b:6.3f} "
            f"{mr:8.4f} {v:12.1f}")
        if b <= B_HEADLINE and a["n_nonfinite"] == 0 and mr < 1.08 and v > best_val:
            best, best_val = s, v
        del a
    log(f"admissible-best s at eta={eta:g} on the search seed: s={best}")
    if best is None:
        return
    log(f"\n--- confirmation of the wider-s pick on seed {CONFIRM_SEED} (R={R_CONF}, "
        f"n_iter={NIT_CONF}) ---")
    W0c = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), R_CONF)
    out = {}
    for tag, alpha, s in (("rev", 0, 0.0), ("nrev", 1, best)):
        a = run_arm(pot, geom, alpha=alpha, s=s, eta=eta, n_iter=NIT_CONF, R=R_CONF,
                    seed=CONFIRM_SEED, ck=CK, W0=W0c)
        out[tag] = dict(res=summarise(a["B"], pot, X_hold, CK), wall=a["wall"],
                        proj=a["proj"], s=s,
                        acc_train=float(accuracy10(ds.X_train, ds.y_train,
                                                   a["B"][::50]).mean()))
        out[tag]["bias"] = bias_of(out[tag]["res"])
        log(f"{tag}: s={s:g} wall={a['wall']:.1f}s proj={a['proj']:.3f} "
            f"bias={out[tag]['bias']:.3f} acc_train={out[tag]['acc_train']:.4f}")
        del a
    log(f"{'fn':>13} {'tau_rev':>9} {'tau_nrev':>9} {'ESSps_rev':>10} {'ESSps_nrev':>11} "
        f"{'d':>9} {'se':>8} {'t':>7} {'x':>6}")
    tab = {}
    for name in TRAIN_ONLY:
        er = out["rev"]["res"][name]["ess"] / out["rev"]["wall"]
        en = out["nrev"]["res"][name]["ess"] / out["nrev"]["wall"]
        d = en - er
        se = d.std(ddof=1) / np.sqrt(d.size)
        tab[name] = dict(tau_rev=float(np.nanmean(out["rev"]["res"][name]["tau_iter"])),
                         tau_nrev=float(np.nanmean(out["nrev"]["res"][name]["tau_iter"])),
                         essps_rev=float(er.mean()), essps_nrev=float(en.mean()),
                         d=float(d.mean()), se=float(se), t=float(d.mean() / se),
                         win=float((d > 0).mean()))
        e = tab[name]
        log(f"{name:>13} {e['tau_rev']:9.1f} {e['tau_nrev']:9.1f} {e['essps_rev']:10.2f} "
            f"{e['essps_nrev']:11.2f} {e['d']:9.2f} {e['se']:8.2f} {e['t']:7.2f} "
            f"{e['essps_nrev']/max(e['essps_rev'],1e-12):6.2f}")
    e = tab[PRIMARY]
    log(f"\nADDENDUM HEADLINE ({PRIMARY}, ESS/s): rev {e['essps_rev']:.2f} -> nrev "
        f"{e['essps_nrev']:.2f}  (x{e['essps_nrev']/e['essps_rev']:.2f})  "
        f"d={e['d']:+.2f} se={e['se']:.2f} t={e['t']:.2f} win={100*e['win']:.1f}%")
    wr = min(tab[k]["essps_rev"] for k in TRAIN_ONLY)
    wn = min(tab[k]["essps_nrev"] for k in TRAIN_ONLY)
    log(f"worst-case test function: {wr:.2f} -> {wn:.2f} (x{wn/wr:.2f})")
    with open(os.path.join(OUT, "mixing_addendum.json"), "w") as fh:
        json.dump(dict(eta=eta, s=best, primary=PRIMARY, table=tab,
                       rev=dict(wall=out["rev"]["wall"], bias=out["rev"]["bias"],
                                proj=out["rev"]["proj"], acc_train=out["rev"]["acc_train"]),
                       nrev=dict(wall=out["nrev"]["wall"], bias=out["nrev"]["bias"],
                                 proj=out["nrev"]["proj"], acc_train=out["nrev"]["acc_train"]),
                       S_EXT=list(S_EXT), R=R_CONF, n_iter=NIT_CONF,
                       confirm_seed=CONFIRM_SEED), fh, indent=1, default=float)
    with open(os.path.join(OUT, "mixing_addendum_log.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    log(f"wrote {OUT}/mixing_addendum.json")


# =============================================================================== robustness
def robust():
    """Robustness re-run of the SAME selected configurations on a THIRD, independent seed
    (4200) with a LONGER burn-in (60% instead of 40%).  Purpose: the confirmation run had
    split-Rhat ~1.055 on the reversible arm, so residual burn-in drift could inflate the
    reversible arm's per-chain IACT and flatter the non-reversible arm.  If the effect is an
    artefact of insufficient burn-in it should shrink here."""
    global BURN_FRAC
    BURN_FRAC = 0.60
    SEED3 = 4200
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    X_hold = ds.X_test[:3]
    geom = make_geom10("ball")
    with open(os.path.join(OUT, "mixing_results.json")) as fh:
        prev = json.load(fh)
    PRIMARY = prev["primary"]
    eta = prev["selection"]["rev"][1]
    s_n = prev["selection"]["nrev"][2]
    log(f"\n=== ROBUSTNESS: seed={SEED3}, burn-in {BURN_FRAC:.0%}, eta={eta:g}, s={s_n:g} ===")
    W0 = init_unit_ball10(np.random.default_rng(SEED3 + 1), R_CONF)
    out = {}
    for tag, alpha, s in (("rev", 0, 0.0), ("nrev", 1, s_n)):
        a = run_arm(pot, geom, alpha=alpha, s=s, eta=eta, n_iter=NIT_CONF, R=R_CONF,
                    seed=SEED3, ck=CK, W0=W0)
        res = summarise(a["B"], pot, X_hold, CK)
        out[tag] = dict(res=res, wall=a["wall"], proj=a["proj"],
                        maxrhat=max(v["rhat"] for v in res.values()),
                        rhat_primary=res[PRIMARY]["rhat"])
        log(f"{tag}: wall={a['wall']:.1f}s proj={a['proj']:.3f} maxRhat={out[tag]['maxrhat']:.4f} "
            f"Rhat({PRIMARY})={out[tag]['rhat_primary']:.4f}")
        del a
    log(f"{'fn':>13} {'tau_rev':>9} {'tau_nrev':>9} {'ESSps_rev':>10} {'ESSps_nrev':>11} "
        f"{'d':>9} {'se':>8} {'t':>7} {'x':>6}")
    tab = {}
    for name in TRAIN_ONLY:
        er = out["rev"]["res"][name]["ess"] / out["rev"]["wall"]
        en = out["nrev"]["res"][name]["ess"] / out["nrev"]["wall"]
        d = en - er
        se = d.std(ddof=1) / np.sqrt(d.size)
        tab[name] = dict(tau_rev=float(np.nanmean(out["rev"]["res"][name]["tau_iter"])),
                         tau_nrev=float(np.nanmean(out["nrev"]["res"][name]["tau_iter"])),
                         essps_rev=float(er.mean()), essps_nrev=float(en.mean()),
                         d=float(d.mean()), se=float(se), t=float(d.mean() / se),
                         win=float((d > 0).mean()))
        e = tab[name]
        log(f"{name:>13} {e['tau_rev']:9.1f} {e['tau_nrev']:9.1f} {e['essps_rev']:10.2f} "
            f"{e['essps_nrev']:11.2f} {e['d']:9.2f} {e['se']:8.2f} {e['t']:7.2f} "
            f"{e['essps_nrev']/max(e['essps_rev'],1e-12):6.2f}")
    e = tab[PRIMARY]
    log(f"\nROBUSTNESS HEADLINE ({PRIMARY}): rev {e['essps_rev']:.3f} -> nrev "
        f"{e['essps_nrev']:.3f} ESS/s (x{e['essps_nrev']/e['essps_rev']:.2f}) "
        f"d={e['d']:+.3f} se={e['se']:.3f} t={e['t']:.2f} win={100*e['win']:.1f}%")
    wr = min(tab[k]["essps_rev"] for k in TRAIN_ONLY)
    wn = min(tab[k]["essps_nrev"] for k in TRAIN_ONLY)
    log(f"worst-case test function: {wr:.3f} -> {wn:.3f} (x{wn/wr:.2f})")
    with open(os.path.join(OUT, "mixing_robust.json"), "w") as fh:
        json.dump(dict(seed=SEED3, burn_frac=BURN_FRAC, eta=eta, s=s_n, primary=PRIMARY,
                       table=tab, R=R_CONF, n_iter=NIT_CONF,
                       rhat=dict(rev=out["rev"]["maxrhat"], nrev=out["nrev"]["maxrhat"])),
                  fh, indent=1, default=float)
    with open(os.path.join(OUT, "mixing_robust_log.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    log(f"wrote {OUT}/mixing_robust.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    {"addendum": addendum, "robust": robust}.get(mode, main)()
