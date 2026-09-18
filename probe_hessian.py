"""probe_hessian: choose the non-reversible J from the Hessian spectrum, not by search.

Hypothesis
----------
Non-reversible perturbations help most when the target is anisotropic.  So derive the
per-block rotation strengths s = (s_1, s_2, s_3) from the Hessian of the smooth anchor U_0
at the constrained mode, by a STATED RULE, instead of searching for them; and consider
re-grouping the nine coefficients into blocks that MIX curvature scales instead of the
fixed triples (1,2,3), (4,5,6), (7,8,9).

What is legitimate here
-----------------------
* A PERMUTATION of coordinates 1..9 is an exact relabelling of the target: the likelihood is
  a sum over rows, the LASSO penalty and both constraint sets (ball, smoothed l_p) are
  permutation-symmetric in w_1..w_9, and the intercept is never moved.  So the permuted chain
  samples the same law and J keeps skewness, div J = 0 and J n = 0 on the boundary.  This is
  what lets us rotate WITHIN a block whose three coordinates have very different curvature.
* The s rule uses only the TRAINING design matrix and the potential -- never test labels.

Rules compared (all with one free budget scalar c; the s vector is normalised to geometric
mean 1 so every rule spends the same "total strength"):
    iso   : s_l  proportional to 1
    hess  : s_l  proportional to 1/sqrt(lambda_max(H_ll))          <- the hypothesis' rule (a)
    hcond : s_l  proportional to cond(H_ll) = lmax/lmin            <- the hypothesis' rule (b):
                                                                      rotate hardest where the
                                                                      block is most anisotropic
    grad  : s_l  proportional to 1/||grad_{I_l} U_0(w*)||          <- known baseline (fact 4)

Outputs: results/probe_hessian/*.json, *.csv, and a stdout log.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from exact_anchored import (Potential, make_potential, run_exact, accuracy10,
                            init_unit_ball10, make_geom10, BLOCKS10, D10)
from nral import build_magic_dataset, build_titanic_dataset

OUT = os.path.join("results", "probe_hessian")
os.makedirs(OUT, exist_ok=True)
SEARCH_SEED = 3000          # every selection run uses this seed
CONFIRM_SEED = 4100         # confirmation only, never used for any choice
RNG_W0 = 91                 # offset for initial states

ETA_GRID = [1e-4, 3e-4, 1e-3, 3e-3]
C_LADDER = [1.0, 3.0, 10.0, 30.0, 100.0]
RULES = ["iso", "hess", "hcond", "grad"]
LAYOUTS = ["natural", "mixed", "homog"]

N_CONFIGS = 0               # counter: every (arm, eta, rule, c, layout) cell we evaluate


# ----------------------------------------------------------------- Hessian of the anchor U_0
def hess_U0(pot: Potential, w: np.ndarray) -> np.ndarray:
    """H = X' diag(p(1-p)) X  +  diag(lam delta^2 / (w_j^2+delta^2)^{3/2})_{j>=1}
            +  1/sigma^2 on the intercept.  Exact second derivative of U_0."""
    p = expit(pot.X @ w)
    H = (pot.X * (p * (1.0 - p))[:, None]).T @ pot.X
    d = np.zeros(D10)
    d[0] = 1.0 / pot.sigma ** 2
    d[1:] = pot.lam * pot.delta ** 2 / (w[1:] ** 2 + pot.delta ** 2) ** 1.5
    return H + np.diag(d)


def constrained_mode(pot: Potential, geom, n_steps: int = 4000, eta: float = None) -> np.ndarray:
    """Projected gradient descent on U_0 inside K (deterministic, no noise, training data only)."""
    w = np.zeros((1, D10))
    if eta is None:
        # 1/L with L = largest eigenvalue of the Hessian at 0 (a safe deterministic choice)
        eta = 1.0 / np.linalg.eigvalsh(hess_U0(pot, np.zeros(D10)))[-1]
    for _ in range(n_steps):
        w, _ = geom.project(w - eta * pot.grad_U0(w))
    return w[0]


def block_stats(H: np.ndarray, g: np.ndarray, blocks) -> dict:
    out = {}
    for l, blk in enumerate(blocks):
        idx = np.asarray(blk)
        Hb = H[np.ix_(idx, idx)]
        ev = np.linalg.eigvalsh(Hb)
        out[l] = dict(coords=[int(i) for i in idx], lmax=float(ev[-1]), lmin=float(ev[0]),
                      cond=float(ev[-1] / max(ev[0], 1e-300)),
                      gnorm=float(np.linalg.norm(g[idx])))
    return out


def s_from_rule(rule: str, bs: dict, c: float) -> tuple:
    if rule == "iso":
        raw = np.ones(3)
    elif rule == "hess":
        raw = 1.0 / np.sqrt([bs[l]["lmax"] for l in range(3)])
    elif rule == "hcond":
        raw = np.array([bs[l]["cond"] for l in range(3)])
    elif rule == "grad":
        raw = 1.0 / np.array([bs[l]["gnorm"] for l in range(3)])
    else:
        raise ValueError(rule)
    raw = raw / np.exp(np.mean(np.log(raw)))          # geometric mean 1: equal budget
    return tuple(float(c * v) for v in raw)


# ----------------------------------------------------------------- permuted-coordinate views
def layout_perm(layout: str, curv: np.ndarray) -> np.ndarray:
    """Return perm such that new coordinate k holds old coordinate perm[k] (index 0 fixed).

    curv[j] = diagonal curvature H_jj for j = 1..9 (index 0 unused).
    natural : identity.
    homog   : blocks are curvature-HOMOGENEOUS (ranks 1-3, 4-6, 7-9).
    mixed   : blocks MIX curvature scales (ranks 1,4,7 | 2,5,8 | 3,6,9).
    """
    if layout == "natural":
        return np.arange(D10)
    order = 1 + np.argsort(-curv[1:])                 # old indices, decreasing curvature
    if layout == "homog":
        new = order
    elif layout == "mixed":
        new = np.concatenate([order[k::3] for k in range(3)])
    else:
        raise ValueError(layout)
    return np.concatenate([[0], new])


def permute_view(pot: Potential, Xtr: np.ndarray, Xte: np.ndarray, perm: np.ndarray):
    """Relabel coordinates.  Returns (pot_p, Xtr_p, Xte_p).  Exactly the same target law."""
    pot_p = dataclasses.replace(pot, X=np.ascontiguousarray(pot.X[:, perm]))
    f = perm[1:] - 1                                   # feature indices (no intercept column)
    return pot_p, np.ascontiguousarray(Xtr[:, f]), np.ascontiguousarray(Xte[:, f])


# ----------------------------------------------------------------- paired experiment driver
def cell(pot, geom, Xtr, ytr, Xte, yte, *, alpha, scales, eta, n_iter, R, seed):
    """One arm.  W0 and the Gaussian stream are fixed by `seed` -> the arms are paired."""
    global N_CONFIGS
    N_CONFIGS += 1
    W0 = init_unit_ball10(np.random.default_rng(seed + RNG_W0), R)
    r = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                  W0=W0, seed=seed, checkpoint_every=max(1, n_iter // 100))
    Ws = r["betas"]
    atr = accuracy10(Xtr, ytr, Ws)
    ate = accuracy10(Xte, yte, Ws)
    h = len(Ws) // 2
    U0e = np.mean([pot.U0(Ws[i]) for i in range(h, len(Ws))], axis=0)   # ergodic avg of U_0
    return dict(train=atr[-1], test=ate[-1],
                train_erg=atr[h:].mean(axis=0),
                test_erg=ate[h:].mean(axis=0),
                U0_erg=U0e,
                U=pot.U(Ws[-1]), proj=r["projection_rate"], nonfinite=r["n_nonfinite"])


def paired(a, b, key):
    d = a[key] - b[key]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    return float(d.mean()), float(se), float(d.mean() / se), float((d > 0).mean()), n


def run_dataset(name, ds, geom_kind, n_iter, R_search, R_confirm, log, c_ladder=None):
    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s); log.append(s)

    pot = make_potential(ds)
    geom = make_geom10(geom_kind, eps=0.20 if name == "magic" else 0.18)
    Xtr, ytr, Xte, yte = ds.X_train, ds.y_train, ds.X_test, ds.y_test
    P(f"\n{'='*92}\n[{name} / {geom_kind}]  n_train={ds.n_train} n_test={ds.n_test} "
      f"lam={pot.lam:.4g} delta={pot.delta:.4g} n_iter={n_iter}\n{'='*92}")

    # ---- 1. Hessian at the constrained mode (theory step, training data only) -------------
    t0 = time.perf_counter()
    wstar = constrained_mode(pot, geom)
    H = hess_U0(pot, wstar)
    g = pot.grad_U0(wstar)
    ev = np.linalg.eigvalsh(H)
    P(f"constrained mode found in {time.perf_counter()-t0:.1f}s, ||w*||={np.linalg.norm(wstar):.4f}, "
      f"g(w*) feasible={bool(geom.feasible(wstar[None])[0])}, ||grad U_0(w*)||={np.linalg.norm(g):.3g}")
    P(f"H eigenvalues: {np.array2string(ev, precision=3, max_line_width=200)}")
    P(f"H condition number = {ev[-1]/ev[0]:.4g}   (lmax={ev[-1]:.4g}, lmin={ev[0]:.4g})")
    curv = np.diag(H).copy()
    P("diag(H) = " + np.array2string(curv, precision=3, max_line_width=200))
    # eigenvector overlap with the natural blocks
    _, V = np.linalg.eigh(H)
    ovl = np.zeros((3, D10))
    for l, blk in enumerate(BLOCKS10):
        ovl[l] = np.sum(V[list(blk), :] ** 2, axis=0)
    P("top-3 eigenvector mass on natural blocks I1/I2/I3:")
    for k in range(1, 4):
        P(f"   eigvec lambda={ev[-k]:.4g}: I1={ovl[0,-k]:.3f} I2={ovl[1,-k]:.3f} "
          f"I3={ovl[2,-k]:.3f} intercept={V[0,-k]**2:.3f}")

    layouts = {}
    for lay in LAYOUTS:
        perm = layout_perm(lay, curv)
        pot_p, Xtr_p, Xte_p = permute_view(pot, Xtr, Xte, perm)
        Hp = H[np.ix_(perm, perm)]
        gp = g[perm]
        bs = block_stats(Hp, gp, BLOCKS10)
        layouts[lay] = dict(perm=perm, pot=pot_p, Xtr=Xtr_p, Xte=Xte_p, bs=bs)
        P(f"layout {lay:8s} perm(feat)={list(perm[1:])}  " +
          "  ".join(f"I{l+1}: lmax={bs[l]['lmax']:.4g} cond={bs[l]['cond']:.3g} "
                    f"|g|={bs[l]['gnorm']:.4g}" for l in range(3)))
    bs_nat = layouts["natural"]["bs"]
    for rule in RULES:
        P(f"  rule {rule:5s} -> s/c = "
          f"{tuple(round(v,4) for v in s_from_rule(rule, bs_nat, 1.0))}")

    rows = []

    # ---- 2. reversible arm gets its OWN best eta, chosen on TRAINING accuracy -------------
    P("\n-- stage 1: reversible arm, own best eta (training accuracy, search seed) --")
    rev = {}
    for eta in ETA_GRID:
        r = cell(pot, geom, Xtr, ytr, Xte, yte, alpha=0, scales=(0, 0, 0), eta=eta,
                 n_iter=n_iter, R=R_search, seed=SEARCH_SEED)
        rev[eta] = r
        rows.append(dict(stage="rev_eta", arm="rev", eta=eta, rule="-", c=0.0, layout="natural",
                         train=r["train"].mean(), test=r["test"].mean(),
                         train_erg=r["train_erg"].mean(), proj=r["proj"]))
        P(f"   eta={eta:<8g} train={r['train'].mean():.4f} (sd {r['train'].std(ddof=1):.4f}) "
          f"proj={r['proj']:.3f}")
    eta_rev = max(ETA_GRID, key=lambda e: rev[e]["train"].mean())
    P(f"   -> reversible best eta = {eta_rev:g}  train={rev[eta_rev]['train'].mean():.4f}")

    # ---- 3. stability ceiling: rule x budget at eta_rev (training accuracy) ---------------
    P("\n-- stage 2: stability ceiling, rules x budgets (natural blocks, eta=eta_rev) --")
    base = rev[eta_rev]["train"].mean()
    ceiling = {}
    best_rc, best_rc_val = None, -1.0
    for rule in RULES:
        ok = []
        for c in (c_ladder or C_LADDER):
            s = s_from_rule(rule, bs_nat, c)
            r = cell(pot, geom, Xtr, ytr, Xte, yte, alpha=1, scales=s, eta=eta_rev,
                     n_iter=n_iter, R=R_search, seed=SEARCH_SEED)
            m = r["train"].mean()
            rows.append(dict(stage="ceiling", arm="nrev", eta=eta_rev, rule=rule, c=c,
                             layout="natural", train=m, test=r["test"].mean(),
                             train_erg=r["train_erg"].mean(), proj=r["proj"]))
            collapsed = m < base - 0.02
            P(f"   {rule:5s} c={c:<6g} s={tuple(round(v,3) for v in s)} train={m:.4f} "
              f"proj={r['proj']:.3f} {'COLLAPSED' if collapsed else ''}")
            if not collapsed:
                ok.append(c)
            if m > best_rc_val:
                best_rc_val, best_rc = m, (rule, c)
        ceiling[rule] = max(ok) if ok else 0.0
    P(f"   stability ceiling (largest non-collapsed budget c): " +
      "  ".join(f"{r}={ceiling[r]:g}" for r in RULES))
    rule_b, c_b = best_rc
    P(f"   -> best (rule, c) by TRAINING accuracy: {rule_b}, c={c_b:g} ({best_rc_val:.4f})")

    # ---- 4. block layout (curvature-mixed vs homogeneous vs natural) ----------------------
    P("\n-- stage 3: rule x block layout at the selected budget c --")
    best_lay, best_rule2, best_lay_val = "natural", rule_b, -1.0
    for lay in LAYOUTS:
        L = layouts[lay]
        for rule in RULES:
            if lay == "natural" and rule == rule_b:
                m = best_rc_val                       # already measured in stage 2
                pr = float("nan")
            else:
                sc = s_from_rule(rule, L["bs"], c_b)
                r = cell(L["pot"], geom, L["Xtr"], ytr, L["Xte"], yte, alpha=1, scales=sc,
                         eta=eta_rev, n_iter=n_iter, R=R_search, seed=SEARCH_SEED)
                m, pr = r["train"].mean(), r["proj"]
                rows.append(dict(stage="layout", arm="nrev", eta=eta_rev, rule=rule, c=c_b,
                                 layout=lay, train=m, test=r["test"].mean(),
                                 train_erg=r["train_erg"].mean(), proj=pr))
            P(f"   layout={lay:8s} rule={rule:6s} s={tuple(round(v,3) for v in s_from_rule(rule, L['bs'], c_b))} "
              f"train={m:.4f} proj={pr:.3f}")
            if m > best_lay_val:
                best_lay_val, best_lay, best_rule2 = m, lay, rule
    rule_b = best_rule2
    P(f"   -> best (rule, layout) by TRAINING accuracy: {rule_b}, {best_lay} ({best_lay_val:.4f})")

    # ---- 5. non-reversible arm gets its OWN best eta too ----------------------------------
    P("\n-- stage 4: non-reversible arm, own best eta at the selected J --")
    L = layouts[best_lay]
    s_b = s_from_rule(rule_b, L["bs"], c_b)
    nrev = {}
    for eta in ETA_GRID:
        r = cell(L["pot"], geom, L["Xtr"], ytr, L["Xte"], yte, alpha=1, scales=s_b, eta=eta,
                 n_iter=n_iter, R=R_search, seed=SEARCH_SEED)
        nrev[eta] = r
        rows.append(dict(stage="nrev_eta", arm="nrev", eta=eta, rule=rule_b, c=c_b,
                         layout=best_lay, train=r["train"].mean(), test=r["test"].mean(),
                         train_erg=r["train_erg"].mean(), proj=r["proj"]))
        P(f"   eta={eta:<8g} train={r['train'].mean():.4f} proj={r['proj']:.3f}")
    eta_nrev = max(ETA_GRID, key=lambda e: nrev[e]["train"].mean())
    P(f"   -> non-reversible best eta = {eta_nrev:g}")

    # ---- 6. CONFIRMATION on an independent seed ------------------------------------------
    P(f"\n-- stage 5: CONFIRMATION, seed {CONFIRM_SEED}, R={R_confirm}, never used above --")
    conf = {}
    # same-eta paired comparison (eta_rev, so both arms see the identical noise stream)
    c_rev = cell(pot, geom, Xtr, ytr, Xte, yte, alpha=0, scales=(0, 0, 0), eta=eta_rev,
                 n_iter=n_iter, R=R_confirm, seed=CONFIRM_SEED)
    c_nrev_same = cell(L["pot"], geom, L["Xtr"], ytr, L["Xte"], yte, alpha=1, scales=s_b,
                       eta=eta_rev, n_iter=n_iter, R=R_confirm, seed=CONFIRM_SEED)
    c_iso_same = cell(pot, geom, Xtr, ytr, Xte, yte, alpha=1,
                      scales=s_from_rule("iso", bs_nat, c_b), eta=eta_rev,
                      n_iter=n_iter, R=R_confirm, seed=CONFIRM_SEED)
    pairs = {}
    for tag, arm in (("sel_vs_rev", c_nrev_same), ("iso_vs_rev", c_iso_same)):
        for key in ("test", "train", "test_erg", "train_erg", "U0_erg"):
            d, se, t, wf, n = paired(arm, c_rev, key)
            pairs[f"{tag}:{key}"] = dict(diff=d, se=se, t=t, win_frac=wf, n=n)
        # variance-reduction metric: squared deviation of the ergodic average from the
        # pooled mean of BOTH arms.  Same invariant law => same target mean; a genuine
        # non-reversible gain should show up as SMALLER squared error (negative diff).
        for key in ("train_erg", "U0_erg"):
            mbar = 0.5 * (arm[key].mean() + c_rev[key].mean())
            d = (arm[key] - mbar) ** 2 - (c_rev[key] - mbar) ** 2
            se = d.std(ddof=1) / np.sqrt(len(d)) + 1e-300
            pairs[f"{tag}:sqerr_{key}"] = dict(diff=float(d.mean()), se=float(se),
                                               t=float(d.mean() / se),
                                               win_frac=float((d < 0).mean()), n=len(d))
    # best-vs-best (different eta -> not paired if eta differs)
    if eta_nrev != eta_rev:
        c_nrev_best = cell(L["pot"], geom, L["Xtr"], ytr, L["Xte"], yte, alpha=1, scales=s_b,
                           eta=eta_nrev, n_iter=n_iter, R=R_confirm, seed=CONFIRM_SEED)
    else:
        c_nrev_best = c_nrev_same
    conf = dict(eta_rev=eta_rev, eta_nrev=eta_nrev, rule=rule_b, c=c_b, layout=best_lay,
                s=list(s_b), ceiling=ceiling,
                rev_test=float(c_rev["test"].mean()), rev_test_sd=float(c_rev["test"].std(ddof=1)),
                rev_train=float(c_rev["train"].mean()),
                nrev_test=float(c_nrev_same["test"].mean()),
                nrev_test_sd=float(c_nrev_same["test"].std(ddof=1)),
                nrev_train=float(c_nrev_same["train"].mean()),
                iso_test=float(c_iso_same["test"].mean()),
                nrev_best_test=float(c_nrev_best["test"].mean()),
                nrev_best_train=float(c_nrev_best["train"].mean()),
                rev_test_erg=float(c_rev["test_erg"].mean()),
                nrev_test_erg=float(c_nrev_same["test_erg"].mean()),
                pairs=pairs)
    P(f"   reversible      eta={eta_rev:g}  train={c_rev['train'].mean():.4f}  "
      f"test={c_rev['test'].mean():.4f} +- {c_rev['test'].std(ddof=1)/np.sqrt(R_confirm):.4f}")
    P(f"   non-rev (hess)  eta={eta_rev:g}  train={c_nrev_same['train'].mean():.4f}  "
      f"test={c_nrev_same['test'].mean():.4f}")
    P(f"   non-rev (iso)   eta={eta_rev:g}  test={c_iso_same['test'].mean():.4f}")
    P(f"   non-rev own-best eta={eta_nrev:g}  train={c_nrev_best['train'].mean():.4f}  "
      f"test={c_nrev_best['test'].mean():.4f}")
    for k, v in pairs.items():
        P(f"   PAIRED {k:24s} diff={v['diff']:+.5f} se={v['se']:.5f} t={v['t']:+.2f} "
          f"win={v['win_frac']:.0%} n={v['n']}")
    surv = c_nrev_best["test"].mean() > c_rev["test"].mean()
    P(f"   best-vs-best on test accuracy: non-reversible {'>' if surv else '<='} reversible")
    return rows, conf


def main():
    t0 = time.perf_counter()
    log = []
    out = {}
    all_rows = []
    ds_t = build_titanic_dataset()
    rows, conf = run_dataset("titanic", ds_t, "ball", 2000, 60, 300, log)
    all_rows += [dict(dataset="titanic_ball", **r) for r in rows]
    out["titanic_ball"] = conf
    print(f"\n[elapsed {time.perf_counter()-t0:.0f}s]")

    if time.perf_counter() - t0 < 200:
        ds_m = build_magic_dataset()
        rows, conf = run_dataset("magic", ds_m, "ball", 1000, 20, 100, log,
                                 c_ladder=[1.0, 10.0, 100.0])
        all_rows += [dict(dataset="magic_ball", **r) for r in rows]
        out["magic_ball"] = conf
        print(f"\n[elapsed {time.perf_counter()-t0:.0f}s]")

    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "grid.csv"), index=False)
    out["_n_configs"] = N_CONFIGS
    out["_eta_grid"] = ETA_GRID
    out["_c_ladder"] = C_LADDER
    out["_rules"] = RULES
    out["_layouts"] = LAYOUTS
    out["_search_seed"] = SEARCH_SEED
    out["_confirm_seed"] = CONFIRM_SEED
    out["_runtime_s"] = time.perf_counter() - t0
    with open(os.path.join(OUT, "confirmation.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(OUT, "log.txt"), "w") as fh:
        fh.write("\n".join(log))
    print(f"\nTOTAL {time.perf_counter()-t0:.0f}s   configurations evaluated: {N_CONFIGS}")


def _entry():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "var":
        return stage6()
    if len(sys.argv) > 1 and sys.argv[1] == "bvb":
        return stage7()
    return main()


# =====================================================================================
# STAGE 6 (run with `python probe_hessian.py var`): the one metric where the theory says
# a non-reversible perturbation CAN win without changing the invariant law -- the
# asymptotic variance of an ergodic average.  Same eta for both arms (so the comparison
# is exactly paired), each arm at the step size the REVERSIBLE arm itself prefers.
#
#   statistic:  d_r = (g^nrev_r - mean_nrev)^2 - (g^rev_r - mean_rev)^2
#   E[d] = Var(g^nrev) - Var(g^rev).  NEGATIVE = the non-reversible chain is the more
#   accurate Monte Carlo estimator.  Each arm is centred on its OWN mean, so a difference
#   in discretisation bias cannot masquerade as a variance gain.
# =====================================================================================
VAR_SEARCH_SEED = 3000
VAR_CONFIRM_SEED = 5200


def erg_scalars(pot, geom, Xtr, ytr, Xte, yte, *, alpha, scales, eta, n_iter, R, seed):
    global N_CONFIGS
    N_CONFIGS += 1
    W0 = init_unit_ball10(np.random.default_rng(seed + RNG_W0), R)
    r = run_exact(pot, geom, alpha=alpha, scales=scales, eta=eta, n_iter=n_iter,
                  W0=W0, seed=seed, checkpoint_every=max(1, n_iter // 200))
    Ws = r["betas"]
    atr = accuracy10(Xtr, ytr, Ws)
    ate = accuracy10(Xte, yte, Ws)
    U0 = np.stack([pot.U0(Ws[i]) for i in range(len(Ws))])
    n = len(Ws)
    out = {}
    for tag, lo in (("half", n // 2), ("q4", 3 * n // 4)):
        out[f"train_{tag}"] = atr[lo:].mean(axis=0)
        out[f"test_{tag}"] = ate[lo:].mean(axis=0)
        out[f"U0_{tag}"] = U0[lo:].mean(axis=0)
    out["proj"] = r["projection_rate"]
    return out


def var_diff(a, b, key, boot=4000, rng=None):
    """Var(a) - Var(b) for paired replicates, with a paired-bootstrap CI."""
    ga, gb = a[key], b[key]
    d = (ga - ga.mean()) ** 2 - (gb - gb.mean()) ** 2
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    rng = rng or np.random.default_rng(12345)
    idx = rng.integers(0, n, size=(boot, n))
    bs = np.var(ga[idx], axis=1, ddof=1) - np.var(gb[idx], axis=1, ddof=1)
    return dict(var_nrev=float(ga.var(ddof=1)), var_rev=float(gb.var(ddof=1)),
                ratio=float(ga.var(ddof=1) / gb.var(ddof=1)),
                diff=float(d.mean()), se=float(se), t=float(d.mean() / se),
                ci_lo=float(np.quantile(bs, 0.025)), ci_hi=float(np.quantile(bs, 0.975)),
                mean_nrev=float(ga.mean()), mean_rev=float(gb.mean()), n=n)


def stage6():
    t0 = time.perf_counter()
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s); log.append(s)

    ds = build_titanic_dataset()
    pot = make_potential(ds)
    geom = make_geom10("ball")
    Xtr, ytr, Xte, yte = ds.X_train, ds.y_train, ds.X_test, ds.y_test
    n_iter, eta = 2000, 1e-4            # eta = the reversible arm's own best (stage 1)
    wstar = constrained_mode(pot, geom)
    bs = block_stats(hess_U0(pot, wstar), pot.grad_U0(wstar), BLOCKS10)

    P(f"\n{'='*92}\nSTAGE 6: asymptotic variance of ergodic averages, titanic/ball, "
      f"eta={eta:g} (reversible arm's own best), n_iter={n_iter}\n{'='*92}")

    # -- selection on TRAINING-side variance only, search seed -----------------------------
    P("-- selection (training ergodic average, search seed, R=120) --")
    rev_s = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=0, scales=(0, 0, 0), eta=eta,
                        n_iter=n_iter, R=120, seed=VAR_SEARCH_SEED)
    best, best_val = None, None
    for rule in ("iso", "hess", "hcond", "grad"):
        for c in (1.0, 3.0, 10.0):
            sc = s_from_rule(rule, bs, c)
            a = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=1, scales=sc, eta=eta,
                            n_iter=n_iter, R=120, seed=VAR_SEARCH_SEED)
            v = var_diff(a, rev_s, "train_half", boot=200)
            P(f"   rule={rule:6s} c={c:<5g} var_nrev/var_rev = {v['ratio']:.4f} "
              f"(diff {v['diff']:+.3e}, t={v['t']:+.2f})")
            if best_val is None or v["ratio"] < best_val:
                best_val, best = v["ratio"], (rule, c)
    P(f"   -> selected on TRAINING variance: rule={best[0]}, c={best[1]:g} (ratio {best_val:.4f})")

    # -- confirmation on an independent seed ------------------------------------------------
    R = 400
    P(f"\n-- CONFIRMATION seed {VAR_CONFIRM_SEED}, R={R}, paired (same W0, same noise) --")
    sc = s_from_rule(best[0], bs, best[1])
    rev_c = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=0, scales=(0, 0, 0), eta=eta,
                        n_iter=n_iter, R=R, seed=VAR_CONFIRM_SEED)
    nrev_c = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=1, scales=sc, eta=eta,
                         n_iter=n_iter, R=R, seed=VAR_CONFIRM_SEED)
    iso_c = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=1,
                        scales=s_from_rule("iso", bs, best[1]), eta=eta,
                        n_iter=n_iter, R=R, seed=VAR_CONFIRM_SEED)
    res = dict(eta=eta, n_iter=n_iter, rule=best[0], c=best[1], s=list(sc), R=R,
               search_seed=VAR_SEARCH_SEED, confirm_seed=VAR_CONFIRM_SEED)
    for tag, arm in (("sel", nrev_c), ("iso", iso_c)):
        for key in ("train_half", "test_half", "U0_half", "train_q4", "test_q4", "U0_q4"):
            v = var_diff(arm, rev_c, key)
            res[f"{tag}:{key}"] = v
            # also the MEAN difference (bias), paired
            d = arm[key] - rev_c[key]
            se = d.std(ddof=1) / np.sqrt(len(d)) + 1e-300
            res[f"{tag}:{key}:mean"] = dict(diff=float(d.mean()), se=float(se),
                                            t=float(d.mean() / se))
            P(f"   {tag:3s} {key:11s} VAR nrev/rev={v['ratio']:.4f}  "
              f"diff={v['diff']:+.4e} se={v['se']:.3e} t={v['t']:+.2f} "
              f"boot95=[{v['ci_lo']:+.3e},{v['ci_hi']:+.3e}] | "
              f"MEAN diff={d.mean():+.5f} t={d.mean()/se:+.2f}")
    res["_n_configs"] = N_CONFIGS
    with open(os.path.join(OUT, "variance.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    with open(os.path.join(OUT, "variance_log.txt"), "w") as fh:
        fh.write("\n".join(log))
    P(f"\nstage6 total {time.perf_counter()-t0:.0f}s, configs {N_CONFIGS}")


# =====================================================================================
# STAGE 7 (`python probe_hessian.py bvb`): BEST vs BEST on estimator quality.
# Stage 6 showed the non-reversible chain has a SMALLER variance but a LARGER bias for the
# ergodic average.  Increasing eta in the REVERSIBLE chain buys exactly the same trade, so
# the only fair question is which arm reaches the smaller mean squared error about the
# eta -> 0 truth.  Reference: a reversible chain at eta = 1e-5 run to the same effective
# time (bias is O(eta), so this reference has ~1/10 of the bias of the cells compared).
# =====================================================================================
BVB_SEARCH_SEED = 5200
BVB_CONFIRM_SEED = 6300
REF_ETA, REF_MULT = 1e-5, 10


def stage7():
    t0 = time.perf_counter()
    log = []

    def P(*a):
        st = " ".join(str(x) for x in a)
        print(st); log.append(st)

    ds = build_titanic_dataset()
    pot = make_potential(ds)
    geom = make_geom10("ball")
    Xtr, ytr, Xte, yte = ds.X_train, ds.y_train, ds.X_test, ds.y_test
    base_iter, base_eta = 2000, 1e-4
    wstar = constrained_mode(pot, geom)
    bs = block_stats(hess_U0(pot, wstar), pot.grad_U0(wstar), BLOCKS10)
    with open(os.path.join(OUT, "variance.json")) as fh:
        v6 = json.load(fh)
    rule_b, c_b = v6["rule"], v6["c"]
    s_b = s_from_rule(rule_b, bs, c_b)
    P(f"\n{'='*92}\nSTAGE 7: best-vs-best on MSE of the ergodic average (titanic/ball). "
      f"J = rule {rule_b}, c={c_b:g}, s={tuple(round(x,3) for x in s_b)}\n{'='*92}")

    P(f"reference: reversible, eta={REF_ETA:g}, n_iter={base_iter*REF_MULT}, R=200 "
      f"(same effective time eta*n_iter)")
    ref = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=0, scales=(0, 0, 0), eta=REF_ETA,
                      n_iter=base_iter * REF_MULT, R=200, seed=7700)
    REF = {k: float(ref[k].mean()) for k in ("train_half", "test_half", "U0_half")}
    P(f"   reference means: train={REF['train_half']:.5f} test={REF['test_half']:.5f} "
      f"U0={REF['U0_half']:.3f}")

    P(f"\n-- eta sweep, both arms, R=400, seed {BVB_SEARCH_SEED} "
      f"(n_iter scaled to keep eta*n_iter fixed) --")
    cells, rowsv = {}, []
    for eta in ETA_GRID:
        n_it = int(round(base_iter * base_eta / eta))
        for tag, alpha, sc in (("rev", 0, (0, 0, 0)), ("nrev", 1, s_b)):
            r = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=alpha, scales=sc, eta=eta,
                            n_iter=n_it, R=400, seed=BVB_SEARCH_SEED)
            cells[(tag, eta)] = r
            d = {}
            for key in ("train_half", "test_half", "U0_half"):
                g = r[key]
                d[key] = dict(mean=float(g.mean()), var=float(g.var(ddof=1)),
                              bias=float(g.mean() - REF[key]),
                              mse=float((g.mean() - REF[key]) ** 2 + g.var(ddof=1)))
            rowsv.append(dict(arm=tag, eta=eta, n_iter=n_it, proj=r["proj"],
                              **{f"{k}_{m}": d[k][m] for k in d for m in
                                 ("mean", "var", "bias", "mse")}))
            P(f"   {tag:4s} eta={eta:<8g} n={n_it:<6d} TRAIN mean={d['train_half']['mean']:.5f} "
              f"bias={d['train_half']['bias']:+.5f} var={d['train_half']['var']:.3e} "
              f"MSE={d['train_half']['mse']:.3e} | TEST MSE={d['test_half']['mse']:.3e}")
    pd.DataFrame(rowsv).to_csv(os.path.join(OUT, "bvb_eta_sweep.csv"), index=False)

    def best_eta(tag):
        return min(ETA_GRID, key=lambda e: (cells[(tag, e)]["train_half"].mean()
                                            - REF["train_half"]) ** 2
                   + cells[(tag, e)]["train_half"].var(ddof=1))
    e_rev, e_nrev = best_eta("rev"), best_eta("nrev")
    P(f"   -> selected on TRAINING MSE only: reversible eta={e_rev:g}, "
      f"non-reversible eta={e_nrev:g}")

    P(f"\n-- CONFIRMATION seed {BVB_CONFIRM_SEED}, R=400, each arm at its OWN best eta --")
    conf = dict(rule=rule_b, c=c_b, s=list(s_b), ref=REF, ref_eta=REF_ETA,
                eta_rev=e_rev, eta_nrev=e_nrev, R=400,
                search_seed=BVB_SEARCH_SEED, confirm_seed=BVB_CONFIRM_SEED)
    fin = {}
    for tag, alpha, sc, eta in (("rev", 0, (0, 0, 0), e_rev), ("nrev", 1, s_b, e_nrev)):
        n_it = int(round(base_iter * base_eta / eta))
        fin[tag] = erg_scalars(pot, geom, Xtr, ytr, Xte, yte, alpha=alpha, scales=sc, eta=eta,
                               n_iter=n_it, R=400, seed=BVB_CONFIRM_SEED)
    for key in ("train_half", "test_half", "U0_half"):
        a, b = fin["nrev"][key], fin["rev"][key]
        mse_a = (a.mean() - REF[key]) ** 2 + a.var(ddof=1)
        mse_b = (b.mean() - REF[key]) ** 2 + b.var(ddof=1)
        d = (a - REF[key]) ** 2 - (b - REF[key]) ** 2          # paired squared error
        se = d.std(ddof=1) / np.sqrt(len(d)) + 1e-300
        conf[key] = dict(mean_nrev=float(a.mean()), mean_rev=float(b.mean()),
                         var_nrev=float(a.var(ddof=1)), var_rev=float(b.var(ddof=1)),
                         mse_nrev=float(mse_a), mse_rev=float(mse_b),
                         sqerr_diff=float(d.mean()), sqerr_se=float(se),
                         sqerr_t=float(d.mean() / se), win_frac=float((d < 0).mean()),
                         n=len(d))
        P(f"   {key:11s} nrev mean={a.mean():.5f} var={a.var(ddof=1):.3e} MSE={mse_a:.3e} | "
          f"rev mean={b.mean():.5f} var={b.var(ddof=1):.3e} MSE={mse_b:.3e} | "
          f"paired sq.err diff={d.mean():+.3e} se={se:.3e} t={d.mean()/se:+.2f} "
          f"win={float((d<0).mean()):.0%}")
    conf["_n_configs"] = N_CONFIGS
    with open(os.path.join(OUT, "best_vs_best.json"), "w") as fh:
        json.dump(conf, fh, indent=2, default=float)
    with open(os.path.join(OUT, "bvb_log.txt"), "w") as fh:
        fh.write("\n".join(log))
    P(f"\nstage7 total {time.perf_counter()-t0:.0f}s, configs {N_CONFIGS}")


if __name__ == "__main__":
    raise SystemExit(_entry())
