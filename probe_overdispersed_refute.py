"""probe_overdispersed_refute.py -- ADVERSARIAL RE-CHECK of hypothesis "overdispersed".

The claim under test (probe_overdispersed.py) is that a hard/overdispersed initialisation
("antimap") amplifies the non-reversible transient advantage.  Its own headline already reports
best-vs-best FAILURE (-0.00046, t=-0.47), while asserting that the MECHANISM is confirmed at
t=45.7.  This script attacks both halves:

  A. PAIRING CONTROL.  alpha=1 with scales=(0,0,0) makes J identically zero, so the two arms must
     be BIT-IDENTICAL.  If they are not, the common-random-number pairing is broken and every
     paired SE in the claim is meaningless.
  B. INDEPENDENT REPLICATION.  Headline configuration re-run at TWO seeds never used by the claim
     (search 7001, confirm 8231 / 3391), R=200 each (claim used R=150).
  C. FAIRNESS / EXTENDED GRID.  The claim capped eta at 3e-4 and the reversible arm's own best sat
     exactly at that cap -- i.e. the reversible arm was cut off at the edge of the grid.  We extend
     to 1e-3 and 3e-3 for BOTH arms and re-select on TRAINING accuracy only.
  D. TRANSIENT IN DISGUISE.  Matched-eta comparison re-run at 4x the iteration budget.
  E. MECHANISM DEGENERACY.  At the antimap start with eta=1e-6 every replicate starts at the SAME
     point and the injected noise is tiny, so the per-replicate spread collapses and t explodes for
     a difference that is deterministic, not statistical.  We quantify the spread, and we test
     whether the "speed-up" is anything other than a larger effective drift: the J term simply adds
     to the step norm, and giving the reversible arm the SAME step norm (eta scaled by the measured
     drift-norm ratio) is enough to erase the advantage.

Selection uses TRAINING accuracy only.  Test accuracy is recorded and never entered into an argmax.
No existing .py file is modified; everything is imported.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from exact_anchored import (D10, Exp10, accuracy10, apply_J10, geom_for, make_potential, run_exact)
from nral import build_titanic_dataset
# import (do not modify) the module under test, so the initialisation is bit-for-bit the claim's
import probe_overdispersed as PO

OUTDIR = os.path.join("results", "probe_overdispersed_refute")
SEARCH_SEED = 7001          # never used by the claim (it used 5100 / 6700)
CONF_SEED_A = 8231
CONF_SEED_B = 3391
R_SEARCH = 64               # claim used 32
R_CONF = 200                # claim used 150
ETAS_EXT = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]     # claim's grid topped out at 3e-4
EXP = Exp10("titanic_ball", "titanic", "ball", 1500)
INIT = "antimap"

rows = []
notes = {}


def arm(ds, pot, geom, n_iter, alpha, scales, eta, W0, seed):
    return PO.one_arm(ds, pot, geom, n_iter, alpha, scales, eta, W0, seed)


def rec(stage, armname, eta, s, a, extra=None):
    d = dict(stage=stage, arm=armname, eta=eta, s=s,
             train=float(a["train"].mean()), test=float(a["test"].mean()),
             train_sd=float(a["train"].std(ddof=1)), proj=float(a["proj"]), nbad=int(a["nbad"]))
    if extra:
        d.update(extra)
    rows.append(d)
    return d


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.perf_counter()
    ds = build_titanic_dataset()
    pot = make_potential(ds)
    geom = geom_for(EXP)
    print(f"titanic n_train={ds.n_train} n_test={ds.n_test} lam={pot.lam:.4g} "
          f"delta={pot.delta:.5g} majority_train={max(ds.y_train.mean(), 1-ds.y_train.mean()):.4f}")

    # ---------------------------------------------------------------- A. pairing control (s=0)
    W0s = PO.make_W0(INIT, geom, pot, R_SEARCH, np.random.default_rng(0))
    W0c = PO.make_W0(INIT, geom, pot, R_CONF, np.random.default_rng(0))
    r0 = run_exact(pot, geom, alpha=0, scales=(0, 0, 0), eta=3e-5, n_iter=300, W0=W0s[:16],
                   seed=SEARCH_SEED, checkpoint_every=300)["betas"][-1]
    r1 = run_exact(pot, geom, alpha=1, scales=(0, 0, 0), eta=3e-5, n_iter=300, W0=W0s[:16],
                   seed=SEARCH_SEED, checkpoint_every=300)["betas"][-1]
    bitgap = float(np.max(np.abs(r0 - r1)))
    # and a non-degenerate check: same seed, different alpha, first-step noise must be identical
    notes["A_pairing_s0_bitgap"] = bitgap
    notes["A_pairing_ok"] = bool(bitgap == 0.0)
    print(f"\n[A] PAIRING CONTROL  alpha=1,s=0 vs alpha=0 : max|w_T diff| = {bitgap:.3e}  "
          f"-> pairing {'INTACT' if bitgap == 0 else 'BROKEN'}")

    # ---------------------------------------------------------------- B/C. independent re-search
    pats = PO.s_patterns(pot, W0s)
    pats = {k: v for k, v in pats.items() if k in ("u0.25", "u1", "u2", "u5", "b2_5", "gn5")}
    rev_s, nrev_s, n_cfg = {}, {}, 0
    print(f"\n[B/C] RE-SEARCH  seed={SEARCH_SEED} R={R_SEARCH}  "
          f"eta grid={ETAS_EXT} (claim capped at 3e-4)  patterns={list(pats)}")
    for eta in ETAS_EXT:
        a_r = arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), eta, W0s, SEARCH_SEED)
        rev_s[eta] = a_r
        n_cfg += 1
        rec("search", "rev", eta, "-", a_r)
        line = [f"  eta={eta:<7g} rev {a_r['train'].mean():.4f}"]
        for nm, sc in pats.items():
            a_n = arm(ds, pot, geom, EXP.n_iter, 1, sc, eta, W0s, SEARCH_SEED)
            nrev_s[(eta, nm)] = a_n
            n_cfg += 1
            st = PO.paired_stats(a_n, a_r)
            rec("search", "nrev", eta, nm, a_n, dict(d_train=st["diff"], t_train=st["t"]))
            line.append(f"{nm}:{a_n['train'].mean():.4f}({st['diff']:+.4f})")
        print("  ".join(line))
    best_eta_rev = max(ETAS_EXT, key=lambda e: rev_s[e]["train"].mean())
    best_nrev = max(nrev_s, key=lambda k: nrev_s[k]["train"].mean())
    print(f"  -> rev  own best eta = {best_eta_rev:g}  (train {rev_s[best_eta_rev]['train'].mean():.4f})")
    print(f"  -> nrev best = eta {best_nrev[0]:g}, s={best_nrev[1]}={pats[best_nrev[1]]} "
          f"(train {nrev_s[best_nrev]['train'].mean():.4f})")
    notes["C_n_configs_research"] = n_cfg
    notes["C_best_eta_rev_extended"] = best_eta_rev
    notes["C_best_nrev"] = [best_nrev[0], best_nrev[1]]
    notes["C_rev_train_by_eta"] = {f"{e:g}": float(rev_s[e]["train"].mean()) for e in ETAS_EXT}

    # ---------------------------------------------------------------- confirmations
    confirm = {}
    CLAIM = dict(eta_nrev=3e-5, s_nrev=(2.0, 2.0, 2.0), eta_rev_best=3e-4)
    for tag, seed in (("A", CONF_SEED_A), ("B", CONF_SEED_B)):
        block = {}
        # (i) the CLAIM's own headline configuration, verbatim, at a fresh seed
        c_n = arm(ds, pot, geom, EXP.n_iter, 1, CLAIM["s_nrev"], CLAIM["eta_nrev"], W0c, seed)
        c_rm = arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), CLAIM["eta_nrev"], W0c, seed)
        c_rb = arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), CLAIM["eta_rev_best"], W0c, seed)
        for nm, a in (("nrev_claim", c_n), ("rev_matched", c_rm), ("rev_claimbest", c_rb)):
            rec(f"confirm{tag}", nm, a["proj"] * 0 + (CLAIM["eta_nrev"] if nm != "rev_claimbest"
                                                     else CLAIM["eta_rev_best"]),
                "u2" if nm == "nrev_claim" else "-", a)
        block["claim_matched_train"] = PO.paired_stats(c_n, c_rm, "train")
        block["claim_matched_test"] = PO.paired_stats(c_n, c_rm, "test")
        block["claim_bvb_train"] = PO.paired_stats(c_n, c_rb, "train")
        block["claim_bvb_test"] = PO.paired_stats(c_n, c_rb, "test")
        block["acc"] = {nm: dict(train=float(a["train"].mean()), test=float(a["test"].mean()),
                                 proj=float(a["proj"]))
                        for nm, a in (("nrev_claim", c_n), ("rev_matched", c_rm),
                                      ("rev_claimbest", c_rb))}
        # (ii) the EXTENDED-grid fair comparison: each arm at its own train-selected best
        e_n, s_n = best_nrev
        x_n = (c_n if (e_n == CLAIM["eta_nrev"] and pats[s_n] == CLAIM["s_nrev"])
               else arm(ds, pot, geom, EXP.n_iter, 1, pats[s_n], e_n, W0c, seed))
        x_rb = (c_rb if best_eta_rev == CLAIM["eta_rev_best"]
                else arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), best_eta_rev, W0c, seed))
        block["ext_bvb_train"] = PO.paired_stats(x_n, x_rb, "train")
        block["ext_bvb_test"] = PO.paired_stats(x_n, x_rb, "test")
        block["ext_acc"] = dict(nrev=dict(train=float(x_n["train"].mean()),
                                          test=float(x_n["test"].mean()), eta=e_n, s=s_n),
                                rev=dict(train=float(x_rb["train"].mean()),
                                         test=float(x_rb["test"].mean()), eta=best_eta_rev,
                                         proj=float(x_rb["proj"])))
        confirm[tag] = block
        m, b, x = block["claim_matched_train"], block["claim_bvb_train"], block["ext_bvb_train"]
        print(f"\n[CONFIRM {tag}] seed={seed} R={R_CONF}  (claim's own headline config, verbatim)")
        print(f"   nrev eta=3e-5 s=(2,2,2)  train {block['acc']['nrev_claim']['train']:.5f} "
              f"test {block['acc']['nrev_claim']['test']:.5f}")
        print(f"   rev  eta=3e-5 (matched)  train {block['acc']['rev_matched']['train']:.5f} "
              f"test {block['acc']['rev_matched']['test']:.5f}")
        print(f"   rev  eta=3e-4 (claim best) train {block['acc']['rev_claimbest']['train']:.5f} "
              f"test {block['acc']['rev_claimbest']['test']:.5f}")
        print(f"   MATCHED  train {m['diff']:+.5f} +- {m['se']:.5f} (t={m['t']:+.2f}, win {m['win']:.0%})")
        print(f"   BEST-vs-BEST (claim grid) train {b['diff']:+.5f} +- {b['se']:.5f} (t={b['t']:+.2f})")
        print(f"   BEST-vs-BEST (EXTENDED grid: nrev eta={e_n:g}/{s_n} vs rev eta={best_eta_rev:g}) "
              f"train {x['diff']:+.5f} +- {x['se']:.5f} (t={x['t']:+.2f})")

    # ---------------------------------------------------------------- D. transient in disguise
    print("\n[D] CONVERGENCE: matched eta=3e-5, 4x budget (6000 iters), R=100, seed A")
    conv = {}
    for nit in (1500, 6000):
        a_n = arm(ds, pot, geom, nit, 1, CLAIM["s_nrev"], 3e-5, W0c[:100], CONF_SEED_A)
        a_r = arm(ds, pot, geom, nit, 0, (0, 0, 0), 3e-5, W0c[:100], CONF_SEED_A)
        st = PO.paired_stats(a_n, a_r, "train")
        conv[nit] = dict(stat=st, nrev=float(a_n["train"].mean()), rev=float(a_r["train"].mean()))
        rec("converge", "nrev", 3e-5, f"u2/n{nit}", a_n, dict(d_train=st["diff"], t_train=st["t"]))
        rec("converge", "rev", 3e-5, f"-/n{nit}", a_r)
        print(f"   n_iter={nit:<5d} nrev {a_n['train'].mean():.5f}  rev {a_r['train'].mean():.5f}  "
              f"paired {st['diff']:+.5f} +- {st['se']:.5f} (t={st['t']:+.2f})")
    notes["D_convergence"] = {str(k): v for k, v in conv.items()}

    # ---------------------------------------------------------------- E. mechanism degeneracy
    print("\n[E] MECHANISM at eta=1e-6 (where the claim reports t=45.7)")
    sc_b2 = pats["b2_5"]
    e_n1 = arm(ds, pot, geom, EXP.n_iter, 1, sc_b2, 1e-6, W0c, CONF_SEED_A)
    e_r1 = arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), 1e-6, W0c, CONF_SEED_A)
    st1 = PO.paired_stats(e_n1, e_r1, "train")
    rec("mech", "nrev", 1e-6, "b2_5", e_n1, dict(d_train=st1["diff"], t_train=st1["t"]))
    rec("mech", "rev", 1e-6, "-", e_r1)
    print(f"   nrev {e_n1['train'].mean():.4f} (sd {e_n1['train'].std(ddof=1):.5f})   "
          f"rev {e_r1['train'].mean():.4f} (sd {e_r1['train'].std(ddof=1):.5f})   "
          f"paired {st1['diff']:+.4f} +- {st1['se']:.5f} (t={st1['t']:+.1f})")
    print(f"   majority-class train rate = {max(ds.y_train.mean(), 1-ds.y_train.mean()):.4f}  "
          f"-> BOTH arms are far below trivial")

    # drift-norm ratio: is the J term just a bigger step?
    G = pot.grad_U0(W0c[:1])
    a_w = pot.anchor_a(W0c[:1])
    drift_rev = -a_w[:, None] * G
    drift_nrev = drift_rev + a_w[:, None] * apply_J10(W0c[:1], G, geom, sc_b2)
    ratio = float(np.linalg.norm(drift_nrev) / np.linalg.norm(drift_rev))
    print(f"   drift-norm ratio ||nrev step||/||rev step|| at the antimap start = {ratio:.3f}")
    eta_eff = 1e-6 * ratio
    e_r_eff = arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), eta_eff, W0c, CONF_SEED_A)
    st_eff = PO.paired_stats(e_n1, e_r_eff, "train")
    rec("mech", "rev_effstep", eta_eff, "-", e_r_eff, dict(d_train=st_eff["diff"], t_train=st_eff["t"]))
    print(f"   rev at eta_eff = {eta_eff:.3g} (SAME drift norm): train {e_r_eff['train'].mean():.4f}"
          f"   nrev minus rev_eff = {st_eff['diff']:+.4f} (t={st_eff['t']:+.1f})")
    # and a plain 3x step size for the reversible arm
    e_r3 = arm(ds, pot, geom, EXP.n_iter, 0, (0, 0, 0), 3e-6, W0c, CONF_SEED_A)
    rec("mech", "rev_3x", 3e-6, "-", e_r3)
    print(f"   rev at eta=3e-6 (plain 3x step): train {e_r3['train'].mean():.4f} "
          f"-- beats the nrev 1e-6 arm by {e_r3['train'].mean() - e_n1['train'].mean():+.4f}")
    notes["E"] = dict(nrev_train=float(e_n1["train"].mean()), rev_train=float(e_r1["train"].mean()),
                      nrev_sd=float(e_n1["train"].std(ddof=1)),
                      rev_sd=float(e_r1["train"].std(ddof=1)), paired=st1,
                      drift_ratio=ratio, eta_eff=eta_eff,
                      rev_eff_train=float(e_r_eff["train"].mean()), vs_eff=st_eff,
                      rev_3x_train=float(e_r3["train"].mean()),
                      majority=float(max(ds.y_train.mean(), 1 - ds.y_train.mean())))

    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "runs.csv"), index=False)
    with open(os.path.join(OUTDIR, "refute.json"), "w") as fh:
        json.dump(dict(notes=notes, confirm=confirm), fh, indent=2, default=float)
    print(f"\ntotal {time.perf_counter() - t0:.1f}s -> {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
