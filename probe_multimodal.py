"""probe_multimodal: does a HARDER target let the non-reversible anchored Langevin
arm genuinely beat the reversible one?

Established beforehand (not re-derived here): J is skew, div J = 0, J n = 0 on the
boundary, so the CONTINUOUS reflected process has the same invariant law for both
alpha = 0 and alpha = 1.  Any difference is therefore a finite-budget / finite-step
effect.  The question this probe asks is whether a legitimately harder version of the
same posterior widens that finite-budget gap enough to survive a BEST-vs-BEST
comparison at a fixed iteration budget.

Three hardness knobs, all declared, none of them touching test labels:
  (H1) lasso weight lam = lasso_frac * n_train pushed far above the default 0.01*n,
       so many coordinates sit at kinks of |w_j|;
  (H2) constraint level shrunk (ball radius^2 r2, or smoothed-l_p Lambda) so the chain
       is pinned to a curved boundary;
  (H3) deliberate ill-conditioning of the design matrix by a declared per-column
       rescaling.  Two shapes:
         "between": block 1 columns x c, block 3 columns x 1/c  (anisotropy ACROSS the
                    three J blocks -- the block cross-product rotation cannot couple them)
         "within" : inside EVERY block, column 1 x c and column 3 x 1/c (anisotropy
                    INSIDE each block -- this is the direction the rotation acts on)
       "within" is the theoretically motivated one: J is block-diagonal, so it can only
       help where the slow and fast directions live in the SAME block.

Everything is paired: both arms get the same W0 and the same seed (hence the same
Gaussian stream).  Every configuration choice is made on TRAINING accuracy; the headline
is a confirmation run on an independent seed.
"""
from __future__ import annotations

import argparse, dataclasses, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from exact_anchored import (make_potential, make_geom10, run_exact, accuracy10,
                            init_unit_ball10, D10)
from nral import build_titanic_dataset, build_magic_dataset

RESDIR = os.path.join(HERE, "results", "probe_multimodal")
os.makedirs(RESDIR, exist_ok=True)

SEARCH_SEED = 3000          # every selection decision uses this seed only
CONFIRM_SEED = 4100         # independent, never used for selection
CONFIRM_SEED2 = 5200


# ------------------------------------------------------------------ hardness knobs
def col_scales(kind: str, c: float) -> np.ndarray:
    """Per-FEATURE multiplier (9 features; the intercept column is never touched)."""
    if kind == "iso":
        return np.ones(9)
    if kind == "within":                       # anisotropy INSIDE each J block
        return np.array([c, 1.0, 1.0 / c] * 3)
    if kind == "between":                      # anisotropy ACROSS J blocks
        return np.array([c] * 3 + [1.0] * 3 + [1.0 / c] * 3)
    raise ValueError(kind)


def rescaled(ds, kind: str, c: float):
    v = col_scales(kind, c)
    return dataclasses.replace(ds, X_train=ds.X_train * v[None, :],
                               X_test=ds.X_test * v[None, :])


@dataclasses.dataclass(frozen=True)
class Cfg:
    tag: str
    dataset: str = "titanic"
    scale_kind: str = "iso"
    c: float = 1.0
    lasso_frac: float = 0.01
    geom: str = "ball"
    r2: float = 2.0
    Lam: float = 4.0
    eps: float = 0.18
    n_iter: int = 2000

    def build(self, base_ds):
        ds = rescaled(base_ds, self.scale_kind, self.c)
        pot = make_potential(ds, lasso_frac=self.lasso_frac)
        g = make_geom10(self.geom, r2=self.r2, p=2.4, eps=self.eps, Lam=self.Lam)
        return ds, pot, g


# ------------------------------------------------------------------ one paired cell
def cell(ds, pot, geom, *, eta, s, n_iter, R, seed, ck=None):
    """Both arms: same W0, same seed -> same Gaussian noise stream."""
    W0 = init_unit_ball10(np.random.default_rng(seed + 1), R)
    if not np.all(geom.feasible(W0)):          # tight constraint sets need a shrunken init
        W0 = W0 * 0.3
        k = 0
        while not np.all(geom.feasible(W0)) and k < 40:
            W0 = W0 * 0.5; k += 1
    ck = ck or max(1, n_iter // 8)
    out = {}
    for tag, alpha in (("rev", 0), ("nrev", 1)):
        t0 = time.perf_counter()
        r = run_exact(pot, geom, alpha=alpha, scales=(s, s, s), eta=eta, n_iter=n_iter,
                      W0=W0, seed=seed, checkpoint_every=ck)
        Ws = r["betas"]
        out[tag] = dict(acc_train=accuracy10(ds.X_train, ds.y_train, Ws),
                        acc_test=accuracy10(ds.X_test, ds.y_test, Ws),
                        U=pot.U(Ws[-1]), checkpoints=r["checkpoints"],
                        proj=r["projection_rate"], nbad=r["n_nonfinite"],
                        wall=time.perf_counter() - t0)
    return out


def paired_stats(a, b, key="acc_train"):
    d = a[key][-1] - b[key][-1]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) + 1e-300
    return float(d.mean()), float(se), float(d.mean() / se), float((d > 0).mean())


def row_of(cfg, eta, s, R, seed, res):
    dtr, setr, ttr, wtr = paired_stats(res["nrev"], res["rev"], "acc_train")
    dte, sete, tte, wte = paired_stats(res["nrev"], res["rev"], "acc_test")
    return dict(tag=cfg.tag, eta=eta, s=s, R=R, seed=seed,
                rev_train=float(res["rev"]["acc_train"][-1].mean()),
                nrev_train=float(res["nrev"]["acc_train"][-1].mean()),
                rev_test=float(res["rev"]["acc_test"][-1].mean()),
                nrev_test=float(res["nrev"]["acc_test"][-1].mean()),
                d_train=dtr, se_train=setr, t_train=ttr, win_train=wtr,
                d_test=dte, se_test=sete, t_test=tte, win_test=wte,
                rev_U=float(res["rev"]["U"].mean()), nrev_U=float(res["nrev"]["U"].mean()),
                proj_rev=res["rev"]["proj"], proj_nrev=res["nrev"]["proj"],
                bad_nrev=res["nrev"]["nbad"], bad_rev=res["rev"]["nbad"],
                wall_rev=res["rev"]["wall"], wall_nrev=res["nrev"]["wall"])


def best_vs_best(df, tag):
    """Each arm picked at ITS OWN best TRAINING accuracy over the whole (eta, s) grid.
    The reversible arm ignores s (alpha = 0), so its grid is just the eta grid."""
    sub = df[df.tag == tag]
    rev = sub.loc[sub.rev_train.idxmax()]
    nre = sub[sub.s > 0]
    if len(nre) == 0:
        return None
    nre = nre.loc[nre.nrev_train.idxmax()]
    return dict(tag=tag,
                rev_eta=float(rev.eta), rev_train=float(rev.rev_train), rev_test=float(rev.rev_test),
                nrev_eta=float(nre.eta), nrev_s=float(nre.s),
                nrev_train=float(nre.nrev_train), nrev_test=float(nre.nrev_test),
                gap_train=float(nre.nrev_train - rev.rev_train),
                gap_test=float(nre.nrev_test - rev.rev_test))


# ------------------------------------------------------------------ stages
ETAS = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
SCREEN_S = [1.0, 5.0]


def stage_screen(args):
    base = build_titanic_dataset()
    cfgs = [
        Cfg("base"),
        Cfg("lam50",     lasso_frac=50.0 / 712),
        Cfg("lam200",    lasso_frac=200.0 / 712),
        Cfg("tight_r2",  r2=0.30),
        Cfg("within10",  scale_kind="within", c=10.0),
        Cfg("within30",  scale_kind="within", c=30.0),
        Cfg("between10", scale_kind="between", c=10.0),
        Cfg("w10_lam50", scale_kind="within", c=10.0, lasso_frac=50.0 / 712),
        Cfg("w10_tight", scale_kind="within", c=10.0, r2=0.30),
        Cfg("w10_lp",    scale_kind="within", c=10.0, geom="lp", Lam=1.2),
    ]
    rows, t0 = [], time.perf_counter()
    for cfg in cfgs:
        ds, pot, geom = cfg.build(base)
        for eta in ETAS:
            for s in [0.0] + SCREEN_S:          # s = 0 gives the reversible reference row
                res = cell(ds, pot, geom, eta=eta, s=s, n_iter=cfg.n_iter,
                           R=args.R, seed=SEARCH_SEED)
                rows.append(row_of(cfg, eta, s, args.R, SEARCH_SEED, res))
        print(f"  [{cfg.tag}] done ({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, "screen.csv"), index=False)
    bvb = [b for b in (best_vs_best(df, t) for t in df.tag.unique()) if b]
    pd.DataFrame(bvb).to_csv(os.path.join(RESDIR, "screen_best_vs_best.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== screen: paired TRAIN gap, per config, best (eta,s) cell by d_train ===")
    for t in df.tag.unique():
        sub = df[(df.tag == t) & (df.s > 0)].sort_values("d_train", ascending=False)
        r = sub.iloc[0]
        print(f"{t:>10}  best d_train {r.d_train:+.4f} (t={r.t_train:+5.2f}) at eta={r.eta:g} s={r.s:g}"
              f"   rev_train={r.rev_train:.4f} nrev_train={r.nrev_train:.4f}")
    print("\n=== screen: BEST-vs-BEST on TRAINING accuracy (each arm at its own best eta) ===")
    print(pd.DataFrame(bvb).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nscreen total {time.perf_counter()-t0:.0f}s   runs={len(rows)*2}")


def stage_refine(args):
    """Refine (eta, s) on the config named by --tag, TRAINING accuracy only."""
    base = build_magic_dataset() if args.dataset == "magic" else build_titanic_dataset()
    cfg = CFG_BY_TAG[args.tag]
    if args.dataset == "magic":
        cfg = dataclasses.replace(cfg, dataset="magic", n_iter=args.n_iter,
                                  lasso_frac=cfg.lasso_frac)
    ds, pot, geom = cfg.build(base)
    etas = [float(x) for x in args.etas.split(",")]
    ss = [float(x) for x in args.ss.split(",")]
    rows, t0 = [], time.perf_counter()
    for eta in etas:
        for s in ss:
            res = cell(ds, pot, geom, eta=eta, s=s, n_iter=cfg.n_iter, R=args.R, seed=SEARCH_SEED)
            rows.append(row_of(cfg, eta, s, args.R, SEARCH_SEED, res))
        print(f"  eta={eta:g} done ({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, f"refine_{args.dataset}_{args.tag}.csv"), index=False)
    pd.set_option("display.width", 240)
    print(df[["eta", "s", "rev_train", "nrev_train", "d_train", "se_train", "t_train",
              "rev_test", "nrev_test", "proj_nrev", "bad_nrev"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    b = best_vs_best(df, cfg.tag)
    print("\nBEST-vs-BEST on TRAINING:", json.dumps(b, indent=2))
    json.dump(b, open(os.path.join(RESDIR, f"refine_{args.dataset}_{args.tag}_bvb.json"), "w"), indent=2)
    print(f"refine total {time.perf_counter()-t0:.0f}s")


def stage_confirm(args):
    """Confirmation on INDEPENDENT seeds, at the (eta, s) chosen by the training search,
    plus the reversible arm at ITS own best eta (also chosen on training)."""
    base = build_magic_dataset() if args.dataset == "magic" else build_titanic_dataset()
    cfg = CFG_BY_TAG[args.tag]
    if args.dataset == "magic":
        cfg = dataclasses.replace(cfg, dataset="magic", lasso_frac=cfg.lasso_frac * 712 / 15216
                                  if args.match_lam else cfg.lasso_frac, n_iter=args.n_iter)
    ds, pot, geom = cfg.build(base)
    out, t0 = {}, time.perf_counter()
    seeds = [CONFIRM_SEED, CONFIRM_SEED2][: args.n_seeds]
    for seed in seeds:
        rec = {}
        res = cell(ds, pot, geom, eta=args.eta, s=args.s, n_iter=cfg.n_iter, R=args.R, seed=seed)
        rec["paired_at_nrev_best"] = row_of(cfg, args.eta, args.s, args.R, seed, res)
        rec["curve_checkpoints"] = res["rev"]["checkpoints"].tolist()
        rec["curve_rev_train"] = res["rev"]["acc_train"].mean(axis=1).tolist()
        rec["curve_nrev_train"] = res["nrev"]["acc_train"].mean(axis=1).tolist()
        rec["curve_rev_test"] = res["rev"]["acc_test"].mean(axis=1).tolist()
        rec["curve_nrev_test"] = res["nrev"]["acc_test"].mean(axis=1).tolist()
        if abs(args.rev_eta - args.eta) > 1e-15:
            res2 = cell(ds, pot, geom, eta=args.rev_eta, s=0.0, n_iter=cfg.n_iter,
                        R=args.R, seed=seed)
            rec["rev_at_own_best"] = dict(
                eta=args.rev_eta,
                train=float(res2["rev"]["acc_train"][-1].mean()),
                train_sd=float(res2["rev"]["acc_train"][-1].std(ddof=1)),
                test=float(res2["rev"]["acc_test"][-1].mean()),
                test_sd=float(res2["rev"]["acc_test"][-1].std(ddof=1)))
            # UNPAIRED cross-eta contrast (different eta -> different chains), reported as such
            dte = res["nrev"]["acc_test"][-1] - res2["rev"]["acc_test"][-1]
            dtr = res["nrev"]["acc_train"][-1] - res2["rev"]["acc_train"][-1]
            rec["bestvsbest_same_W0_same_seed"] = dict(
                d_test=float(dte.mean()), se_test=float(dte.std(ddof=1) / np.sqrt(len(dte))),
                t_test=float(dte.mean() / (dte.std(ddof=1) / np.sqrt(len(dte)) + 1e-300)),
                d_train=float(dtr.mean()), se_train=float(dtr.std(ddof=1) / np.sqrt(len(dtr))),
                t_train=float(dtr.mean() / (dtr.std(ddof=1) / np.sqrt(len(dtr)) + 1e-300)))
        else:
            rec["rev_at_own_best"] = dict(eta=args.eta, note="same eta as the non-reversible arm")
        out[str(seed)] = rec
        r = rec["paired_at_nrev_best"]
        print(f"[seed {seed}] eta={args.eta:g} s={args.s:g} R={args.R}  "
              f"rev_train={r['rev_train']:.4f} nrev_train={r['nrev_train']:.4f} "
              f"d_train={r['d_train']:+.4f} (t={r['t_train']:+.2f})  |  "
              f"rev_test={r['rev_test']:.4f} nrev_test={r['nrev_test']:.4f} "
              f"d_test={r['d_test']:+.4f} (t={r['t_test']:+.2f})", flush=True)
        if "bestvsbest_same_W0_same_seed" in rec:
            bb = rec["bestvsbest_same_W0_same_seed"]; rb = rec["rev_at_own_best"]
            print(f"           rev at ITS own best eta={rb['eta']:g}: train={rb['train']:.4f} "
                  f"test={rb['test']:.4f}   ->  nrev - rev_best: test {bb['d_test']:+.4f} "
                  f"+- {bb['se_test']:.4f} (t={bb['t_test']:+.2f}), train {bb['d_train']:+.4f} "
                  f"(t={bb['t_train']:+.2f})", flush=True)
        print(f"           wall: rev {res['rev']['wall']:.2f}s  nrev {res['nrev']['wall']:.2f}s  "
              f"proj rev {res['rev']['proj']:.3f} nrev {res['nrev']['proj']:.3f}", flush=True)
    meta = dict(cfg=dataclasses.asdict(cfg), eta=args.eta, s=args.s, rev_eta=args.rev_eta,
                R=args.R, dataset=args.dataset, seeds=seeds, results=out)
    json.dump(meta, open(os.path.join(RESDIR, args.out), "w"), indent=2, default=float)
    print(f"confirm total {time.perf_counter()-t0:.0f}s -> {args.out}")



def stage_budget(args):
    """How much EXTRA iteration budget does the reversible arm need, at ITS own best step
    size, to match the non-reversible arm's accuracy at the shared budget?  This is the
    honest price of the win: the two arms share an invariant law, so the reversible arm
    must catch up eventually -- the question is at what cost."""
    base = build_titanic_dataset()
    cfg = CFG_BY_TAG[args.tag]
    ds, pot, geom = cfg.build(base)
    N = cfg.n_iter
    rows, t0 = [], time.perf_counter()
    res = cell(ds, pot, geom, eta=args.eta, s=args.s, n_iter=N, R=args.R, seed=CONFIRM_SEED)
    nrev_train = float(res["nrev"]["acc_train"][-1].mean())
    nrev_test = float(res["nrev"]["acc_test"][-1].mean())
    per_iter_rev = res["rev"]["wall"] / N
    per_iter_nrev = res["nrev"]["wall"] / N
    rows.append(dict(arm="nrev", eta=args.eta, s=args.s, mult=1, n_iter=N,
                     train=nrev_train, test=nrev_test, wall=res["nrev"]["wall"]))
    rows.append(dict(arm="rev", eta=args.rev_eta, s=0.0, mult=1, n_iter=N,
                     train=float(res["rev"]["acc_train"][-1].mean()),
                     test=float(res["rev"]["acc_test"][-1].mean()), wall=res["rev"]["wall"]))
    for mult in [int(m) for m in args.mults.split(",")]:
        W0 = init_unit_ball10(np.random.default_rng(CONFIRM_SEED + 1), args.R)
        r = run_exact(pot, geom, alpha=0, scales=(0.0, 0.0, 0.0), eta=args.rev_eta,
                      n_iter=N * mult, W0=W0, seed=CONFIRM_SEED, checkpoint_every=N)
        tr = accuracy10(ds.X_train, ds.y_train, r["betas"])[-1].mean()
        te = accuracy10(ds.X_test, ds.y_test, r["betas"])[-1].mean()
        rows.append(dict(arm="rev", eta=args.rev_eta, s=0.0, mult=mult, n_iter=N * mult,
                         train=float(tr), test=float(te), wall=r["runtime"]))
        print(f"  rev x{mult} ({N*mult} iters): train {tr:.4f} test {te:.4f}   "
              f"(nrev at x1: train {nrev_train:.4f} test {nrev_test:.4f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, f"budget_{args.tag}.csv"), index=False)
    print(f"\nper-iteration wall clock: rev {per_iter_rev*1e3:.4f} ms, "
          f"nrev {per_iter_nrev*1e3:.4f} ms  (overhead {100*(per_iter_nrev/per_iter_rev-1):.1f}%)")
    print(f"budget total {time.perf_counter()-t0:.0f}s")


CFG_BY_TAG = {}


def main():
    global CFG_BY_TAG
    for c in [Cfg("base"), Cfg("lam50", lasso_frac=50.0 / 712),
              Cfg("lam200", lasso_frac=200.0 / 712), Cfg("tight_r2", r2=0.30),
              Cfg("within10", scale_kind="within", c=10.0),
              Cfg("within30", scale_kind="within", c=30.0),
              Cfg("between10", scale_kind="between", c=10.0),
              Cfg("w10_lam50", scale_kind="within", c=10.0, lasso_frac=50.0 / 712),
              Cfg("w10_tight", scale_kind="within", c=10.0, r2=0.30),
              Cfg("w10_lp", scale_kind="within", c=10.0, geom="lp", Lam=1.2),
              Cfg("within100", scale_kind="within", c=100.0)]:
        CFG_BY_TAG[c.tag] = c
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["screen", "refine", "confirm", "budget"])
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--tag", default="within10")
    ap.add_argument("--etas", default="1e-4,3e-5,1e-5")
    ap.add_argument("--ss", default="0,0.5,2,5,15")
    ap.add_argument("--eta", type=float, default=1e-5)
    ap.add_argument("--rev_eta", type=float, default=1e-5)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--dataset", default="titanic")
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--match_lam", action="store_true")
    ap.add_argument("--out", default="confirm.json")
    ap.add_argument("--mults", default="2,4,8,16")
    args = ap.parse_args()
    {"screen": stage_screen, "refine": stage_refine, "confirm": stage_confirm,
     "budget": stage_budget}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
