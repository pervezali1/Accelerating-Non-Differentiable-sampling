"""Does changing the REGULARIZER help the non-reversible arm beat the reversible one?

The answer is fixed in advance for stationary quantities and open for rates, and this script
separates the two.

WHY NO REGULARIZER CAN WIN ON ACCURACY.  J's three conditions -- J^T = -J, div J = 0, and Jn = 0
on the boundary -- involve only the constraint set K, never the potential.  The anchor
cancellation needs only a = exp(U - U_0).  So for ANY regularizer both arms share the invariant
law pi_K, and accuracy at stationarity is common to them.  Nothing below can change that, and
nothing below is meant to.

WHAT THE REGULARIZER DOES CONTROL.  For a Gaussian target N(0, Sigma) a skew perturbation of the
drift improves the spectral gap by an amount that grows with the ANISOTROPY of Sigma, and is
exactly zero when Sigma is isotropic.  Our J rotates only WITHIN the blocks (1,2,3), (4,5,6),
(7,8,9).  Prediction:

    the non-reversible speed-up tracks WITHIN-block anisotropy and ignores BETWEEN-block
    anisotropy.

THE DESIGN.  One elastic-net family, so the anchor is held fixed and only the curvature moves:

    U   = loglik + w_0^2/(2 sigma^2) + lam1 * sum_j |w_j| + (lam2/2) * sum_j c_j w_j^2
    U_0 = the same with |w_j| -> sqrt(w_j^2 + delta^2)
    a   = exp(-lam1 * sum_j [sqrt(w_j^2 + delta^2) - |w_j|])          <- depends on lam1 only

Because a depends on lam1 and delta alone, every variant below has the IDENTICAL anchor and the
identical a-range.  The only thing that changes is the vector c, which reshapes the curvature:

    lasso    lam2 = 0                                  (the notebook's regularizer)
    iso      c = (1,...,1)                             isotropic ridge
    between  c = (1/k,1/k,1/k, 1,1,1, k,k,k)           anisotropic ACROSS blocks, flat inside
    within   c = (1/k,1,k) repeated in each block      anisotropic INSIDE every block

`between` and `within` at the same k have the same multiset of c_j -- the same PENALTY
anisotropy -- placed differently relative to the block structure.  They do NOT end up with the
same posterior anisotropy, because the likelihood Hessian is not block-aligned; the measured
posterior numbers are reported rather than assumed.  That pair is the experiment, and `within`
is also run at three values of k for a dose-response.

Everything is paired: both arms start from the SAME stationary state and share the Gaussian
stream, so the s = 0 control is bit-identical and the standard errors are honest.
"""
from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exact_nb as X                                                    # noqa: E402
from exact_nb import D10, design_with_intercept, softplus               # noqa: E402
from nral import _wrap, build_titanic_dataset                           # noqa: E402
from probe_mixing import iact_geyer, split_rhat                         # noqa: E402
from scipy.special import expit                                         # noqa: E402

OUT = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results", "regularizer")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

QUICK = os.environ.get("REG_MODE", "full").lower() == "quick"
R_CH = 24 if QUICK else 96              # coupled chains
N_BURN = 1000 if QUICK else 6000        # reach stationarity before the measured run
N_MEAS = 3000 if QUICK else 24000       # measured run
CK = 2                                  # store every CK iterations
ETA = 1e-4
S_BLK = (5.0, 5.0, 5.0)
SEED = 7100                             # used nowhere else in the repository
LAM1_FRAC = 0.01                        # lam1 = LAM1_FRAC * n_train  (as in the notebook)
LAM2_FRAC = 0.05                        # lam2 = LAM2_FRAC * n_train
BLOCKS = X.BLOCKS10


class ElasticPotential:
    """Elastic net on w_1..w_9 with per-coordinate ridge weights c; Gaussian prior on w_0.

    U_0 smooths ONLY the l_1 part, so a = exp(U - U_0) depends on (lam1, delta) alone and is
    identical across every variant in this study.
    """

    def __init__(self, Xd, y, sigma, lam1, lam2, c, delta):
        self.X, self.y = Xd, y.astype(float)
        self.sigma, self.lam1, self.lam2, self.delta = sigma, lam1, lam2, delta
        self.c = np.asarray(c, dtype=float)
        assert self.c.shape == (D10 - 1,)

    def _loglik(self, W):
        z = np.atleast_2d(W) @ self.X.T
        return np.sum(softplus(z) - self.y[None, :] * z, axis=1)

    def _ridge(self, W2):
        return 0.5 * self.lam2 * np.sum(self.c[None, :] * W2[:, 1:] ** 2, axis=1)

    def U(self, W):
        W2 = np.atleast_2d(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam1 * np.sum(np.abs(W2[:, 1:]), axis=1) + self._ridge(W2))
        return out[0] if np.ndim(W) == 1 else out

    def U0(self, W):
        W2 = np.atleast_2d(W)
        out = (self._loglik(W2) + W2[:, 0] ** 2 / (2 * self.sigma ** 2)
               + self.lam1 * np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2), axis=1)
               + self._ridge(W2))
        return out[0] if np.ndim(W) == 1 else out

    def grad_U0(self, W):
        W2 = np.atleast_2d(W)
        g = (expit(W2 @ self.X.T) - self.y[None, :]) @ self.X
        g[:, 0] += W2[:, 0] / self.sigma ** 2
        g[:, 1:] += (self.lam1 * W2[:, 1:] / np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2)
                     + self.lam2 * self.c[None, :] * W2[:, 1:])
        return g[0] if np.ndim(W) == 1 else g

    def a(self, W):
        W2 = np.atleast_2d(W)
        gap = np.sum(np.sqrt(W2[:, 1:] ** 2 + self.delta ** 2) - np.abs(W2[:, 1:]), axis=1)
        out = np.exp(-self.lam1 * gap)
        return out[0] if np.ndim(W) == 1 else out

    @property
    def a_lower(self):
        return float(np.exp(-self.lam1 * 9 * self.delta))


def c_vector(kind: str, k: float) -> np.ndarray:
    if kind == "iso":
        return np.ones(9)
    if kind == "within":                       # spread INSIDE each block
        return np.array([1.0 / k, 1.0, k] * 3)
    if kind == "between":                      # same multiset, spread ACROSS blocks
        return np.array([1.0 / k] * 3 + [1.0] * 3 + [k] * 3)
    raise ValueError(kind)


VARIANTS = (
    dict(key="lasso",       kind="iso",     k=1.0,  lam2_frac=0.0),
    dict(key="iso",         kind="iso",     k=1.0,  lam2_frac=LAM2_FRAC),
    dict(key="between k=10", kind="between", k=10.0, lam2_frac=LAM2_FRAC),
    dict(key="within k=3",  kind="within",  k=3.0,  lam2_frac=LAM2_FRAC),
    dict(key="within k=10", kind="within",  k=10.0, lam2_frac=LAM2_FRAC),
    dict(key="within k=30", kind="within",  k=30.0, lam2_frac=LAM2_FRAC),
)


# The accuracy difference at a fixed eta is finite-step bias, not a difference in the target.
# Re-running one variant at eta/3 says whether it behaves like bias (shrinks) or not.
# Iterations are scaled by ETA/eta so the SIMULATED TIME matches: a third of the step with the
# same iteration count would simply be a third of the trajectory, which proves nothing.
ETA_CHECK = (
    dict(key="within k=10 @eta/3", kind="within", k=10.0, lam2_frac=LAM2_FRAC, eta=ETA / 3.0),
)


def anisotropy(Sig: np.ndarray):
    """Within-block and between-block anisotropy of the posterior covariance of w_1..w_9."""
    S = Sig[1:, 1:]
    within = []
    tr = []
    for b, blk in enumerate(BLOCKS):
        idx = [j - 1 for j in blk]
        Sb = S[np.ix_(idx, idx)]
        ev = np.linalg.eigvalsh(Sb)
        within.append(float(ev.max() / max(ev.min(), 1e-300)))
        tr.append(float(np.trace(Sb) / 3.0))
    ev_all = np.linalg.eigvalsh(S)
    total = float(ev_all.max() / max(ev_all.min(), 1e-300))
    # geometric mean of the three within-block condition numbers; ratio of block scales; and the
    # condition number of the whole 9x9 block, which `within` and `between` share by construction
    return (float(np.exp(np.mean(np.log(within)))), float(max(tr) / max(min(tr), 1e-300)),
            total, within)


def ess_per_chain(S: np.ndarray, ck: int) -> np.ndarray:
    """S is (T, R) stored every `ck` iterations. Returns ESS in ITERATIONS per chain."""
    T, R = S.shape
    out = np.empty(R)
    for r in range(R):
        tau = iact_geyer(S[:, r])
        out[r] = np.nan if not np.isfinite(tau) else T / (2.0 * max(tau, 0.5))
    return out


def main() -> int:
    ds = build_titanic_dataset()
    Xd, y = design_with_intercept(ds.X_train), ds.y_train
    lam1 = LAM1_FRAC * ds.n_train
    delta = -np.log(X.A_LOWER) / (9.0 * lam1)
    geom = X.make_geom("ball")
    print(f"titanic n_train={ds.n_train}  lam1={lam1:.4g}  delta={delta:.4g}  "
          f"a in [{np.exp(-lam1 * 9 * delta):.3f}, 1]  (identical for every variant)")
    print(f"eta={ETA:g}  s={S_BLK}  R={R_CH} chains  burn={N_BURN}  measure={N_MEAS}  "
          f"seed={SEED}\n")

    rows, per_coord, t0 = [], [], time.perf_counter()
    for v in list(VARIANTS) + list(ETA_CHECK):
        eta = v.get("eta", ETA)
        scale = ETA / eta                       # keep eta * n_iter, i.e. simulated time, fixed
        n_burn = int(round(N_BURN * scale))
        n_meas = int(round(N_MEAS * scale))
        ck = max(1, int(round(CK * scale)))     # and keep the stored series the same length
        c = c_vector(v["kind"], v["k"])
        lam2 = v["lam2_frac"] * ds.n_train
        pot = ElasticPotential(Xd, y, X.SIGMA, lam1, lam2, c, delta)

        # --- reach stationarity once, then hand the SAME states to both arms
        W0 = X.init_unit_ball(np.random.default_rng(SEED + 1), R_CH)
        burn = X.run_chain(pot, geom, alpha=0, scales=S_BLK, eta=eta, n_iter=n_burn,
                           W0=W0, seed=SEED, checkpoint_every=max(1, n_burn // 2))
        Wstat = burn["W"][-1]

        res = {}
        for tag, alpha in (("rev", 0), ("nrev", 1)):
            res[tag] = X.run_chain(pot, geom, alpha=alpha, scales=S_BLK, eta=eta,
                                   n_iter=n_meas, W0=Wstat, seed=SEED + 99,
                                   checkpoint_every=ck)
        assert res["rev"]["n_nonfinite"] == 0 and res["nrev"]["n_nonfinite"] == 0

        # --- posterior covariance and its anisotropy, from the REVERSIBLE arm only
        Wr = res["rev"]["W"]                                    # (T, R, 10)
        flat = Wr.reshape(-1, D10)
        Sig = np.cov(flat.T)
        aniso_w, aniso_b, aniso_tot, within_list = anisotropy(Sig)

        # --- ESS per coordinate, paired across chains
        ratios, rhats = {}, {}
        ess_tot = {"rev": 0.0, "nrev": 0.0}
        logr = []
        for j in range(1, D10):
            e = {}
            for tag in ("rev", "nrev"):
                S = res[tag]["W"][:, :, j]
                e[tag] = ess_per_chain(S, ck)
                ess_tot[tag] += float(np.nansum(e[tag]))
                rhats[(tag, j)] = split_rhat(S)
            ok = np.isfinite(e["rev"]) & np.isfinite(e["nrev"]) & (e["rev"] > 0)
            lr = np.log(e["nrev"][ok] / e["rev"][ok])
            ratios[j] = float(np.exp(lr.mean()))
            logr.append(lr)
            per_coord.append(dict(variant=v["key"], coord=j, block=1 + (j - 1) // 3,
                                  c_j=float(c[j - 1]), ess_rev=float(np.nanmean(e["rev"])),
                                  ess_nrev=float(np.nanmean(e["nrev"])),
                                  ratio=ratios[j], rhat_rev=rhats[("rev", j)],
                                  rhat_nrev=rhats[("nrev", j)]))
        L = np.concatenate(logr)
        se = L.std(ddof=1) / np.sqrt(L.size)

        accs = {}
        for tag in ("rev", "nrev"):
            A = X.accuracy(ds.X_test, ds.y_test, res[tag]["W"])
            accs[tag] = A.mean(axis=0)                           # post-burn-in style average
        d_acc = X.paired(accs["nrev"], accs["rev"])

        rows.append(dict(
            variant=v["key"], kind=v["kind"], k=v["k"], lam2=lam2, eta=eta, n_meas=n_meas,
            aniso_within=aniso_w, aniso_between=aniso_b, aniso_total=aniso_tot,
            ess_ratio_geo=float(np.exp(L.mean())), ess_ratio_se=float(se),
            ess_ratio_t=float(L.mean() / se), ess_ratio_total=ess_tot["nrev"] / ess_tot["rev"],
            ratio_min=float(min(ratios.values())), ratio_max=float(max(ratios.values())),
            acc_rev=float(accs["rev"].mean()), acc_nrev=float(accs["nrev"].mean()),
            d_acc_pts=d_acc[0] * 100, t_acc=d_acc[2],
            proj_rev=res["rev"]["projection_rate"], proj_nrev=res["nrev"]["projection_rate"],
            rhat_max=float(np.nanmax([rhats[kk] for kk in rhats]))))
        print(f"[{v['key']:<13}] within-aniso {aniso_w:7.2f}  total-aniso {aniso_tot:8.2f}  "
              f"ESS ratio {np.exp(L.mean()):.3f} (t={L.mean()/se:+6.1f})  "
              f"acc {d_acc[0]*100:+.3f} pts (t={d_acc[2]:+5.2f})  "
              f"Rhat<={np.nanmax([rhats[kk] for kk in rhats]):.3f}  "
              f"[{time.perf_counter()-t0:.0f}s]")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(RES, "regularizer_summary.csv"), index=False)
    pd.DataFrame(per_coord).to_csv(os.path.join(RES, "regularizer_per_coordinate.csv"),
                                   index=False)
    print()
    print(tbl[["variant", "aniso_within", "aniso_between", "aniso_total", "ess_ratio_geo",
               "ess_ratio_t", "acc_rev", "acc_nrev", "d_acc_pts", "t_acc"]]
          .to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(dict(eta=ETA, s=S_BLK, R=R_CH, n_burn=N_BURN, n_meas=N_MEAS, ck=CK,
                       seed=SEED, lam1=lam1, delta=delta, lam2_frac=LAM2_FRAC,
                       rows=rows), f, indent=2, default=float)
    print(f"\ntotal {time.perf_counter()-t0:.0f}s -> {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
