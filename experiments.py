"""
experiments.py -- the numerical study "can non-reversibility beat reversible anchored Langevin?"

Run `python experiments.py --quick` for a smoke test or `python experiments.py` for the full study.
Results (CSV/JSON) go to results/, figures to figures/.  Every experiment is seeded; the two Euler
arms always share the initial states and the Gaussian stream (common random numbers).

Design principles, each tied to the paper:
  * The theory (Thm 3.2/3.3, Thm 4.2/4.9, Thm 5.1) compares *rates*: convergence to pi, the level-2
    rate function, and the asymptotic variance of ergodic averages.  We therefore measure
      - the energy distance between the replicate cloud and an exact reference sample vs. time,
      - the asymptotic variance / ESS of time averages (Thm 5.1; exact in Prop. 6.7),
      - the mean-squared error of the posterior-mean estimate at a fixed gradient budget,
    and we report the finite-eta bias next to every ESS number (Remark 5.4).
  * The exact reference is a Metropolis-adjusted anchored chain (nald.anchored_mala): it targets
    exactly pi ~ e^{-U} 1_K, so it is the same law both Euler arms approximate.
  * Tuning is done on a *selection* seed; every headline number is re-measured on a disjoint
    *confirmation* seed with the configuration frozen.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

# the matrices here are small (hundreds x tens); multithreaded BLAS only adds contention
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nald  # noqa: E402
from nald import (Ball, L1Logistic, PenalisedRegression, QuadraticClock, StudentClock,  # noqa: E402
                  anchored_mala, block_J2, energy_curve, ess_from_replicates, lasso_penalty,
                  load_titanic, make_regression_data, mcp_penalty, mse_curve, run_nald,
                  scad_penalty, spectral_J)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "results", "cache")
for _d in (RESULTS, FIGURES, CACHE):
    os.makedirs(_d, exist_ok=True)

REV, NREV, NREV2, INK, INK2, GRIDC = "#1f5fbf", "#1a9850", "#d95f02", "#1f2328", "#57606a", "#e4e6ea"


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIDC,
        "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
        "ytick.color": INK2, "legend.frameon": False, "grid.color": GRIDC, "grid.linewidth": 0.7,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def cached_reference(key: str, builder, force: bool = False) -> np.ndarray:
    f = os.path.join(CACHE, key + ".npz")
    if os.path.exists(f) and not force:
        return np.load(f)["refs"]
    refs = builder()
    np.savez(f, refs=refs)
    return refs


# ==============================================================================================
# Experiment 1: heavy tails -- exact theory (Prop. 6.7) and the Student-t clock (Cor. 6.5)
# ==============================================================================================


def exp_heavy_tailed(quick: bool) -> Dict:
    log("E1 heavy-tailed clocks")
    R, n, thin = (600, 6000, 4) if quick else (3000, 16000, 4)
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    # --- quadratic clock, d = 4, beta = 4 ---------------------------------------------------
    qc = QuadraticClock(4, 4.0)
    J = block_J2(4)
    eta = 0.004
    W0 = np.random.default_rng(1).standard_normal((R, 4)) * 0.3
    burn = 500
    m2 = 4 * qc.second_moment()
    st = StudentClock(4, 6.0, 1.5)
    eta_s = 0.002
    W0s = np.random.default_rng(2).standard_normal((R, 4)) * 0.3
    for integ in ("euler", "exp"):
        for al in alphas:
            r = run_nald(qc, W0, eta=eta, n_iter=n, alpha=al, J=J, seed=11, thin=thin, integrator=integ)
            x = r.traj[burn:]
            T = x.shape[0]
            n_steps = (T - 1) * thin + 1
            obs = {"x1": x[..., 0], "x1+x2": x[..., 0] + x[..., 1], "|x|^2": np.sum(x ** 2, -1),
                   "x1*x3": x[..., 0] * x[..., 2], "sin(2x1)": np.sin(2 * x[..., 0])}
            for name, v in obs.items():
                tavg = v.mean(0)
                s2 = n_steps * tavg.var(ddof=1) * eta                   # continuous-time asymptotic var
                th = qc.sigma2_linear(np.eye(4)[0], al, J) if name == "x1" else (
                    qc.sigma2_linear(np.array([1, 1, 0, 0.0]), al, J) if name == "x1+x2" else np.nan)
                rows.append(dict(model="quadratic clock (d=4, beta=4)", integrator=integ, alpha=al, observable=name,
                                 sigma2=s2, sigma2_theory=th, mean=float(tavg.mean()),
                                 se_rel=math.sqrt(2 / (R - 1)), eta=eta, n_steps=n_steps, R=R,
                                 second_moment=float(np.mean(obs["|x|^2"])), second_moment_exact=m2))
            log(f"  qclock [{integ}] alpha={al}: sigma2(x1)={rows[-5]['sigma2']:.4f} theory={rows[-5]['sigma2_theory']:.4f} "
                f"sigma2(|x|^2)={rows[-3]['sigma2']:.4f} E|x|^2={np.mean(obs['|x|^2']):.3f} (exact {m2:.3f}) nonfinite={r.n_nonfinite}")
        # --- Student-t clock, d = 4, theta = 6, zeta = 1.5 -----------------------------------
        for al in alphas:
            r = run_nald(st, W0s, eta=eta_s, n_iter=n, alpha=al, J=J, seed=12, thin=thin, integrator=integ)
            x = r.traj[burn:]
            T = x.shape[0]
            n_steps = (T - 1) * thin + 1
            obs = {"x1": x[..., 0], "x1+x2": x[..., 0] + x[..., 1], "|x|^2": np.sum(x ** 2, -1),
                   "x1*x3": x[..., 0] * x[..., 2], "sin(2x1)": np.sin(2 * x[..., 0])}
            for name, v in obs.items():
                tavg = v.mean(0)
                rows.append(dict(model="Student-t clock (d=4, theta=6, zeta=1.5)", integrator=integ, alpha=al, observable=name,
                                 sigma2=n_steps * tavg.var(ddof=1) * eta_s, sigma2_theory=np.nan,
                                 mean=float(tavg.mean()), se_rel=math.sqrt(2 / (R - 1)), eta=eta_s,
                                 n_steps=n_steps, R=R, second_moment=float(np.mean(obs["|x|^2"])),
                                 second_moment_exact=4 * st.second_moment()))
            log(f"  student [{integ}] alpha={al}: sigma2(x1)={rows[-5]['sigma2']:.4f} sigma2(|x|^2)={rows[-3]['sigma2']:.4f} "
                f"E|x|^2={np.mean(obs['|x|^2']):.3f} (exact {4 * st.second_moment():.3f}) nonfinite={r.n_nonfinite}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e1_heavy_tailed.csv"), index=False)
    fig_heavy_tailed(df)
    return dict(table=df)


def fig_heavy_tailed(df: pd.DataFrame) -> None:
    plt = setup_mpl()
    models = list(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    cols = {"x1": NREV, "x1+x2": NREV2, "x1*x3": "#7570b3", "sin(2x1)": "#e7298a", "|x|^2": INK2}
    for i, integ in enumerate(("euler", "exp")):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = df[(df["model"] == model) & (df["integrator"] == integ)]
            base = sub[sub["alpha"] == 0].set_index("observable")["sigma2"]
            for obs, col in cols.items():
                s = sub[sub["observable"] == obs].sort_values("alpha")
                ax.plot(s["alpha"], s["sigma2"] / base[obs], "o-", color=col, ms=4.5, lw=1.6, label=f"observable {obs}")
            al = np.linspace(0, 3, 200)
            ax.plot(al, 1 / (1 + al ** 2), "--", color=INK, lw=1.2, label="theory $1/(1+\\alpha^2)$ (Prop. 6.7, linear $g$)")
            ax.axhline(1.0, color=REV, lw=1.4, label="reversible anchored ($\\alpha=0$)")
            ax.set_yscale("log"); ax.set_xlabel("non-reversible strength $\\alpha$")
            ax.set_ylabel("$\\sigma^2_{g,\\alpha}/\\sigma^2_{g,0}$")
            ttl = "Euler chord (7.1)" if integ == "euler" else "exact rotation (exponential integrator)"
            ax.set_title(f"{model} -- {ttl}", loc="left", fontsize=9.5); ax.grid(True, alpha=0.9)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Heavy-tailed anchored clocks: circulation with $J^2=-I$ shrinks the asymptotic variance of ergodic "
                 "averages of every non-radial observable; the radial $|x|^2$ (in $\\ker A$) is untouched in continuous time.\n"
                 "Top: plain Euler, whose chord of a rotation inflates $|x|$ in the heavy tail (Remark 5.4). "
                 "Bottom: the same runs with the rotation applied exactly.", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGURES, "e1_heavy_tailed.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "e1_heavy_tailed.pdf"), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================================
# Experiment 2: unconstrained penalised regression -- Lasso / MCP / SCAD (Fig. 1-2 of the paper)
# ==============================================================================================


def regression_problem(penalty: str, data_seed: int, *, n=100, d=20, rho=0.9, k=5, a_lower=0.5):
    data = make_regression_data(n=n, d=d, rho=rho, n_nonzero=k, sigma=1.0, seed=data_seed)
    lam = 5.0
    if penalty == "lasso":
        pen = lasso_penalty(lam)
    elif penalty == "mcp":
        pen = mcp_penalty(lam, 3.0)
    elif penalty == "scad":
        pen = scad_penalty(lam, 3.7)
    else:
        raise ValueError(penalty)
    # smoothing width: the Gaussian smoothing gap is at most lam*mu*sqrt(2/pi) per coordinate for
    # the Lasso (and below that for MCP/SCAD), so this keeps a >= a_lower everywhere
    mu = -math.log(a_lower) / (d * lam * math.sqrt(2 / math.pi))
    pot = PenalisedRegression(data["X"], data["y"], 1.0, pen, mu)
    return data, pot


def build_regression_reference(pot: PenalisedRegression, eta: float, quick: bool, seed: int) -> np.ndarray:
    R, n, thin, burn = (200, 8000, 20, 100) if quick else (400, 30000, 20, 300)
    w_mode = pot.mode_U0()
    r = anchored_mala(pot, np.tile(w_mode, (R, 1)), eta=0.5 * eta, n_iter=n, seed=seed, thin=thin)
    log(f"    reference aMALA acceptance {r.extra['accept_rate']:.3f} ({(n // thin - burn) * R} samples)")
    return r.traj[burn:].reshape(-1, pot.d)


def init_states(kind: str, R: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "normal":          # N(0, 10 I)
        return rng.standard_normal((R, d)) * math.sqrt(10.0)
    if kind == "uniform":         # Uniform(-5, 5)^d
        return rng.uniform(-5, 5, (R, d))
    raise ValueError(kind)


def arm_configs(d: int, alphas: Sequence[float], Hcov: Optional[np.ndarray] = None) -> List[Tuple[str, float, Optional[np.ndarray]]]:
    cfgs: List[Tuple[str, float, Optional[np.ndarray]]] = [("reversible", 0.0, None)]
    J2 = block_J2(d)
    for al in alphas:
        cfgs.append((f"J2 alpha={al:g}", al, J2))
    if Hcov is not None:
        for s in (1.0, 2.0):
            Js, _ = spectral_J(Hcov, strength=s)
            cfgs.append((f"spectral(cov) s={s:g}", 1.0, Js))
    return cfgs


def evaluate_arm(pot, W0, refs, *, eta, n_iter, alpha, J, seed, thin, burn_steps, geom=None,
                 current="stream", energy_every=None, energy_frames=None, **kw) -> Dict:
    ref_mean, ref_var = refs.mean(0), refs.var(0)
    r = run_nald(pot, W0, eta=eta, n_iter=n_iter, alpha=alpha, J=J, seed=seed, thin=thin, geom=geom,
                 current=current, **kw)
    burn_saved = burn_steps // thin
    st = ess_from_replicates(r.traj, ref_var, burn_saved, thin)
    bias = (st["mean"] - ref_mean) / np.sqrt(ref_var)
    mse = mse_curve(r.traj, ref_mean, ref_var, burn_saved)          # after burn-in
    out = dict(ess_per_step=st["ess_per_step"], ess_min=float(st["ess_per_step"].min()),
               ess_med=float(np.median(st["ess_per_step"])), ess_bm_med=float(np.median(st["ess_bm"]) / st["n_steps"]),
               bias_rms=float(np.sqrt(np.mean(bias ** 2))), bias_max=float(np.max(np.abs(bias))),
               mse_curve=mse, mse_final=float(mse[-1]), proj_rate=r.proj_rate, nonfinite=r.n_nonfinite,
               runtime=r.runtime, traj=r.traj)
    if energy_every is not None:
        nf = energy_frames if energy_frames is not None else r.traj.shape[0]
        idx, ed = energy_curve(r.traj[:nf], refs[:3000], every=energy_every)
        out["energy_steps"] = idx * thin
        out["energy"] = ed
    return out


def exp_regression(quick: bool) -> Dict:
    log("E2 unconstrained penalised regression (Lasso / MCP / SCAD)")
    R, n_iter, thin = (100, 12000, 10) if quick else (200, 60000, 10)
    burn_steps = n_iter // 4
    alphas_screen = [1.0, 2.0, 3.0, 5.0, 8.0]
    SEL_DATA, CONF_DATA = 7, 8            # data seeds
    SEL_CHAIN, CONF_CHAIN = 5, 6          # Gaussian-stream seeds
    rows, curves = [], {}
    chosen: Dict[str, float] = {}
    for penalty in ("lasso", "mcp", "scad"):
        for phase, dseed, cseed in (("selection", SEL_DATA, SEL_CHAIN), ("confirmation", CONF_DATA, CONF_CHAIN)):
            data, pot = regression_problem(penalty, dseed)
            H = pot.hess_U0(pot.mode_U0())
            eta = 0.8 / np.linalg.eigvalsh(H).max()
            refs = cached_reference(f"reg_{penalty}_{dseed}_{'q' if quick else 'f'}",
                                    lambda: build_regression_reference(pot, eta, quick, 99 + dseed))
            Hcov = np.linalg.inv(np.cov(refs.T))
            log(f"  [{penalty}/{phase}] d={pot.d} a in [{pot.a_range[0]:.3f},{pot.a_range[1]:.3f}] eta={eta:.2e} "
                f"cond(anchor H)={np.linalg.cond(H):.0f} cond(posterior cov^-1)={np.linalg.cond(Hcov):.0f}")
            if phase == "selection":
                cfgs = arm_configs(pot.d, alphas_screen, Hcov)
            else:
                cfgs = [("reversible", 0.0, None), (f"J2 alpha={chosen[penalty]:g}", chosen[penalty], block_J2(pot.d))]
            for init in (("normal", "uniform") if phase == "confirmation" else ("normal",)):
                W0 = init_states(init, R, pot.d, cseed + 1)
                for name, al, J in cfgs:
                    t0 = time.perf_counter()
                    ev = evaluate_arm(pot, W0, refs, eta=eta, n_iter=n_iter, alpha=al, J=J, seed=cseed,
                                      thin=thin, burn_steps=burn_steps, energy_every=20,
                                      energy_frames=min(r_ := (n_iter // thin) + 1, 3001))
                    rows.append(dict(penalty=penalty, phase=phase, init=init, arm=name, alpha=al,
                                     eta=eta, R=R, n_iter=n_iter, ess_min=ev["ess_min"], ess_med=ev["ess_med"],
                                     ess_bm_med=ev["ess_bm_med"], bias_rms=ev["bias_rms"], bias_max=ev["bias_max"],
                                     mse_final=ev["mse_final"], nonfinite=ev["nonfinite"], runtime=ev["runtime"]))
                    curves[(penalty, phase, init, name)] = dict(mse=ev["mse_curve"], energy=ev["energy"],
                                                                 energy_steps=ev["energy_steps"])
                    log(f"    {init:7s} {name:22s} ESS/step min={ev['ess_min']:.4f} med={ev['ess_med']:.4f} "
                        f"bias rms={ev['bias_rms']:.3f} max={ev['bias_max']:.3f} MSE={ev['mse_final']:.2e} "
                        f"nonfinite={ev['nonfinite']} [{time.perf_counter() - t0:.0f}s]")
            if phase == "selection":
                # selection rule, fixed in advance: among J2 arms whose max |bias| <= 2x the reversible
                # arm's (and never above 0.1 sd), take the one with the largest minimum ESS/step.
                sub = pd.DataFrame([r for r in rows if r["penalty"] == penalty and r["phase"] == "selection"])
                rev_bias = float(sub[sub["arm"] == "reversible"]["bias_max"].iloc[0])
                ok = sub[(sub["arm"].str.startswith("J2")) & (sub["bias_max"] <= max(2 * rev_bias, 0.1))
                         & (sub["nonfinite"] == 0)]
                best = ok.sort_values("ess_min", ascending=False).iloc[0] if len(ok) else sub[sub["arm"].str.startswith("J2")].sort_values("bias_max").iloc[0]
                chosen[penalty] = float(best["alpha"])
                log(f"  [{penalty}] selected alpha = {chosen[penalty]:g} (rule: bias_max <= max(2x rev, 0.1 sd), max min-ESS)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e2_regression.csv"), index=False)
    np.savez(os.path.join(RESULTS, "e2_regression_curves.npz"),
             **{"|".join(k): np.asarray(v["mse"]) for k, v in curves.items()},
             **{"E|" + "|".join(k): np.asarray(v["energy"]) for k, v in curves.items()},
             **{"S|" + "|".join(k): np.asarray(v["energy_steps"]) for k, v in curves.items()})
    with open(os.path.join(RESULTS, "e2_selected_alpha.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    fig_regression(df, curves, chosen, thin)
    return dict(table=df, curves=curves, chosen=chosen)


def fig_regression(df: pd.DataFrame, curves: Dict, chosen: Dict[str, float], thin: int) -> None:
    plt = setup_mpl()
    # Figure A: selection screen -- ESS/step (min over coordinates) vs alpha, bias alongside
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for j, pen in enumerate(("lasso", "mcp", "scad")):
        sub = df[(df["penalty"] == pen) & (df["phase"] == "selection")]
        rev = sub[sub["arm"] == "reversible"].iloc[0]
        j2 = sub[sub["arm"].str.startswith("J2")].sort_values("alpha")
        sp = sub[sub["arm"].str.startswith("spectral")]
        ax = axes[0, j]
        ax.axhline(rev["ess_min"], color=REV, lw=1.6, label="reversible (min over coords)")
        ax.axhline(rev["ess_med"], color=REV, lw=1.0, ls=":", label="reversible (median)")
        ax.plot(j2["alpha"], j2["ess_min"], "o-", color=NREV, label="non-reversible, block $J_2$, min over coords")
        ax.plot(j2["alpha"], j2["ess_med"], "o:", color=NREV, mfc="white", label="non-reversible, block $J_2$, median")
        for _, r in sp.iterrows():
            ax.scatter([0.35 if "s=1" in r["arm"] else 0.6], [r["ess_min"]], marker="s", color=NREV2, zorder=5,
                       label="spectrally matched $J$ (cov), min" if "s=1" in r["arm"] else None)
        ax.axvline(chosen[pen], color=INK2, lw=0.8, ls="--")
        ax.set_yscale("log"); ax.set_title(f"{pen.upper()}: ESS per gradient evaluation", loc="left", fontsize=10)
        ax.set_xlabel("$\\alpha$"); ax.grid(True, alpha=0.9)
        ax = axes[1, j]
        ax.axhline(rev["bias_max"], color=REV, lw=1.6, label="reversible")
        ax.plot(j2["alpha"], j2["bias_max"], "o-", color=NREV, label="non-reversible, block $J_2$")
        for _, r in sp.iterrows():
            ax.scatter([0.35 if "s=1" in r["arm"] else 0.6], [r["bias_max"]], marker="s", color=NREV2, zorder=5)
        ax.axvline(chosen[pen], color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{pen.upper()}: max finite-$\\eta$ bias of the mean (in posterior sd)", loc="left", fontsize=10)
        ax.set_xlabel("$\\alpha$"); ax.grid(True, alpha=0.9); ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=7)
    fig.suptitle("Selection screen (data seed 7, chain seed 5): larger circulation buys ESS until the Euler bias "
                 "grows; dashed line = the $\\alpha$ frozen for the confirmation run", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGURES, "e2_selection_screen.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "e2_selection_screen.pdf"), bbox_inches="tight")
    plt.close(fig)
    # Figure B: confirmation -- energy distance to the exact reference and MSE of the running mean
    for init in ("normal", "uniform"):
        fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6))
        for j, pen in enumerate(("lasso", "mcp", "scad")):
            for name, col, lab in (("reversible", REV, "reversible anchored Langevin"),
                                   (f"J2 alpha={chosen[pen]:g}", NREV, f"non-reversible, $\\alpha={chosen[pen]:g}$")):
                c = curves.get((pen, "confirmation", init, name))
                if c is None:
                    continue
                axes[0, j].plot(c["energy_steps"], c["energy"], color=col, lw=1.7, label=lab)
                steps = np.arange(len(c["mse"])) * thin
                axes[1, j].plot(steps, c["mse"], color=col, lw=1.7, label=lab)
            axes[0, j].set_yscale("log"); axes[0, j].set_xscale("log")
            axes[0, j].set_title(f"{pen.upper()}: energy distance to exact reference", loc="left", fontsize=10)
            axes[0, j].set_xlabel("Euler steps"); axes[0, j].grid(True, alpha=0.9)
            axes[1, j].set_yscale("log"); axes[1, j].set_xscale("log")
            axes[1, j].set_title(f"{pen.upper()}: MSE of running posterior-mean estimate", loc="left", fontsize=10)
            axes[1, j].set_xlabel("Euler steps after burn-in"); axes[1, j].grid(True, alpha=0.9)
        axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
        ini = "$w_0\\sim N(0,10 I)$" if init == "normal" else "$w_0\\sim$ Uniform$(-5,5)^d$"
        fig.suptitle(f"Confirmation run (fresh data seed 8, fresh chain seed 6), initial distribution {ini}: "
                     f"same $\\eta$, same number of gradient evaluations, common random numbers", fontsize=9.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(FIGURES, f"e2_confirmation_{init}.png"), dpi=220, bbox_inches="tight")
        fig.savefig(os.path.join(FIGURES, f"e2_confirmation_{init}.pdf"), bbox_inches="tight")
        plt.close(fig)


# ==============================================================================================
# Experiment 3: constrained l1-logistic regression on the ball (Titanic)
# ==============================================================================================


def titanic_problem(a_lower: float = 0.5):
    ds = load_titanic(os.path.join(HERE, "data", "titanic.csv"))
    X, y = ds["X_train"], ds["y_train"]
    n, d = X.shape
    lam = 0.01 * n
    delta = -math.log(a_lower) / ((d - 1) * lam)
    pot = L1Logistic(X, y, 10.0, lam, delta)
    return ds, pot, Ball(2.0)


def build_titanic_reference(pot, ball, eta, quick, seed) -> np.ndarray:
    R, n, thin, burn = (200, 10000, 20, 100) if quick else (400, 40000, 20, 300)
    r = anchored_mala(pot, np.zeros((R, pot.d)), eta=0.3 * eta, n_iter=n, geom=ball, seed=seed, thin=thin)
    log(f"    reference aMALA acceptance {r.extra['accept_rate']:.3f}")
    return r.traj[burn:].reshape(-1, pot.d)


def exp_titanic(quick: bool) -> Dict:
    log("E3 constrained l1-logistic regression on the ball (Titanic)")
    R, n_iter, thin = (100, 12000, 10) if quick else (200, 60000, 10)
    burn_steps = n_iter // 4
    ds, pot, ball = titanic_problem()
    H = pot.hess_U0(pot.mode_U0())
    eta = 0.8 / np.linalg.eigvalsh(H).max()
    refs = cached_reference(f"titanic_{'q' if quick else 'f'}", lambda: build_titanic_reference(pot, ball, eta, quick, 99))
    Hcov = np.linalg.inv(np.cov(refs.T))
    log(f"  d={pot.d} lam={pot.lam:.3f} delta={pot.delta:.4f} eta={eta:.2e} |w|^2 on boundary frac="
        f"{np.mean(np.sum(refs ** 2, 1) > 1.99):.3f}")
    d = pot.d
    blocks = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    def cfgs(alphas_J2, spectral, cross):
        out = [("reversible", dict(alpha=0.0))]
        for al in alphas_J2:
            out.append((f"stream J2 alpha={al:g}", dict(alpha=al, J=block_J2(d), current="stream")))
        for s in spectral:
            out.append((f"stream spectral(cov) s={s:g}", dict(alpha=1.0, J=spectral_J(Hcov, strength=s)[0], current="stream")))
        for s in cross:
            out.append((f"cross-product s={s:g}", dict(alpha=1.0, current="cross", cross_scale=s, cross_blocks=blocks)))
        return out
    rows, curves = [], {}
    SEL_CHAIN, CONF_CHAIN = 5, 6
    chosen_name, chosen_kw = None, None
    for phase, cseed in (("selection", SEL_CHAIN), ("confirmation", CONF_CHAIN)):
        rng = np.random.default_rng(cseed + 1)
        Z = rng.standard_normal((R, d))
        W0 = (rng.random(R) ** (1 / d))[:, None] * Z / np.linalg.norm(Z, axis=1, keepdims=True)   # uniform in unit ball
        arms = cfgs([1.0, 2.0, 4.0, 8.0], [1.0, 2.0], [5.0, 20.0]) if phase == "selection" else \
            [("reversible", dict(alpha=0.0)), (chosen_name, chosen_kw)]
        for name, kw in arms:
            t0 = time.perf_counter()
            ev = evaluate_arm(pot, W0, refs, eta=eta, n_iter=n_iter, seed=cseed, thin=thin, burn_steps=burn_steps,
                              geom=ball, energy_every=20, energy_frames=min((n_iter // thin) + 1, 3001),
                              **{k: v for k, v in kw.items() if k != "J"}, J=kw.get("J"))
            # posterior predictive test accuracy from the post-burn-in chain (a stationary quantity:
            # both arms must agree on it -- reported as a sanity check, not as a metric)
            post = ev["traj"][burn_steps // thin:].reshape(-1, d)[::max(1, (n_iter // thin) // 200)]
            p_test = np.mean(1 / (1 + np.exp(-(post @ ds["X_test"].T))), axis=0)
            acc_test = float(np.mean((p_test >= 0.5) == (ds["y_test"] > 0.5)))
            rows.append(dict(phase=phase, arm=name, eta=eta, R=R, n_iter=n_iter, ess_min=ev["ess_min"], ess_med=ev["ess_med"],
                             ess_bm_med=ev["ess_bm_med"], bias_rms=ev["bias_rms"], bias_max=ev["bias_max"], mse_final=ev["mse_final"],
                             proj_rate=ev["proj_rate"], nonfinite=ev["nonfinite"], acc_test_bma=acc_test, runtime=ev["runtime"]))
            curves[(phase, name)] = dict(mse=ev["mse_curve"], energy=ev["energy"], energy_steps=ev["energy_steps"])
            log(f"    {phase:12s} {name:28s} ESS/step min={ev['ess_min']:.4f} med={ev['ess_med']:.4f} bias rms={ev['bias_rms']:.3f} "
                f"max={ev['bias_max']:.3f} MSE={ev['mse_final']:.2e} proj={ev['proj_rate']:.3f} acc={acc_test:.4f} [{time.perf_counter() - t0:.0f}s]")
        if phase == "selection":
            sub = pd.DataFrame([r for r in rows if r["phase"] == "selection"])
            rev_bias = float(sub[sub["arm"] == "reversible"]["bias_max"].iloc[0])
            ok = sub[(sub["arm"] != "reversible") & (sub["bias_max"] <= max(2 * rev_bias, 0.1)) & (sub["nonfinite"] == 0)]
            best = ok.sort_values("ess_min", ascending=False).iloc[0]
            chosen_name = best["arm"]
            chosen_kw = dict(arms)[chosen_name]
            log(f"  selected arm: {chosen_name}")
    ref_acc = float(np.mean((np.mean(1 / (1 + np.exp(-(refs[::10] @ ds["X_test"].T))), axis=0) >= 0.5) == (ds["y_test"] > 0.5)))
    df = pd.DataFrame(rows)
    df["acc_test_reference"] = ref_acc
    df.to_csv(os.path.join(RESULTS, "e3_titanic.csv"), index=False)
    fig_titanic(df, curves, chosen_name, thin)
    return dict(table=df, curves=curves, chosen=chosen_name)


def fig_titanic(df: pd.DataFrame, curves: Dict, chosen: str, thin: int) -> None:
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    sub = df[df["phase"] == "selection"].copy()
    sub["kind"] = sub["arm"].str.split(" ").str[0]
    ax = axes[0]
    rev = sub[sub["arm"] == "reversible"].iloc[0]
    ax.axhline(rev["ess_min"], color=REV, lw=1.6, label="reversible (min over coords)")
    order = sub[sub["arm"] != "reversible"]
    xs = np.arange(len(order))
    cols = {"stream": NREV, "cross-product": NREV2}
    ax.bar(xs, order["ess_min"], color=[cols.get(k.split()[0], INK2) if "spectral" not in k else "#7570b3" for k in order["arm"]])
    ax.set_xticks(xs); ax.set_xticklabels([a.replace("stream ", "").replace("cross-product", "cross") for a in order["arm"]], rotation=45, ha="right", fontsize=7)
    ax.set_yscale("log"); ax.set_title("selection: min ESS per gradient evaluation", loc="left", fontsize=10); ax.grid(True, axis="y", alpha=0.9)
    ax.legend(fontsize=7.5)
    for ax, key, ttl, xl in ((axes[1], "energy", "energy distance to exact reference", "Euler steps"),
                             (axes[2], "mse", "MSE of running posterior-mean estimate", "Euler steps after burn-in")):
        for name, col, lab in (("reversible", REV, "reversible anchored Langevin"), (chosen, NREV, f"non-reversible: {chosen}")):
            c = curves.get(("confirmation", name))
            if c is None:
                continue
            if key == "energy":
                ax.plot(c["energy_steps"], c["energy"], color=col, lw=1.7, label=lab)
            else:
                ax.plot(np.arange(len(c["mse"])) * thin, c["mse"], color=col, lw=1.7, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.grid(True, alpha=0.9)
        ax.set_title("confirmation: " + ttl, loc="left", fontsize=10); ax.legend(fontsize=7.5)
    fig.suptitle("Titanic, $\\ell_1$-logistic regression on the ball $|w|^2\\le 2$ with normal reflection (projection): "
                 "stream construction $\\psi=e^{-U_0}(|w|^2-R^2)/2$ admits any constant $J$", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGURES, "e3_titanic.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, "e3_titanic.pdf"), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated subset of e1,e2,e3")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else {"e1", "e2", "e3"}
    t0 = time.perf_counter()
    if "e1" in only:
        exp_heavy_tailed(args.quick)
    if "e2" in only:
        exp_regression(args.quick)
    if "e3" in only:
        exp_titanic(args.quick)
    log(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
