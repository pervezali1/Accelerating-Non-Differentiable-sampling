"""Build ``accuracy_curves_beat.ipynb`` (and a Colab variant that carries the modules).

The notebook contains ONLY training/test accuracy curves for the two methods --
reversible (alpha = 0) and non-reversible (alpha = 1) anchored Langevin -- on the
configuration found by ``nonreversible_beats_reversible.py``; every other output
is a table.  Run ``python make_beat_notebook.py`` and then, for the executed copy,

    jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 \
        accuracy_curves_beat.ipynb --output accuracy_curves_beat.executed.ipynb
"""
from __future__ import annotations

import nbformat as nbf

INTRO = r"""# Non-reversible beats reversible anchored Langevin — accuracy curves only

Two figures, **training and test accuracy against iteration**, one per constraint
set. No other plots. The vertical axis is fixed to $[0.4, 0.9]$ — it is **not**
windowed on the plateau.

## Target and update

$$U(w)=\underbrace{\sum_j\bigl[\mathrm{softplus}(x_j^\top w)-y_jx_j^\top w\bigr]+\frac{w_0^2}{2\sigma^2}}_{f,\ \text{differentiable}}
\;+\;\underbrace{\lambda_{\mathrm{lasso}}\sum_{j\ge1}|w_j|}_{g,\ \text{NOT differentiable}}$$

The anchor $U_0=f+g_\delta$ replaces $|w_j|$ by $\sqrt{w_j^2+\delta^2}$ and
$a(w)=e^{U(w)-U_0(w)}\in[e^{-8\lambda\delta},1]$. Both methods use the **exact** gradient:

$$w_{k+1}=\Pi_K\!\Bigl(w_k-\eta\,a(w_k)\nabla U_0(w_k)+\eta\,\alpha\,a(w_k)\,J_s(w_k)\nabla U_0(w_k)+\sqrt{2\eta\,a(w_k)}\,\xi_{k+1}\Bigr)$$

$\alpha=0$ is the reversible method, $\alpha=1$ the non-reversible one. Within a
replicate the two chains share the initial point and every Gaussian increment; the
only difference is $\alpha$. $J_s$ is block diagonal on the coordinate triples
$(0,1,2),(3,4,5),(6,7,8)$, block $I$ being the cross-product matrix $[v_I]_\times$ with
$v_I=s\,w_I$ (ball) or $v_I=-s\nabla_I g_\varepsilon(w)$ (smoothed $L_1$ ball).

## Why this design, in one paragraph

$[v_I]_\times$ rotates only in the plane **perpendicular to its axis** $v_I$. Linearising
the drift $-\eta a(I-J)H$ at the mode, a slow Hessian eigen-direction $q_3$ that lies in
that plane, with fast partner $q_2$, has its rate replaced by $(\lambda_2+\lambda_3)/2$
once $\sigma=s|v_I|\ge\sigma^*=(\lambda_2-\lambda_3)/(2\sqrt{\lambda_2\lambda_3})$ — a
speed-up of $(\kappa+1)/2$, $\kappa=\lambda_2/\lambda_3$ — while a slow direction *along*
the axis is untouched and an oblique one is capped at $\sim1/\cos^2\theta$. In discrete
time the rotation is stable while $\sigma^2<(\lambda_a+\lambda_b)/(\eta a\lambda_a\lambda_b)-1$
for the pair spanning the rotated plane. Under the $L_1$ geometry the axis is the
soft-sign of $w_I$, i.e. $\approx(1,1,1)/\sqrt3$ when the three coefficients are positive.
So the design ("unstandardised covariates") gives each slope triple the covariance
$v_{\rm axis}q_1q_1^\top+v_{\rm fast}q_2q_2^\top+v_{\rm slow}q_3q_3^\top$ with
$q_1=(1,1,1)/\sqrt3$, $q_2=(0,1,-1)/\sqrt2$ (fast, $q_2\cdot\beta=0$: no signal) and
$q_3=(2,-1,-1)/\sqrt6$ (slow, in the rotated plane, and carrying signal because
$\beta_I=(b,e,e)$ with $b>e$). The reversible chain needs $\sim1/(\eta a\lambda_3)$
iterations along $q_3$; the non-reversible one needs $\sim2/(\eta a\lambda_2)$. Because the
$q_3$ residual carries an $O(1)$ share of the logit variance, the accuracy gap during
that transient is large. It is a **convergence-speed** effect: both chains reach the same
plateau, and the gap closes once the reversible chain has converged.

Single-iterate accuracy: at checkpoint $k$ every replicate predicts with its own current
$w_k$. Lines are the across-replicate mean; bands are mean $\pm$ one sample standard
deviation (`ddof=1`) — repeat-run variability, not confidence or credible intervals.
"""

SETUP = '''import os, sys, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

sys.path.insert(0, os.path.abspath("."))
import anchored_lasso as lasso
import anchored_sgld as nral
import nonreversible_beats_reversible as beat   # sets the Agg backend; figures are shown with display(fig)

pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:,.5f}")
QUICK  = os.environ.get("NRAL_QUICK", "0") == "1"
MODE   = "QUICK MODE" if QUICK else "FULL"
HELD_OUT = (101,) if QUICK else (101, 202, 303, 404)
OUT = "results_beat"; os.makedirs(OUT, exist_ok=True)
print(MODE)
print(pd.DataFrame(beat.CONFIGS).T)'''

DATA = '''setups = {tag: beat.setup(tag, QUICK) for tag in ("l1", "ball")}
for tag, (cfg, dataset, target, geometry) in setups.items():
    z = dataset.X_test @ dataset.beta_true
    ceiling = float(np.mean(np.maximum(nral.expit(z), 1 - nral.expit(z))))
    print(f"{geometry.name:<16} feasible(beta_true) {bool(geometry.feasible(dataset.beta_true))}   "
          f"|beta|_1 {np.abs(dataset.beta_true).sum():.2f}  |beta|_2 {np.linalg.norm(dataset.beta_true):.2f}   "
          f"Bayes ceiling on the test set {ceiling:.4f}   a in [{target.a_lower_bound:.3f}, 1]   "
          f"eta*L = {cfg.eta * target.lipschitz_constant():.3f}   R = {cfg.n_repeats}, {cfg.n_iterations} iterations")'''

RUN = '''results = {}
for tag, (cfg, dataset, target, geometry) in setups.items():
    t0 = time.time(); print(f"[{MODE}] {geometry.name}", flush=True)
    results[tag] = lasso.run_both(dataset, target, geometry, cfg, verbose=False)
    nr = results[tag]["Non-reversible anchored Langevin"]
    print(f"   {time.time() - t0:.0f} s;  projection rate NR {nr.projection_rate:.4f};  non-finite {nr.n_nonfinite}")'''

FIGURE = '''STYLE = {"Reversible anchored Langevin":     {"color": "#0173B2", "ls": "--", "lw": 2.0},
         "Non-reversible anchored Langevin": {"color": "#CC3311", "ls": "-",  "lw": 2.6}}
Y_LIMITS = (0.40, 0.90)          # fixed for every panel: NOT windowed on the plateau

for tag, (cfg, dataset, target, geometry) in setups.items():
    runs = results[tag]; p = beat.CONFIGS[tag]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, which, title in ((axes[0], "train_accuracy", f"Training accuracy  (n = {cfg.n_train})"),
                             (axes[1], "test_accuracy",  f"Test accuracy  (n = {cfg.n_test})")):
        for name, run in runs.items():
            st = STYLE[name]; mean, sd = run.mean_std(which)
            ax.fill_between(run.checkpoints, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                            color=st["color"], alpha=0.15, lw=0)
            ax.plot(run.checkpoints, mean, color=st["color"], ls=st["ls"], lw=st["lw"], label=name)
        ax.set_ylim(*Y_LIMITS); ax.set_xlabel("Iterations"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=11); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Non-reversible beats reversible anchored Langevin — {geometry.name}  [{MODE}]", fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.text(0.5, 0.012,
             f"s = {p['s']:g}, eta = {cfg.eta:.1e}, R = {cfg.n_repeats}, lambda = {cfg.lambda_lasso:g}, delta = {cfg.delta_anchor}; "
             f"design: per slope triple {p['v_axis']:g} q1q1' + {p['v_fast']:g} q2q2' + {p['v_slow']:g} q3q3', "
             f"beta_triple = ({p['b']:g}, {p['e']:g}, {p['e']:g});  y-axis fixed to {Y_LIMITS}, not windowed.",
             ha="center", fontsize=7.5)
    png = os.path.join(OUT, f"notebook_accuracy_{tag}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight"); plt.close(fig)
    display(Image(filename=png))          # backend-independent: the saved figure is shown inline'''

TABLE = '''rows = []
for tag, (cfg, dataset, target, geometry) in setups.items():
    rev = results[tag]["Reversible anchored Langevin"]; nr = results[tag]["Non-reversible anchored Langevin"]
    for k in sorted({k for k in (50, 100, 150, 200, 400, 600) if k < cfg.n_iterations} | {cfg.n_iterations}):
        i = int(np.argmin(np.abs(rev.checkpoints - k)))
        m, se, t = lasso.paired_difference(nr.test_accuracy[i], rev.test_accuracy[i])
        rows.append({"geometry": geometry.name, "iteration": int(rev.checkpoints[i]),
                     "reversible": rev.test_accuracy[i].mean(), "non_reversible": nr.test_accuracy[i].mean(),
                     "paired_diff": m, "paired_se": se, "t_stat": t})
paired_table = pd.DataFrame(rows); display(paired_table)'''

HELD = '''confirm, verdicts = [], {}
for tag, (cfg, dataset, target, geometry) in setups.items():
    evaluate_at = beat.CONFIGS[tag]["evaluate_at"]; diffs, ses = [], []
    for off in (0,) + HELD_OUT:
        runs = results[tag] if off == 0 else lasso.run_both(dataset, target, geometry, cfg, seed_offset=off, verbose=False)
        rev = runs["Reversible anchored Langevin"]; nr = runs["Non-reversible anchored Langevin"]
        i = int(np.argmin(np.abs(rev.checkpoints - evaluate_at)))
        m, se, t = lasso.paired_difference(nr.test_accuracy[i], rev.test_accuracy[i])
        confirm.append({"geometry": geometry.name, "seed_offset": off, "held_out": off != 0, "iteration": int(rev.checkpoints[i]),
                        "reversible": rev.test_accuracy[i].mean(), "non_reversible": nr.test_accuracy[i].mean(),
                        "paired_diff": m, "paired_se": se, "t_stat": t})
        if off != 0: diffs.append(m); ses.append(se)
    pooled = float(np.mean(diffs)); pooled_se = float(np.sqrt(np.sum(np.square(ses))) / len(ses))
    verdicts[geometry.name] = {"iteration": evaluate_at, "pooled_diff": pooled, "pooled_t": pooled / pooled_se,
                               "all_positive": all(d > 0 for d in diffs),
                               "confirmed": all(d > 0 for d in diffs) and pooled / pooled_se > 2}
confirm = pd.DataFrame(confirm); display(confirm)
for name, v in verdicts.items():
    print(f"{name:<16} held-out pooled diff {v['pooled_diff']:+.4f} at iteration {v['iteration']} (t = {v['pooled_t']:+.2f}), "
          f"all positive: {v['all_positive']}  ->  {'CONFIRMED' if v['confirmed'] else 'NOT confirmed'}")'''

RESULT = r"""## Result

**The non-reversible method beats the reversible one by a wide margin during the
transient, on both constraint sets, and the margin survives held-out sampler seeds.**
On the $L_1$ ball the test-accuracy gap at iteration 100 is about $+0.16$ (reversible
$\approx0.56$, non-reversible $\approx0.72$; pooled held-out $t\approx29$ with $R=100$ per
seed); on the unit ball it is about $+0.13$ at iteration 90. Both chains reach the same
plateau afterwards, and the paired difference at the end of the run is zero within noise —
the effect is faster convergence, not a different stationary accuracy.

## Where the win lives — the parameter range

From the one-factor-at-a-time map in `results_beat/parameter_map.md` (all with the $L_1$
geometry unless stated; "wins" means peak gap $\ge0.03$ with $t\ge3$):

| factor | wins | ties | loses / unstable |
|---|---|---|---|
| anisotropy $\kappa=v_{\rm fast}/v_{\rm slow}$ | $\kappa\ge4$; the gap grows with $\kappa$ and saturates at $\kappa\gtrsim32$ ($+0.18$) | $\kappa\le2$ | — |
| block strength $s$ | $0.5\le s\le12$ (gap $+0.03\to+0.21$) | $s=0$ | $s\ge16$: the projection fires on a third of the steps, late deficit |
| step size $\eta$ | $2\cdot10^{-6}\le\eta\le6\cdot10^{-5}$; the peak sits at $\approx0.7/(\eta\,a\,\lambda_{\rm slow})$ iterations | — | $\eta=10^{-4}$: $\eta a\lambda_{\max}>2$, unstable |
| $\lambda_{\rm lasso}$, $\delta$ | every value tried ($\lambda\in[0,30]$, $\delta\in[0.002,0.1]$) | — | — |
| sample size $n$ | every value tried ($250$–$8000$); the peak iteration scales as $1/n$ | — | — |
| initialisation | uniform on $K$ ($+0.18$), antipodal ($+0.65$) | origin ($+0.01$: the residual is then a pure shrinkage of the signal, which barely changes predictions) | — |
| slow-direction coefficient $b$ | $b\ge0.5$; the gap tracks the logit variance carried by $q_3$ | $b=0.25$ | — |
| geometry | $L_1$ (axis $\approx$ soft-sign, $q_3$ in the rotated plane); ball needs $\kappa\ge64$ and $s\ge8$ with $\eta\le5\cdot10^{-6}$ ($+0.10$ to $+0.13$) | — | ball with the slow coordinate along $w^*_I$ (axis theorem) |
| where the slow direction points | in the plane $\perp$ the rotation axis (this design) | isotropic design ($\kappa=1$); slow direction signal-free (`aligned`) | slow direction on a coordinate axis, oblique to the axis: $\le+0.06$, with a late deficit for $s\ge4$ |
| evaluation iteration | $0.15\lesssim\eta a\lambda_{\rm slow}\,k\lesssim1.5$ | later: both converged | — |
"""


def build(colab: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO)]
    if colab:
        cells.append(nbf.v4.new_markdown_cell(
            "## Colab bootstrap\n\nThe three modules are written next to the notebook so that "
            "`import anchored_lasso` works without cloning the repository."))
        for module in ("anchored_sgld.py", "anchored_lasso.py", "nonreversible_beats_reversible.py"):
            with open(module) as handle:
                cells.append(nbf.v4.new_code_cell(f"%%writefile {module}\n" + handle.read()))
    cells += [
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_markdown_cell("## Data, target and the two constraint sets\n\nThe ball has radius 1, so its "
                                 "`beta` is halved and `v_axis`, `v_fast` and the block-1 variance quadrupled while "
                                 "`v_slow` is only doubled: a more anisotropic design (`v_fast/v_slow = 64`), smaller "
                                 "logits (Bayes ceiling 0.74 vs 0.82) and about five times the curvature, hence the smaller `eta`."),
        nbf.v4.new_code_cell(DATA),
        nbf.v4.new_markdown_cell("## Run both methods on both constraint sets"),
        nbf.v4.new_code_cell(RUN),
        nbf.v4.new_markdown_cell("## Figures — training and test accuracy only\n\nVertical axis fixed to "
                                 "$[0.4, 0.9]$ on every panel."),
        nbf.v4.new_code_cell(FIGURE),
        nbf.v4.new_markdown_cell("## The numbers\n\nPaired per-replicate differences of single-iterate test "
                                 "accuracy (both methods share starting points and Gaussian increments within a "
                                 "replicate)."),
        nbf.v4.new_code_cell(TABLE),
        nbf.v4.new_markdown_cell("### Held-out sampler seeds\n\nThe configuration is re-run on seeds that took "
                                 "no part in the search, at the iteration where the search-seed gap peaks."),
        nbf.v4.new_code_cell(HELD),
        nbf.v4.new_markdown_cell(RESULT),
    ]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    for colab in (False, True):
        nb = build(colab)
        nbf.validate(nb)
        path = "accuracy_curves_beat_colab.ipynb" if colab else "accuracy_curves_beat.ipynb"
        nbf.write(nb, path)
        print("wrote", path, f"({len(nb.cells)} cells)")
