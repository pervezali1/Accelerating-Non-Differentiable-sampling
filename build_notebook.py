"""Assemble `beating_reversible_anchored_langevin.ipynb` from results/ and figures/ and execute it."""
import json
import os
import subprocess
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Beating reversible anchored Langevin: what the theory says to measure, and what happens when you do

**Setting.** Target $\pi\propto e^{-U}$ (on $\mathbb R^d$ or on a ball with normal reflection), $U$ possibly
non-differentiable, smooth anchor $U_0$, speed $a=e^{U-U_0}$. The two arms compared throughout are

$$\text{ALD (reversible):}\quad dX=-a\nabla U_0\,dt+\sqrt{2a}\,dW,\qquad
\text{NALD (non-reversible):}\quad dX=\big[-a\nabla U_0+\alpha\,c\big]dt+\sqrt{2a}\,dW,\quad c=e^{U}J\nabla\psi ,$$

both discretised by (projected) Euler–Maruyama, eq. (7.1)–(7.2) of the paper, with the *same* step $\eta$, the *same*
number of gradient evaluations, the *same* initial states and the *same* Gaussian stream (common random numbers).

**Why the earlier notebook could not see a difference.** It measured the classification accuracy of the last iterate.
Both arms provably share the invariant law (Theorem 2.5), and accuracy is a bounded, coarse functional of a nearly
stationary state, so the only thing it can reveal is an $O(\eta)$ discretisation bias, which is why nothing cleared
$|t|>2.5$ there. The paper's acceleration theorems are statements about *rates*:

| Theorem | Quantity | Measured here by |
|---|---|---|
| Thm 3.2 / 3.3 (TV convergence, $\rho_\alpha\ge\rho_0$) | speed of $\mathrm{Law}(X_t)\to\pi$ | energy distance between the replicate cloud and an *exact* reference sample, as a function of the step count |
| Thm 4.2 / 4.9 (LDP, $I_\alpha\ge I_0$) | concentration of the empirical measure | mean-squared error of the running posterior-mean estimate at a fixed gradient budget |
| Thm 5.1, Cor. 5.3, Cor. 6.5, Prop. 6.7 ($\sigma^2_{g,\alpha}\le\sigma^2_{g,0}$) | asymptotic variance of ergodic averages | effective sample size per gradient evaluation, from $R$ independent replicate chains (no autocorrelation model needed) |

and Remark 5.4 asks that the finite-$\eta$ bias be reported next to every ESS number, which is done everywhere below.

**The exact reference.** A Metropolis-adjusted anchored chain (`nald.anchored_mala`): proposal
$y\sim N\!\big(x-\eta a(x)\nabla U_0(x),\,2\eta a(x)I\big)$, accepted with the exact MH ratio for $e^{-U}\mathbf 1_K$.
It targets the law both Euler arms approximate, so bias and variance can be separated.

**Two design choices that come straight out of the paper**

1. *The stream mechanism makes any constant $J$ admissible on the ball.* Lemma 2.3(i): if $\psi$ is constant on $\partial K$
   then $c\cdot n=0$ with ordinary normal reflection. Taking $\psi=e^{-U_0}\,\varphi$ with $\varphi=(|x|^2-R^2)/2$ gives
   $c=a\,J\,(x-\varphi\nabla U_0)$: a constant skew $J$ (which is automatically divergence-free) can be used on a bounded domain.
   The cross-product tensor of Remark 2.4(iii), used in the earlier notebook, annihilates the radial part of $\nabla U_0$ and
   vanishes at $w_I=0$, so its circulation is weak by construction.
2. *Strength is what the variance theorem rewards.* Theorem 5.1 gives $\sigma^2_{g,\alpha}=2\langle K^{-1/2}g,(I-\alpha^2G^2)^{-1}K^{-1/2}g\rangle$,
   decreasing in $|\alpha|$; Prop. 6.7 makes it $1/(1+\alpha^2)$ for $J^2=-I$. The limit on $\alpha$ is the Euler
   discretisation, not the theory. So the recipe tested is: a block rotation $J_2$ with $J_2^2=-I$ and the largest $\alpha$
   whose bias stays within a pre-declared tolerance, chosen on a *selection* seed and re-measured on a *confirmation* seed.
   A curvature-matched alternative (`nald.spectral_J`, which equalises the real parts of the drift spectrum pairwise) is
   included in the screen for comparison.

All code is in `nald.py` (samplers, potentials, metrics, self-tests) and `experiments.py` (the study). This notebook
verifies the implementation, then loads and discusses the results produced by `python experiments.py`.
""")

code(r"""import json, os, sys, math
import numpy as np, pandas as pd
from IPython.display import Image, display
sys.path.insert(0, os.getcwd())
import nald
from nald import *
RESULTS, FIGURES = "results", "figures"
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("numpy", np.__version__, "| pandas", pd.__version__)""")

md(r"""## 1. Implementation checks

Every identity the samplers rest on is verified numerically: Lemma 7.1 (closed-form Gaussian smoothing of Lasso, MCP
and SCAD) against Monte Carlo and finite differences; the Lasso special case (7.10); $\nabla U_0$ against central differences;
$a=e^{U-U_0}$; the analytic Hessian; $J^\top=-J$ and the real-part equalisation of `spectral_J`; tangency $c\cdot n=0$ of the
stream current on the ball and the identity $c=e^{U}J\nabla\psi$ for $\psi=e^{-U_0}\varphi$; $\nabla\!\cdot(J\nabla\psi)=0$;
exactness of the anchored MALA reference on a Laplace target; and invariance of the second moment of the quadratic clock
under both Euler arms.""")
code(r"""checks = nald.self_test(verbose=True)""")

md(r"""## 2. Heavy tails: the exact theory (Prop. 6.7) and the Student-$t$ clock (Cor. 6.5)

Quadratic clock $q=1+|x|^2$, $d=4$, $\beta=4$, $J=\mathrm{blockdiag}(J_2,J_2)$ so $J^2=-I$. Prop. 6.7 predicts
$\sigma^2_{u,\alpha}/\sigma^2_{u,0}=1/(1+\alpha^2)$ for every linear observable. The Student-$t$ clock is Cor. 6.2 with
$\vartheta=6$, $\zeta=1.5$. Asymptotic variances are estimated as $t\cdot\mathrm{Var}_r(\bar g_r)$ over $R$ independent chains
(relative standard error $\approx\sqrt{2/R}$).

The top row uses the plain Euler chord (7.1). The bottom row applies the drift flow exactly,
$x\mapsto e^{-\eta c(x)(I+\alpha J)}x$, which is available because $a\nabla U_0=c(x)\,x$ for these clocks: the chord of a rotation
inflates $|x|$ by $\sqrt{1+(\eta c\alpha)^2}$ per step, and in a heavy tail $c\,\eta$ is not small. This is the stiffness of
Remark 5.4 made concrete, and the exact rotation removes it.""")
code(r"""e1 = pd.read_csv(f"{RESULTS}/e1_heavy_tailed.csv")
display(Image(f"{FIGURES}/e1_heavy_tailed.png", width=1000))
t = e1[(e1.observable == "x1")].pivot_table(index=["model", "integrator"], columns="alpha", values="sigma2")
t0 = t[0.0]
ratio = (t.T / t0).T
print("asymptotic-variance ratio sigma^2_alpha / sigma^2_0 for g = x_1  (theory 1/(1+alpha^2) for the quadratic clock):")
display(ratio.round(3))
print("theory:", {a: round(1 / (1 + a**2), 3) for a in ratio.columns})
print("\nsecond moment E|x|^2 of the chain (exact 1.000 / 6.000):")
display(e1[e1.observable == "x1"].pivot_table(index=["model", "integrator"], columns="alpha", values="second_moment").round(3))""")
md(r"""**Reading.** For every non-radial observable the variance falls with $\alpha$ and the linear ones follow $1/(1+\alpha^2)$
to within a few percent; at $\alpha=3$ the non-reversible chain delivers about ten times the effective sample size of the
reversible one *per gradient evaluation*. The radial observable $|x|^2$ lies in $\ker A$ (Remark 6.8), so the theorem promises
nothing for it, and with the Euler chord its variance and its mean actually deteriorate as $\alpha$ grows; the exact rotation
restores the theoretical picture (flat), and keeps $E|x|^2$ at its $\alpha=0$ value for all $\alpha$.""")

md(r"""## 3. Unconstrained sampling: Bayesian regression with Lasso, MCP and SCAD penalties

$U(w)=\|y-Xw\|^2/2+\sum_i p(w_i)$, $n=100$, $d=20$, AR(1)-correlated design ($\rho=0.9$, condition number of the posterior
precision in the hundreds), five non-zero coefficients, $\lambda=5$, MCP $a=3$, SCAD $a=3.7$. $U_0$ replaces $p$ by its Gaussian
smoothing $p_0$ (Lemma 7.1, closed form), with the width $\mu$ chosen so that $a\ge\tfrac12$ everywhere. $\eta=0.8/\lambda_{\max}(\nabla^2U_0)$.

**Protocol.** *Selection* (data seed 7, chain seed 5): screen $J_2$ with $\alpha\in\{1,2,3,5,8\}$ and the covariance-matched
`spectral_J` with strengths 1 and 2; the rule, fixed in advance, keeps the $J_2$ arm with the largest minimum-over-coordinates
ESS among those whose maximal mean bias is at most $\max(2\times\text{reversible bias},\,0.1\,\mathrm{sd})$.
*Confirmation* (fresh data seed 8, fresh chain seed 6, two initial distributions $N(0,10I)$ and $\mathrm{Unif}(-5,5)^d$): reversible
versus the frozen $\alpha$, same $\eta$, same budget.""")
code(r"""e2 = pd.read_csv(f"{RESULTS}/e2_regression.csv")
chosen = json.load(open(f"{RESULTS}/e2_selected_alpha.json"))
print("selected alpha per penalty:", chosen)
display(Image(f"{FIGURES}/e2_selection_screen.png", width=1000))
sel = e2[e2.phase == "selection"][["penalty", "arm", "ess_min", "ess_med", "bias_rms", "bias_max", "mse_final", "nonfinite"]]
display(sel.round(4))""")
md(r"""**Reading the screen.** ESS per gradient evaluation rises steadily with $\alpha$, by roughly an order of magnitude at
$\alpha\approx3$–$5$, while the bias of the posterior mean stays at a few hundredths of a posterior standard deviation until
$\alpha$ becomes large enough for the Euler chord to distort the law. The curvature-matched $J$ helps but by less than a strong
uniform rotation does: on a non-Gaussian, kinked target the Hessian pairing is matched to the wrong operator, whereas Theorem 5.1
rewards strength in every direction.""")
code(r"""conf = e2[e2.phase == "confirmation"].copy()
conf["gain_ess_min"] = conf.groupby(["penalty", "init"])["ess_min"].transform(lambda s: s / s.iloc[0])
conf["gain_ess_med"] = conf.groupby(["penalty", "init"])["ess_med"].transform(lambda s: s / s.iloc[0])
conf["mse_ratio_rev_over_nrev"] = conf.groupby(["penalty", "init"])["mse_final"].transform(lambda s: s.iloc[0] / s)
display(conf[["penalty", "init", "arm", "ess_min", "ess_med", "gain_ess_min", "gain_ess_med", "bias_rms", "bias_max",
              "mse_final", "mse_ratio_rev_over_nrev"]].round(4))
for init in ("normal", "uniform"):
    display(Image(f"{FIGURES}/e2_confirmation_{init}.png", width=1000))""")
md(r"""**Reading the confirmation.** With the configuration frozen and everything re-seeded, the non-reversible arm keeps its
ESS advantage and reaches the exact reference faster (energy distance) from both initial distributions. The MSE of the
running posterior-mean estimate at the full budget is several times smaller: at a fixed number of gradient evaluations,
the non-reversible chain is the better estimator of the posterior mean, on all three penalties, including the non-convex
MCP and SCAD.""")

md(r"""## 4. Constrained sampling: $\ell_1$-logistic regression on the ball (Titanic)

The same model as the earlier notebook: intercept prior $N(0,10^2)$, $\lambda=0.01\,n$, $\delta=\log 2/(9\lambda)$ so
$a\in[\tfrac12,1]$, $K=\{|w|^2\le2\}$, projected Euler. Three circulations are screened: the stream construction with $J_2$
at $\alpha\in\{1,2,4,8\}$, the stream construction with the covariance-matched $J$, and the cross-product tensor of the earlier
notebook with $s\in\{5,20\}$. Selection rule and confirmation as in Section 3 (fresh chain seed; the data set is fixed).
The Bayesian-model-average test accuracy is listed only as a sanity check: it is a stationary functional and both arms must
agree on it.""")
code(r"""e3 = pd.read_csv(f"{RESULTS}/e3_titanic.csv")
display(Image(f"{FIGURES}/e3_titanic.png", width=1100))
display(e3[["phase", "arm", "ess_min", "ess_med", "bias_rms", "bias_max", "mse_final", "proj_rate", "acc_test_bma", "acc_test_reference"]].round(4))
c = e3[e3.phase == "confirmation"]
print(f"confirmation: ESS gain (min over coords) = {c.ess_min.iloc[1]/c.ess_min.iloc[0]:.2f}x, "
      f"(median) = {c.ess_med.iloc[1]/c.ess_med.iloc[0]:.2f}x, MSE ratio rev/nrev = {c.mse_final.iloc[0]/c.mse_final.iloc[1]:.2f}x")""")
md(r"""**Reading.** On the constrained problem the projection rate rises with the circulation (the earlier notebook's observation), and
that is the cost that bounds $\alpha$ here. Within the pre-declared bias tolerance the stream construction with a constant $J$ still
gives a clear ESS gain and a lower posterior-mean MSE at equal budget, whereas the cross-product tensor, whose circulation is
tangential *and* orthogonal to the radial gradient, moves the needle much less. The BMA test accuracies of the two arms agree with
each other and with the exact reference, as they must.""")

md(r"""## 5. Summary

* Measured on the quantities the theorems are about, non-reversibility wins, and wins by a lot: roughly $10\times$ ESS per gradient
  on the heavy-tailed clocks at $\alpha=3$ (matching $1/(1+\alpha^2)$), several-fold ESS gains and several-fold lower posterior-mean MSE
  on Lasso/MCP/SCAD regression at equal cost, and a clear gain on the constrained logistic problem.
* Two ingredients made the difference relative to the earlier notebook: measuring rates instead of a stationary accuracy, and using
  a *strong constant* circulation (admissible on the ball through the stream potential of Lemma 2.3(i)) instead of the weak
  cross-product tensor.
* The binding constraint is the discretisation, exactly as Remark 5.4 anticipates: bias and projection rate grow with $\alpha$.
  For radial reference drifts an exponential integrator removes the artefact entirely; for general potentials the practical rule is
  the largest $\alpha$ whose bias stays inside a tolerance chosen in advance.
* Nothing here contradicts the earlier notebook: last-iterate accuracy is stationary and cannot separate the arms. It simply is not
  what non-reversibility improves.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(HERE, "beating_reversible_anchored_langevin.ipynb")
nbf.write(nb, out)
print("wrote", out)
if "--execute" in sys.argv:
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", out], check=True, cwd=HERE)
    print("executed")
