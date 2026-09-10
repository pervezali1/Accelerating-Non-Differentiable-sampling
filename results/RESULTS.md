# Results

All numbers below are reproduced by the scripts in `experiments/`; the raw JSON
is in `results/data/` and the figures in `results/figures/`.

Two kinds of number appear and they must not be confused.

* **Exact** — from the closed-form second-moment analysis of the discretisation
  (`skewanchor/analysis.py`).  No Monte Carlo error.
* **Measured** — from ensemble simulation, with bootstrap bands over
  replications and always shown against the estimator's own noise floor.

---

## 1. The paper's own heavy-tailed target is inert to $J$

Section 6.4 of the paper uses $\pi(x)\propto(1+\|x\|^2)^{-\iota}$ with
$\iota = 2$, anchor $\beta = 1$, $\eta = 0.01$, 5 000 particles.  Because
$\iota > 1 + d/2$, that forces $d = 1$, where the only skew-symmetric matrix is
$0$.  Lifting the same family to $d = 3$ makes skew matrices available but the
target stays radial, and the added drift then generates rotations that act
unitarily on $L^2(\pi)$.

Exact (`exp1`):

| target | gap at $J=0$ | ceiling over all skew $J$ | best speed-up |
|---|---|---|---|
| $d=1$, $\iota=2$ | 2.000 | 2.000 | 1.0000× |
| $d=3$, $\iota=3$ | 4.000 | 4.000 | 1.0000× |
| $d=5$, $\iota=4$ | 6.000 | 6.000 | 1.0000× |
| $d=10$, $\iota=7$ | 12.000 | 12.000 | 1.0000× |

This is a negative control, not a failure: it says the skew extension has to be
tested on **anisotropic** heavy-tailed targets, which is what the rest does.

### Measuring on this target at all is hard

The $\nu = 3$ tail makes the empirical 2-Wasserstein distance a slow estimator.
Its floor — the same statistic evaluated on exact i.i.d. draws — is

| $n$ | 500 | 1 000 | 5 000 | 20 000 |
|---|---|---|---|---|
| $W_2$ floor | 0.337 | 0.316 | 0.265 | 0.186 |

decaying like $n^{-1/6}$, exactly the rate the tail index predicts.  At the
paper's $n = 5\,000$ nothing below $\approx 0.23$ is signal.  A naive midpoint
quantile estimator reports 0.138 for the same samples — 1.7× smaller — because
it truncates the two tail cells.  Both estimators are implemented
(`metrics.axis_sliced_w2` and `metrics.sliced_w2_midpoint`); the tail-resolving
one is the default.

---

## 2. Anisotropic heavy tails: the exact speed-up

Student-t targets, comparison at **equal stationary covariance bias (2 %) and
equal cost per iteration**.  Exact (`exp1`).

| $\kappa(\Sigma)$ | $d=2$ | $d=3$ | $d=5$ | $d=10$ |
|---|---|---|---|---|
| 1 | 1.00× | 1.00× | 1.00× | 1.00× |
| 3 | 1.00× | 1.01× | 1.02× | 1.03× |
| 10 | 1.20× | 1.19× | 1.13× | 1.16× |
| 30 | 2.38× | 1.82× | 1.35× | 1.21× |
| 100 | 6.83× | 3.57× | 2.08× | 1.53× |
| 300 | 19.6× | 7.41× | 3.82× | 2.01× |
| 1000 | 64.2× | 20.3× | 9.49× | 4.91× |

Two readings.  The gain grows fast with the condition number and **falls with
dimension** — at $d = 10$, $\kappa = 100$ it is only 1.53×.  And it is far below
what the continuous-time spectral gap would suggest: at $d=2$, $\kappa=1000$ the
gap ceiling is 500×, the realisable per-iteration gain 64×.

The result is not sensitive to the bias level held fixed: 6.64× at 0.5 % bias,
6.83× at 2 %, 7.19× at 5 % ($d=2$, $\kappa=100$).

---

## 3. Where the acceleration goes (`exp5`)

For $d=2$, $\nu=5$, $\kappa=100$, along the optimal $J$ direction:

| $\|J\|_2$ | SDE second-moment rate | ×  | equal-bias $\eta$ | per-iteration rate | × |
|---|---|---|---|---|---|
| 0 | 1.499 | 1.0 | 1.07e-3 | 0.00160 | 1.00 |
| 1.48 | 4.794 | 3.2 | 8.97e-4 | 0.00428 | 2.68 |
| 3.46 | 18.78 | 12.5 | 5.18e-4 | 0.00967 | 6.05 |
| 4.95 | 32.86 | 21.9 | 3.36e-4 | 0.01092 | **6.83** |
| 8.41 | 51.07 | 34.1 | 1.45e-4 | 0.00729 | 4.56 |
| 14.85 | 57.78 | 38.5 | 4.6e-5 | 0.00289 | 1.81 |

The **dynamics** keeps getting faster as $\|J\|$ grows — up to 38× here, and up
to 380× at $\kappa = 1000$.  The **algorithm** does not, because the larger
drift shrinks the mean-square-stable stepsize.  There is an interior optimum,
and a criterion that ignores the stepsize will sit far to the right of it.

The isotropic control passes here too: the SDE rate is 2.0000 both at $J=0$ and
at $\|J\|=4$.

---

## 4. Simulation confirms it (`exp2`)

$d=2$, $\nu=5$, $\kappa=100$, 5 000 particles, prior $\mathcal N(0,10I)$,
10 replications, equal-bias stepsizes.  The estimator floor is
$0.075 \pm 0.018$.

| $\|J\|_2$ | $\eta$ | predicted rate | iterations to $2\times$ floor | iterations to 10 % slow-direction error | final $W_2$ |
|---|---|---|---|---|---|
| 0 | 1.07e-3 | 0.00160 | 2233 | 3316 | 0.091 |
| 0.49 | 1.05e-3 | 0.00195 | 2233 | 2721 | 0.087 |
| 1.48 | 8.97e-4 | 0.00428 | 1234 | 1503 | 0.077 |
| 2.47 | 6.95e-4 | 0.00731 | 831 | 1012 | 0.068 |
| 3.46 | 5.18e-4 | 0.00967 | **682** | **831** | 0.068 |
| 4.95 | 3.36e-4 | 0.01092 | 682 | 831 | 0.072 |
| 8.41 | 1.45e-4 | 0.00729 | 1234 | 1234 | 0.073 |

No replication diverged.  The measured gain is **4.0×** on the slow-direction
metric and **3.3×** on Wasserstein, against 6.8× predicted asymptotically — the
transient is not purely the asymptotic rate, so the measured number is the
conservative one.  All final $W_2$ values sit inside the floor's uncertainty,
which is the check that the equal-bias protocol really did equalise accuracy.

At a **common** stepsize instead ($\eta = 1.07\times10^{-3}$ for everyone), the
skew variant reaches $2\times$ floor in 254 iterations against 2233 — 8.8× —
but at a larger bias, which is precisely why the equal-bias protocol is the one
to quote.

### Against tuned baselines

Each method at its own best stepsize from a grid (2 000 particles, floor 0.089;
the whole grid is saved, not just the winner).  Selection is by iterations to
$2\times$ floor, so it favours speed over final accuracy — both columns are
shown for that reason.

| method | best $\eta$ | iterations to $2\times$ floor | final $W_2$ |
|---|---|---|---|
| skew-anchored $\|J\|=4.95$ | 4.3e-3 | **169** | 0.224 |
| MALA | 6.9e-2 | 516 | 0.101 |
| anchored ($J=0$) | 4.3e-3 | 645 | 0.103 |
| underdamped Langevin | 6.9e-2 | 1259 | 0.086 |
| ULA | 1.7e-2 | 3841 | 0.130 |
| skew-ULA | 1.7e-2 | never | 0.404 |

Read honestly, this says three things.  The skew variant is the fastest to a
given accuracy, by 3.8× over the reversible anchored method.  A **well-tuned
MALA is a strong baseline** on this target — better than anchored at $J=0$ —
so the anchored family's advantage over Metropolis-adjusted methods is smaller
than a common-stepsize comparison suggests.  And the unscaled skew perturbation
of ULA (drift $J\nabla U$ rather than $e^{U-U_0}J\nabla U_0$) is actively
harmful here: it never reaches the floor.

---

## 5. Cost per iteration is identical

For the log-quadratic family the step folds ``J`` into a precomputed matrix,
``M = ((J - I) Sigma^{-1})^T``, and then does one ``z @ M`` per iteration --
exactly the operation the ``J = 0`` step already does.  Adding the skew matrix
costs nothing per step, so "per iteration" and "per unit work" are the same
comparison here.  The measured wall-clock times bear this out: across the seven
equal-bias configurations the per-run times were 59-81 s with no trend in
``||J||``.

For a general anchored potential (`samplers.generic_skew_anchored_step`) the
skew term does add one matrix-vector product per step, which is negligible
beside the gradient evaluation.

## 6. Threats to validity

* **Equal covariance bias is not equal bias in every functional.**  The protocol
  equalises the second moment exactly; higher moments are only equalised to the
  extent the final $W_2$ values agree, which here they do to within the floor.
* **The exact analysis needs the canonical anchor.**  It relies on
  $\beta = \iota - 1$, which makes the drift linear.  For other $\beta$, and for
  the composite non-smooth target, stepsizes are chosen empirically.
* **Dimension.**  Everything is $d \le 10$.  The trend with $d$ is unfavourable
  and we do not claim the method helps in high dimension.
* **The optimal $J$ needs $\Sigma$.**  `lnp_optimal` is built from
  $\Sigma^{-1}$.  A practitioner without it would need to estimate the
  covariance online; that is not studied here.
* **Baselines.**  ULA, skew-ULA, MALA and underdamped Langevin are included, each
  at its own tuned stepsize with the full grid saved.  Well-tuned ULA is much
  better than ULA at the anchored method's stepsize, and tuned MALA beats
  anchored at $J=0$, so a common-stepsize comparison flatters the anchored
  family; the tuned columns are the honest ones.
* **The transient gets worse before it gets better.**  Starting from
  $\mathcal N(0,10I)$, the multiplicative noise $q^{1/2}$ inflates the ensemble
  first, and the rotation mixes the badly scaled fast direction's excess into
  the slow one, so a larger $\|J\|$ overshoots further before converging much
  sooner.  The net effect is still a gain, but the curves are not monotone and
  a summary taken early in the run would mislead.
* **Detecting divergence needs a magnitude test.**  A mean-square-unstable
  scheme can grow geometrically to $10^{29}$ and stay finite for a whole run.
  The isotropic control at $\|J\|=8$ and the paper's $\eta = 0.01$ does exactly
  this, as the exact analysis predicts; runs are now flagged by magnitude as
  well as by non-finiteness.
