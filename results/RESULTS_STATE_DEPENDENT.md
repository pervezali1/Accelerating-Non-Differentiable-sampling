# State-dependent skew fields: results

Second round. The first round used a constant skew matrix $J$ and got 4–7× on
the anisotropic heavy-tailed target. This round asks what a *state-dependent*
$J$ buys, and finds that the answer depends entirely on doing it the right way.

Setting throughout: $d = 2$, $\nu = 5$, $\kappa(\Sigma) = 100$, anchored
Langevin with the canonical anchor. Every method is run at a stepsize chosen so
that all methods sit at the **same accuracy**, then compared on how many
iterations they need from the $\mathcal N(0, 10I)$ prior. Cost per iteration is
the same for all of them.

## 1. The diagnosis that reframed the problem

The first round assumed the stepsize was limited by stability. It is not:

| $\|J\|_2$ | stability limit | equal-bias stepsize | ratio |
|---|---|---|---|
| 0 | 1.50e-2 | 1.07e-3 | 14× |
| 3.46 | 1.44e-2 | 5.18e-4 | 28× |
| 4.95 | 1.29e-2 | 3.36e-4 | 38× |
| 14.85 | 2.61e-3 | 5.1e-5 | 51× |

The binding constraint is **discretisation bias**, and the part of it that grows
with $\|J\|$ is explicit Euler's error on a rotation, which amplifies by
$\sqrt{1+\theta^2}$ per step. That single observation is what the rest follows
from.

## 2. Naive state dependence makes things worse

Keeping the drift in the form $e^{U-U_0}J(x)\nabla U_0$ forces
$\langle\operatorname{div}J,\nabla U_0\rangle = 0$, whose complete solution in
two dimensions is a radial profile $\psi(q)J_0$. At matched average rotation
strength:

| field | speed-up |
|---|---|
| constant | 7.72× |
| radial, decaying in $q$ | 1.15× |
| radial, concentrated on the mode | 0.66× |
| radial, growing in $q$ | diverged |

That is the expected answer, not a bug. At fixed mean rotation strength the
acceleration goes with the mean of $\psi$ and the bias with its second moment,
so spreading $\psi$ out can only lose. And $q$ cannot tell the stiff axis from
the soft one, so a radial profile has no way to be selective about *where* it
rotates.

## 3. The correction, and what it unlocks

Adding $-e^{U-U_0}\operatorname{div}J$ removes the admissibility condition
entirely (see [`../docs/STATE_DEPENDENT.md`](../docs/STATE_DEPENDENT.md)). In
$d = 2$ the complete family is then $c = e^{U}J_0\nabla\Phi$ for an arbitrary
stream function. Taking

$$\Phi = -\delta\,q^{-\beta}\Big(1 + a\,\frac{u_1^2-u_2^2}{q}\Big),
\qquad u = \Sigma^{-1/2}x/\sqrt\nu,$$

the tilt $a$ makes the rotation stronger on the soft axis ($a < 0$) or the stiff
one ($a > 0$); $a = 0$ is the constant field.

**Why the sign matters.** The rotation's job is to carry mass from the soft axis
onto the stiff one, where the reversible drift — a hundred times faster there —
contracts it. Rotating *while on the stiff axis* undoes that, carrying mass back
out before it has been contracted. A constant field does both in equal measure.
A field tilted towards the soft axis is a ratchet: it feeds the fast direction
and then gets out of the way. `results/figures/fig8_flow_field_light.png` shows
the two flows.

## 4. Measured speed-ups

Iterations from the prior until the slow direction's variance is within 10 % of
the truth, at matched accuracy, relative to Euler with $J=0$ (2685 iterations):

| method | $\|J\|_2$ | tilt $a$ | $\eta$ | iterations | speed-up |
|---|---|---|---|---|---|
| Euler, $J=0$ | 0 | — | 1.07e-3 | 2685 | 1.00× |
| Euler, constant | 4.95 | 0 | 8.30e-4 | 348 | 7.72× |
| Euler, constant | 9.9 | 0 | 4.18e-4 | 399 | 6.73× |
| Euler, constant | 19.8 | 0 | 1.15e-4 | 1359 | 1.98× |
| Cayley, constant | 9.9 | 0 | 7.09e-4 | 232 | 11.57× |
| Cayley, constant | 19.8 | 0 | 6.66e-4 | 202 | **13.29×** |
| Euler, stream | 4.95 | +0.5 | 7.11e-4 | 789 | 3.40× |
| Euler, stream | 4.95 | −0.5 | 8.97e-4 | 202 | 13.29× |
| Euler, stream | 4.95 | −1.0 | 9.29e-4 | 102 | 26.32× |
| Euler, stream | 4.95 | −2.0 | 9.38e-4 | 59 | 45.51× |
| Euler, stream | 4.95 | −3.0 | 9.34e-4 | 52 | 51.63× |
| Euler, stream | 9.9 | −2.0 | 5.91e-4 | 45 | 59.67× |
| Cayley, stream | 4.95 | −6.0 | 4.69e-4 | 52 | 51.63× |
| Cayley, stream | 9.9 | −4.0 | 4.44e-4 | 39 | 68.85× |
| Cayley, stream | 9.9 | −3.0 | 6.34e-4 | 34 | **78.97×** |

The two effects compose almost exactly: the integrator is worth about 1.7×
(7.7 → 13.3 with a constant field) and the tilt about 6× (13.3 → 79.0 with the
same integrator).

The tilt curve has a clear interior optimum near $a \approx -3$: tilting the
wrong way ($a>0$) is worse than a constant field, and tilting too far eventually
turns back down.

Two separate effects, and they compose. The integrator alone takes the constant
field from 7.7× to 13.3×. State dependence alone takes it from 7.7× to 59.7×
with the *same* explicit Euler scheme. Together: **79.0×**.

## 5. Matched effective strength: the comparison that isolates state dependence

A tilted field rotates harder in some places than a constant one, so part of the
gain above could be "more rotation" rather than "better-placed rotation". To
separate them, define the effective strength of a field as the root-mean-square
drift magnitude under the target, expressed as the equivalent constant $\|J\|$,
and compare fields with the same effective strength under the same integrator
(Cayley):

| effective $\|J\|$ | constant field | tilted stream field |
|---|---|---|
| 10.5 | 10.1× (at $\|J\|$ = 14.9) | **45.5×** ($a = -4$) |
| 14.7 | 10.1× (at $\|J\|$ = 14.9) | **51.6×** ($a = -6$) |
| 19.1 | 11.6× (at $\|J\|$ = 24.8) | 45.5× ($a = -8$) |

So at equal rotation strength, equal accuracy, equal cost per iteration and the
same integrator, placing the rotation well is worth a further **4 to 5 times**.
That is the part of the result that is genuinely about state dependence.

Resolution: convergence is read off a log-spaced grid with a ratio of 1.15
between points, and the stepsize is set from a bias estimate with a few per cent
of noise, so individual entries carry roughly 20 % uncertainty. The constant
column varies between 8.8× and 13.3× across all strengths measured, which is
the size of that noise; the gap to the tilted column is an order of magnitude
larger.

## 6. The same result on the paper's own metric

The table above uses a slow-direction proxy. Repeating the top configurations
with the sliced 2-Wasserstein distance against exact quantiles — the paper's
Section 6.4 metric — with 5000 particles and 8 replications, counting iterations
to twice the estimator floor (0.075 ± 0.018):

| method | iterations to 2× floor | final $W_2$ | speed-up |
|---|---|---|---|
| Euler, $J = 0$ | 2470 | 0.069 | 1.00× |
| Euler, constant $\|J\|$=4.95 | 350 | 0.073 | 7.06× |
| Cayley, constant $\|J\|$=19.8 | 246 | 0.082 | 10.04× |
| Cayley, stream $a=-0.9$, $\|J\|$=9.9 | 101 | 0.078 | 24.5× |
| Euler, stream $a=-2.0$, $\|J\|$=4.95 | 59 | 0.076 | **41.9×** |

Every final $W_2$ sits inside the floor's uncertainty, so the comparison really
is at equal accuracy and the fast methods are converged, not merely passing
through.

## 7. Bias audit: the criteria are not interchangeable

At each method's chosen stepsize, four ways of asking how accurate it is:

| method | Frobenius covariance | worst per-direction variance | quantiles 50–90 | quantile 99 |
|---|---|---|---|---|
| Euler, $J=0$ | 0.017 | 0.076 | 0.043 | 0.032 |
| Euler, constant $\|J\|$=4.95 | 0.049 | 0.095 | 0.045 | 0.047 |
| Cayley, constant $\|J\|$=19.8 | 0.031 | 0.099 | 0.047 | 0.051 |
| Euler, stream $a=-3$ | 0.105 | 0.106 | 0.048 | 0.073 |
| Euler, stream $a=-2$, $\|J\|$=9.9 | 0.071 | 0.104 | 0.047 | 0.056 |

The criterion the comparison equalises — quantiles 50 to 90 — is matched to
within 10 % across every method, as it should be. **Nothing else is.** The
covariance error of the tilted field is six times the baseline's, and its
99th-percentile error twice. So the faster methods are not uniformly as accurate
as the baseline: their error is concentrated in the tail, and the bulk criterion
cannot see it.

For the baseline the worst direction is the stiff one, which barely enters the
Frobenius norm; for the tilted field the worst direction is the soft one, which
dominates it. That is why the two criteria rank the methods differently, and it
is why Section 8 reports the speed-ups under the tail-sensitive criterion as
well.

## 8. The same comparison under the tail-sensitive criterion

Re-selecting every stepsize so that the **stationary covariance** error is 2 %
instead, and verifying the achieved value rather than trusting the
extrapolation:

| method | $\eta$ | achieved covariance bias | iterations | speed-up |
|---|---|---|---|---|
| Euler, $J=0$ | 1.27e-3 | 0.0203 | 2374 | 1.00× |
| Euler, constant $\|J\|$=4.95 | 3.40e-4 | 0.0186 | 678 | 3.50× |
| Cayley, constant $\|J\|$=19.8 | 4.30e-4 | 0.0192 | 362 | 6.56× |
| Cayley, stream $a=-3$, $\|J\|$=9.9 | 1.09e-4 | 0.0171 | 319 | 7.44× |
| Euler, stream $a=-3$, $\|J\|$=4.95 | 1.78e-4 | 0.0146 | 219 | **10.84×** |

**The ordering survives, the magnitudes do not.** A tilted state-dependent field
is still the best method, still beats the best constant field, and still beats
it by more than the integrator does on its own. But 79× at matched bulk accuracy
becomes 10.8× at matched covariance accuracy. Both numbers are real; they answer
different questions.

Which to quote depends on the application. If the tail is the point — and on a
heavy-tailed target it usually is — the 10.8× figure is the honest one. Against
the first round's best constant field under this same criterion (3.5× measured,
6.8× exact), state dependence is worth roughly a further 2 to 3 times.

Two details. The two stream rows came in *under* the 2 % target (0.0146 and
0.0171), so they were given slightly conservative stepsizes and 10.84× is if
anything a small underestimate. And under this criterion the Cayley variant is
*worse* than Euler for the tilted field, because the bias is then dominated by
the nonlinear remainder that the split step takes explicitly — advancing the
linear part exactly no longer buys anything.

## 9. Exact numbers for the constant field

For a constant field the second-moment recursion is exact for any integrator, so
these involve no Monte Carlo at all. Speed-up at 2 % stationary covariance bias:

| target | Euler | Cayley | exponential | gap ceiling |
|---|---|---|---|---|
| $d=2$, $\kappa=100$ | 6.8× | 17.2× | 20.4× | 50× |
| $d=2$, $\kappa=1000$ | 64.2× | 169.9× | 204.9× | 500× |
| $d=5$, $\kappa=100$ | 2.1× | 4.7× | 5.5× | 29× |
| $d=10$, $\kappa=100$ | 1.6× | 3.8× | 4.9× | 25× |

The integrator roughly triples the exact speed-up everywhere, and closes about
40 % of the gap to the continuous-time ceiling.

## 10. Threats to validity

* **Two accuracy criteria disagree, by a factor of seven.** 79× at matched bulk
  accuracy, 10.8× at matched covariance accuracy, for the same method. Sections
  7 and 8 give both. The ordering of methods is the same under either, so the
  qualitative conclusion is safe; any single speed-up number is not, unless the
  criterion is stated with it.
* **The tilt sweep is not closed.** The speed-up is still improving at the edge
  of the range in places, and there is no analogue of the
  $\operatorname{Tr}(A)/d$ ceiling for a state-dependent field, so how far the
  tuned member is from optimal is unknown.
* **Two dimensions only.** The stream-function family is a $d=2$ construction.
  In $d \ge 3$ the divergence-free curl family exists and is implemented, but is
  untuned.
* **The tilt needs to know $\Sigma$.** As does the optimal constant field.
  Estimating it online is not studied.
* **State dependence costs exact integrability.** A constant field gives a
  linear drift that the exponential integrator advances exactly; a tilted field
  does not, so only its constant part is advanced exactly and the remainder is
  explicit. That is why Euler plus a large tilt can beat Cayley plus a small one.
