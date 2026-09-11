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
| constant | 2.95× |
| radial, decaying in $q$ | 0.40× |
| radial, decaying faster | 0.40× |
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
| Euler, constant | 4.95 | 0 | 7.43e-4 | 348 | 7.72× |
| Euler, constant | 19.8 | 0 | 1.21e-4 | 1359 | 1.98× |
| Cayley, constant | 9.9 | 0 | 6.49e-4 | 232 | 11.57× |
| Cayley, constant | 19.8 | 0 | 6.44e-4 | 202 | **13.29×** |
| Euler, stream | 4.95 | −0.9 | 8.69e-4 | 134 | 20.04× |
| Euler, stream | 4.95 | −2.0 | 8.92e-4 | 59 | 45.51× |
| Euler, stream | 4.95 | −4.0 | 7.79e-4 | 45 | **59.67×** |
| Cayley, stream | 9.9 | −0.9 | 6.44e-4 | 102 | 26.32× |

Two separate effects, and they compose. The integrator alone takes the constant
field from 7.7× to 13.3×. State dependence alone takes it from 7.7× to 59.7×
with the *same* explicit Euler scheme.

## 5. The same result on the paper's own metric

The table above uses a slow-direction proxy. Repeating the top configurations
with the sliced 2-Wasserstein distance against exact quantiles — the paper's
Section 6.4 metric — with 5000 particles and 8 replications, counting iterations
to twice the estimator floor (0.075 ± 0.018):

| method | iterations to 2× floor | final $W_2$ | speed-up |
|---|---|---|---|
| Euler, $J = 0$ | 2470 | 0.069 | 1.00× |
| Euler, constant $\|J\|$=4.95 | 350 | 0.073 | 7.06× |
| Cayley, constant $\|J\|$=19.8 | 246 | 0.082 | 10.04× |
| Euler, stream $a=-2.0$ | 59 | 0.076 | **41.9×** |

Every final $W_2$ sits inside the floor's uncertainty, so the comparison really
is at equal accuracy and the fast methods are converged, not merely passing
through.

## 6. Exact numbers for the constant field

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

## 7. Threats to validity

* **Two accuracy criteria disagree.** Matching the stationary *covariance* and
  matching stationary *quantiles* pick different stepsizes for a skew method,
  because its bias sits mostly in the tail. Covariance is exactly computable but
  needs fourth moments to estimate, which at $\nu = 5$ makes the sample version
  very noisy; the quantile criterion is what the simulations use, calibrated to
  agree with the covariance criterion at $J=0$. Under the covariance criterion
  the constant-field Euler speed-up is 6.8× rather than 7.7×.
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
