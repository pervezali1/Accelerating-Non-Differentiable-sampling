# State-dependent skew fields

The first round used a constant skew matrix $J$. This note works out what
happens when $J$ is allowed to depend on $x$: what is admissible, what the
complete families are, and what actually helps.

## 1. The naive condition, and why it is the wrong one

Keep the drift in the form the constant case uses,
$c(x) = e^{(U-U_0)(x)} J(x)\nabla U_0(x)$. Writing $\rho_0 = e^{-U_0}$,

$$\operatorname{div}(\pi c) \;\propto\; \operatorname{div}\big(\rho_0 J\nabla U_0\big)
 = \underbrace{-\rho_0\langle \nabla U_0, J\nabla U_0\rangle}_{=\,0}
 + \underbrace{\rho_0\langle J, \nabla^2 U_0\rangle_F}_{=\,0}
 - \rho_0\langle \operatorname{div} J, \nabla U_0\rangle,$$

with $(\operatorname{div} J)_j := \sum_i \partial_i J_{ji}$. So the target is
preserved **iff**

$$\langle \operatorname{div} J(x), \nabla U_0(x)\rangle = 0 \quad \text{for all } x. \tag{INV}$$

A constant $J$ satisfies (INV) trivially. The next simplest family is
$J(x) = \psi(q(x))J_0$, a radial modulation of the rotation strength, which
satisfies (INV) for every profile $\psi$ because $\nabla q$ and $\nabla U_0$ are
parallel and $\langle v, J_0 v\rangle = 0$.

In $d = 2$ that family is not merely an example, it is everything. Any
$J(x) = \psi(x)J_0$ obeying (INV) needs $\psi$ constant along the orbits of
$x \mapsto J_0\Sigma^{-1}x$, and those orbits are exactly the level sets of
$q$, since $\Sigma^{-1}J_0\Sigma^{-1}$ is skew. So under (INV) the only freedom
in two dimensions is one scalar profile of one variable — and that profile is
**blind to direction**, because $q$ cannot tell the stiff axis from the soft one.

That is a bad place to be, and the experiments confirm it: at matched average
rotation strength every non-constant radial profile is *worse* than constant,
some catastrophically.

## 2. The correction that removes the condition

(INV) is an artefact of insisting on the form $e^{U-U_0}J\nabla U_0$. Add one
term and it disappears entirely:

$$\boxed{\;c(x) = e^{(U-U_0)(x)}\Big[\,J(x)\nabla U_0(x) \;-\; (\operatorname{div} J)(x)\,\Big]\;}\tag{COR}$$

preserves $\pi$ for **every** skew matrix field $J$, with no constraint.

*Proof.* Two divergences cancel:
$\operatorname{div}(\rho_0 J\nabla U_0) = -\rho_0\langle\operatorname{div}J,\nabla U_0\rangle$
from the computation above, and

$$\operatorname{div}(\rho_0 \operatorname{div}J)
 = -\rho_0\langle\nabla U_0,\operatorname{div}J\rangle
 + \rho_0\underbrace{\textstyle\sum_{ij}\partial_i\partial_j J_{ji}}_{=\,0},$$

the last sum vanishing because a symmetric second derivative is contracted with
an antisymmetric matrix. Subtracting gives $\operatorname{div}(\pi c) = 0$. $\square$

Equivalently: $\pi c = \operatorname{div} S$ for the antisymmetric matrix field
$S = -e^{-U_0}J$, and the divergence of the divergence of an antisymmetric field
is identically zero. Every $\pi$-divergence-free drift arises this way, so (COR)
is not one construction among many — it is the general one.

Two things change. The corrected drift is **no longer tangent to the level sets
of $U_0$**: $\langle\nabla U_0, c\rangle = -e^{U-U_0}\langle\nabla U_0,
\operatorname{div}J\rangle$ need not vanish. And the design space becomes the
full set of skew matrix fields.

## 3. Complete families

**In $d = 2$.** Any divergence-free planar field is a skew gradient, so

$$c(x) = e^{U(x)}\,J_0\,\nabla\Phi(x) \tag{2D}$$

for an arbitrary scalar stream function $\Phi$, and this is the complete family.
The constant field of norm $\delta$ is $\Phi = -\delta e^{-U_0}$. Writing
$u = \Sigma^{-1/2}x/\sqrt\nu$ and $q = 1 + |u|^2$, the experiments use

$$\Phi = -\delta\, q^{-\beta}\Big(1 + a\,\frac{u_1^2-u_2^2}{q} + b\,\frac{2u_1u_2}{q}\Big),$$

whose $a = b = 0$ member is the constant field. The quadrupole term $a$ tilts
the rotation towards the soft axis ($a<0$) or the stiff one ($a>0$). Neither
tilt is admissible as a bare $J(x)$ field; both become legal through (COR).

**In $d \ge 3$.** $J = \operatorname{div}A$ for a totally antisymmetric
3-tensor $A$ has $\operatorname{div}J \equiv 0$, so it satisfies (INV) with no
correction and preserves *any* target with *any* anchor. For $d = 3$ this is
$J(x)v = \nabla f(x)\times v$ for an arbitrary scalar $f$.

In $d = 3$ that family is not merely *sufficient* — it is everything:

> **The curl family is complete in $d=3$.** Skew $3\times3$ matrices and
> vectors in $\mathbb R^3$ are the same three-dimensional space, so every skew
> field is $J(x)v = a(x)\times v$ for a unique vector field $a$. By § 3b3,
> $\operatorname{div}J = \operatorname{curl}a$. Hence
> $\operatorname{div}J \equiv 0 \iff \operatorname{curl}a \equiv 0 \iff
> a = \nabla f$ for some scalar $f$ ($\mathbb R^3$ is simply connected). So the
> correction-free skew fields in $d = 3$ are *exactly* the curl fields, one per
> scalar $f$.

Choosing a correction-free field in three dimensions is therefore choosing one
scalar function, and nothing outside that family exists to be found. It also
says where the remaining freedom is: the correction term (COR) of § 2 buys
access to the fields with $\operatorname{curl}a \ne 0$, and nothing else.

The contrast with the plane is sharp rather than incidental. Skew $2\times2$
matrices are one-dimensional, $J(x) = \varphi(x)J_0$, so
$(\operatorname{div}J)_k = (J_0^{\!\top}\nabla\varphi)_k$ and, $J_0$ being
invertible, $\operatorname{div}J = 0$ forces $\nabla\varphi = 0$: **in $d = 2$
the only correction-free skew field is a constant one.** Every state-dependent
planar field in this work is legal only through (COR). That is why two
dimensions are confined to radial modulation under (INV) alone.

## 3b. The curl family in three dimensions, and what it can and cannot do

In $d \ge 3$ the field $J(x)v = \nabla f(x)\times v$ has $\operatorname{div}J
\equiv 0$ for **every** scalar $f$, so it satisfies (INV) outright: it goes
straight into the anchored drift with no correction term and preserves any
target with any anchor. Two familiar objects are members:

| $f$ | field | conserves besides $U_0$ |
|---|---|---|
| linear | a **constant** $J$ | nothing |
| $s\|x\|^2/2$ | $J_s(x)v = s\,(x\times v)$ | $\|x\|$ |
| $s\,x_{\text{stiff}}x_{\text{soft}}$ | hyperbolic | $x_{\text{stiff}}x_{\text{soft}}$ |

**Every** field of the form $c = e^{U-U_0}J\nabla U_0$ conserves $U_0$, because
$\langle\nabla U_0, J\nabla U_0\rangle = 0$ for skew $J$. The curl family
conserves a *second* function as well, namely $f$, since
$\nabla f\times\nabla U_0$ is orthogonal to both. Two conserved functions in
three dimensions pin the flow to a one-dimensional curve, and which curve
decides whether the field is any use.

For $J_s(x)v = s(x\times v)$ the second conserved quantity is $\|x\|$, and with
the canonical anchor the drift reduces exactly to the quadratic field
$c(x) = s(2\beta/\nu)(x\times\Sigma^{-1}x)$. That is an attractive object — it
vanishes identically on an isotropic target, which is precisely where no skew
perturbation can help, and grows with the anisotropy. But conserving $\|x\|$ is
fatal for the mechanism that actually accelerates. The useful transport carries
mass from the soft axis to the stiff axis **at fixed $q$**, where the reversible
drift is $\kappa$ times faster; those two points differ in $\|x\|$ by a factor
$1/\sqrt\kappa$. A field that conserves $\|x\|$ cannot connect them.

The measurements agree. On the three-dimensional anisotropic Student-t
($\nu=6$, $\kappa=100$), at stepsizes chosen to equalise accuracy with an
adaptive search, every field carrying the warm-up ramp it needs and every row
converging in all five replications ([`exp10`](../experiments/exp10_curl_potentials.py)):

| curl potential $f$ | conserves | $s$ | $\eta$ | iterations to 2× floor | speed-up |
|---|---|---|---|---|---|
| linear (a **constant** $J$), $\|J\|=6$ | — | — | 8.52e-4 | **421** | **5.36×** |
| $s\,x_{\text{stiff}}x_{\text{soft}}$ | $x_{\text{stiff}}x_{\text{soft}}$ | 0.3 | 1.14e-3 | 1153 | 1.96× |
| $s\,x_{\text{stiff}}x_{\text{soft}}$ | ″ | 0.5 | 1.11e-3 | 824 | 2.74× |
| $s\,x_{\text{stiff}}x_{\text{soft}}$ | ″ | 0.7 | 1.09e-3 | 589 | 3.83× |
| $s\,x_{\text{stiff}}x_{\text{soft}}$ | ″ | **1.0** | 1.10e-3 | **498** | **4.53×** |
| $s\,x_{\text{stiff}}x_{\text{soft}}$ | ″ | 1.5 | 1.69e-4 † | 2256 | 1.00× |
| $s\|x\|^2/2$ (cross product) | $\|x\|$ | 0.1 | 1.16e-3 | 1907 | 1.18× |
| $s\|x\|^2/2$ | $\|x\|$ | 0.3 | 1.83e-4 † | 12087 | 0.19× |
| $s\|x\|^2/2$ | $\|x\|$ | 0.5 | 1.76e-4 | 12087 | 0.19× |
| $J = 0$ | — | — | 1.17e-3 | 2256 | 1.00× |

† the equal-accuracy stepsize was unstable from the wide prior, so it was backed
off (2.5× and 6.2×) until every replication survived. A backed-off row runs at a
*smaller* stepsize than equal accuracy requires, hence more accurately: it is
charged the extra iterations and given no credit for the extra accuracy.

Every row here runs the paper's explicit Euler scheme, which is what caps them
all — see § 3b2, where changing only the integrator beats the best of them.

![curl potentials](../results/figures/fig11_curl_potentials_light.png)

The cross-product curve lies almost exactly on top of $J=0$, which is why it is
dashed.

Two things had to be got right before these numbers meant anything, and both
were got wrong first — see § 3c.

**The verdict on each potential.** The cross-product field $J_s$ is inert at
small $s$ (1.18× at $s=0.1$: the added drift is under a fifth of the reversible
one in root-mean-square) and actively harmful above it, because the quadratic
growth costs more in stepsize than the rotation returns. It never beats $J=0$
by more than the measurement noise. The diagnosis holds: it conserves $\|x\|$,
and the transport that accelerates has to change $\|x\|$ by $1/\sqrt\kappa$.

Breaking that conservation works, and works well. The hyperbolic potential
$f = s\,x_{\text{stiff}}x_{\text{soft}}$, whose level sets contain both
endpoints, does break it — $\langle x, c\rangle$ goes from $5\times10^{-13}$ to
$2.2\times10^{3}$ — and climbs monotonically with $s$ to **4.53×** at $s=1$,
four and a half times the reversible anchored sampler, with the best constant
field 18 % faster still. Past $s=1$ the equal-accuracy stepsize goes unstable and
the back-off costs more than the extra rotation returns.

So the conclusion is quantitative rather than qualitative. The best member of
the three-dimensional curl family we found is still the linear potential — a
*constant* $J$, at 5.36× under Euler and 7.50× once it is given the integrator
it can use (§ 3b2) — but it leads a tuned state-dependent member
by only 18 %, not by the factor of two it looked like before the ramp and the
divergence accounting were fixed. State dependence pays much more in two
dimensions, through the stream construction of § 3, precisely because there the
correction term of § 2 frees the field from the second conservation law
entirely. The curl family buys freedom from the correction term and pays for it
with an extra invariant.

**Independent seeds.** The ordering above is not a seed artefact;
[`exp11`](../experiments/exp11_seed_robustness.py) re-asks it of three disjoint
seed blocks, five replications each:

| field | blk 500 | blk 20500 | blk 40500 |
|---|---|---|---|
| $J=0$ | 2137 | 1844 | 1844 |
| constant $\|J\|=6$ | 364 (**5.87×**) | 364 (**5.07×**) | 422 (**4.37×**) |
| hyperbolic $s=0.5$ | 882 (2.42×) | 882 (2.09×) | 761 (2.42×) |
| hyperbolic $s=0.7$ | 657 (3.25×) | 657 (2.81×) | 657 (2.81×) |
| hyperbolic $s=1$ | 489 (4.37×) | 489 (3.77×) | 489 (3.77×) |
| hyperbolic $s\ge 1.5$ | diverges | diverges | diverges |

Every cell that reports a number had no divergences; $s \ge 1.5$ reports none
because it diverged in all three blocks. The constant field is ahead in every
block, by 1.16× to 1.34×, and the level of each speed-up moves by up to 15 %
between blocks — which is the honest error bar on any single number here.

Note the dimensions differ: the 29–79× figures elsewhere are $d = 2$ stream
fields, and are not comparable with any of these. Every number in this section
is $d = 3$ and like-for-like with every other number in it.

## 3b2. The integrator is worth more than the potential

Everything above runs the explicit Euler scheme of the paper. That is the wrong
default, and it is what caps the whole $d = 3$ comparison.

Euler amplifies a rotation by $\sqrt{1 + \theta^2}$ per step, so the stronger
the field the smaller the stepsize that equal accuracy allows, and the constant
field's optimum lands at $\|J\| \approx 3.5$. The Cayley transform has unit
modulus on the skew part and the matrix exponential integrates the linear drift
outright, so neither pays that penalty. From the exact second-moment recursion,
at 2 % stationary covariance bias — no Monte Carlo in these numbers:

| $\|J\|$ | 3.5 | 7 | 12 | 20 | 30 | 60 | 90 |
|---|---|---|---|---|---|---|---|
| Euler | **3.57×** | 2.98× | 1.94× | 0.60× | 0.34× | — | — |
| Cayley | 3.54× | 6.21× | **7.33×** | 7.03× | 6.26× | 4.41× | 3.33× |
| exponential | 3.54× | 6.32× | 7.74× | 8.07× | 8.15× | 8.19× | **8.19×** |

Euler peaks early and then *loses*; the exponential integrator climbs to a
plateau and stays there, because its stepsize is set by the noise treatment
alone and stops caring about $\|J\|$. The continuous-time rate for such a field
is 15.7× the reversible one, so the exponential integrator recovers a little
over half of what the SDE has to offer, and Euler under a quarter.

Measured, at the same equal-accuracy protocol as the table in § 3b:

| method | ramp | rise | iterations | speed-up | ms/iter | **wall clock** |
|---|---|---|---|---|---|---|
| $J = 0$, Euler | — | 1.00× | 2256 | 1.00× | 1.037 | 1.00× |
| constant $\|J\|=6$, Euler | 10 | 1.00× | 421 | 5.36× | 1.094 | 5.08× |
| **constant $\|J\|=60$, exponential** | **25** | **1.00×** | **301** | **7.50×** | **0.522** | **14.88×** |
| $f = s\,x_{\rm stiff}x_{\rm soft}$, $s=1$, Euler | 40 | 1.01× | 498 | 4.53× | 1.267 | 3.71× |

Two separate gains, and the second is the larger:

* **Fewer iterations, 7.50× against 5.36×.** The field can be run ten times
  stronger without the stepsize collapsing.
* **Cheaper iterations, 2.0× .** For a constant field on a log-quadratic target
  with the canonical anchor the drift is linear, so the whole step is
  `x @ M.T` plus noise — one $d\times d$ multiply, with no per-particle
  gradient, anchor scale or diffusion coefficient. The propagator is built once
  and reused, including after the warm-up ramp saturates.

Together, **14.9× in wall clock against 5.08×**, and the curve is monotone.

Three cautions. The stronger field has a larger transient, so it needs a
25-relaxation ramp rather than 10 — at ramp 10 it reaches the floor just as
fast but rises 1.38× on the way, and at ramp 20 it still rises 1.05×. On three
disjoint seed sets the ordering holds in every one (6.27×, 6.56×, 5.74× against
Euler's 5.50×, 5.50×, 4.42× on a grid fine enough to separate them). And the
exponential integrator does *not* help a reversible sampler: at $J=0$ it needs
a smaller stepsize for the same bias and its rate is 0.52× Euler's, which the
halved cost per iteration almost exactly cancels. It pays only against the
rotation, which is precisely the part Euler gets wrong.

**What it cannot do.** Cayley and the exponential need the drift's linear part
to be a fixed matrix, so they take a constant or a radially modulated field and
nothing else. The curl potentials of § 3b are neither, so they are stuck on
Euler — and that, not the extra conserved quantity, is now the main reason the
best of them trails. Splitting the exponential reversible part from an explicit
skew part would give them the same benefit; that is not implemented here.

## 3b3. Why the curl family is divergence free, and what has to be checked

For a field of the form $J(x)v = a(x)\times v$ the matrix is
$J_{ik} = \varepsilon_{ijk}a_j$, so

$$(\operatorname{div}J)_k = \sum_i \partial_i J_{ik}
  = \varepsilon_{ijk}\,\partial_i a_j = (\operatorname{curl}a)_k .$$

**div $J$ is the curl of the generating vector field.** It vanishes exactly
when $a$ is curl free, and $a = \nabla f$ always is: writing it out,
$(\operatorname{curl}\nabla f)_k = \varepsilon_{kij}H_{ij} = 0$ because
$\varepsilon$ is antisymmetric in $(i,j)$ while the Hessian $H$ is symmetric.

That is why every member of this family needs no correction term, with no
condition on $f$, on $M$, on the strength, or on the target:

| field | generator $a$ | potential $f$ |
|---|---|---|
| constant $J_a$ | $v$ | $\langle v, x\rangle$ |
| $J_s$ (cross product) | $s\,x$ | $s\|x\|^2/2$ |
| $J_M$ | $Mx$, $M$ symmetric | $x^\top Mx/2$ |
| improved $J_s$ | $sMx/\sqrt{r^2+x^\top Mx}$ | $s\sqrt{r^2 + x^\top Mx}$ |

For the improved field the Hessian is
$H = s\big[M/\sqrt{u} - (Mx)(Mx)^\top/u^{3/2}\big]$ with $u = r^2 + x^\top Mx$,
manifestly symmetric, so $\operatorname{div}J = 0$ identically. Measured: the
analytic Hessian's asymmetry is $0$ to the last bit, and $|\operatorname{div}J|$
by complex-step differentiation — which has no truncation error at all — is
$7\times10^{-15}$ against a field of size $30$, i.e. machine precision. A
central difference reports $\sim10^{-9}$ on the same field, and that number is
the difference scheme's error, not the field's.

$M$ must be symmetric, since only the symmetric part of a general $M$ is a
gradient; and the saturating form additionally needs $M \succeq 0$ and $r > 0$,
or the radicand goes negative and $\nabla f$ is NaN.

**The check that has to exist.** `divergence()` returns zero for these classes,
but that is a *precondition* on the generator, not a computation — a
non-gradient $a$ passed by mistake gives a sampler that quietly fails to
preserve the target, which surfaces as a wrong answer rather than an exception.
`skewfield.assert_gradient_field` now runs at construction and rejects one;
`skewfield.curl` exposes the underlying quantity. On a skew generator $a = Ax$
the check returns $|\operatorname{curl}a|/|\nabla a| = 2$ and raises.

## 3c. Two ways to fool yourself, both of which we did

The first version of the table above reported the hyperbolic potential at
$s = 1$ converging in **301** iterations, a 7.50× that beat the constant field.
It was wrong, and it is worth recording exactly how.

**The mean over replications hid the divergences.** A convergence curve here is
the mean over five chains, and `np.nanmean` stays finite as long as *one* chain
survives. At the stepsize in question, four chains out of five blew up; the
curve reported was the survivor's, and the survivor happened to be fast. Asked
of three disjoint seed sets, that configuration diverges in five replications
out of five in every one of them. The fix is to count divergences and treat a
field as stable only when *every* replication is — which is what the table
above does, and why it carries no "diverged" column: nothing in it diverged.

**The ramp was too short for a field that grows with $\|x\|$.** The warm-up
length is computed from the target's stiff relaxation time, a property of the
*linear* drift. Both non-constant potentials here have $\nabla f$ linear in
$x$, so their field strength is proportional to $\|x\|$: started from an
$\mathcal N(0, 10I)$ prior they are an order of magnitude stronger through the
transient than they will ever be at stationarity, and a hold long enough for a
constant field is not long enough for them. Ten relaxations leave a 1.18×
overshoot on $s=0.5$ and let $s\ge 0.7$ blow up; forty leave nothing and keep
the equal-accuracy stepsize usable, which is where most of the improvement in
the table comes from — $s=0.7$ goes from 1.96× to 3.83× purely by no longer
needing its stepsize backed off. Measured on $s=0.5$:

| ramp | ramp iterations | rise out of a trough | iterations to 2× floor |
|---|---|---|---|
| 10 | 77 | 1.18× | 697 |
| 20 | 154 | 1.03× | 824 |
| 40 | 308 | **1.00×** | 824 |
| 80 | 617 | 1.00× | 975 |

Unlike a constant field, where the ramp is free, here it is a real trade — and
one worth making, because forty relaxations is also what buys the stability.

## 3d. The regulariser is not a neutral bystander, and neither is the stepsize

Asked to "play with the stepsize or the regulariser $\lambda$" on the MCP
target while holding the two matrices fixed, the sweep produced two findings,
and the larger one belongs to $J = 0$.

**The stepsize was the binding protocol, not the matrix.** Every earlier
experiment here gave each field its *equal-bias* stepsize: the largest $\eta$
whose exact stationary covariance bias is 2%. That is the right protocol for
comparing convergence rates at a fixed asymptotic accuracy, but 2% is far
tighter than the accuracy a run of $n = 5000$ particles can resolve. The
sliced-$W_2$ estimator's own two-sample floor is about $0.045$ there, and the
measured plateau sits on that floor until the bias is several times 2%: at
$\lambda = 0.5$ the plateau reads $0.047$, $0.050$, $0.056$, $0.065$ for
covariance biases of $0.03$, $0.07$, $0.11$, $0.17$. Tuning each field's
stepsize against the accuracy actually being measured, $J = 0$ goes

| $\eta/\eta_0$ | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| iterations to $2\times$ floor | 2482 | 1165 | **798** | never |
| exact covariance bias ($\lambda = 1$) | 0.032 | 0.069 | 0.167 | 0.654 |

so the baseline this repository had been quoting was undertuned by $3.1\times$.
At $8\eta_0$ the bias finally overtakes the floor and the chain never reaches
the accuracy at all, which is exactly where the exact stand-in says it should
stop. Nothing about this is specific to $J$; it is a correction to the
baseline, and it applies at every $\lambda$ tested ($0.25$, $0.5$, $1$ all give
798).

**$\lambda$ moves the ceiling, but in the wrong direction, and the tridiagonal
$J_a$ cannot reach it anyway.** The smoothed penalty is separable and even, so
near the origin it adds the *same* curvature to every coordinate,

$$(p^\varepsilon_\lambda)''(0) \;=\; \frac{\lambda}{\varepsilon}
  \;-\; \frac{\lambda}{\sqrt{a^2\lambda^2+\varepsilon^2}},$$

an isotropic term in the anchor's Hessian. Replacing the penalty by that
curvature gives a log-quadratic stand-in (`nonsmooth.gaussianised`) on which
every exact quantity is available, and it shows the regulariser **rounding the
target off**:

| $\lambda$ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| effective $\mathrm{cond}$ | 100.0 | 37.4 | 21.4 | 15.1 | 11.8 |
| best speedup, any skew $J$ | 4.38× | 2.03× | 1.46× | 1.25× | 1.15× |
| best speedup, tridiagonal $J_a$ | 1.13× | 1.07× | 1.02× | 1.00× | 1.00× |

A skew perturbation earns its keep by moving mass between directions of
*different* stiffness, so what it can earn is set by the anisotropy — and the
regulariser destroys anisotropy. Since $\varepsilon$ enters only through
$\lambda/\varepsilon$, it is the stronger of the two knobs: at $\lambda = 0.5$
the effective condition number is $11.8$ for $\varepsilon = 0.05$ but $89.0$
for $\varepsilon = 0.5$. Lowering $\lambda$ therefore does restore room for a
skew field, but only for one that couples the stiff axis to the soft one
directly. The tridiagonal $J_a$ couples $x_1\!\leftrightarrow\!x_2$ and
$x_2\!\leftrightarrow\!x_3$ with the *same* strength $a$; with
$\Sigma = \mathrm{diag}(0.01, 0.1, 1)$ that is stiff$\leftrightarrow$middle and
middle$\leftrightarrow$soft, and the stiff$\leftrightarrow$middle entry is the
one that forces the stepsize down while buying almost no rate. At a 2% bias its ceiling
is $\le 1.13\times$ *everywhere* in the $(\lambda, \varepsilon)$ plane, and
loosening the bias tolerance all the way to 50% lifts that only to
$1.31\times$. The cross-product $J_s$ conserves $\|x\|$, and the stiff and soft directions
live at different $\|x\|$, so it is blocked outright at every $\lambda$.

The two findings interact in a way that is easy to misread. At a *common*
stepsize $\eta_0$ the tridiagonal field looks good — the exact rate ratio is
$1.78\times$ at $a = 2$, and the measured crossing moves from 2482 to 1292
iterations. All of that is the baseline's slack, not the field: give $J = 0$
the same freedom and it reaches the accuracy in 798 iterations.

Tuned against tuned at $\lambda = 0.5$, the sweep's single best cell for the
tridiagonal field is $a = 1$ at $2.83\eta_0$: 620 iterations against 798, a
$1.29\times$. That number is a measurement artefact, and it is worth being
careful about, because it is the sort of thing that gets reported.

The exact stand-in, maximising the rate over *all* stepsizes at each bias
tolerance, allows the tridiagonal at most

| bias tolerance | 0.02 | 0.05 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|---|
| best over every $a$ | 1.02× | 1.04× | 1.04× | 1.08× | 1.12× |
| at $a = 1$ | 0.91× | 0.94× | 0.95× | 1.02× | 1.10× |

so there is no accuracy at which $1.29\times$ is on offer. The measurement
agrees once it is asked more than once. A crossing is read off a log-spaced
grid whose points are 9% apart, from a few chains against a single reference
draw, and the estimator floor that sets the threshold is itself a heavy-tailed
average — it reads 0.0442, 0.0450, 0.0577 on three seed blocks. Re-running the
*identical* $J = 0$ configuration at $2.83\eta_0$ on two of those blocks gives
1165 and 704 iterations: the noise on one configuration is $1.65\times$, larger
than any field-versus-$J = 0$ difference in the sweep. Taking only ratios
measured against $J = 0$ *inside* the same block, across three seed blocks, both
priors and four accuracy levels (`exp16 --aggregate`):

| field | cells | geometric mean | range |
|---|---|---|---|
| $J_a$, $a = 1$ | 15 | **1.03×** | 0.68–1.29× |
| $J_a$, $a = 2$ | 8 | 0.69× | 0.41–1.13× |
| $J_s$, $s = 0.05$ | 15 | 0.84× | 0.41–1.00× |
| $J_s$, $s = 0.1$ | 8 | 0.54× | 0.28–0.88× |

$1.03\times$ for the best tridiagonal setting, against an exact ceiling of
$1.02$–$1.12\times$. The two agree, and neither is worth having.

So the answer to the question as asked is that the stepsize is worth $3.1\times$
and belongs to $J = 0$; that $\lambda$ sets how much a skew field could win but
the tridiagonal $J_a$ is capped near 1 wherever $\lambda$ is put; and that
$J_s$, which conserves $\|x\|$, is not neutral but mildly harmful — at
$s = 0.05$ it reproduces $J = 0$'s crossings (2482, 1499, 1165, 798) and then
loses its stability a grid point sooner, and at $s = 0.1$ it is half the speed.

## 3e. The single plane that does work

Everything in §3d says the tridiagonal $J_a$ is capped near 1 because it spends
half its strength on the stiff$\leftrightarrow$middle pair. The obvious repair
is to spend none of it there:

$$J_{13} = \begin{pmatrix} 0 & 0 & a \\ 0 & 0 & 0 \\ -a & 0 & 0\end{pmatrix},
\qquad \Sigma = \mathrm{diag}(0.01,\,0.1,\,1),$$

one plane, stiffest axis against softest. Same target, same MCP penalty, same
protocol: every field including $J = 0$ tuned over the stepsize against a fixed
sliced-$W_2$ accuracy, five replications, and the whole thing repeated on a
disjoint seed block. Taking only within-block ratios, over two seed blocks and
four accuracy levels, from the $N(0, 10I_d)$ start:

| field | $\lambda = 0.25$ | $\lambda = 0.5$ | exact rate ratio at $4\eta_0$ ($\lambda = 0.25$) |
|---|---|---|---|
| $J_{13}$, $a = 1$ | 1.85× (1.46–2.42) | 1.81× (1.29–2.74) | 1.94× |
| $J_{13}$, $a = 1.3$ | 2.20× (1.46–3.11) | 2.33× (1.88–2.74) | 2.50× |
| $J_{13}$, $a = 2$ | **3.11×** (1.88–4.53) | **2.95×** (2.13–4.53) | 3.55× |
| $J_a$ tridiagonal, $a = 1$ | 1.13× (0.88–1.88) | 1.13× (0.88–1.46) | 1.46× |

The tridiagonal row is re-measured here under the identical protocol -- five
replications, same seed blocks, same stepsize grid -- so the two are directly
comparable; it comes out at 1.13× rather than the 1.03× of §3d's wider sweep,
which is the same answer within the resolution. Every $J_{13}$ cell is above
1, which none of the tridiagonal's were, and
the measured means land just under the exact same-stepsize rate ratios, and
$\lambda$ between 0.25 and 0.5 barely moves them.

**Not yet measured: the uniform start.** Everything above is the
$N(0, 10I_d)$ ensemble. The $\mathrm{Uniform}(-5,5)^d$ start is the one that
caught out the hyperbolic state-dependent field of §3c -- 5.31× Gaussian
against 2.18× uniform -- so it is the check this field still owes, even though
a constant matrix has no reason to be prior-fragile the way a field growing
with $\|x\|$ does. The run is `exp16 ... --prior uniform5`.

The last column is where the two matrices part company, and it is not where
§3d guessed. Both fields are faster than $J = 0$ at a common $4\eta_0$ — the
tridiagonal by 1.46×, the plane by 3.55× — so raw rate is only half the gap.
The other half is that at $4\eta_0$ the tridiagonal's measured plateau is
0.093, above the 0.087 threshold, so it cannot use that stepsize and falls back
to $2.83\eta_0$ and its 1.03×; the plane's plateau at the same stepsize is
0.077 and fits.

That ordering is a measured fact and the exact stand-in does **not** reproduce
it: on the stand-in the plane at $a = 2$ carries *more* covariance bias at
$4\eta_0$ (0.238) than the tridiagonal does (0.185), which would predict the
opposite. Two things differ between the scalar covariance bias and what the
sliced $W_2$ sees — it is a distance between full one-dimensional marginals,
weighted by each axis's own scale, not a summary of second moments — and the
stand-in replaces the MCP by its curvature at the origin, an approximation a
field that transports mass along the soft axis will strain harder than one that
does not. Where the two disagree the measurement is the authority; the
stand-in's same-stepsize *rates* are what it gets right here (3.55× predicted
against 3.52× and 4.53× measured on the two blocks).

**$a$ trades against the accuracy target.** The stand-in's strict
*equal-covariance-bias* ceiling for this plane is 2.03× at $\lambda = 0.25$,
below the measured 3.11×, for the reason just given: the run is not comparing at
equal bias but at equal measured $W_2$, which at $n = 5000$ cannot separate
plateaus of 0.077 and 0.070 under a threshold of 0.087. Ask for more accuracy
and $a = 2$ is the first to fail — at $1.2\times$ the floor its plateau does not
fit at any stepsize — while $a = 1.3$ still returns 1.88× and 2.42× on the two
blocks. So $a \approx 1.3$ is the setting to quote when the accuracy target is
not known in advance, and $a = 2$ only when a bias of a few tenths is
acceptable.

The comparison to keep in mind is with the smooth Student-t core, where the
same plane is worth 4.38×. The regulariser costs it about a third of that at
$\lambda = 0.25$ and nearly all of it by $\lambda = 1$ — the isotropising
effect of §3d applies to this field exactly as it does to the others. What
changes is that this field can use whatever anisotropy is left, and the
tridiagonal cannot.

## 4. The price

A constant $J$ makes the anchored drift linear in $x$ (for the canonical anchor
$\beta = \iota - 1$). Any genuinely state-dependent field makes it nonlinear:
a linear skew drift $e^{U}J_0\nabla\Phi$ requires $\nabla\Phi = q^{-\iota}Lx$
to be curl-free, which forces $L \propto \Sigma^{-1}$, i.e. back to a constant
field.

That matters because the binding constraint on the whole method is
**discretisation bias, not stability** — the equal-bias stepsize sits a factor
of 14 to 50 below the mean-square stability limit. Linearity is what lets the
drift be integrated exactly, and exact integration is what removes the bias.
So state-dependence buys expressiveness and pays for it in integrability, and
the experiments have to settle which wins.

The implementation splits the difference: the constant part of the field is
advanced exactly and only the nonlinear remainder is taken explicitly.
