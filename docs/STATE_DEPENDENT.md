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
$J(x)v = v\times\nabla f(x)$ for an arbitrary scalar $f$. No such family exists
in $d = 2$, which is precisely why two dimensions are confined to radial
modulation under (INV).

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
adaptive search, and with the warm-up ramp:

| field | iterations to 2× floor | speed-up |
|---|---|---|
| $J = 0$ | 1907 | 1.00× |
| constant, $\|J\|=6$ | 421 | **4.53×** |
| $J_s$, $s = 0.15$ | 2256 | 0.85× |
| $J_s$, $s = 0.2$ | 1907 | 1.00× |
| $J_s$, $s = 0.3$ | 4415 | 0.43× |
| $J_s$, $s \ge 0.5$ | diverged | — |

Below $s\approx 0.2$ the added drift is under a fifth of the reversible one in
root-mean-square and changes nothing; above it the quadratic growth costs more
in stepsize than the rotation returns. The divergences are transient, from the
wide $\mathcal N(0,10I)$ prior rather than at stationarity, so a longer ramp
than the three relaxations used here might rescue the larger $s$; that is
untested.

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
