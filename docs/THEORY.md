# Skew-anchored Langevin dynamics: what is true, and what is not

This note states the mathematics behind the code, separating what is proved,
what is computed exactly, and what is only observed.  It extends
*Anchored Langevin Algorithms* (Gürbüzbalaban, Nguyen, Zhang and Zhu,
arXiv:2509.19455), whose numbering is used throughout: Eq. (5)–(7) for the
anchored SDE, Theorem 2/3 for ergodicity, Theorem 11 for the time change,
Assumption 12 and Theorem 14 for the 2-Wasserstein bound, and Section 6.4 for
the heavy-tailed experiment.

## 1. The dynamics

The anchored Langevin SDE of the paper is

$$dX_t = -\nabla U_0(X_t)\,e^{(U-U_0)(X_t)}\,dt + \sqrt2\, e^{(U-U_0)(X_t)/2}\,dW_t .$$

We insert a **constant skew-symmetric** matrix $J = -J^\top$:

$$\boxed{\;dX_t = e^{(U-U_0)(X_t)}\,(J-I)\,\nabla U_0(X_t)\,dt
        + \sqrt2\, e^{(U-U_0)(X_t)/2}\,dW_t\;}\tag{SAL}$$

with Euler–Maruyama discretisation

$$x_{k+1} = x_k + \eta\, e^{(U-U_0)(x_k)}(J-I)\nabla U_0(x_k)
          + \sqrt{2\eta}\, e^{(U-U_0)(x_k)/2}\,\xi_{k+1}.$$

$J = 0$ recovers the paper exactly.  Implemented in
`skewanchor/samplers.py:skew_anchored_step`.

The generator is

$$\mathcal L_J f = e^{U-U_0}\Big(\Delta f - \langle\nabla U_0,\nabla f\rangle
  + \langle J\nabla U_0,\nabla f\rangle\Big).$$

## 2. Invariance — proved, for every skew $J$

Write $G = e^{U-U_0}$.  The one structural identity behind everything is

$$G\,e^{-U} = e^{-U_0}. \tag{2.1}$$

The added drift field is $c(x) = G(x)\,J\nabla U_0(x)$, and

$$\nabla\!\cdot\!\big(e^{-U}c\big) = \nabla\!\cdot\!\big(e^{-U_0}J\nabla U_0\big)
 = e^{-U_0}\Big(\operatorname{Tr}\!\big(J\,\nabla^2U_0\big)
   - \langle\nabla U_0, J\nabla U_0\rangle\Big) = 0,$$

because $\operatorname{Tr}(JH) = 0$ for skew $J$ and symmetric $H$, and
$\langle v, Jv\rangle = 0$.  So $c$ is $\pi$-divergence free and $\pi \propto
e^{-U}$ remains invariant, under exactly the hypotheses the paper already uses
for $J=0$ ($U_0\in C^2$, $U\in C^1$, $\int e^{-U}<\infty$, non-explosion).

In $L^2(\pi)$ the added term is **antisymmetric**, so the Dirichlet form
$\mathcal E(f) = \int e^{U-U_0}\|\nabla f\|^2\,d\pi$ of the paper's Lemma 6 is
unchanged.  The perturbation therefore cannot slow $L^2(\pi)$ convergence and
generically strictly accelerates it.

The unscaled alternative $c'(x) = J\nabla U(x)$ also preserves $\pi$, but it
destroys the time-change structure of Section 3; `skew_ula` uses it and is
included only as a baseline.

*Checked numerically*: `tests/test_correctness.py::test_target_is_invariant_for_every_dynamics`
starts every sampler at stationarity and confirms the moments do not drift.

## 3. Random time change — proved

Let $Z$ solve the **irreversible** reference Langevin SDE
$dZ_t = (J-I)\nabla U_0(Z_t)\,dt + \sqrt2\,d\widetilde W_t$, whose invariant
measure is $\propto e^{-U_0}$, and let $\ell$ solve
$\ell'(t) = e^{(U-U_0)(Z_{\ell(t)})}$, $\ell(0)=0$.  Then $X_t := Z_{\ell(t)}$
solves (SAL).  This is the paper's Theorem 11 with the reference dynamics made
non-reversible; the time change itself does not see $J$ at all.

The discrete analogue of the paper's Theorem 15 also survives verbatim: under
synchronous coupling the Euler–Maruyama scheme above and

$$\ell_{k+1} = \ell_k + \eta e^{(U-U_0)(z_k)},\qquad
  z_{k+1} = z_k + \Delta\ell_k (J-I)\nabla U_0(z_k) + \sqrt{2\Delta\ell_k}\,\xi_{k+1}$$

produce **identical** trajectories.  `tests/test_correctness.py::test_time_change_equivalence_is_exact`
confirms agreement to $10^{-15}$.

## 4. Geometric ergodicity — proved, with no smallness condition on $J$

The paper's Assumption 1 uses $V(x) = 1+\|x\|^2$, and there the skew drift does
*not* drop out: the drift condition acquires the term
$\langle x, J\nabla U_0(x)\rangle$, which vanishes only when $\nabla U_0$ is
parallel to $x$.

The clean repair is to use a Lyapunov function **built from the anchor**:

> **Proposition.** Let $V = \Psi\circ U_0$ with $\Psi\in C^2$ increasing.  Then
> the skew drift contributes exactly zero to $\mathcal L_J V$, because
> $\langle J\nabla U_0, \nabla V\rangle = \Psi'(U_0)\,\langle J\nabla U_0,\nabla U_0\rangle = 0$.

So any drift condition the paper verifies with such a $V$ transfers to every
skew $J$ unchanged.  For the anisotropic Student-t the same conclusion follows
from the geometry-adapted $V_\Sigma(x) = 1 + (x-\mu)^\top\Sigma^{-1}(x-\mu)$,
since $\Sigma^{-1}J\Sigma^{-1}$ is again skew.

## 5. What does **not** transfer: the 2-Wasserstein bound

Assumption 12 of the paper asks for
$\langle b(x)-b(y), x-y\rangle \le -m\|x-y\|^2$,
$\|b(x)-b(y)\| \le L\|x-y\|$,
$\|\sigma(x)I - \sigma(y)I\|_{HS} \le \sqrt\alpha\|x-y\|$ with $0<\alpha<m$.
For the anisotropic Student-t with the canonical anchor,

$$m_J = \tfrac{2\beta}{\nu}\lambda_{\min}\!\big(\Sigma^{-1}-S\big),\quad
  S = \operatorname{sym}(J\Sigma^{-1}) = \tfrac12[J,\Sigma^{-1}],\quad
  \alpha = \tfrac{d\,\lambda_{\max}(\Sigma^{-1})}{\nu}.$$

Two facts follow, both implemented in `skewanchor/analysis.py:assumption12`.

1. Define $\rho_J = \lambda_{\max}(\Sigma^{1/2}S\,\Sigma^{1/2})$.  Then
   $\rho_J = 0$ **iff** $J$ commutes with $\Sigma$ — and a commuting $J$ is
   exactly the one that cannot accelerate (Section 6).  Acceleration and the
   Euclidean contraction constant are in direct tension, and $m_J>0$ requires
   $\rho_J<1$.
2. Even at $J = 0$, $\alpha<m$ reduces to $\beta > \tfrac d2\kappa(\Sigma)$ —
   the paper's own Corollary 13 condition.  For $d=2$, $\kappa=100$ that needs
   $\nu > 200$.  **The paper's Theorem 14 does not cover ill-conditioned
   heavy-tailed targets at all**, with or without $J$.

We therefore do not claim a Wasserstein bound.  We replace it with an exact
computation (next section), which needs no synchronous coupling.

## 6. The exact second-moment analysis (this work's main technical tool)

For the log-quadratic target with the canonical anchor $\beta = \iota - 1$, the
anchored drift is *linear* and the diffusion coefficient is the scalar
$q(x)^{1/2}$:

$$x_{k+1} = (I-\eta B)x_k + \sqrt{2\eta}\,q(x_k)^{1/2}\xi_{k+1},\qquad
  B = \tfrac{2\beta}{\nu}(I-J)\Sigma^{-1}.$$

Since $q$ is exactly quadratic and the noise is isotropic given $x$, the
second-moment matrix obeys an **exact affine recursion** — no closure:

$$C_{k+1} = M C_k M^\top + 2\eta\Big(1 + \tfrac{\operatorname{Tr}(\Sigma^{-1}C_k)}{\nu}\Big)I,
\qquad M = I-\eta B. \tag{6.1}$$

In vectorised form the linear part is
$L = M\otimes M + \tfrac{2\eta}{\nu}\operatorname{vec}(I)\operatorname{vec}(\Sigma^{-1})^\top$.
Three exact quantities follow.

* **Stability.** The scheme is mean-square stable iff $\rho(L)<1$, giving an
  exact maximum stepsize.  It is *much* smaller than the drift-only Euler limit,
  because the multiplicative noise $q^{1/2}$ grows linearly in $\|x\|$.
  `max_stable_stepsize` vs `deterministic_stepsize_limit`.
* **Bias.** The chain's stationary covariance is the fixed point of (6.1), so
  the discretisation bias is exact, not an $O(\eta)$ bound.
* **Rate.** $-\log\rho(L)$ is the exact per-iteration convergence rate.

The continuous-time counterpart
$\dot C = -BC - CB^\top + 2(1+\operatorname{Tr}(\Sigma^{-1}C)/\nu)I$
is Hurwitz for every skew $J$ we tested, which is the honest statement of
ergodicity in the regime where Assumption 12 fails.

Two sanity checks worth recording.  The SDE's stationary covariance is
$\tfrac{\nu}{\nu-2}\Sigma$ **for every** $J$ — the $J$ terms cancel in
$BC+CB^\top$ — which is what makes an equal-bias comparison across $J$
meaningful.  And `tests/test_correctness.py::test_second_moment_recursion_matches_simulation`
confirms the predicted stationary covariance against a 150 000-particle
simulation.

## 7. Choosing $J$, and the ceiling

$\operatorname{Tr}(JA) = 0$ for skew $J$ and symmetric $A$, so the eigenvalues
of $(I-J)A$ always sum to $\operatorname{Tr}(A)$ and

$$\min_i \operatorname{Re}\lambda_i\big((I-J)A\big) \le \frac{\operatorname{Tr}(A)}{d}.$$

The reversible choice $J=0$ gives $\lambda_{\min}(A)$, so the best conceivable
speed-up of the continuous-time gap is $\overline\lambda(A)/\lambda_{\min}(A)$ —
large precisely when $\Sigma$ is ill conditioned.

`skewanchor/skew.py:lnp_optimal` **attains** this ceiling.  Since $(I+K)A$ is
similar to $A + K'$ with $K' = A^{1/2}KA^{1/2}$ skew, it suffices to rotate to a
basis where $A$ has constant diagonal $\bar a = \operatorname{Tr}(A)/d$ (a
Bendel–Mickey/Davies–Higham Givens sweep) and there subtract the strictly lower
triangle while adding its transpose — a skew move that leaves an upper
triangular matrix with diagonal $\bar a$.  Verified to 4–5 digits for
$d \in \{2,3,5,10,20\}$.

## 8. The no-go theorem for isotropic targets

If $U$ and $U_0$ are radial — which is exactly the paper's own Section 6.4
target $\pi \propto (1+\|x\|^2)^{-\iota}$ — then a constant skew $J$ gives
**exactly zero** acceleration.  The added drift is a generator of rotations,
which commutes with $\mathcal L$ and acts unitarily on $L^2(\pi)$, so
$\|e^{t\mathcal L_J}f\|_{L^2(\pi)} = \|e^{t\mathcal L}f\|_{L^2(\pi)}$ for every
$f$ and $t$.  In $d = 1$ the point is starker still: the only skew $1\times1$
matrix is $0$, and the paper's Figure 8 uses $\iota=2$, which forces $d=1$.

Consequences for the experiments, all of which are run as controls: the
spectral gap is unchanged for every $J$ (`test_isotropic_target_is_a_negative_control`),
the continuous-time second-moment rate is unchanged to machine precision
(`exp5`), and the measured Wasserstein curves coincide.  **Any apparent gain on
a radial target is discretisation noise.**

## 9. Summary: claims and non-claims

Claimed and proved:

* $\pi$ is invariant for (SAL) for every constant skew $J$.
* The random-time-change representation and the exact equivalence of the two
  discretisations extend verbatim.
* Geometric ergodicity transfers to every skew $J$ under any Lyapunov function
  of the form $\Psi\circ U_0$.
* The eigenvalue-sum ceiling $\operatorname{Tr}(A)/d$, and an explicit $J$
  attaining it.
* No acceleration is possible for radial targets.

Computed exactly (not bounded):

* Mean-square stability threshold, stationary bias and per-iteration rate of the
  discretisation, and the continuous-time second-moment rate.

**Not** claimed:

* No extension of the paper's Theorem 14 (2-Wasserstein) — its hypotheses fail
  in the interesting regime even at $J=0$.
* Nothing about non-constant $J(x)$, about $s \ne 1$ anchors beyond what the
  code computes numerically, or about the composite non-smooth target in
  Section 6.4 of the code (there the exact second-moment analysis does not
  apply and stepsizes are chosen empirically).
* No claim that the skew perturbation helps in high dimension: the realised
  speed-up falls steadily with $d$ (see `results/`).
