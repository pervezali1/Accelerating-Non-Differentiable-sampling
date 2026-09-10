# Lyapunov drift and geometric ergodicity for the **skew-anchored Langevin SDE**

Extension of Assumption 1 / Theorem 2 / Theorem 3 of
*Anchored Langevin Algorithms* (Gürbüzbalaban, Nguyen, Zhang, Zhu; arXiv:2509.19455)
to the non-reversible ("skew") anchored dynamics.

Slug: `lyapunov`.  All algebra below has been verified numerically (scripts in the same
directory: `chk1_algebra.py`, `chk2_T2.py`, `chk3_invariance.py`, `chk4_rates.py`,
`chk5_scan.py`, `chk6_mono.py`, `chk7_sim.py`, `chk8_phi.py`, `chk9_inv_mc.py`,
`chk10_lmi.py`, `chk12_extra.py`, `chk13_table.py`, `chk14_unique.py`).

---

## 0. Setup, notation, standing hypotheses

Throughout, `U, U0 : R^d -> R`, `J in R^{d x d}` is a **constant** matrix with `J^T = -J`,
`sym(A) := (A + A^T)/2`, `[A,B] := AB - BA`, `lam_min/lam_max` denote extreme eigenvalues,
`kappa(Sigma) := lam_max(Sigma^{-1})/lam_min(Sigma^{-1})` is the condition number.

**(H0)** `U0 in C^2(R^d)`, `U in C^1(R^d)`, `M := int e^{-U} < infinity`. (Same as the paper.)

**Skew-anchored Langevin SDE (SAL-SDE).**

    dX_t = e^{(U-U0)(X_t)} (J - I) grad U0(X_t) dt + sqrt(2) e^{(U-U0)(X_t)/2} dW_t.     (5_J)

Equivalently `dX = b_J(X)dt + sqrt(2) sigma(X) dW` with

    b_J(x) = e^{(U-U0)(x)} (J - I) grad U0(x),      sigma(x) = e^{(U-U0)(x)/2}.          (6_J)

Its generator is

    L_J f = sigma^2 Laplacian f + <b_J, grad f>
          = e^{U-U0} [ Laplacian f - <grad U0, grad f> + <J grad U0, grad f> ]
          = L f + e^{U-U0} <J grad U0, grad f>,                                          (7_J)

where `L` is the paper's anchored generator. Write `L_J = e^{U-U0} L_0^J` with
`L_0^J f = Laplacian f - <grad U0, grad f> + <J grad U0, grad f>` the generator of the
*standard* non-reversible Langevin dynamics for `pi_0 ∝ e^{-U0}` (Hwang–Hwang–Sheu).
So the construction is: **anchor first, then add the Hwang-type skew field to the anchor
potential `U0`, and let the anchoring factor `e^{U-U0}` multiply it as well.**

### 0.1 pi ∝ e^{-U} is still invariant (this is what makes the construction work)

**Proposition 0.** Under (H0) and non-explosion, `pi(dx) = e^{-U(x)}dx / M` is an invariant
measure of (5_J), for **every** constant skew `J`.

*Proof.* The formal Fokker–Planck adjoint is `L_J^* rho = Laplacian(sigma^2 rho) - div(b_J rho)`.
With `rho = e^{-U}`:

* `sigma^2 rho = e^{U-U0} e^{-U} = e^{-U0}`;
* `b_J rho = e^{U-U0}(J-I) grad U0 · e^{-U} = (J-I) grad U0 · e^{-U0}`.

Hence

    L_J^* e^{-U} = Laplacian e^{-U0} + div( grad U0 · e^{-U0} ) - div( J grad U0 · e^{-U0} ).

The first two terms cancel exactly (the paper's computation, since
`Laplacian e^{-U0} = div(-grad U0 e^{-U0})`).  For the third, note
`J grad U0 · e^{-U0} = - J grad(e^{-U0})`, so

    div( J grad U0 · e^{-U0} ) = - sum_i d_i ( sum_j J_ij d_j e^{-U0} )
                               = - sum_{i,j} J_ij d_i d_j e^{-U0} = 0,

because `J` is skew-symmetric and constant while the Hessian of `e^{-U0}` is symmetric. □

**Where a hypothesis is needed.** (i) `J` **constant**. For a state-dependent `J(x)` the
same computation leaves the residue `- sum_j ( sum_i d_i J_ij(x) ) d_j e^{-U0}`, so one must
additionally require each **column-divergence of `J` to vanish**,
`sum_i d_i J_ij(x) = 0` for all `j` (or, more weakly, that this vector be orthogonal to
`grad U0` everywhere). (ii) `U0 in C^2` so that `d_i d_j e^{-U0}` exists and is symmetric.

Equivalently: the added drift `gamma(x) := e^{U-U0} J grad U0` is **`pi`-divergence free**,
`div(gamma e^{-U}) = 0`, because `gamma e^{-U} = -J grad(e^{-U0})`. This is the anchored
analogue of the classical `J grad U` perturbation; note it is **not** `e^{U-U0} J grad U`,
and it is **not** `J grad U` — the anchoring factor and the anchor gradient must appear
together for the cancellation to happen.

*Numerical check (`chk3_invariance.py`):* `L_J^* e^{-U}` evaluated by centred finite
differences at random points, for `d = 2,3`, `alpha in {1, 1.5}`, `nu in {6,8,10}` and
`||J||_2` up to 6.7, is `O(10^{-8})` against a term scale `O(10^{-1})`; a direct
Euler–Maruyama simulation of (5_J) with `theta = 0` reproduces the target Student-`t`
(empirical Cov `[1.487, 0.495]` vs. target `[1.5, 0.5]`; KS statistics `0.0046`, `0.0035`
against the `t_6` marginals with `n = 6*10^4`, below the 95% critical value `0.0056`).

### 0.2 What breaks and what survives from Section 3 of the paper

**Lemma 5 (reversibility) FAILS — by design.** For `f, g` in the domain,

    int f L_J g dpi = (int e^{-U0} / int e^{-U}) int f L_0^J g dpi_0,

and the added piece is antisymmetric in `L^2(pi_0)`:

    int f <J grad U0, grad g> dpi_0 = - int g <J grad U0, grad f> dpi_0,

(integrate by parts and use `div(J grad U0 e^{-U0}) = 0` again). Hence `L_J` is symmetric in
`L^2(pi)` **iff `J grad U0 == 0`**. So (5_J) is genuinely non-reversible whenever
`J grad U0 not== 0`. **This matters for the proof of Theorem 2:** the paper's proof of
Theorem 2 does *not* use Lemma 5 — it uses only the Lyapunov bound `L V <= -2c_0||x||^{2+r}+2c_1`,
positivity/continuity of `sigma`, and (Xie & Zhang 2020, Thm 7.4), which is stated for
general (non-symmetric) Itô SDEs. So the whole of Theorem 2 transfers, provided the drift
condition is re-derived — which is the content of §1 below.

**Lemma 6 and Proposition 7 survive verbatim.** Setting `f = g` above kills the
antisymmetric part, so the Dirichlet form is unchanged:

    E_J(f) := - int f L_J f dpi = int e^{U-U0} ||grad f||^2 dpi  =  E(f).

Since `d/dt chi^2(mu_t||pi) = -2 E_J(h_t)` also for non-reversible generators
(`h_t = dmu_t/dpi`, `d_t h_t = L_J^dagger h_t`, and `<h, L_J^dagger h> = <L_J h, h>`),
Proposition 7 holds with **the same constant**. This is an honest limitation to record:
*the `chi^2`/Poincaré route is blind to `J`*; any acceleration must be proved by
non-symmetric arguments (or, as in §6 below, computed exactly).

---

## 1. The drift condition for `V(x) = 1 + ||x||^2`

`grad V = 2x`, `Laplacian V = 2d`, so from (7_J)

    L_J V(x) = e^{(U-U0)(x)} [ 2d + 2 <(J - I) grad U0(x), x> ]
             = 2 [ d - <x, grad U0(x)> + <x, J grad U0(x)> ] e^{(U-U0)(x)}.              (*)

**Assumption 1_J.** For some `c_0, c_1 > 0` and `r > -1`,

    [ d - <x, grad U0(x)> + <x, J grad U0(x)> ] e^{(U-U0)(x)}  <=  -c_0 ||x||^{2+r} + c_1,
                                                                  for all x in R^d.     (8_J)

This is exactly the drift condition requested in the task statement, and (*) shows that
(8_J) is equivalent to `L_J V <= -2c_0 ||x||^{2+r} + 2c_1`.

**Theorem 2_J.** Assume (H0) and Assumption 1_J. Then `pi ∝ e^{-U}` is the unique invariant
measure of the semigroup `P_t^J` of (5_J), and

  (i) if `r >= 0`, there are `lam, C > 0` with
      `sup_{||phi/V||_inf <= 1} |P_t^J phi(x) - pi(phi)| <= C e^{-lam t} V(x)`;
  (ii) if `r > 0`, there are `lam, C > 0` with `sup_x ||P_t^J(x,·) - pi||_var <= C e^{-lam t}`.

*Proof sketch and the hypotheses it needs.* (a) `L_J V <= -2c_0||x||^{2+r} + 2c_1 <= 2c_1`
with `V -> infinity` gives non-explosion. (b) `sigma > 0` is continuous, `b_J` is locally
bounded measurable, so (5_J) has a unique weak solution which is strong Feller and
irreducible on `R^d` (uniform ellipticity on compacts). (c) For `r >= 0`,
`||x||^{2+r} >= ||x||^2 = V - 1` so `L_J V <= -2c_0 V + 2(c_0+c_1)`: a Foster–Lyapunov
geometric drift. For `r > 0`, `||x||^{2+r} >= c V^{1+r/2}` outside a ball, which is the
"coming down from infinity" drift giving uniform-in-`x` TV convergence. (d) Conclude by
(Xie & Zhang 2020, Thm 7.4) exactly as in the paper.
**Hypothesis flag:** step (d) is the *only* import from the literature, and it is a result
about general Itô SDEs; it does not use reversibility. Nothing else in the paper's proof of
Theorem 2 does either. If one prefers a self-contained route, (c)+(b) plus
Down–Meyn–Tweedie (1995) give the same conclusions.

**Theorem 3_J (heavy tails).** With `U0 = beta log q`, `U = (beta+1) log q`
(so `e^{U-U0} = q` and `grad U0 = beta grad q / q`), (8_J) reads

    d q(x) - beta <x, grad q(x)> + beta <x, J grad q(x)>  <=  -c_0||x||^{2+r} + c_1.     (8'_J)

If moreover `int q^{-1-beta} < infinity`, then `pi = q^{-1-beta}/int q^{-1-beta}` is the
unique stationary distribution of (5_J) and Theorem 2_J applies.

**The skew term does not vanish in general.** Since `<x, J grad U0(x)> = - <Jx, grad U0(x)>`,
the extra term is zero for all `x` **iff `grad U0(x) ⟂ Jx` for every `x`**. In particular it
vanishes identically when `grad U0(x) = g(x) x` is *radial* (then `<x, Jx> = 0` because `J`
is skew), which is §2; it does **not** vanish for an anisotropic quadratic anchor, which is §3.

*Numerical check (`chk1_algebra.py`):* `L_J V` computed by finite differences of the full
generator agrees with (*) to relative error `10^{-7}`–`10^{-9}` on random `(d, Sigma, J, x)`.

---

## 2. Target (T1): the isotropic heavy-tailed target of §6.4 — Assumption 1 is *skew-blind*

**Proposition 1 (isotropic anchor absorbs any skew `J`).**
Let `U(x) = iota log(1 + ||x||^2)` and `U0(x) = beta log(1 + ||x||^2)` with
`iota > 1 + d/2`, `beta > d/2` (the parameters of §6.4). Then for **every** constant
skew-symmetric `J`:

  (a) `grad U0(x) = 2 beta x / (1 + ||x||^2)` is parallel to `x`, hence
      `<x, J grad U0(x)> = 2 beta <x, Jx>/(1+||x||^2) = 0` **exactly, for all x**;

  (b) consequently

        [ d - <x, grad U0> + <x, J grad U0> ] e^{U-U0}
              = [ d - (2beta - d)||x||^2 ] (1 + ||x||^2)^{iota - beta - 1};

  (c) so Assumption 1_J holds **with exactly the same constants `c_0, c_1` and the same
      exponent** `r = 2(iota - beta - 1)` as Assumption 1 for the reversible case `J = 0`.
      The admissibility conditions are unchanged:
      `r > -1  <=>  iota - beta > 1/2`, `r >= 0  <=>  iota >= beta + 1`,
      `r > 0  <=>  iota > beta + 1`. Quantitatively, for any `c_0 in (0, 2beta - d)` there
      is a finite `c_1 = c_1(c_0, d, beta, iota)` such that (8_J) holds on all of `R^d`
      (the function `t -> [d-(2beta-d)t](1+t)^{iota-beta-1} + c_0 t^{iota-beta}`,
      `t = ||x||^2`, is continuous and tends to `-infinity`, hence is bounded above).

  (d) **Therefore Assumption 1, Theorem 2 and Theorem 3 of the paper hold verbatim for the
      skew-anchored SDE, for every skew `J`, with no condition whatsoever on `J`.**

The same is true for the Theorem-3 form: if `q(x) = (1+||x||^2)^alpha`, then
`grad q ∝ x`, so `<x, J grad q(x)> = 0` and (8'_J) reduces to the paper's condition, with
`r = 2 alpha - 2` (Example 1 with `Sigma = I`, `mu = 0`).

*Numerical check (`chk1_algebra.py`):* `<x, J grad U0(x)> = O(10^{-9})` (finite-difference
noise) for `d = 2,3,5` and `||J||` up to 6; the closed form in (b) matches the numerical
generator to `10^{-5}`–`10^{-8}` relative error.

**Honest caveat: for (T1) the extension is correct but *inert*.** The added field
`e^{U-U0} · 2 beta Jx/(1+||x||^2)` is tangent to the spheres `{||x|| = const}` and `pi`,
`U`, `U0` are radial. Hence the radial part of the dynamics — which carries the slow,
tail-limited modes for a heavy-tailed target — is *completely unchanged*; `J` only stirs
mass within spheres. Concretely, in §6 we compute the exact `L^2` relaxation rate for the
Student-`t` family and find it is **numerically identical for all `theta` when `Sigma ∝ I`**
(`chk4_rates.py`: rate `= 1.33333` for `theta in {0, 0.25, ..., 500}` at `d=2, nu=6, Sigma=I`).
So §6.4's target is exactly the case where non-reversibility cannot help. **Any experiment
demonstrating a benefit of `J` must use an anisotropic target — i.e. (T2).**

---

## 3. Target (T2): anisotropic Student-`t`, Euclidean `V = 1 + ||x||^2`

Fix `mu = 0` (translate otherwise), `Sigma > 0`, `nu > 0`, `alpha >= 1` and set

    p(x) = 1 + (1/nu) x^T Sigma^{-1} x,   q = p^alpha,
    U0 = beta log q = alpha*beta log p,   U = (beta+1) log q,
    beta = (d+nu)/(2 alpha) - 1,          b := alpha*beta = (d+nu)/2 - alpha.

(For `alpha = 1` this is exactly Example 2 / the (T2) of the task: `beta = (d+nu)/2 - 1`,
`e^{U-U0} = q = p`, `b_J(x) = (2beta/nu)(J - I)Sigma^{-1}x` is **linear**, `sigma = p^{1/2}`.)
Note `2b - d = nu - 2 alpha`.

Since `grad p = (2/nu)Sigma^{-1}x` and `grad U0 = b grad p / p`,

    <x, grad U0> = (2b/nu) x^T Sigma^{-1} x / p,
    <x, J grad U0> = (2b/nu) x^T J Sigma^{-1} x / p = (2b/nu) x^T sym(J Sigma^{-1}) x / p,

and `e^{U-U0} = p^alpha`. Using `d p = d + (d/nu)x^T Sigma^{-1} x`,

    [ d - <x, grad U0> + <x, J grad U0> ] e^{U-U0}
        = p^{alpha-1} [ d p - (2b/nu) x^T( Sigma^{-1} - sym(J Sigma^{-1}) ) x ]
        = p^{alpha-1} [ d - (1/nu) x^T M_J x ],                                          (11)

with

    M_J := (2b - d) Sigma^{-1} - 2b · sym(J Sigma^{-1})
         = (nu - 2 alpha) Sigma^{-1} - b [J, Sigma^{-1}].                                (12)

**The commutator identity.** Because `J^T = -J` and `Sigma^{-1}` is symmetric,

    sym(J Sigma^{-1}) = ( J Sigma^{-1} + Sigma^{-1} J^T )/2 = ( J Sigma^{-1} - Sigma^{-1} J )/2
                      = (1/2) [ J, Sigma^{-1} ].                                         (13)

**Proposition 2 (drift condition for (T2), Euclidean `V`).** Let `alpha >= 1` and `nu > 2 alpha`.
Then Assumption 1_J holds **for some `r > -1` iff** the symmetric matrix

    M_J = (2b - d) Sigma^{-1} - 2b·sym(J Sigma^{-1})    is positive definite,

equivalently `(d - 2b)Sigma^{-1} + 2b·sym(J Sigma^{-1}) ≺ 0` — exactly the condition in the
task statement with `2b = 2 alpha beta` — and then the best exponent is `r = 2 alpha - 2`.
(*Necessity:* if `u^T M_J u <= 0` for some unit `u`, then by (11)
`LHS(Ru) = p^{alpha-1}[d - R^2 u^T M_J u/nu] >= d > 0` for every `R`, while the right-hand
side of (8_J) tends to `-infinity`; so (8_J) fails for every `r > -1`.)
Explicitly, writing `m := lam_min(M_J) > 0`:

* `alpha = 1`:  `L_J V(x) = 2[ d - (1/nu) x^T M_J x ] <= 2d - (2m/nu)||x||^2`, i.e.
  `c_0 = 2m/nu`, `c_1 = 2d`, `r = 0`; and the geometric drift
  `L_J V <= -(2m/nu) V + (2d + 2m/nu)`.
* `alpha > 1`: from (11), `L_J V = 2 p^{alpha-1}[ d - (1/nu)x^T M_J x ]`. For
  `||x||^2 >= R^2 := 2 nu d/m` the bracket is `<= -(m/(2nu))||x||^2`, while
  `p^{alpha-1} >= (lam_min(Sigma^{-1})||x||^2/nu)^{alpha-1}`, so
  `L_J V <= -c_0 ||x||^{2 alpha}` with `c_0 = (m/nu)(lam_min(Sigma^{-1})/nu)^{alpha-1}`,
  and `c_1` absorbs the (compact) ball `||x|| <= R`. So `r = 2 alpha - 2 > 0` and
  Theorem 2_J(ii) applies.

**Sufficient smallness conditions on `J`.** From `lam_min(M_J) >= (2b-d)lam_min(Sigma^{-1}) - 2b·lam_max(sym(J Sigma^{-1}))`:

    lam_max( sym(J Sigma^{-1}) ) < (2b - d) lam_min(Sigma^{-1}) / (2b)
                                 = (nu - 2 alpha) lam_min(Sigma^{-1}) / (d + nu - 2 alpha).   (14)

This is precisely the condition requested in the task (with `2b = 2 alpha beta`). Two
usable consequences:

* **Operator norm.** `lam_max(sym(J Sigma^{-1})) <= ||J||_2 lam_max(Sigma^{-1})`, so (14)
  holds whenever

        ||J||_2  <  (nu - 2 alpha) / [ (d + nu - 2 alpha) · kappa(Sigma) ].               (15)

* **Sharper, commutator-aware.** In an eigenbasis of `Sigma^{-1}` with eigenvalues `lam_i`,
  `([J,Sigma^{-1}])_{ij} = J~_{ij}(lam_j - lam_i)`. Hence
  `||[J,Sigma^{-1}]||_F <= (lam_max(Sigma^{-1}) - lam_min(Sigma^{-1})) ||J||_F` and

        ||J||_F  <  2 (nu - 2 alpha) lam_min(Sigma^{-1})
                    / [ (d + nu - 2 alpha)(lam_max(Sigma^{-1}) - lam_min(Sigma^{-1})) ].  (16)

  In particular, **if `Sigma ∝ I`, or more generally if `J` commutes with `Sigma^{-1}`, then
  `[J, Sigma^{-1}] = 0` and there is no condition at all** — consistent with §2.
  (Over `R`, a nonzero skew `J` commuting with `Sigma^{-1}` exists only when `Sigma^{-1}`
  has a repeated eigenvalue; so for a generic anisotropic `Sigma` the useful `J`'s are
  exactly the non-commuting ones.)

**The degradation is not an artifact of a lazy bound.** `Tr( sym(J Sigma^{-1}) ) = Tr(J Sigma^{-1}) = 0`
(trace of skew × symmetric), so `lam_max(sym(J Sigma^{-1})) >= 0`, **with equality iff
`[J, Sigma^{-1}] = 0`**. Hence for every non-commuting `J`, `lam_min(M_J)` is *strictly*
smaller than at `J = 0`, and `M_J` is indefinite once `J` is scaled up far enough. So the
Euclidean-`V` drift condition genuinely fails for large `J` — see §5 for what that means.

**Exact `d = 2` threshold.** For `d = 2`, `Sigma^{-1} = diag(lam_1, lam_2)` and
`J = theta Omega` with `Omega = [[0,1],[-1,0]]`: `sym(J Sigma^{-1}) = (theta/2)(lam_2-lam_1)·[[0,1],[1,0]]`,
so `M_J = (nu-2alpha)diag(lam_1,lam_2) - b theta (lam_2-lam_1)[[0,1],[1,0]]` and
`det M_J > 0` gives the sharp threshold

    theta* = 2 (nu - 2 alpha) sqrt(lam_1 lam_2) / [ (d + nu - 2 alpha) |lam_2 - lam_1| ],
    (alpha = 1, d = 2:  theta* = 2(nu-2)sqrt(lam_1 lam_2) / (nu |lam_2 - lam_1|) ).       (17)

*Numerical checks (`chk2_T2.py`, `chk4_rates.py`):* (11)–(12) verified against the
finite-difference generator to `10^{-8}`–`10^{-9}` relative error for
`(d,alpha,nu) in {(2,1,7),(3,1,9),(3,1.4,11),(4,1,13)}`; identity (13) and
`Tr sym(J Sigma^{-1}) = 0` to machine precision; formula (17) matched exactly by a bisection
on `lam_min(M_J)` (e.g. `Sigma^{-1}=diag(1,10)`, `nu = 6`: `theta* = 0.4685`;
`Sigma^{-1}=diag(1,3)`, `nu = 6`: `theta* = 1.1547`).

---

## 4. KEY IMPROVEMENT: the geometry-adapted Lyapunov function is *exactly* skew-blind

### 4.1 The structural lemma

**Lemma A (skew-blind Lyapunov functions).** Let `Phi in C^2(R)` and set `V = Phi ∘ U0`.
Then for **every** constant skew `J` and every `U, U0` (with `U0 in C^2`),

    L_J V = e^{U-U0} [ Phi''(U0) ||grad U0||^2 + Phi'(U0) ( Laplacian U0 - ||grad U0||^2 ) ]
          = L V,     i.e. L_J V does not depend on J at all.                             (18)

*Proof.* `grad V = Phi'(U0) grad U0`, so
`<J grad U0, grad V> = Phi'(U0) <J grad U0, grad U0> = 0`, because `<Jv, v> = 0` for every
skew `J` and every `v`. The remaining terms are the `J = 0` computation. □

*Geometric reading:* the added drift `e^{U-U0} J grad U0` is **tangent to the level sets of
`U0`** (it is orthogonal to `grad U0`). Any Lyapunov function that is constant on those level
sets therefore sees no transport from it — the skew field only *transports along* level sets.

This single lemma explains and unifies everything:

* **(T1) is Corollary A:** `V = 1 + ||x||^2 = Phi(U0)` with `Phi(u) = e^{u/beta}` when
  `U0 = beta log(1+||x||^2)`. That is *why* §2 works.
* **(T2) is Corollary B:** `V_Sigma(x) = 1 + (x-mu)^T Sigma^{-1}(x-mu) = Phi(U0)` with
  `Phi(u) = 1 + nu(e^{u/b} - 1)` when `U0 = b log p`.
* And directly, for `V_Sigma`: `grad V_Sigma = 2 Sigma^{-1} y` (`y = x - mu`) so
  `<J grad U0, grad V_Sigma> ∝ <J Sigma^{-1} y, Sigma^{-1} y> = y^T Sigma^{-1} J^T Sigma^{-1} y = 0`
  because **`Sigma^{-1} J Sigma^{-1}` is skew-symmetric**
  (`(Sigma^{-1}J Sigma^{-1})^T = Sigma^{-1}J^T Sigma^{-1} = -Sigma^{-1}J Sigma^{-1}`), exactly
  as the task anticipated.

**Uniqueness among quadratics.** For `V_A = 1 + y^T A y` (`A = A^T`) one gets
`<J grad U0, grad V_A> = -(4b/(nu p)) y^T sym(Sigma^{-1} J A) y`; this vanishes *for every
skew `J`* iff `Sigma^{-1} J A = A J Sigma^{-1}` for every skew `J`, which forces
`A ∝ Sigma^{-1}`. So `V_Sigma` is **the** quadratic Lyapunov function in the skew-blind class.
*(Verified in `chk14_unique.py`: solving the linear system over a full basis of skew `J`'s,
the nullspace has dimension exactly 1 for `d = 2,3,4` and is spanned by `Sigma^{-1}` to
`3·10^{-16}`.)*

*Numerical check (`chk8_phi.py`):* for `Phi in {id, exp, square, log}` and both the
Student-`t` `U0` and a deliberately generic non-radial
`U0(x) = 0.3 x^T A x + sin(c·x) + 0.1 sum x_i^4`, `L_J V` is constant to `10^{-6}` across
`||J||` scales `0, 1, 4, 20`, whereas for `V = 1 + ||x||^2` the same sweep changes `L_J V`
by hundreds of units.

### 4.2 The drift condition for `V_Sigma` on (T2)

With `y = x - mu`, `V_Sigma = 1 + y^T Sigma^{-1} y = 1 + nu(p-1)`, and
`Laplacian V_Sigma = 2 Tr(Sigma^{-1})`, `grad V_Sigma = 2Sigma^{-1}y`:

    L_J V_Sigma = p^alpha [ 2 Tr(Sigma^{-1}) - (4b/(nu p)) y^T Sigma^{-2} y ]
                = 2 p^{alpha-1} [ Tr(Sigma^{-1}) p - (2b/nu) y^T Sigma^{-2} y ]
                = 2 p^{alpha-1} [ Tr(Sigma^{-1}) - (1/nu) y^T G y ],                     (19)

    G := 2 b Sigma^{-2} - Tr(Sigma^{-1}) Sigma^{-1}.                                     (20)

`G` is diagonal in the eigenbasis of `Sigma^{-1}` with eigenvalues
`lam_i ( 2 b lam_i - Tr(Sigma^{-1}) )`, so

    G ≻ 0  <=>  2 b · lam_min(Sigma^{-1}) > Tr(Sigma^{-1}),                              (C_Sigma)
    and then  lam_min(G) = lam_min(Sigma^{-1}) ( 2 b lam_min(Sigma^{-1}) - Tr(Sigma^{-1}) ).

**No `J` appears anywhere in (19)–(20).**

**Theorem 4 (skew-anchored geometric ergodicity in the `Sigma^{-1}` geometry).**
Let `alpha >= 1`, `Sigma ≻ 0`, `nu > 2 alpha`, `beta = (d+nu)/(2 alpha) - 1`,
`b = alpha beta = (d+nu)/2 - alpha`, `q = p^alpha`, `U0 = beta log q`, `U = (beta+1) log q`,
so that `pi` is the `d`-dimensional Student-`t(nu, mu, Sigma)` law. Assume

    (C_Sigma):    ( d + nu - 2 alpha ) lam_min(Sigma^{-1})  >  Tr(Sigma^{-1}).

Then, with `V_Sigma(x) = 1 + (x-mu)^T Sigma^{-1}(x-mu)` and `g := lam_min(G) > 0`, for
**every** constant skew-symmetric `J in R^{d x d}`:

    L_J V_Sigma(x)  <=  - c_0 ||x - mu||^{2 + r} + c_1,       r = 2 alpha - 2 >= 0,      (21)

with `c_0, c_1` depending only on `(d, nu, alpha, Sigma)` — **independent of `J`**:

* `alpha = 1` (`r = 0`):  `L_J V_Sigma <= 2 Tr(Sigma^{-1}) - (2g/nu)||y||^2`, i.e.
  `c_0 = 2g/nu`, `c_1 = 2 Tr(Sigma^{-1})`, and the geometric drift

        L_J V_Sigma  <=  - lam_* V_Sigma + b_*,
        lam_* = 2 lam_min(Sigma^{-1})[(d+nu-2)lam_min(Sigma^{-1}) - Tr(Sigma^{-1})]
                 / ( nu · lam_max(Sigma^{-1}) ),        b_* = 2Tr(Sigma^{-1}) + lam_*.  (22)

* `alpha > 1` (`r = 2alpha-2 > 0`): `c_0 = (g/nu)(lam_min(Sigma^{-1})/nu)^{alpha-1}` and
  `c_1 = c_0 R^{2 alpha} + 2 Tr(Sigma^{-1})(1 + lam_max(Sigma^{-1})R^2/nu)^{alpha-1}`
  with `R^2 = 2 nu Tr(Sigma^{-1})/g`.

Consequently `pi` is the unique invariant measure of `P_t^J` and

  (i) if `r >= 0`, i.e. `alpha >= 1`: there are `lam, C > 0` with
      `sup_{||phi/V_Sigma||_inf <= 1} |P_t^J phi(x) - pi(phi)| <= C e^{-lam t} V_Sigma(x)`;
      since `min(1,lam_min(Sigma^{-1})) V <= V_Sigma <= max(1,lam_max(Sigma^{-1})) V`, this
      is equivalent (up to constants) to the paper's statement with `V = 1 + ||x||^2`;
  (ii) if `r > 0`, i.e. `alpha > 1`: `sup_x ||P_t^J(x,·) - pi||_var <= C e^{-lam t}`.

**A fully `J`-uniform quantitative corollary.** From (22) alone (no small-set constants
involved), Dynkin + Grönwall give, for every `J` and every starting point,

    E_x[ V_Sigma(X_t) ]  <=  e^{-lam_* t} V_Sigma(x) + b_*/lam_*,                        (23)

with `lam_*, b_*` **independent of `J`**. (The full `V`-uniform constants `C, lam` in (i)
also involve a minorization on a compact set, which does depend on `J`; it holds for every
`J` by uniform ellipticity but is not automatically uniform in `J`. So the clean `J`-free
quantitative statement is (23), and geometric ergodicity itself holds for every `J`.)

**Comparison with the paper's own hypotheses.** For `alpha = 1`, (C_Sigma) is
`(d+nu-2)lam_min(Sigma^{-1}) > Tr(Sigma^{-1})`; since `Tr(Sigma^{-1}) <= d lam_max(Sigma^{-1})`,
it is **implied by** the paper's Example-2 condition `d + nu > 2 + d kappa(Sigma)`
(needed there for Corollary 13 / Theorem 14). In `d = 2` it reads `nu > 1 + kappa(Sigma)`.
The price relative to the Euclidean `V` at `J = 0` (which needs only `nu > 2 alpha`) is the
extra `Sigma`-conditioning; the gain is complete freedom in `J`.

**Refinement (a one-parameter skew-blind family).** `V_s := p^s` is also `Phi ∘ U0`, hence
skew-blind, and

    L_J p^s = (2s/nu) p^{alpha+s-2} [ Tr(Sigma^{-1}) - (1/nu) y^T G_s y ],
    G_s := 2(b + 1 - s) Sigma^{-2} - Tr(Sigma^{-1}) Sigma^{-1},   0 < s < b+1.           (24)

(`s = 1` is `V_Sigma` up to the affine map `V_Sigma = nu p - (nu-1)`.) The condition
`G_s ≻ 0` is `2(b+1-s) lam_min(Sigma^{-1}) > Tr(Sigma^{-1})`, weakest as `s -> 0+`. Since
`L_J V_s ≍ -||y||^{2(alpha+s)-2}` while `V_s ≍ ||y||^{2s}`, the geometric drift
`L_J V_s <= -lam V_s + b` holds precisely when `alpha >= 1`. Hence:

**Theorem 4' .** If `alpha >= 1` and

    (C'_Sigma):   ( d + nu - 2 alpha + 2 ) lam_min(Sigma^{-1})  >  Tr(Sigma^{-1})

(strictly), then for every constant skew `J` the SAL-SDE is geometrically ergodic with the
`J`-free Lyapunov function `V_s = p^s` for any small enough `s > 0`. For `alpha = 1` this is
`(d + nu) lam_min(Sigma^{-1}) > Tr(Sigma^{-1})`. (The price is a *weaker weight*: `V_s`
controls only `||x||^{2s}`-growth test functions, not `1 + ||x||^2`.)

*Numerical checks.*
* (`chk2_T2.py`) (19) reproduced by the finite-difference generator for
  `(d,alpha,nu) in {(2,1,9),(3,1,11),(3,1.3,13),(5,1,15)}`, with the value **identical across
  `||J||` scales `0, 0.5, 2, 10, 60`** (spread `<= 10^{-5}`); the predicted
  `lam_min(G) = lam_min(Sigma^{-1})(2b lam_min(Sigma^{-1}) - Tr(Sigma^{-1}))` matched exactly
  on nine `(d, nu, Sigma)` cases.
* (`chk12_extra.py`) the **explicit constants** of Theorem 4 were tested pointwise. For
  `alpha > 1`, `(d,alpha,nu) in {(2,1.6,9),(3,1.3,12),(4,2,16)}`, inequality (21) with the
  stated `c_0, c_1` held at all `20000` random points per case (`0` violations, with a
  comfortable margin: `max(LHS-RHS) <= -3.06`). For `alpha = 1`, the geometric drift (22)
  with the stated `lam_*, b_*` held at all `50000` random points per case for
  `(d,nu,Sigma^{-1}) in {(2,6,diag(1,3)), (3,9,diag(1,2,3)), (2,12,diag(1,10))}`
  (`0` violations).

---

## 5. Is the Euclidean-`V` smallness condition on `J` real, or an artifact?

**Short answer: for this target family it is an artifact of the Lyapunov function — and a
severe one. The dynamics not only stays geometrically ergodic for arbitrarily large `J`, its
true `L^2` relaxation rate *increases monotonically* in `||J||`.**

The reason we can be categorical here is that for `alpha = 1` the (T2) model is **exactly
solvable at the level of second moments**: the drift `b_J(x) = -Bx` is linear with
`B = (2beta/nu)(I - J)Sigma^{-1}`, and `sigma^2(x) = q(x) = 1 + x^T Sigma^{-1}x/nu` is
quadratic, so `S_t := E[X_t X_t^T]` obeys the **closed linear ODE**

    dS/dt = A*(S) + 2I,     A*(S) := -BS - SB^T + (2/nu) Tr(Sigma^{-1} S) I.             (25)

Facts (all verified numerically, `chk3_invariance.py`, `chk4_rates.py`, `chk5_scan.py`):

1. **Consistency.** `S_infinity = (nu/(nu-2)) Sigma` solves `A*(S)+2I = 0` for *every* `J`
   (residual `< 5·10^{-14}` over `d = 2,3,4`, `||J||` up to 32) — an independent confirmation
   of Proposition 0.
2. **Sharp criterion.** `E||X_t||^2` converges exponentially **iff `A*` is Hurwitz**, and
   `L_J V_A = 2Tr(A) + x^T A(A) x` where `A(·)` is the adjoint of `A*`. So
   **{exists a quadratic Lyapunov function `V_A = 1 + x^T A x`, `A ≻ 0`, with a geometric
   drift} <=> {`A*` Hurwitz}.** Both `A = I` (§3) and `A = Sigma^{-1}` (§4) are particular,
   generally *sub-optimal*, choices.
3. **A certificate is one linear solve away.** `A*` positivity-preserving on the PSD cone
   implies the same for `e^{A t}`, so whenever `A*` is Hurwitz the solution of the linear
   system `A(A) = -I` is automatically `A ≻ 0`. Example (`chk10_lmi.py`), `d = 2`, `nu = 6`,
   `Sigma^{-1} = diag(1,10)`, `theta = 50` — where **both** `A = I` (threshold `theta* = 0.4685`)
   and `A = Sigma^{-1}` ((C_Sigma): `6 > 11` false) fail:

        A = [[0.0752, 0.0113],[0.0113, 0.7498]] ≻ 0,  giving  L_J V_A <= -V_A/lam_max(A) + const,
        a guaranteed drift rate 1.3334 (the exact rate is 7.3314).

   The **optimal** quadratic Lyapunov function is the Perron eigenvector of `A` in the PSD
   cone, `A(A_*) = -rho A_*`; then `L_J V_{A_*} = -rho V_{A_*} + (2Tr(A_*)+rho)` with `rho`
   equal *exactly* to the second-moment rate.
4. **No smallness is needed, ever (numerically).** Over `4000` random draws of
   `d in {2,3,4}`, `nu in (2, 12]` (down to `nu = 2.001`), `Sigma` with condition numbers up
   to `10^3`, and `||J||_2 in [0.1, 10^3]`, the rate was **strictly positive in every case**;
   over `2000` further draws (`d <= 5`) we never found a `theta` with
   `rate(theta) < rate(0)` — the minimum of `min_theta rate(theta)/rate(0)` was `1.000001`.
   Conversely for `nu < 2` (where `pi` has no second moment) the rate is negative for all
   `theta`, and `nu = 2` is exactly critical.
5. **`J` accelerates, a lot, exactly when `Sigma` is ill-conditioned.** In `d = 2`, with
   `J = theta Omega`, the exact rate increases monotonically from `rate(0)` to the closed-form
   limit

        lim_{theta -> infinity} rate(theta) = Tr(Sigma^{-1}) (nu - 2)/nu                 (26)

   (verified to 5 decimals on random `Sigma, nu`). Acceleration factors `rate(inf)/rate(0)`:

        kappa \ nu     3        4        6       12       30
             1      1.000    1.000    1.000    1.000    1.000
             2      1.183    1.230    1.297    1.386    1.452
             3      1.477    1.577    1.699    1.841    1.934
             5      2.117    2.310    2.525    2.756    2.901
            10      3.766    4.175    4.604    5.046    5.317
            30     10.422   11.669   12.934   14.212   14.984
           100     33.751   37.917   42.100   46.296   48.817

   (The mechanism is the classical one: the eigenvalues of `B = (2beta/nu)(I-J)Sigma^{-1}`
   move from `∝ {lam_1, lam_2}` to a complex-conjugate pair with common real part
   `∝ (lam_1+lam_2)/2`; the slow direction is destroyed.)
   For `kappa = 1` there is **no** acceleration, matching §2's caveat.

6. **The deterministic part is unconditionally stable.** *Lemma B.* For every `Sigma ≻ 0`
   and every skew `J`, every eigenvalue `mu` of `(I-J)Sigma^{-1}` satisfies
   `Re(mu) >= lam_min(Sigma^{-1}) > 0`. *Proof:* if `(I-J)Sigma^{-1}v = mu v`, put
   `w := Sigma^{-1}v`; then `(I-J)w = mu Sigma w`, so `w^*w - w^*Jw = mu · w^* Sigma w`, and
   `w^* J w` is purely imaginary (`(w^*Jw)^* = w^*J^T w = -w^*Jw`), whence
   `Re(mu) = ||w||^2/(w^* Sigma w) >= 1/lam_max(Sigma) = lam_min(Sigma^{-1})`. □
   (Verified on `3000` random `(d <= 5, Sigma, J)` with `||J||` up to `10^3`: the observed
   `min Re(mu)/lam_min(Sigma^{-1})` was `1.000000`, i.e. the bound is attained.)
   So *only the multiplicative noise* `sigma^2 = q ∝ ||x||^2` can destabilise the second
   moment, and — empirically — it does so exactly when `nu <= 2`, i.e. exactly when `pi`
   itself has no second moment. This is the structural reason we expect the sharp condition
   to be `nu > 2 alpha` with no reference to `J`.

**So what, exactly, is the status of the smallness condition (14)–(16)?**

* It is a *sufficient* condition attached to a *particular* Lyapunov function. Its failure
  says nothing about the process. Because `Tr(sym(J Sigma^{-1})) = 0`, the Euclidean-`V`
  drift certificate is *guaranteed* to break for large `J` — the certificate is structurally
  incapable of covering the regime where `J` is most useful.
* There *is* a real phenomenon behind it, but it is not loss of ergodicity: it is that with
  a large `J` the flow **is no longer contracting in the Euclidean metric**. Indeed
  `<b_J(x)-b_J(y), x-y> = -(2b/nu)(x-y)^T[Sigma^{-1} - sym(J Sigma^{-1})](x-y)`, so the
  one-sided Lipschitz constant `m` in the paper's **Assumption 12(18)** is positive only if
  `Sigma^{-1} - sym(J Sigma^{-1}) ≻ 0` — a smallness condition of exactly the same shape.
  The relaxation is *rotational/oscillatory* (the spectrum of `A*` becomes complex for
  `theta >~ 1` in our example), so any Euclidean synchronous-coupling / contraction argument
  — and hence the paper's non-asymptotic Theorem 14 as literally stated — will keep a
  smallness condition. **Ergodicity is `J`-free; Euclidean contractivity is not.**
* **Practice (large `J` at fixed `Sigma`).** (a) Do *not* read (15)–(16) as a tuning rule for
  `J`; they are certificate limits, not stability limits. (b) In the `Sigma^{-1}` geometry
  there is no limit at all under (C_Sigma), and empirically none under `nu > 2`.
  (c) The useful diagnostic is `M_J ≻ 0 / G ≻ 0 /` Hurwitz-ness of (25): compute
  `lam_min(M_J)`, `lam_min(G)`, and (for `alpha = 1`) the spectral abscissa of `A*`; the last
  is a `d(d+1)/2 x d(d+1)/2` eigenvalue problem and gives the exact rate. (d) The gains
  saturate: `rate(theta)` is within a few percent of its `theta -> infinity` limit already at
  `theta ≈ 5` in our examples, while the discretization step must shrink like
  `eta = O(1/||B||) = O(1/theta)`; so **there is an optimal finite `theta`** for the
  *algorithm* even though the SDE keeps improving. Practically, choose `theta` of order the
  inverse of the slowest time scale, `theta ≈ kappa(Sigma)^{1/2}`–`kappa(Sigma)`, and verify
  with the exact rate computation.

**What is genuinely open.** We have a *proof* of `J`-free geometric ergodicity only under
(C_Sigma) (or (C'_Sigma)). The numerical evidence says the correct statement should be
"`nu > 2 alpha` suffices, for every `J`". Note that the natural test functional reproduces
exactly (C_Sigma): applying `Tr(Sigma^{-1} ·)` to (25) gives
`Tr(Sigma^{-1}A*(S)) = -(4b/nu)Tr(Sigma^{-2}S) + (2/nu)Tr(Sigma^{-1})Tr(Sigma^{-1}S)`
(the `J`-dependence cancels because `Sigma^{-1}B + B^T Sigma^{-1} = (4b/nu)Sigma^{-2}`), and
bounding `Tr(Sigma^{-2}S) >= lam_min(Sigma^{-1})Tr(Sigma^{-1}S)` yields (C_Sigma) again. A
proof of the sharp statement therefore needs a `J`-adapted (non-skew-blind) Lyapunov
function — e.g. the Perron eigenvector `A_*` of §5.3.

---

## 6. Numerics for (T2), `d = 2`

Setting: `d = 2`, `alpha = 1`, `mu = 0`, `beta = (d+nu)/2 - 1 = nu/2`, `J = theta Omega`,
`Omega = [[0,1],[-1,0]]`, `B = (I - theta Omega) Sigma^{-1}` (since `2beta/nu = 1` when `d = 2`).

### 6.1 Direct check of the drift conditions (`chk7_sim.py`, part A)

`nu = 6`, `Sigma^{-1} = diag(1,10)` (so `kappa = 10`, `2beta = 6`, `Tr(Sigma^{-1}) = 11`,
`theta* = 0.4685`, and (C_Sigma) *fails*: `6 < 11`, `lam_min(G) = -5`):
minimising `u^T M_J u` and `u^T G u` over the unit circle and evaluating `L_J V` at `||x|| = 50`:

    theta      min_u u'M_J u      L_J V at |x|=50        Euclidean drift?
    0.000            4.0000              -3329.3               OK
    0.250            2.7760              -2309.3               OK
    0.4685          -0.0001                  +4.1              FAILS (exactly at theta*)
    0.600          -2.2165               +1851.0               FAILS
    1.000         -10.4499               +8712.2               FAILS
    5.000        -114.1947              +95166.2               FAILS

So for `theta > theta*` the Euclidean drift condition fails *strongly* — `L_J V` is large and
positive in the bad direction at large `||x||` — yet (next subsection) the process is
geometrically ergodic with a *better* rate. A second setting, `Sigma^{-1} = diag(1,3)`,
`nu = 6`, has `theta* = 1.1547` and (C_Sigma) *holds* (`6 > 4`, `lam_min(G) = 2`), giving the
`J`-free guaranteed rate `lam_* = 2·2/(6·3) = 0.2222` from (22) — valid for all `theta`.

### 6.2 Exact rate, and Monte-Carlo estimate by fitting the decay (`chk7_sim.py`, part B)

Test statistic: `D(t) := || E[X_t X_t^T] - S_infinity ||_F`, `S_infinity = (nu/(nu-2))Sigma`,
started from the deterministic point `x_0 = (6,6)`; `n` Euler–Maruyama particles, step
`eta = min(2.5·10^{-4}, 0.004/||B||_2)`; `rate` fitted by least squares on `log D(t)`.
`nu = 6`, `Sigma^{-1} = diag(1,10)`, `n = 6·10^4`:

    theta   eig(A*)                                  exact rate   MC fit   ODE fit
    0.0     -16.740, -11.000,  -1.593                  1.5930      1.5907   1.6025
    0.5     -15.416, -11.922,  -1.995                  1.9950      1.9549   1.9882
    1.0     -13.086 ± 3.789i,  -3.161                  3.1608      3.1752   3.1537
    2.0     -11.746 ± 10.637i, -5.840                  5.8404      6.5858   6.4599
    5.0     -11.103 ± 30.770i, -7.127                  7.1273      5.8439   5.9428
    theta -> infinity limit  Tr(Sigma^{-1})(nu-2)/nu = 7.3333

The MC fits track the exact rates closely for `theta <= 1`. For `theta >= 2` the spectrum of
`A*` is **complex** (oscillatory relaxation — the signature of non-reversibility), and a
single log-linear fit over a short window is biased; note the *exact ODE* (25) fitted the same
way gives essentially the same biased numbers (6.46 vs 5.84; 5.94 vs 7.13), which shows the
discrepancy is a **fitting artifact of the oscillation, not a Monte-Carlo or discretization
error** — MC and ODE agree to `<= 2%` throughout. Fitting over several oscillation periods
(`chk11_final.py`) removes the bias.

Summary of the practically relevant numbers at `nu = 6`, `kappa = 10`: `theta = 0 -> 1.59`,
`theta = 1 -> 3.16` (2.0x), `theta = 5 -> 7.13` (4.5x), ceiling `7.33` (4.6x).

### 6.3 Sanity checks

* Invariance under discretization (`chk9_inv_mc.py`, `d = 2`, `nu = 6`,
  `Sigma = diag(1, 1/3)`, `eta = 2·10^{-4}`, `n = 6·10^4`, `2.5·10^4` steps): empirical
  covariance `diag(1.487, 0.495)` vs target `diag(1.5, 0.5)`; KS distances to the `t_6`
  marginals `0.0046` and `0.0035`, and to the `F(d,nu)` law of `x^T Sigma^{-1}x / d`
  `0.0043`, all at/below the 95% critical value `1.36/sqrt(n) = 0.0056`.
* Isotropic control: `Sigma = I`, `d = 2`, `nu in {3,6,12}` — the exact rate is *exactly*
  constant in `theta` over `theta in [0, 500]`, confirming §2's "correct but inert".

---

## 7. Implementation directives for the experiment code

1. **Dynamics.** `x_{k+1} = x_k + eta e^{(U-U0)(x_k)}(J - I) grad U0(x_k) + sqrt(2 eta) e^{(U-U0)(x_k)/2} xi_{k+1}`,
   `xi ~ N(0, I_d)`. Build `J` once, assert `norm(J + J.T) < 1e-12` (a non-constant or
   non-skew `J` breaks invariance — Proposition 0).
2. **For (T2), `alpha = 1`, hard-code the closed forms** (do not autodiff):
   `beta = (d+nu)/2 - 1`; `q(x) = 1 + (x-mu)^T Sigma^{-1} (x-mu)/nu`;
   drift `= (2 beta/nu) (J - I) Sigma^{-1} (x - mu)` (linear, `J`-dependent, *no* `q` factor);
   `sigma(x) = sqrt(q(x))`. Precompute `Sigma^{-1}` and `Bmat = (2 beta/nu)(I - J)Sigma^{-1}`
   so the step is one `matvec` plus one `sqrt`.
3. **Use an anisotropic `Sigma`.** With `Sigma ∝ I` the skew term is provably inert
   (§2, §5.5): the exact rate is independent of `theta` to machine precision. Use
   `kappa(Sigma) >= 5` (e.g. `Sigma^{-1} = diag(1, 10)` in `d = 2`) or the acceleration will
   not be visible. Always include an isotropic run as a *negative control*.
4. **Report `V_Sigma`, not `1 + ||x||^2`.** Diagnostics, drift plots and any Lyapunov-based
   claim should use `V_Sigma(x) = 1 + (x-mu)^T Sigma^{-1}(x-mu)`; the Euclidean `V` gives a
   spuriously negative verdict for `theta > theta*` (17).
5. **Certificates to compute and print** for each `(Sigma, nu, alpha, J)`:
   `lam_min(M_J)` with `M_J = (nu-2alpha)Sigma^{-1} - alpha beta [J, Sigma^{-1}]` (Euclidean),
   `lam_min(G)` with `G = 2 alpha beta Sigma^{-2} - Tr(Sigma^{-1})Sigma^{-1}` (`J`-free), and
   `r = 2 alpha - 2`.
6. **Exact rate (`alpha = 1` only).** Assemble the `d(d+1)/2 x d(d+1)/2` matrix of
   `A*(S) = -B S - S B^T + (2/nu)Tr(Sigma^{-1}S) I` in an orthonormal basis of symmetric
   matrices; `rate = -max Re eig(A*)`. Use it to (a) set simulation horizons `T ≈ 8/rate`,
   (b) validate the MC fit, (c) sweep `theta` cheaply. The Perron eigenvector of the adjoint
   `A` is the *optimal* quadratic Lyapunov matrix `A_*` (then `L_J V_{A_*} = -rate·V_{A_*} + const`).
7. **Step size.** `||Bmat||_2 ≈ (2beta/nu)sqrt(1+theta^2) lam_max(Sigma^{-1})`; take
   `eta <= 0.004/||Bmat||_2` (used throughout §6) — so `eta = O(1/theta)`. Report iteration
   counts, not just continuous time, or the acceleration will be overstated.
8. **Rate fitting.** For `theta` above roughly `theta*` the spectrum of `A*` is complex and
   the relaxation oscillates. Fit `log||E[X X^T] - S_infinity||_F` over a window spanning
   **several oscillation periods** (`Im(eig)` gives the period), or fit the envelope; a short
   window biases the estimate by `10`–`20%` in either direction. Always cross-check against
   the exact ODE (25) integrated with the same fitting procedure — the difference between
   the two is the genuine MC + discretization error.
9. **Estimator variance.** `E||X_t||^2` has finite Monte-Carlo variance only for `nu > 4`.
   Use `nu >= 6` for moment-based rate fits; for `nu in (2,4]` use the exact ODE (25) or a
   bounded statistic (e.g. `||x||^2/(1+||x||^2)`, or quantiles).
10. **Correctness test to run first.** Long-run empirical covariance must match
    `(nu/(nu-2)) Sigma` and the marginals must pass a KS test against `t_nu`, *for every*
    `theta` used. This is the operational form of Proposition 0 and catches sign errors in
    `J - I` immediately.

---

## 8. Summary of what was proved, and under which hypotheses

| Statement | Hypotheses | Depends on `J`? |
|---|---|---|
| `pi ∝ e^{-U}` invariant for (5_J) | `U0 in C^2`, `J` constant skew | no |
| `L_J` non-symmetric in `L^2(pi)` | `J grad U0 not== 0` | — |
| Dirichlet form / Prop. 7 unchanged | (H0) | no |
| `L_J V = 2[d - <x,gradU0> + <x,J gradU0>]e^{U-U0}` | `V = 1+||x||^2` | yes |
| Assumption 1 & Thms 2,3 hold verbatim for (T1) | `iota > 1+d/2`, `beta > d/2` | **no** (exact cancellation) |
| (T2), Euclidean `V`: `M_J ≻ 0` needed, `r = 2alpha-2` | `alpha >= 1`, `nu > 2alpha` | **yes** (smallness (14)–(16)) |
| `L_J (Phi ∘ U0) = L(Phi ∘ U0)` | `Phi in C^2`, `U0 in C^2`, `J` constant skew | **no** |
| (T2), `V_Sigma`: drift (21), Theorem 4, `r = 2alpha-2` | `alpha >= 1`, `nu > 2alpha`, (C_Sigma) | **no** |
| `E_x V_Sigma(X_t) <= e^{-lam_* t}V_Sigma(x) + b_*/lam_*` | as Theorem 4, `alpha = 1` | **no** |
| geometric ergodicity for all `J` under only `nu > 2alpha` | — | conjecture (strong numerics) |
