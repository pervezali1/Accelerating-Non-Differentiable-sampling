# Invariance of $\pi\propto e^{-U}$ for the skew-anchored Langevin SDE (SAL-SDE)

**Slug:** `invariance`.
**Object of study.** With $U,U_0:\mathbb R^d\to\mathbb R$, $M:=\int_{\mathbb R^d}e^{-U}\,dx<\infty$,
$\pi(dx)=M^{-1}e^{-U(x)}dx$, and a **constant** matrix $J\in\mathbb R^{d\times d}$ with $J^\top=-J$:

$$
\boxed{\;dX_t \;=\; e^{(U-U_0)(X_t)}\,(J-I)\nabla U_0(X_t)\,dt \;+\;\sqrt2\,e^{(U-U_0)(X_t)/2}\,dW_t\;}
\tag{SAL}
$$

For $J=0$ this is exactly the anchored Langevin SDE (AL-SDE), Eq. (5)–(7) of
Gürbüzbalaban–Nguyen–Zhang–Zhu (arXiv:2509.19455).

**Verdict on invariance: TRUE.** $\pi\propto e^{-U}$ is invariant for (SAL) for *every* constant
skew-symmetric $J$, under exactly the regularity/integrability hypotheses the paper already uses for
$J=0$, plus a non-explosion condition that *does* acquire a $J$-dependent term in general (§7) but
is **unchanged** whenever the anchor $U_0$ is radial about the origin — which covers the paper's own
heavy-tailed targets (T1) and the isotropic/centred Student-$t$ (T2). All of §1–§6 is proved below and
verified numerically in §9.

> **Two warnings that matter more than the invariance itself, both proved below.**
>
> 1. **(§4.5, a theorem.)** If $U$ and $U_0$ are *radial* **and** the anchored drift is linear, i.e.
>    $e^{(U-U_0)(x)}\nabla U_0(x)=\kappa x$ — which is exactly the paper's own §6.4 heavy-tailed target
>    (T1) in its canonical form $\iota=\beta+1$, and any isotropic Student-$t$ — then a constant skew
>    $J$ gives **exactly zero** acceleration:
>    $\|e^{tL_J}f\|_{L^2(\pi)}=\|e^{tL}f\|_{L^2(\pi)}$ for every $f$ and every $t$, because
>    $\Gamma=\kappa\mathcal R$ is a multiple of the rotation generator, which commutes with $L$ and
>    acts unitarily on $L^2(\pi)$. (For a radial anchor with a *nonlinear* anchored drift one still
>    gets $L_J=L$ on all radial functions, so no gain can come from the radial modes.) Experiments on
>    radial heavy-tailed targets will show no gain, and any apparent gain is discretisation noise.
> 2. **(§4.4.)** Even for anisotropic targets the acceleration **saturates** as $\|J\|\to\infty$: by
>    Fact B, the added drift conserves $U_0$ exactly, so the family is not "relaxation enhancing" in
>    the sense of Constantin–Kiselev–Ryzhik–Zlatoš. §9.6 shows a $\times5.5$ plateau.
>
> The extension is therefore worth doing, but must be tested on an **anisotropic** target.

### TL;DR for whoever writes the experiment code

1. $\pi\propto e^{-U}$ **is** invariant for the SAL-SDE, for every constant skew $J$ — proved in §2–§3,
   verified to $|z|\le1.4$ over $2.4\times10^7$ exact samples and to machine precision at the discrete
   level (§9.4, §9.6).
2. Use **(T2) with anisotropic $\Sigma$** as the demo target (§5.5). Drift is linear:
   $b_J(x)=\tfrac{2\beta}{\nu}(J-I)\Sigma^{-1}(x-\mu)$, diffusion $\sqrt{2q(x)}$.
3. Use **(T1)** and **isotropic (T2)** only as *negative controls*: §4.5 proves the gain there is
   **exactly zero**.
4. Sanity-check any implementation with the two zero-tests of §9.3–§9.4 *including their controls*
   (a symmetric matrix in place of $J$; a state-dependent $J(x)$ violating (6.1)) — without controls
   these tests have no power.
5. Sweep $\|J\|$ geometrically and expect a **plateau** (§4.4, §9.6: $\times5.5$ at $\kappa(\Sigma)=12$,
   $d=2$), not unbounded gain. Also expect the Euler bias to grow with $\|J\|$ ($\approx1.6\times$ for
   the $J$ used in §9.10), so compare at matched bias, not matched $\eta$.

---

## 0. Notation and standing hypotheses

Write throughout

$$
G(x):=e^{(U-U_0)(x)}>0,\qquad \rho_0(x):=e^{-U_0(x)}>0,\qquad \rho(x):=e^{-U(x)}>0 .
$$

The **single structural identity** on which everything rests is

$$
G(x)\,\rho(x)=e^{(U-U_0)(x)}e^{-U(x)}=e^{-U_0(x)}=\rho_0(x),
\qquad\text{i.e.}\qquad G\,\pi=\tfrac1M\rho_0 .
\tag{0.1}
$$

*The diffusion coefficient of the anchored dynamics is precisely the Radon–Nikodym factor that turns
the target $\pi$ into the (possibly non-normalisable) anchor measure $\rho_0\,dx$.* Everything below
is a consequence of (0.1) together with two lines of linear algebra.

**Hypotheses.** We name them so that we can point at exactly which one each step needs.

* **(H1) Regularity.** $U_0\in C^2(\mathbb R^d)$ and $U\in C^1(\mathbb R^d)$.
  (Same as the paper: "Under (8) and the assumptions $U_0\in C^2$ and $U\in C^1$…".) Consequently
  $\rho_0\in C^2$, $G\in C^1$, $\sigma=G^{1/2}\in C^1$ and $\sigma>0$.
* **(H2) Normalisability of the target.** $M=\int e^{-U}<\infty$. **No integrability of $\rho_0=e^{-U_0}$
  is required anywhere.**
* **(H3) Well-posedness / non-explosion.** (SAL) has a unique (weak or strong) solution for every
  starting point, with explosion time $\zeta=\infty$ a.s. See §7 for the Lyapunov condition; this is
  the *only* place where $J$ can genuinely cost something.
* **(H4) No boundary flux.** For the class $\mathcal D$ of test functions used, the integration by
  parts on $\mathbb R^d$ produces no term at infinity. Concretely: **(H4) holds for
  $\mathcal D=C_c^\infty(\mathbb R^d)$ with no extra assumption**, since all integrands are then compactly
  supported. For non-compactly-supported $f$ one needs, on $B_R^c$,
  $\oint_{\|x\|=R}\rho_0\big(|\nabla f|+|f|\,\|J\nabla U_0\|\big)\,dS\to0$ along a sequence $R_n\uparrow\infty$.
  All statements are first proved on $C_c^\infty$ and then extended by the usual core/closure argument
  (see the remark at the end of §2).

The **generator-level** statements (§1–§6) need only (H1),(H2),(H4). (H3) is what upgrades
"$\int L_Jf\,d\pi=0$ for all $f\in C_c^\infty$" to "$\pi$ is an invariant probability measure of the
Markov semigroup $P_t^J$".

---

## 1. The generator of (SAL)

(SAL) is the Itô SDE $dX_t=b_J(X_t)dt+\Sigma(X_t)dW_t$ with

$$
b_J(x)=G(x)(J-I)\nabla U_0(x),\qquad \Sigma(x)=\sqrt2\,G(x)^{1/2}I_d .
$$

The diffusion matrix is $a(x):=\tfrac12\Sigma\Sigma^\top=G(x)I_d$, so for $f\in C^2$

$$
L_Jf=\langle b_J,\nabla f\rangle+\operatorname{Tr}\!\big(a\,\nabla^2f\big)
     =G\big\langle (J-I)\nabla U_0,\nabla f\big\rangle+G\,\Delta f .
$$

Splitting $(J-I)=J-I$,

$$
\boxed{\;L_Jf \;=\; e^{(U-U_0)}\Big(\Delta f-\langle\nabla U_0,\nabla f\rangle+\langle J\nabla U_0,\nabla f\rangle\Big)\;}
\tag{1.1}
$$

as claimed. Write

$$
L_Jf=Lf+\Gamma f,\qquad
Lf:=G\big(\Delta f-\langle\nabla U_0,\nabla f\rangle\big),\qquad
\Gamma f:=\langle c,\nabla f\rangle,\quad c:=G\,J\nabla U_0 .
\tag{1.2}
$$

$L$ is exactly the anchored generator of the paper ("$Lf=e^{U-U_0}(\Delta f-\langle\nabla U_0,\nabla f\rangle)$").
Two further ways of writing (1.1) are used repeatedly:

**(a) Time-change form.** With $L_0f:=\Delta f-\langle\nabla U_0,\nabla f\rangle$ (the generator of the
overdamped Langevin diffusion for the *anchor* potential $U_0$) and
$L_0^Jf:=\Delta f+\langle(J-I)\nabla U_0,\nabla f\rangle$ (the **Hwang-type non-reversible Langevin
generator for $U_0$**),

$$
L=G\cdot L_0,\qquad L_J=G\cdot L_0^{J}.
\tag{1.3}
$$

**(b) Divergence form.** Using (0.1), for $f\in C^2$,

$$
\rho\,L_Jf=\rho_0\big(\Delta f-\langle\nabla U_0,\nabla f\rangle\big)+\rho_0\langle J\nabla U_0,\nabla f\rangle
        =\nabla\!\cdot\!\big(\rho_0\nabla f\big)+\big\langle V,\nabla f\big\rangle,
\qquad V:=\rho_0J\nabla U_0=-J\nabla\rho_0 ,
\tag{1.4}
$$

because $\nabla\cdot(\rho_0\nabla f)=\rho_0\Delta f+\langle\nabla\rho_0,\nabla f\rangle
=\rho_0(\Delta f-\langle\nabla U_0,\nabla f\rangle)$ and
$\rho_0J\nabla U_0=-J\nabla\rho_0$ (since $\nabla\rho_0=-\rho_0\nabla U_0$).

*Hypotheses used in §1:* (H1) only (Itô's formula needs $f\in C^2$, $b_J,\Sigma$ locally bounded
measurable; $b_J$ continuous by (H1)).

---

## 2. $\displaystyle\int L_Jf\,d\pi=0$

### 2.1 Two pieces of linear algebra

> **Fact A.** If $J^\top=-J$ and $H^\top=H$ (both $d\times d$) then $\operatorname{Tr}(JH)=0$.
> *Proof.* $\operatorname{Tr}(JH)=\operatorname{Tr}\big((JH)^\top\big)=\operatorname{Tr}(H^\top J^\top)=-\operatorname{Tr}(HJ)=-\operatorname{Tr}(JH)$. $\square$

> **Fact B.** If $J^\top=-J$ then $\langle v,Jv\rangle=0$ for every $v\in\mathbb R^d$.
> *Proof.* $\langle v,Jv\rangle=\langle J^\top v,v\rangle=-\langle Jv,v\rangle=-\langle v,Jv\rangle$. $\square$

Fact A is what makes the *rotational* field divergence-free; Fact B is what makes the perturbation
*energy-neutral* (it is tangent to the level sets of $U_0$) and is what gives antisymmetry in $L^2(\pi)$
and the $J$-free Lyapunov estimate for radial anchors.

### 2.2 The vector field $V:=\rho_0\,J\nabla U_0=-J\nabla\rho_0$ is divergence free

$$
\nabla\!\cdot\!V=-\sum_{i}\partial_i\Big(\sum_j J_{ij}\partial_j\rho_0\Big)
=-\sum_{i,j}J_{ij}\,\partial_i\partial_j\rho_0
=-\operatorname{Tr}\!\big(J\,\nabla^2\rho_0\big)=0
\tag{2.1}
$$

by **Fact A**, since the Hessian $\nabla^2\rho_0$ is symmetric. **This is exactly where $U_0\in C^2$
(H1) is needed** — the interchange $\partial_i\partial_j\rho_0=\partial_j\partial_i\rho_0$ (Schwarz)
requires $\rho_0\in C^2$. (For $U_0$ merely $C^1$, (2.1) still holds in the sense of distributions:
$\langle\nabla\cdot V,\varphi\rangle=\sum_{i,j}J_{ij}\int\rho_0\,\partial_i\partial_j\varphi=0$ for
$\varphi\in C^\infty_c$, again by Fact A. So the invariance survives $U_0\in C^1$ with distributional
Hessian; only the pointwise statement needs $C^2$.)

### 2.3 The computation

Let $f\in C_c^\infty(\mathbb R^d)$. By (1.4),

$$
M\!\int L_Jf\,d\pi=\int_{\mathbb R^d}\nabla\!\cdot\!\big(\rho_0\nabla f\big)\,dx
\;+\;\int_{\mathbb R^d}\big\langle V,\nabla f\big\rangle\,dx .
\tag{2.2}
$$

*First term.* $\rho_0\nabla f$ is $C^1$ with compact support, so
$\int\nabla\cdot(\rho_0\nabla f)\,dx=0$ (divergence theorem on a large ball; this is (H4)). This is the
$J=0$ statement — the paper's Theorem 2 / Lemma 5.

*Second term.* $\langle V,\nabla f\rangle=\nabla\cdot(fV)-f\,\nabla\cdot V=\nabla\cdot(fV)$ by (2.1),
and $fV$ is $C^1$ with compact support, so the integral is $0$ (again (H4)).

Hence

$$
\boxed{\;\int_{\mathbb R^d}L_Jf\,d\pi=0\qquad\text{for all }f\in C_c^\infty(\mathbb R^d).\;}
\tag{2.3}
$$

**Exactly which facts were used.** (i) the structural identity (0.1) [$G\pi\propto\rho_0$];
(ii) **Fact A**, $\operatorname{Tr}(J\nabla^2\rho_0)=0$, i.e. skewness of $J$ against symmetry of a Hessian;
(iii) $U_0\in C^2$ (H1) so that the Hessian is symmetric; (iv) $M<\infty$ (H2) so that $\pi$ is a
probability measure; (v) (H4) no flux at infinity. **Fact B is not needed for (2.3)** — it enters in
§4 (antisymmetry with $f=g$), §6 (state-dependent $J$), and §7 (Lyapunov).

### 2.4 From (2.3) to invariance of the semigroup

(2.3) says $\pi$ is *infinitesimally invariant*. To conclude that $\pi P_t^J=\pi$ for the Markov
semigroup one needs (H3): if $(X_t)$ is non-explosive and $f\in C_c^\infty$, Dynkin/Itô gives
$P_t^Jf(x)-f(x)=\int_0^tP_s^JL_Jf(x)\,ds$; integrating against $\pi$ and using Fubini
(legitimate because $\|L_Jf\|_\infty<\infty$ for $f\in C_c^\infty$, $G$ continuous) plus (2.3) applied
to the (again $C_b$, by Fubini and a standard approximation) functions $P^J_sf$ yields
$\int P_t^Jf\,d\pi=\int f\,d\pi$. The clean way to avoid circularity is the standard
Echeverría–Ethier–Kurtz theorem: *if $C_c^\infty$ is a core for the generator, the martingale problem
is well posed, and $\int L_Jf\,d\pi=0$ for all $f\in C_c^\infty$, then $\pi$ is invariant*
(Echeverría 1982; Ethier–Kurtz, *Markov Processes*, Thm 4.9.17). Well-posedness of the martingale
problem holds here because $a=G\,I_d$ is continuous and **locally uniformly elliptic** ($G>0$
continuous) and $b_J$ is continuous, plus non-explosion (H3) (Stroock–Varadhan).
This is the same route the paper takes for $J=0$.

**Remark (extension beyond $C_c^\infty$).** Since $\pi$ is invariant, $P^J_t$ extends to a
$C_0$-contraction semigroup on $L^2(\pi)$; $C_c^\infty$ is a core for its generator under (H1),(H3)
and local ellipticity, so (2.3), and the identities of §4, extend to the whole domain by closure.
For explicit non-compactly-supported $f$ one may instead verify (H4) directly.

---

## 3. Fokker–Planck: $c$ is a $\pi$-divergence-free perturbation

The forward (Fokker–Planck/Kolmogorov) equation for the law $p_t$ of (SAL) is

$$
\partial_tp=-\nabla\!\cdot\!\big(b_Jp\big)+\sum_{i,j}\partial_i\partial_j\big(a_{ij}p\big)
=-\nabla\!\cdot\!\big(b_Jp\big)+\Delta\big(Gp\big)
=-\nabla\!\cdot\!\underbrace{\Big(b_Jp-\nabla(Gp)\Big)}_{=:\ \mathcal J[p]} .
\tag{3.1}
$$

Insert $p=\pi$. By (0.1), $G\pi=\rho_0/M$, so
$\nabla(G\pi)=\tfrac1M\nabla\rho_0=-\tfrac1M\rho_0\nabla U_0=-G\pi\,\nabla U_0$. Therefore

$$
\mathcal J[\pi]=G\pi(J-I)\nabla U_0+G\pi\,\nabla U_0=G\pi\,J\nabla U_0=\pi\,c ,
\qquad c(x)=e^{(U-U_0)(x)}J\nabla U_0(x).
\tag{3.2}
$$

Hence

$$
\partial_t\pi=-\nabla\!\cdot\!\big(\pi c\big),\qquad\text{and}\qquad
\pi\ \text{stationary}\iff \nabla\!\cdot\!\big(\pi c\big)=0 .
\tag{3.3}
$$

And indeed, by (0.1) again, $\pi c=\tfrac1M\rho_0J\nabla U_0=-\tfrac1MJ\nabla\rho_0=\tfrac1MV$, so
$\nabla\cdot(\pi c)=\tfrac1M\nabla\cdot V=0$ by (2.1) — **Fact A**. Equivalently, in the "$\pi$-weighted
divergence" form used by Hwang et al.,

$$
\nabla\!\cdot\!(\pi c)=\pi\Big(\nabla\!\cdot\!c-\langle\nabla U,c\rangle\Big)=0
\quad\Longleftrightarrow\quad
\nabla\!\cdot\!c=\langle\nabla U,c\rangle .
\tag{3.4}
$$

Three comments.

1. **$J=0$ gives detailed balance.** $\mathcal J[\pi]\equiv0$: the stationary probability current
   vanishes identically. That is the analytic form of "the anchored Langevin SDE is reversible"
   (paper, Lemma 5).
2. **$J\ne0$ gives a genuine non-equilibrium steady state.** $\mathcal J[\pi]=\tfrac1MV=-\tfrac1MJ\nabla\rho_0\not\equiv0$
   whenever $J\nabla U_0\not\equiv0$: the stationary current is a nonzero but divergence-free
   (solenoidal) circulation. So (SAL) is **not** reversible w.r.t. $\pi$ — which is the whole point.
3. **General form.** Any field of the form $\pi c=\nabla\cdot S$ with $S=S(x)$ a $C^1$ *skew-symmetric
   matrix field*, $(\nabla\cdot S)_i:=\sum_j\partial_jS_{ij}$, is automatically divergence free:
   $\nabla\cdot\nabla\cdot S=\sum_{i,j}\partial_i\partial_jS_{ij}=0$ by Fact A applied to the symmetric
   "matrix" $(\partial_i\partial_j\cdot)$. Our case is $S=-\tfrac1M\rho_0J$ (constant $J$):
   $(\nabla\cdot S)_i=-\tfrac1M\sum_jJ_{ij}\partial_j\rho_0=\tfrac1MV_i$. ✓
   This is the canonical parametrisation of all $\pi$-preserving drift perturbations.

---

## 4. $L^2(\pi)$ decomposition, Dirichlet form, and why the rate cannot degrade

Work in $\mathcal H:=L^2(\pi)$, $\langle f,g\rangle_\pi=\int fg\,d\pi$, on the core $\mathcal D=C_c^\infty$.

### 4.1 $L$ is symmetric; $\Gamma$ is antisymmetric

For $f,g\in\mathcal D$, by (1.4) and integration by parts (H4),

$$
\langle g,Lf\rangle_\pi=\tfrac1M\!\int g\,\nabla\!\cdot\!(\rho_0\nabla f)\,dx
=-\tfrac1M\!\int\rho_0\langle\nabla f,\nabla g\rangle\,dx
=-\int e^{U-U_0}\langle\nabla f,\nabla g\rangle\,d\pi .
\tag{4.1}
$$

Symmetric in $(f,g)$ $\Rightarrow$ $L=L^{*}$ (on the core) — the paper's Lemmas 5 and 6.

For $\Gamma f=\langle c,\nabla f\rangle=\tfrac{1}{M\pi}\langle V,\nabla f\rangle$ with $\nabla\cdot V=0$:

$$
\langle g,\Gamma f\rangle_\pi=\tfrac1M\!\int g\,\langle V,\nabla f\rangle\,dx
=\tfrac1M\!\int\Big[\nabla\!\cdot\!(fgV)-fg\underbrace{\nabla\!\cdot\!V}_{=0}-f\langle V,\nabla g\rangle\Big]dx
=-\langle f,\Gamma g\rangle_\pi .
\tag{4.2}
$$

So $\Gamma^*=-\Gamma$. Consequently the (unique) decomposition into symmetric and antisymmetric parts is

$$
\boxed{\;L_J=\underbrace{L}_{=L_{\rm sym}}+\underbrace{\Gamma}_{=L_{\rm anti}},\qquad
L_{\rm sym}=\tfrac12(L_J+L_J^*)=L,\qquad L_{\rm anti}=\tfrac12(L_J-L_J^*)=\Gamma. }
\tag{4.3}
$$

**$L_{\rm sym}$ is exactly the original reversible anchored generator** — the skew perturbation
contributes nothing to it.

### 4.2 The Dirichlet form is unchanged

Take $g=f$ in (4.2): $\langle f,\Gamma f\rangle_\pi=-\langle f,\Gamma f\rangle_\pi=0$. (Pointwise this is
$\int\langle V,\nabla \tfrac{f^2}{2}\rangle=0$; if one instead takes $f=U_0$ one sees the *pointwise*
version of the same phenomenon, $\langle J\nabla U_0,\nabla U_0\rangle=0$ by **Fact B** — the added
drift is everywhere tangent to the level sets of $U_0$, so it neither ascends nor descends $U_0$.)
Therefore

$$
\boxed{\;\mathcal E_J(f):=-\int f\,L_Jf\,d\pi=-\int fLf\,d\pi=\int e^{U-U_0}\|\nabla f\|^2\,d\pi=\mathcal E(f)\;}
\tag{4.4}
$$

— **identical to the paper's Lemma 6, for every skew $J$.** More generally the symmetric bilinear form
$\mathcal E_J(f,g)=-\tfrac12(\langle f,L_Jg\rangle_\pi+\langle g,L_Jf\rangle_\pi)=\int G\langle\nabla f,\nabla g\rangle\,d\pi$
is $J$-independent.

### 4.3 Consequence: never slower, at every time

Let $\mu_t=\text{Law}(X_t)$, $h_t:=d\mu_t/d\pi$. The density $h_t$ solves the backward-adjoint equation
$\partial_th_t=L_J^{*}h_t=(L-\Gamma)h_t$ (the $L^2(\pi)$-adjoint of $L_J$ is $L-\Gamma$ by (4.3)).
Since $\int h_t\,d\pi=1$ for all $t$,

$$
\frac{d}{dt}\chi^2(\mu_t\|\pi)=\frac{d}{dt}\|h_t-1\|_{L^2(\pi)}^2
=2\big\langle h_t-1,(L-\Gamma)h_t\big\rangle_\pi
=-2\mathcal E(h_t)+0 = -2\int e^{U-U_0}\|\nabla h_t\|^2d\pi .
\tag{4.5}
$$

The antisymmetric part contributes **exactly zero** to the dissipation. Two corollaries.

**(i) Every $\chi^2$/Poincaré guarantee of the paper transfers verbatim, with the same constant.**
If $\pi$ satisfies a Poincaré inequality with constant $C_P$ and $a:=e^{\inf_x(U-U_0)(x)}\in(0,\infty)$,
then $\mathcal E(h)\ge a\int\|\nabla h\|^2d\pi\ge (a/C_P)\operatorname{Var}_\pi(h)$, and (4.5) gives

$$
\chi^2(\mu_t\|\pi)\le\chi^2(\mu_0\|\pi)\,e^{-2at/C_P}\qquad\text{for every skew }J,
$$

which is *literally* Proposition 7 of the paper. More sharply, let
$\lambda_1:=\inf\{\mathcal E(f)/\operatorname{Var}_\pi(f):f\in\mathcal D,\ \operatorname{Var}_\pi(f)>0\}$ be the
*optimal* spectral gap of the reversible anchored generator. Then (4.5) gives, **for all $t\ge0$**,

$$
\big\|P^J_t\big\|_{L^2_0(\pi)\to L^2_0(\pi)}\le e^{-\lambda_1t},
\tag{4.6}
$$

which is the *sharp* rate for $J=0$. So the skew term can never make the $L^2(\pi)$/$\chi^2$ bound worse
— **not even transiently**: $P^J_t$ is a contraction on $L^2_0(\pi)$ for every $t$, because
$\frac{d}{dt}\|h_t\|^2=-2\mathcal E(h_t)\le0$ exactly. (This rules out the "non-normal transient
amplification" objection *in this norm*; it does not rule it out in other norms, e.g. $L^\infty$ or
$W_2$.)

**(ii) Generic strict acceleration.** Write $L_J=S+\Gamma$ with $S=L$ self-adjoint $\le0$ and
$\Gamma$ skew. Let $\lambda(J):=-\sup\operatorname{Re}\,\operatorname{spec}\big(L_J|_{L^2_0(\pi)}\big)$
be the asymptotic $L^2$ decay rate. By (4.6), $\lambda(J)\ge\lambda(0)=\lambda_1$ always.
The classical *no-improvement* criterion (Hwang–Hwang-Ma–Sheu, *Accelerating diffusions*, Ann. Appl.
Probab. 15 (2005) 1433–1444; see also Lelièvre–Nier–Pavliotis, J. Stat. Phys. 152 (2013) 237–274,
and Duncan–Lelièvre–Pavliotis, J. Stat. Phys. 163 (2016) 457–491) is:

> Assume $S$ has compact resolvent, spectral gap $\lambda_1$, eigenspace
> $\mathcal H_1=\ker(S+\lambda_1)\subset L^2_0(\pi)$ (finite dimensional). Then
> $\lambda(J)=\lambda_1$ **iff** $\Gamma\mathcal H_1\subseteq\mathcal H_1$
> (equivalently, iff the complexification of $\mathcal H_1$ contains a joint eigenvector of $S$ and $\Gamma$).

*The easy direction, which we prove.* Suppose $\Gamma\mathcal H_1\subseteq\mathcal H_1$. Then
$\Gamma|_{\mathcal H_1}$ is a real skew operator on a finite-dimensional space, hence its
complexification has an eigenvector $\varphi\in\mathcal H_1^{\mathbb C}\setminus\{0\}$,
$\Gamma\varphi=i\theta\varphi$ with $\theta\in\mathbb R$. Then
$L_J\varphi=(-\lambda_1+i\theta)\varphi$, so $\|e^{tL_J}\varphi\|=e^{-\lambda_1t}\|\varphi\|$: no gain.
$\square$ The converse requires the spectral hypotheses above and is the content of the cited theorem.

*Interpretation.* $\Gamma\mathcal H_1\subseteq\mathcal H_1$ is a codimension-positive algebraic
coincidence: the rotation must map the slowest reversible mode(s) back into the slowest eigenspace.
For a generic skew $J$ (and generic $U_0$) it fails, and then the inequality $\lambda(J)>\lambda_1$ is
strict. Hence: **the skew term can never hurt the $L^2(\pi)$ rate and generically strictly helps.**

> **However — "generic" is doing a lot of work here, and our family is systematically non-generic.**
> §4.4 shows the gain always *saturates*, and §4.5 shows that for the paper's own radial heavy-tailed
> targets the equality case $\Gamma\mathcal H_1\subseteq\mathcal H_1$ holds **exactly** (indeed
> $[L,\Gamma]=0$), so the gain is exactly zero. Read §4.3(ii) as "no harm, possible help", and §4.4–§4.5
> for when the help is actually there.
Two standard companion facts, both consequences of $\mathcal E_J=\mathcal E$ and $\Gamma^*=-\Gamma$:

* *Asymptotic variance is never larger.* For the ergodic average of $f\in L^2_0(\pi)$,
  $\sigma^2_J(f)=2\langle f,(-L_J)^{-1}f\rangle_\pi$, and $\sigma^2_J(f)\le\sigma^2_0(f)$
  (Duncan–Lelièvre–Pavliotis 2016). *Proof (four lines, uses only $S$ symmetric, $\Gamma$ skew).*
  Put $\mathcal A:=-S\succ0$ on $L^2_0$ and let $\phi$ solve $\mathcal A\phi-\Gamma\phi=f$. Then, using
  $\langle\phi,\Gamma\phi\rangle=0$ and $\langle\mathcal A^{-1}\Gamma\phi,\mathcal A\phi\rangle=\langle\Gamma\phi,\phi\rangle=0$,
  $$
  \tfrac12\sigma_0^2(f)=\langle \mathcal A^{-1}f,f\rangle
  =\langle\phi-\mathcal A^{-1}\Gamma\phi,\ \mathcal A\phi-\Gamma\phi\rangle
  =\langle\phi,\mathcal A\phi\rangle+\langle\mathcal A^{-1}\Gamma\phi,\Gamma\phi\rangle
  \;\ge\;\langle\phi,\mathcal A\phi\rangle=\langle f,\phi\rangle=\tfrac12\sigma_J^2(f). \square
  $$
* *Large deviations.* The empirical-measure LDP rate function is (weakly) increased
  (Rey-Bellet–Spiliopoulos, Nonlinearity 28 (2015) 2081).

### 4.4 A structural ceiling: this family of perturbations **cannot** give unbounded acceleration

There is a genuine limit to how much $J$ can buy, and it is caused by **Fact B**. For *any* (possibly
state-dependent) skew $J(\cdot)$, the added drift satisfies

$$
\big\langle c(x),\nabla U_0(x)\big\rangle=e^{(U-U_0)(x)}\big\langle J(x)\nabla U_0(x),\nabla U_0(x)\big\rangle=0
\qquad\text{for every }x,
\tag{4.7}
$$

i.e. **the flow $\dot x=c(x)$ conserves $U_0$ exactly** — it only stirs *within* the level sets of the
anchor. Consequently every $\phi=h(U_0)$ satisfies $\Gamma\phi=h'(U_0)\langle c,\nabla U_0\rangle=0$:
$\Gamma$ has a huge supply of non-constant eigenvectors (eigenvalue $0$) lying in
$H^1(\pi)=\mathcal D(\mathcal E)^{1/2}$ (take $h$ smooth, compactly supported in $U_0$-value). By the
*relaxation-enhancement* criterion of Constantin–Kiselev–Ryzhik–Zlatoš (Ann. of Math. 168 (2008)
643–674) — which says that for $S+\tfrac1\varepsilon\Gamma$ the decay rate tends to $\infty$ as
$\varepsilon\to0$ **iff** $\Gamma$ has no non-constant eigenvector in $H^1$ — the family
$\{sJ:s>0\}$ is **not relaxation enhancing**: $\sup_{s>0}\lambda(sJ)<\infty$.

*(Hypotheses: CKRZ is stated for $\mathcal A=-S$ with compact resolvent on the relevant space, so this
applies whenever the anchored generator has discrete spectrum; for heavy-tailed $\pi$ without a
spectral gap the statement must be re-derived. But the mechanism — (4.7) — is unconditional.)*

**What the $\|J\|\to\infty$ limit converges to (heuristic, not proved here).** When the rotation is
fast, the standard averaging picture says the effective dynamics is the diffusion obtained by
averaging $L$ over the orbits of the flow of $c$ — i.e. over the level sets of $U_0$. So
$\lim_{s\to\infty}\lambda(sJ)$ should equal the spectral gap of the **averaged (for a radial anchor,
one-dimensional radial) generator** acting on functions of $U_0$ alone. §9.6 supports this: the
eigenvalues move as $-2.326\pm i\theta s$ — the imaginary part grows linearly in $s$ while the real
part is pinned at $2.3265$.

The same holds for the alternative $c'=J\nabla U$, which conserves $U$. **Design consequence:** to
get an unbounded gain one must leave the family $c=\pi^{-1}\nabla\cdot(\rho_0 J)$ and use a general
skew matrix *field* $S(x)$ with $\pi c=\nabla\cdot S$, chosen so that the resulting flow has no
non-constant conserved quantity in $H^1$. §9.6 exhibits the saturation empirically: the gap increases
$\times1.01,\times1.07,\times1.29,\times2.30,\times5.48$ and then stops at $\approx\times5.49$ while
the eigenvalues acquire ever-larger imaginary parts.

**Where Fact B was used in §4:** (i) $\langle f,\Gamma f\rangle_\pi=0$, hence $\mathcal E_J=\mathcal E$
(4.4); (ii) pointwise, $\langle c,\nabla U_0\rangle=0$ (4.7), hence the saturation of §4.4 and the
exact no-gain theorem of §4.5. Fact A was used only in §2–§3 (invariance).

### 4.5 A sharp NEGATIVE result: for a spherically symmetric problem the gain is *exactly zero*

This is the most consequential consequence of Fact B for the present paper, and it is a theorem,
not a heuristic.

> **Theorem (no gain for radial $U,U_0$).** Assume $U(x)=\phi(\|x\|)$ and $U_0(x)=\psi(\|x\|)$ are
> radial, so $\nabla U_0(x)=\tfrac{\psi'(r)}{r}x$ with $r=\|x\|$. Let $J$ be constant skew and let
> $\mathcal R f(x):=\langle Jx,\nabla f(x)\rangle=\tfrac{d}{dt}\big|_{0}f(e^{tJ}x)$ be the generator of
> the one-parameter rotation group $R_t:=e^{tJ}\in SO(d)$. Then:
>
> **(a)** $\Gamma f=g(r)\,\mathcal Rf$ with $g(r)=e^{(\phi-\psi)(r)}\psi'(r)/r$.
>
> **(b)** $\Gamma$ annihilates every radial function, and both $L$ and $\Gamma$ leave each
> spherical-harmonic sector $\mathcal H_\ell=\{h(r)Y(x/r):Y\in\mathcal Y_\ell\}$ invariant. In particular
> $L_J=L$ *identically* on the radial subspace. **So if the $L^2(\pi)$ spectral gap of $L$ is attained
> on radial functions, no constant $J$ can improve it at all.**
>
> **(c)** If moreover $g\equiv\kappa$ is constant, then $\Gamma=\kappa\mathcal R$ **commutes with $L$**
> (because $L$ has radial coefficients, hence commutes with the unitary rotation group
> $U_tf:=f\circ R_t$ on $L^2(\pi)$, hence with its generator $\mathcal R$). Therefore
> $$
> e^{tL_J}=e^{t\kappa\mathcal R}\,e^{tL}=U_{t\kappa}\,e^{tL},
> $$
> and since $U_{t\kappa}$ is a **unitary** operator on $L^2(\pi)$ ($\pi$ is rotation invariant),
> $$
> \big\|e^{tL_J}f\big\|_{L^2(\pi)}=\big\|e^{tL}f\big\|_{L^2(\pi)}\qquad\text{for every }f\text{ and every }t\ge0 .
> $$
> **The skew perturbation has literally zero effect on $L^2(\pi)$/$\chi^2$ convergence — at every time,
> not merely asymptotically. Its only effect is to rigidly rotate the law.**

*Proof.* (a) $\Gamma f=e^{U-U_0}\langle J\nabla U_0,\nabla f\rangle=e^{(\phi-\psi)(r)}\tfrac{\psi'(r)}{r}\langle Jx,\nabla f\rangle$.
(b) For radial $f$, $\nabla f\parallel x$ and $\langle Jx,x\rangle=0$ (**Fact B**), so $\Gamma f=0$.
$\mathcal R$ is a rotation generator, hence maps $\mathcal Y_\ell$ to itself; $g(r)$ is a radial multiplier;
$L$ has radial coefficients. (c) $L$ commutes with $U_t$, hence with $\mathcal R$; two commuting
generators exponentiate as stated; $U_s$ is unitary on $L^2(\pi)$ because $\pi$ is rotation invariant
and Lebesgue measure is $SO(d)$-invariant. $\square$

**When is $g$ constant?** Exactly when $e^{U-U_0}\nabla U_0(x)=\kappa x$, i.e. when the *anchored drift
is linear*. That is precisely the paper's canonical heavy-tailed setting:

| target | $g(r)$ | constant? | $\kappa$ |
|---|---|---|---|
| **(T1)**, $U_0=\beta\log q$, $U=\iota\log q$, $q=1+r^2$ | $2\beta(1+r^2)^{\iota-\beta-1}$ | **yes iff $\iota=\beta+1$** (Thm 3's choice) | $2\beta$ |
| **(T2) isotropic** ($\mu=0,\Sigma=\varsigma^2I$) | $2\beta/(\nu\varsigma^2)$ | **always** | $2\beta/(\nu\varsigma^2)$ |
| **(T2) anisotropic** | not radial — theorem does not apply | — | — |

So: **the paper's own Section 6.4 target (T1) and any isotropic Student-$t$ get *nothing* from a
constant $J$.** (For (T1) with $\iota\neq\beta+1$, part (b) still applies: the radial sector is
untouched, and the gain — if any — can only come from the angular sectors.)

**Numerically confirmed (§9.8).** For the $d=2$ isotropic Student-$t$ ($\nu=8$, $\beta=4$, $\Sigma=I$),
the discrete spectral gap is $1.000208,\,1.000217,\,1.000238,\,1.000311,\,1.000530,\,1.000987$ for
$j=0,0.5,1,2,4,8$ — flat to $10^{-3}$ (the residual drift is the square-box discretisation breaking
rotational symmetry), while the imaginary part is *exactly* $j$: $0.50009,1.00017,2.00033,4.00056,8.00070$,
i.e. $\mathrm{Im}=\kappa j m$ with $\kappa=2\beta/\nu=1$ and angular index $m=1$. And
$[L,\Gamma]=0$ to $2\times10^{-7}$ relative in the isotropic case versus $0.23$–$0.87$ relative for
anisotropic $\Sigma$.

**Practical upshot.** *A constant skew $J$ is only worth adding when the problem is genuinely
anisotropic (or multimodal / non-radial).* For the paper's heavy-tail experiments as written, the
correct prediction is **no speed-up whatsoever**, and any observed difference would be discretisation
noise. To get a gain from a heavy-tailed radial target one must either (i) use an anisotropic
$\Sigma$ / non-radial target, or (ii) leave this family of perturbations altogether (§4.4, last
paragraph).

---

## 5. The alternative perturbation $c'(x)=J\nabla U(x)$

### 5.1 It also preserves $\pi$

Define $L'_J:=L+\Gamma'$, $\Gamma'f:=\langle c',\nabla f\rangle$, $c'=J\nabla U$, i.e. the SDE

$$
dX_t=\Big(-e^{(U-U_0)(X_t)}\nabla U_0(X_t)+J\nabla U(X_t)\Big)dt+\sqrt2\,e^{(U-U_0)(X_t)/2}dW_t .
\tag{5.1}
$$

Then $\pi c'=\tfrac1M e^{-U}J\nabla U=-\tfrac1MJ\nabla\rho$ (with $\rho=e^{-U}$), so, **provided
$U\in C^2$** (a strictly stronger hypothesis than (H1), which only asked $U\in C^1$!),

$$
\nabla\!\cdot\!(\pi c')=-\tfrac1M\operatorname{Tr}\!\big(J\,\nabla^2\rho\big)=0
$$

by **Fact A**. All of §2–§4 goes through verbatim with $\rho_0\rightsquigarrow\rho$: $\Gamma'$ is
$L^2(\pi)$-antisymmetric, $\mathcal E_{J'}=\mathcal E$, invariance holds. (As in §2.2 one can drop to
$U\in C^1$ by reading $\nabla^2\rho$ distributionally.)

### 5.2 Which one preserves the random-time-change structure? — **$c$ does; $c'$ does not (cleanly)**

Recall the paper's Theorem 11: $X_t=Z_{\ell(t)}$ where $dZ_s=-\nabla U_0(Z_s)ds+\sqrt2\,d\widetilde W_s$
and $\ell'(t)=e^{(U-U_0)(Z_{\ell(t)})}$. The mechanism is the elementary *time-change lemma*: if $Z$ has
generator $A$ and $\ell$ is the inverse of the additive functional
$s\mapsto\int_0^s e^{(U_0-U)(Z_r)}dr$, then $t\mapsto Z_{\ell(t)}$ has generator $e^{U-U_0}A$
(Ethier–Kurtz Thm 6.1.3; Revuz–Yor Ch. V). Comparing with (1.3):

* **Variant $c=e^{U-U_0}J\nabla U_0$.** $L_J=G\cdot L_0^J$ with
  $L_0^Jf=\Delta f+\langle(J-I)\nabla U_0,\nabla f\rangle$. So

  $$
  \boxed{\;X_t=Z^J_{\ell(t)},\qquad dZ^J_s=(J-I)\nabla U_0(Z^J_s)\,ds+\sqrt2\,d\widetilde W_s,\qquad
  \ell'(t)=e^{(U-U_0)(Z^J_{\ell(t)})}\;}
  \tag{5.2}
  $$

  i.e. **(SAL) is exactly the paper's random time change applied to the classical *Hwang non-reversible
  Langevin diffusion for the anchor potential $U_0$*.** The inner process still uses **only $\nabla U_0$**;
  the clock still uses only $e^{U-U_0}$. Everything the paper builds on top of Theorem 11 survives:
  * The proof of Lemma 10 / Theorem 11 is unchanged except that Assumption 9 ($d-\langle x,\nabla U_0\rangle\le-c_0$)
    becomes $d-\langle x,(I-J)\nabla U_0(x)\rangle\le -c_0$, which is **identical** when $\nabla U_0(x)\parallel x$ (radial anchor, Fact B).
  * **Theorem 15 extends verbatim.** Replace (34) by
    $z_{\ell_{k+1}}=z_{\ell_k}+\Delta\ell_k\,(J-I)\nabla U_0(z_{\ell_k})+\sqrt{2\Delta\ell_k}\,\xi_{k+1}$ with
    (33) unchanged, $\Delta\ell_k=\eta e^{(U-U_0)(z_{\ell_k})}$. Then
    $\Delta\ell_k(J-I)\nabla U_0=\eta\,b_J$ and $\sqrt{2\Delta\ell_k}=\sqrt{2\eta}\,\sigma$, so under
    synchronous coupling the two discretisations are **pathwise identical**, exactly as in Theorem 15.
* **Variant $c'=J\nabla U$.** Now
  $L'_J=G\big(\Delta+\langle-\nabla U_0+e^{U_0-U}J\nabla U,\nabla\cdot\rangle\big)=G\cdot A'$, so it is
  *still* a time change by the same clock, of the diffusion
  $dZ'_s=\big(-\nabla U_0(Z'_s)+e^{(U_0-U)(Z'_s)}J\nabla U(Z'_s)\big)ds+\sqrt2\,d\widetilde W_s$
  (whose invariant measure is again $\rho_0dx$, since
  $\nabla\cdot(\rho_0e^{U_0-U}J\nabla U)=\nabla\cdot(e^{-U}J\nabla U)=0$).
  But the inner dynamics is **no longer a $U_0$-only dynamics**: it needs $\nabla U$ *and* the ratio
  $e^{U_0-U}$ at every step. That destroys the two selling points of anchoring:
  (a) the method is supposed to work when $U$ is non-smooth (§5 of the paper: choose $U_0$ smooth and
  never differentiate $U$) — $c'$ requires $\nabla U$; (b) the inner process is supposed to be a
  *standard, well-understood* Langevin diffusion for $U_0$ — with $c'$ it is a bespoke
  non-gradient diffusion for which none of the Hwang/LNP theory applies off the shelf.

  **Verdict: $c=e^{U-U_0}J\nabla U_0$ is the structurally correct choice.**

**Remark (relation between the two Dirichlet forms).** If in addition $M_0:=\int e^{-U_0}dx<\infty$
(true for (T1),(T2) but *not* required anywhere else), then with $\pi_0:=M_0^{-1}e^{-U_0}dx$,

$$
\mathcal E_\pi(f)=\int e^{U-U_0}\|\nabla f\|^2d\pi=\tfrac1M\!\int\rho_0\|\nabla f\|^2dx
=\tfrac{M_0}{M}\int\|\nabla f\|^2d\pi_0=\tfrac{M_0}{M}\,\mathcal E_{\pi_0}(f).
$$

The two Dirichlet forms are proportional, but the *reference measures* ($\pi$ vs $\pi_0$) differ, so
the Poincaré constants — hence the spectral gaps — are **not** simply related: the anchored gap is a
gap of the pair $(\mathcal E,\pi)$, not of $(\mathcal E,\pi_0)$. Concretely, the acceleration theory of
Hwang et al. applies directly to the *inner* process $Z^J$ in $L^2(\pi_0)$, and transfers to the
*outer* process only through the (state-dependent, hence non-uniform) time change. §4.3 is proved
directly for the outer process and does not rely on this transfer.

### 5.3 A second, quantitative reason to prefer $c$: tail behaviour

The ratio of rotational to dissipative speed is

$$
\frac{\|c(x)\|}{\|b(x)\|}=\frac{\|J\nabla U_0(x)\|}{\|\nabla U_0(x)\|}\quad(\text{$J$-only, scale free}),
\qquad
\frac{\|c'(x)\|}{\|b(x)\|}=e^{(U_0-U)(x)}\frac{\|J\nabla U(x)\|}{\|\nabla U_0(x)\|}.
$$

For heavy tails $e^{U_0-U}\to0$ as $\|x\|\to\infty$ (that is the entire point of the anchor: $e^{U-U_0}$
is the *blow-up* factor that accelerates the dynamics in the tails). So **$c'$ switches the
non-reversible push off exactly in the tails**, where the mixing is hardest, whereas $c$ keeps it in
fixed proportion to the dissipative drift everywhere.

*(This argues that $c$ is the better of the two, not that either is useful: for a radial problem
§4.5 shows both give exactly nothing, since the "push" is purely angular while the slow modes are
radial. The comparison bites only for non-radial targets.)*

### 5.4 Target (T1): $U=\iota\log(1+\|x\|^2)$, $U_0=\beta\log(1+\|x\|^2)$

Here $q(x):=1+\|x\|^2$, $\nabla U=\dfrac{2\iota x}{q}$, $\nabla U_0=\dfrac{2\beta x}{q}$, so
$\nabla U=\tfrac\iota\beta\nabla U_0$ — **the two gradients are parallel** (both radial). Consequently
the two perturbation fields are *parallel at every point*, both equal to a scalar multiple of the
rigid rotation field $Jx$, and differ only by a radial speed factor:

$$
c(x)=q^{\,\iota-\beta}\cdot\frac{2\beta}{q}Jx=2\beta\,(1+\|x\|^2)^{\iota-\beta-1}\,Jx,
\qquad
c'(x)=\frac{2\iota}{1+\|x\|^2}\,Jx,
\qquad
\frac{c(x)}{c'(x)}=\frac\beta\iota\,(1+\|x\|^2)^{\iota-\beta}.
\tag{5.3}
$$

Reading them as rigid rotations $\dot x=\omega(\|x\|)Jx$ (angular velocity $\omega$ in the $J$-planes):

| | angular velocity $\omega(r)$ at $\|x\|=r$ | $r\to\infty$ |
|---|---|---|
| $c=e^{U-U_0}J\nabla U_0$ | $2\beta\,(1+r^2)^{\iota-\beta-1}$ | $\to2\beta$ if $\iota=\beta+1$; $\to\infty$ if $\iota>\beta+1$; $\to0$ if $\iota<\beta+1$ |
| $c'=J\nabla U$ | $2\iota/(1+r^2)$ | $\to0$ like $r^{-2}$ |

The paper's canonical heavy-tail choice is exactly $\iota=\beta+1$ (Theorem 3: $U_0=\beta\log q$,
$U=(\beta+1)\log q$; and $\iota>1+d/2\iff\beta>d/2$). In that case

$$
c(x)=q\cdot J\Big(\beta\tfrac{\nabla q}{q}\Big)=\beta\,J\nabla q(x)=2\beta\,Jx :
$$

**the added drift is a rigid rotation at constant angular velocity $2\beta$, uniformly over $\mathbb R^d$.**

**But beware — this is exactly the case §4.5 shows is useless.** $c(x)=2\beta Jx$ means
$\Gamma=2\beta\mathcal R$ with $\mathcal R$ the rotation generator; $U$ and $U_0$ are radial, so
$[L,\Gamma]=0$ and $\|e^{tL_J}f\|_{L^2(\pi)}=\|e^{tL}f\|_{L^2(\pi)}$ for all $t$: a rigid rotation of a
rotationally symmetric problem does **nothing** except rotate the law. So for (T1) as the paper states
it, the constant-$J$ extension is provably worthless — a genuinely useful thing to know before
running the experiment. By contrast $c'$ rotates only near the origin and dies out in the tails (and
is equally worthless here, for the same reason). (Note also that $d\ge2$ is needed for a nonzero skew $J$; the paper's Fig. 8 uses
$\iota=2,\beta=1$, which forces $d=1$, so an experiment must take e.g. $d=2$, $\beta=1.2$, $\iota=2.2$.)

**Free bonus for (T1) (and any radial anchor):** $\langle x,c(x)\rangle\propto\langle x,Jx\rangle=0$
by **Fact B**. This is what makes the Lyapunov condition $J$-free — see §7.

### 5.5 Target (T2): anisotropic Student-$t$

$q(x)=1+\tfrac1\nu(x-\mu)^\top\Sigma^{-1}(x-\mu)$, $U=\tfrac{d+\nu}2\log q$, $U_0=\beta\log q$ with
$\beta=\tfrac{d+\nu}2-1$, so $e^{U-U_0}=q$, $\nabla U_0=\beta\nabla q/q$, $\nabla q=\tfrac2\nu\Sigma^{-1}(x-\mu)$:

$$
b_J(x)=\beta(J-I)\nabla q(x)=\frac{2\beta}{\nu}(J-I)\Sigma^{-1}(x-\mu),\qquad \sigma(x)=q(x)^{1/2},
$$
$$
c(x)=\frac{2\beta}{\nu}J\Sigma^{-1}(x-\mu)\ \ (\text{linear, unbounded}),\qquad
c'(x)=\frac{d+\nu}{\nu}\frac{J\Sigma^{-1}(x-\mu)}{q(x)}\ \ (\text{$O(1/\|x\|)$}),
$$

and $L_Jf=q\,\Delta f+\tfrac{2\beta}{\nu}\big\langle(J-I)\Sigma^{-1}(x-\mu),\nabla f\big\rangle$.
Again $c=\tfrac{2\beta}{d+\nu}\,q\,c'$: same direction, ratio $\propto q(x)\sim\|x\|^2$.

**This is the target the experiments should use.** It is heavy-tailed (so the anchoring machinery is
needed), it has a *linear* drift and a scalar diffusion (so it is cheap and exactly analysable), and
— crucially — for **anisotropic** $\Sigma$ it escapes the no-gain theorem of §4.5, so a genuine
acceleration exists to be measured (§9.6: $\times5.5$ at $\kappa(\Sigma)=12$, $d=2$). The isotropic
case $\Sigma=\varsigma^2I$ and the paper's (T1) both fall inside §4.5 and will show **nothing**;
they are useful only as *negative controls*.

---

## 6. State-dependent $J(x)$

Let $J:\mathbb R^d\to\mathbb R^{d\times d}$ be $C^1$ with $J(x)^\top=-J(x)$ **pointwise**, and set
$c(x)=e^{(U-U_0)(x)}J(x)\nabla U_0(x)$, so that
$L_Jf=e^{U-U_0}\big(\Delta f-\langle\nabla U_0,\nabla f\rangle+\langle J(x)\nabla U_0,\nabla f\rangle\big)$.
Note that the generator formula (1.1) is unchanged: only the drift changes, not the diffusion, so no
Itô correction appears.

**Invariance criterion.** By (3.3) we need $\nabla\cdot(\pi c)=0$. With (0.1),
$M\pi c=\rho_0J(x)\nabla U_0=-J(x)\nabla\rho_0$, so componentwise
$M(\pi c)_i=-\sum_jJ_{ij}(x)\,\partial_j\rho_0$ and

$$
M\,\nabla\!\cdot\!(\pi c)
=-\sum_{i}\partial_i\Big(\sum_jJ_{ij}\partial_j\rho_0\Big)
=-\underbrace{\sum_{i,j}\big(\partial_iJ_{ij}\big)\partial_j\rho_0}_{\text{new}}
\;-\;\underbrace{\sum_{i,j}J_{ij}\,\partial_i\partial_j\rho_0}_{=\operatorname{Tr}(J(x)\nabla^2\rho_0)=0\ \text{(Fact A, pointwise)}} .
$$

Hence, defining the **column divergence** $(\operatorname{div}J)_j:=\sum_i\partial_iJ_{ij}=\nabla\cdot J_{\cdot j}$
(the ordinary divergence of the $j$-th column of $J$), the exact necessary and sufficient condition is

$$
\boxed{\;\sum_{i,j}\partial_iJ_{ij}(x)\;\partial_j\rho_0(x)=0\ \ \forall x
\iff \big\langle \operatorname{div}J(x),\,\nabla\rho_0(x)\big\rangle=0
\iff \big\langle \operatorname{div}J(x),\,\nabla U_0(x)\big\rangle=0\ \ \forall x\;}
\tag{6.1}
$$

(the last equivalence because $\nabla\rho_0=-\rho_0\nabla U_0$ and $\rho_0>0$). Remarks:

* Skewness of $J(x)$ *alone is not enough*: it only kills the Hessian term. The extra requirement is
  (6.1). (Verified numerically in §9: $J(x)=(1+\|x\|^2)J_0$ is pointwise skew yet violates (6.1) and
  demonstrably breaks invariance.)
* **Sufficient condition 1:** $\operatorname{div}J\equiv0$, i.e. every *column* of $J$ is a
  divergence-free vector field. Constant $J$ is the trivial case.
* **Sufficient condition 2 (useful family):** $J(x)=h(U_0(x))\,J_0$ with $J_0$ constant skew and
  $h\in C^1(\mathbb R)$. Indeed
  $(\operatorname{div}J)_j=\sum_i\partial_i\big(h(U_0)\big)(J_0)_{ij}=(J_0^\top\nabla h(U_0))_j=-(J_0\nabla h(U_0))_j$,
  and $\nabla h(U_0)=h'(U_0)\nabla U_0$, so
  $\langle\operatorname{div}J,\nabla U_0\rangle=-h'(U_0)\langle J_0\nabla U_0,\nabla U_0\rangle=0$ by **Fact B**.
  More generally $J(x)=h(x)J_0$ works iff $\langle J_0\nabla h,\nabla U_0\rangle=0$, i.e. iff $\nabla h$
  lies in the $J_0$-orthogonal complement of $\nabla U_0$; $h=h(U_0)$ is the canonical solution.
  *This is a genuinely useful knob*: e.g. $h(u)=e^{-\kappa u}$ damps the rotation where $\pi$ is small.
* **Sign conventions coincide.** The row divergence $r_i:=\sum_j\partial_jJ_{ij}$ satisfies
  $r=-\operatorname{div}J$ by skewness ($J_{ij}=-J_{ji}$), so (6.1) can equivalently be written
  $\langle r,\nabla U_0\rangle=0$; only the sign differs.
* **Complete parametrisation.** As in §3 remark 3, the *general* $\pi$-preserving additive drift is
  $c=\pi^{-1}\nabla\cdot S$ for a skew matrix field $S$. Our ansatz is the sub-family
  $S=-\tfrac1M\rho_0J(x)$; then
  $(\nabla\cdot S)_i=-\tfrac1M\sum_j\partial_j(\rho_0J_{ij})=-\tfrac1M\sum_jJ_{ij}\partial_j\rho_0-\tfrac1M\rho_0\,r_i$,
  which reproduces $\pi c$ exactly iff $r\equiv0$, i.e. $\operatorname{div}J\equiv0$ —
  **sufficient but strictly stronger than (6.1)**, which only demands
  $\operatorname{div}J\perp\nabla U_0$. (6.1) is the sharp condition.
* If one wants to keep the *time-change* structure (5.2), the inner process becomes
  $dZ=(J(Z)-I)\nabla U_0(Z)ds+\sqrt2 d\widetilde W$, and (6.1) is exactly the condition for $\rho_0dx$
  to be invariant for it — consistent, as it must be.
* **Regularity needed:** $J\in C^1$ for the pointwise statement; $J\in L^\infty_{loc}$ with (6.1)
  distributionally otherwise. Non-explosion (H3) must be re-checked, since $J(x)$ unbounded can
  destroy it.

---

## 7. What genuinely changes: well-posedness, non-explosion, ergodicity

This is the only place where the skew term is not free, and it must be stated honestly.

**7.1 Lyapunov condition.** With $V(x)=1+\|x\|^2$, $\nabla V=2x$, $\Delta V=2d$, (1.1) gives

$$
L_JV(x)=2\,e^{(U-U_0)(x)}\Big[d-\langle x,\nabla U_0(x)\rangle+\langle x,J\nabla U_0(x)\rangle\Big].
\tag{7.1}
$$

So the paper's **Assumption 1** must be replaced by

> **Assumption 1$'$.** For some $c_0,c_1>0$, $r>-1$ and all $x$,
> $\Big[d-\big\langle x,(I-J)\nabla U_0(x)\big\rangle\Big]e^{(U-U_0)(x)}\le-c_0\|x\|^{2+r}+c_1 .$

Under (H1), Assumption 1$'$ and local ellipticity ($e^{U-U_0}>0$ continuous), the proof of the paper's
Theorem 2 applies **verbatim** (the drift condition is used only through $L V\le -cV^{1+r/2}+c'$, plus
irreducibility + aperiodicity from ellipticity, plus a standard Foster–Lyapunov / Down–Meyn–Tweedie
argument): $\pi$ is the **unique** invariant measure, with $V$-uniform exponential ergodicity for
$r\ge0$ and total-variation exponential ergodicity for $r>0$. Likewise Assumption 9 becomes
$d-\langle x,(I-J)\nabla U_0(x)\rangle\le-c_0$ for $\|x\|^2\ge K$.

**7.1b A general $J$-free family of Lyapunov functions (this is the right fix).**

> **Proposition.** Let $W=h(U_0)$ for any $h\in C^2(\mathbb R)$. Then, for **every** skew $J(x)$
> (constant or state-dependent),
> $$
> L_JW=LW\qquad\text{pointwise on }\mathbb R^d .
> $$
> *Proof.* $\nabla W=h'(U_0)\nabla U_0$, so the extra term is
> $e^{U-U_0}\langle J\nabla U_0,\nabla W\rangle=h'(U_0)\,e^{U-U_0}\langle J\nabla U_0,\nabla U_0\rangle=0$
> by **Fact B**. $\square$
>
> **Corollary.** If the paper's drift condition can be verified with *some* Lyapunov function that is a
> function of the anchor $U_0$ alone, then Assumption 1$'$ $\equiv$ Assumption 1 with **no condition on
> $J$ whatsoever**, and Theorem 2 extends for every skew $J$.

This is the correct generalisation of §7.2. Two instances:

* **(T1)** $U_0=\beta\log(1+\|x\|^2)$: the paper's own $V(x)=1+\|x\|^2=e^{U_0(x)/\beta}$ **is** a
  function of $U_0$. So Assumption 1 is $J$-free, exactly as in §7.2.
* **(T2), general $\mu,\Sigma$:** take the **Mahalanobis** Lyapunov function
  $$
  V_\Sigma(x):=1+(x-\mu)^\top\Sigma^{-1}(x-\mu)=\nu q(x)-(\nu-1)=\nu e^{U_0(x)/\beta}-(\nu-1),
  $$
  again a function of $U_0$. **Condition (7.2) then disappears entirely.** The price is that
  $V_\Sigma$ has a different, $J$-free drift condition: $\Delta V_\Sigma=2\operatorname{Tr}(\Sigma^{-1})$ and
  $\langle\nabla U_0,\nabla V_\Sigma\rangle=\tfrac{4\beta}{\nu q}(x-\mu)^\top\Sigma^{-2}(x-\mu)$, so
  $$
  L_JV_\Sigma=2q\operatorname{Tr}(\Sigma^{-1})-\tfrac{4\beta}{\nu}(x-\mu)^\top\Sigma^{-2}(x-\mu)
  \le -c_0\|x-\mu\|^2+c_1\quad\text{iff}\quad 2\beta\,\lambda_{\min}(\Sigma^{-1})>\operatorname{Tr}(\Sigma^{-1}),
  \tag{7.3}
  $$
  for which a sufficient form is $d+\nu-2>d\,\kappa(\Sigma)$ — **the same shape as the paper's own
  Example 2 / Corollary 13 condition.** (With $W=q^{s}$, $0<s<1+\beta$, one gets the slightly weaker
  $ (d+\nu)\lambda_{\min}(\Sigma^{-1})>\operatorname{Tr}(\Sigma^{-1})$ in the limit $s\downarrow0$, at the price
  of only sub-geometric drift.)

So for (T2) there are **two** certificates and one may use either:

| Lyapunov function | condition | depends on $J$? |
|---|---|---|
| $V(x)=1+\|x\|^2$ | $\lambda_{\max}(\tfrac12(J\Sigma^{-1}-\Sigma^{-1}J))<\lambda_{\min}(\Sigma^{-1})$ — (7.2) | **yes** |
| $V_\Sigma(x)=1+(x-\mu)^\top\Sigma^{-1}(x-\mu)$ | $2\beta\lambda_{\min}(\Sigma^{-1})>\operatorname{Tr}(\Sigma^{-1})$ — (7.3) | **no** |

*(Honest caveat: in the §9 test problem, $\kappa(\Sigma)=12$, and **both** fail for $J_{\rm big}$:
(7.2) has margin $-1.248$, and (7.3) reads $3.000>5.333$, false. Yet the chain is empirically stable
(§9.10). So the sufficient conditions are far from sharp for strongly anisotropic $\Sigma$; a genuinely
sharp criterion is open.)*

Verified numerically (`check_lyap.py`): the $J$-contribution to $L_JW$ is $\le2.5\times10^{-13}$ for
$W\in\{V_\Sigma,\ q^{0.3},\ U_0,\ e^{-U_0/5}\}$ against a drift-term scale of $10^0$–$10^3$, while for
$W=1+\|x\|^2$ (not a function of $U_0$ when $\Sigma\neq\varsigma^2I$) it is $6.5\times10^{2}$ against a
scale of $1.03\times10^{3}$ — i.e. comparable to the whole drift.

**7.2 Radial anchors: the $J$-term vanishes identically** (overlapping with, but not implied by,
§7.1b — here we need only $\nabla U_0(x)\parallel x$, not that $V$ be a function of $U_0$).

> **Proposition.** If $U_0(x)=\psi(\|x\|)$ is radial (more generally, if $\nabla U_0(x)\parallel x$ for
> every $x$), then $\langle x,J\nabla U_0(x)\rangle=0$ for every skew $J$ (**Fact B**), so
> **Assumption 1$'$ $\equiv$ Assumption 1** and the paper's Theorem 2 holds verbatim for every
> constant skew $J$, with the *same* $c_0,c_1,r$ (the drift function $L_JV=LV$ is literally
> unchanged). The resulting $\lambda,C$ come from the same Foster-Lyapunov / Down-Meyn-Tweedie
> argument, but their numerical values depend on the minorisation constants of the transition
> kernel, which do depend on $J$ - so "same rate constants" is *not* claimed, only "same proof and
> same drift condition".

This covers **(T1)** ($U_0=\beta\log(1+\|x\|^2)$, $\nabla U_0=2\beta x/q$) and **(T2) with $\mu=0$,
$\Sigma=\varsigma^2I$. It also covers Theorem 3 whenever $q$ is radial.**

**7.3 Non-radial anchors: an explicit matrix condition (T2).** For (T2) with general $\mu,\Sigma$,
$(I-J)\nabla U_0\,e^{U-U_0}=\tfrac{2\beta}{\nu}(I-J)\Sigma^{-1}(x-\mu)$, and

$$
-\big\langle x,\tfrac{2\beta}{\nu}(I-J)\Sigma^{-1}(x-\mu)\big\rangle
=-\tfrac{2\beta}{\nu}\,x^\top\!\Big[\Sigma^{-1}-\tfrac12\big(J\Sigma^{-1}-\Sigma^{-1}J\big)\Big]x+O(\|x\|),
$$

so Assumption 1$'$ **with the quadratic Lyapunov function $V=1+\|x\|^2$ and $r=0$** holds *if and
only if*

$$
\boxed{\;\lambda_{\max}\!\Big(\tfrac12\big(J\Sigma^{-1}-\Sigma^{-1}J\big)\Big)<\lambda_{\min}(\Sigma^{-1})\;}
\tag{7.2}
$$

(the displayed matrix is symmetric: $(J\Sigma^{-1}-\Sigma^{-1}J)^\top=J\Sigma^{-1}-\Sigma^{-1}J$).
Note:

* If $J$ **commutes** with $\Sigma^{-1}$ (in particular $\Sigma=\varsigma^2I$), the left side is $0$
  and (7.2) holds for **every** $J$, of every size.
* Otherwise (7.2) fails for $J$ large enough. A crude sufficient bound:
  $\lambda_{\max}(\tfrac12(J\Sigma^{-1}-\Sigma^{-1}J))\le\|J\|_2\|\Sigma^{-1}\|_2$, so (7.2) holds
  whenever $\|J\|_2<\lambda_{\min}(\Sigma^{-1})/\lambda_{\max}(\Sigma^{-1})=1/\kappa(\Sigma)$ —
  **the admissible rotation strength scales like the inverse condition number**. (This bound is
  conservative: in §9, $\kappa=12$ so it would allow only $\|J\|_2<0.083$, whereas $J_{\rm small}$
  with $\|J\|_2=0.159$ still satisfies the sharp condition (7.2), margin $+0.096$.)
* **(7.2) is only sufficient for ergodicity**, being tied to the quadratic Lyapunov function. In our numerical example
  (§9) the "large" $J$ violates (7.2) yet all eigenvalues of $(J-I)\Sigma^{-1}$ still have negative
  real part and the chain is manifestly stable; a Lyapunov function $V(x)=1+x^\top Px$ adapted to $J$
  (solving a Lyapunov equation) restores the argument in that case. **Invariance of $\pi$ (§2–§4) is
  entirely unaffected by any of this** — it is an algebraic/PDE statement, valid for every skew $J$.

**7.4 Uniqueness.** Infinitesimal invariance alone does not give uniqueness. Uniqueness of the
invariant measure follows from Assumption 1$'$ + local ellipticity (irreducibility) exactly as for
$J=0$.

---

## 8. The discretisation

$$
x_{k+1}=x_k+\eta\,e^{(U-U_0)(x_k)}(J-I)\nabla U_0(x_k)+\sqrt{2\eta}\;e^{(U-U_0)(x_k)/2}\,\xi_{k+1}
\tag{8.1}
$$

is Euler–Maruyama for (SAL); **it does not preserve $\pi$ exactly** (weak order 1, bias $O(\eta)$),
exactly as for $J=0$. The paper's Theorem 14 (mean-square analysis of Li et al.) extends, with
explicit $J$-dependence of Assumption 12. For (T2), $b_J$ is linear and $\sigma$ is $J$-independent:

* $\alpha$ (Eq. 20) is **unchanged** by $J$: $\sigma=q^{1/2}$ does not involve $J$. One may take
  $\sqrt\alpha=\sqrt d\,\lambda_{\max}(\Sigma^{-1})/\sqrt{\nu\lambda_{\min}(\Sigma^{-1})}$.
* $m$ (Eq. 18, one-sided Lipschitz) becomes
  $m_J=\tfrac{2\beta}{\nu}\Big[\lambda_{\min}(\Sigma^{-1})-\lambda_{\max}\big(\tfrac12(J\Sigma^{-1}-\Sigma^{-1}J)\big)\Big]$
  — **the same quantity as in (7.2)**, decreasing in $\|J\|$ unless $[J,\Sigma^{-1}]=0$.
* $L$ (Eq. 19) becomes $L_J=\tfrac{2\beta}{\nu}\|(J-I)\Sigma^{-1}\|$ — **increasing** in $\|J\|$.

*(Caveat, worth stating plainly: Assumption 12 is restrictive already at $J=0$. Example 2 of the paper
requires $d+\nu>2+d\,\kappa(\Sigma)$. Our §9 test problem has $\kappa(\Sigma)=12$, $d=3$, $\nu=8$, i.e.
"$11>38$" — false — so Theorem 14 does **not** apply there; indeed $\alpha=18\gg m=0.375$. Nothing in
§2–§6 depends on Assumption 12, but any Wasserstein-rate extension of Theorem 14 must be stated for
near-isotropic $\Sigma$, or in a $\Sigma$-adapted norm.)*

Since Theorem 14 requires $\alpha<m$ and has $\eta_{\max}\propto\min\{L^{-2},(m-\alpha),\dots\}$ and
$C\propto(m-\alpha)^{-1}$, **there is a real trade-off**: increasing $\|J\|$ improves the
continuous-time rate (§4) but degrades the admissible stepsize and the discretisation constant, unless
$[J,\Sigma^{-1}]=0$ (isotropic case) in which case $m$ is untouched and only $L$ grows.
This is the discrete-time analogue of the well-known fact that the LNP "$\|J\|\to\infty$" acceleration
is not free once you discretise.

> **The central tension of the whole extension.** $\tfrac12(J\Sigma^{-1}-\Sigma^{-1}J)$ is the symmetric
> part of $J\Sigma^{-1}$, and it vanishes **iff $J$ commutes with $\Sigma$**. So:
> * $[J,\Sigma]=0$ (in particular any $J$ when $\Sigma=\varsigma^2I$): the one-sided Lipschitz constant
>   $m$ is untouched, the discretisation is as stable as at $J=0$ — but by **§4.5 there is no gain at
>   all**.
> * $[J,\Sigma]\neq0$: this is exactly where a gain exists (§9.6, $\times5.5$) — and exactly where the
>   Euclidean contraction constant degrades and (7.2) can fail.
>
> **The cheap direction is the useless one and the useful direction is the expensive one.** The way
> out is to stop measuring contraction in the Euclidean norm: §7.1b's $V_\Sigma$ is $J$-free, and the
> natural conjecture is that the mean-square analysis of Theorem 14 should be redone in the
> $\Sigma^{-1}$-weighted norm, where the drift matrix becomes $\widetilde J-\Sigma^{-1}$ with
> $\widetilde J:=\Sigma^{-1/2}J\Sigma^{-1/2}$ *still skew*, so its symmetric part is $-\Sigma^{-1}\prec0$
> **independently of $J$**. That is the right formulation; it is not carried out here.

By §5.2 the *random-time-change* discretisation (33)–(34) with $\nabla U_0\rightsquigarrow(I-J)\nabla U_0$
is **pathwise identical** to (8.1) under synchronous coupling (Theorem 15 extends).

---

## 9. Numerical verification

Code: `invariance_num.py`, `common.py`, `check_1b.py`, `check_generic_anchor.py`,
`check_mc_invariance.py`, `sim.py` (same directory). Setup: **(T2)** with
$d=3$, $\nu=8$, $\beta=(d+\nu)/2-1=4.5$, $\mu=(1,-0.5,0.3)$, $\Sigma$ anisotropic with eigenvalues
$(3,1,0.25)$ in a random orthonormal basis (so $\kappa(\Sigma)=12$). Skew matrices:
$J_{\rm big}=\mathrm{skew}(0.8,-0.35,0.6)$ (violates (7.2): margin $-1.248$) and
$J_{\rm small}=0.15\,J_{\rm big}$ (satisfies (7.2): margin $+0.096$).
Test functions $f\in\{\sin(a_1^\top x),\cos(a_2^\top x),e^{-\|x-v_0\|^2/2},1/q(x),\tanh(a_1^\top x)\}$
(all bounded with bounded derivatives; the resulting $L_Jf$ has finite variance under $\pi$ because
$\nu=8>4$). Exact $\pi$-sampling by $X=\mu+\sqrt{\nu/W}\,LZ$, $Z\sim N(0,I)$, $W\sim\chi^2_\nu$,
$LL^\top=\Sigma$ (verified: empirical mean/cov of $4\times10^6$ draws match $\mu$ and $\tfrac{\nu}{\nu-2}\Sigma$ to $10^{-3}$).

<<<RESULTS>>>

---

## 10. Summary of hypotheses

| Statement | Needs |
|---|---|
| Generator formula (1.1) | $U_0\in C^2$ (or $C^1$), $U\in C^1$; $f\in C^2$ |
| $\operatorname{Tr}(J\nabla^2\rho_0)=0$, hence $\nabla\cdot(\pi c)=0$ | $J^\top=-J$ **constant**; $U_0\in C^2$ (Schwarz) — or distributional |
| $\int L_Jf\,d\pi=0$ for $f\in C_c^\infty$ | above + $M=\int e^{-U}<\infty$ |
| $\pi$ invariant for $P^J_t$ | above + non-explosion (Assumption 1$'$) + well-posed martingale problem |
| $\pi$ **unique** invariant + exponential ergodicity | Assumption 1$'$ with $r\ge0$ (resp. $r>0$ for TV) + ellipticity |
| $\Gamma^*=-\Gamma$, $\mathcal E_J=\mathcal E$ | $\nabla\cdot V=0$ + no boundary flux |
| $\chi^2$ rate $\ge$ reversible rate, all $t$ | $\mathcal E_J=\mathcal E$ + Poincaré for $\pi$ + $\inf(U-U_0)>-\infty$ |
| strict acceleration | $\Gamma\mathcal H_1\not\subseteq\mathcal H_1$ (+ compact resolvent for the converse) |
| $c'=J\nabla U$ preserves $\pi$ | $U\in C^2$ (stronger than the paper's $U\in C^1$!) |
| state-dependent $J(x)$ | $J(x)^\top=-J(x)$ **and** $\langle\operatorname{div}J(x),\nabla U_0(x)\rangle=0$ |
| gap saturates as $\|J\|\to\infty$ (§4.4) | Fact B (the flow of $c$ conserves $U_0$) + CKRZ hypotheses on $-L$ |
| **zero** gain, at every $t$ (§4.5) | $U,U_0$ radial **and** $e^{U-U_0}\nabla U_0(x)=\kappa x$ (i.e. linear anchored drift) |
| $L_J=L$ on radial functions (§4.5b) | $U_0$ radial only |
| Assumption 1$'$ $\equiv$ Assumption 1, **no condition on $J$** (§7.1b) | the drift condition holds for some Lyapunov function $W=h(U_0)$ — Fact B |
| ↳ instance (T1) | $V=1+\|x\|^2=e^{U_0/\beta}$ |
| ↳ instance (T2) | $V_\Sigma=1+(x-\mu)^\top\Sigma^{-1}(x-\mu)$, needs $2\beta\lambda_{\min}(\Sigma^{-1})>\operatorname{Tr}(\Sigma^{-1})$ (7.3) |
| Assumption 1$'$ for (T2) with $V=1+\|x\|^2$ | $\lambda_{\max}(\tfrac12(J\Sigma^{-1}-\Sigma^{-1}J))<\lambda_{\min}(\Sigma^{-1})$ (7.2) |

---

## 11. Open questions / things this note does *not* settle

1. **Quantitative rate.** §4.3 gives "$\ge$, generically $>$". It gives **no explicit** improved
   $\lambda(J)$ for (T1)/(T2). Getting one probably means running the Lelièvre–Nier–Pavliotis
   machinery on the *inner* $U_0$-diffusion and then paying for the time change.
2. **Optimal $J$.** For (T2) the natural conjecture is that $J$ should be chosen relative to
   $\Sigma$ (the LNP optimal $J$ for the Gaussian is built from the covariance), but here the
   invariant measure of the inner process is $\propto q^{-\beta}$, not Gaussian, so the LNP formula
   does not directly apply. §7.3/§8 say that $[J,\Sigma^{-1}]$ is what costs; the optimum is a
   trade-off between the two.
3. **[PARTLY RESOLVED.] A $J$-adapted Lyapunov function.** §7.1b shows that any $W=h(U_0)$ gives a
   completely $J$-free drift condition — in particular the Mahalanobis $V_\Sigma$ for (T2), which
   removes (7.2). But its own condition (7.3) fails for strongly anisotropic $\Sigma$, and both
   certificates fail for $\kappa(\Sigma)=12$ even though the chain is empirically stable. A sharp
   ergodicity criterion for anisotropic (T2) with large $J$ is still open.
4. **Discretisation bias.** Does the $O(\eta)$ bias constant of (8.1) grow with $\|J\|$ fast enough
   to cancel the continuous-time gain? §9.10(i) shows a $\approx1.6\times$ increase for $J_{\rm big}$.
   The right comparison is bias-at-fixed-wall-clock, not bias-at-fixed-$\eta$.
5. **[RESOLVED — negatively.] Whether $J$ helps *heavy tails* specifically.** §4.5 proves that for
   radial $U,U_0$ the answer is **no gain at all**, exactly. This settles (T1) and isotropic (T2).
   What remains open is the *quantitative* question for genuinely anisotropic targets: §9.6 gets
   $\times5.5$ for $\kappa(\Sigma)=12$ in $d=2$; how the plateau scales with $\kappa$, $d$ and $\nu$
   is unknown. A natural conjecture from the OU analogy is that the plateau is governed by
   $\kappa(\Sigma)$, so the gain is $O(\kappa)$ — worth testing.
6. **Beyond the $J\nabla U_0$ family.** §4.4 shows the whole family saturates. Constructing a
   $\pi$-preserving perturbation $\pi c=\nabla\cdot S$ (general skew matrix field $S$) with no
   non-constant conserved quantity in $H^1(\pi)$ — hence unbounded relaxation enhancement — is the
   natural next step, and is *not* covered by anything in this note.
7. **Sharp ergodicity for large $\|J\|$.** Both certificates (7.2)/(7.3) fail for $\kappa(\Sigma)=12$
   although the chain is stable; the true condition is unknown. Note the eigenvalues of
   $(J-I)\Sigma^{-1}$ all have negative real part in that case — a spectral, not Euclidean-contraction,
   argument is presumably what is needed.
8. **State-dependent $J(x)$ with the time change.** (6.1) preserves the structure, but the inner
   scheme (34) then needs $J(z)$; the pathwise equivalence of Theorem 15 still holds, but
   Assumption 9/1$'$ must be re-verified with an $x$-dependent $J$.
9. **Non-smooth $U$ (the paper's §5).** The whole point of $c=e^{U-U_0}J\nabla U_0$ is that it never
   touches $\nabla U$. Combined with Gaussian smoothing ($U_0=f+g_0$), the skew term costs nothing
   extra per iteration. Untested.
