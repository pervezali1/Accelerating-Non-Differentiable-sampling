# Random Time Change for the Skew-Anchored Langevin SDE

Extension of Section 3.1 (Lemma 10, Theorem 11) and Section 5.1 (Eqs. 32–34, Theorem 15) of
*Anchored Langevin Algorithms* (Gürbüzbalaban–Nguyen–Zhang–Zhu, arXiv:2509.19455) to the
non-reversible, skew-perturbed dynamics.

All numerical claims below were verified with the scripts in
`/tmp/claude-0/-home-user/a27bb4e7-3347-59a6-87c3-52bea1592b16/scratchpad/code/`
(`skew_time_change.py`, `skew_time_change2.py`, `part1_fixed.py`).

---

## 0. Notation, standing objects, standing assumptions

Throughout, $J\in\mathbb R^{d\times d}$ is a **constant** matrix with $J^\top=-J$, and

$$\gamma(x):=e^{(U-U_0)(x)}>0,\qquad \sigma(x):=\gamma(x)^{1/2},\qquad
b_J(x):=\gamma(x)\,(J-I)\nabla U_0(x).$$

**Skew-anchored Langevin SDE (SAL-SDE).**

$$dX_t=\gamma(X_t)(J-I)\nabla U_0(X_t)\,dt+\sqrt2\,\gamma(X_t)^{1/2}\,dW_t. \tag{S}$$

**Irreversible reference SDE (Z-process).**

$$dZ_s=(J-I)\nabla U_0(Z_s)\,ds+\sqrt2\,d\widetilde W_s,\qquad Z_0=X_0. \tag{R}$$

Their generators (Itô convention $\mathcal Lf=\langle b,\nabla f\rangle+\tfrac12\operatorname{tr}(aa^\top\nabla^2f)$,
here $a=\sqrt2\,\sigma I$, so the second-order part is $\sigma^2\Delta$):

$$\boxed{\ \mathcal L_0f=\Delta f+\langle (J-I)\nabla U_0,\nabla f\rangle,\qquad
\mathcal Lf=\gamma\,\mathcal L_0f.\ } \tag{0.1}$$

The whole of Section 3.1 is the statement that **multiplying a generator by a positive function
is a random time change**; the skew field $J\nabla U_0$ rides along untouched. Everything below
makes that precise.

**Assumption (A) — regularity.** $U_0\in C^2(\mathbb R^d)$ with $\nabla U_0$ locally Lipschitz;
$U\in C^1(\mathbb R^d)$; hence $\gamma\in C^1$, $\gamma>0$, and $b_J,\sigma$ are locally Lipschitz.
(This is exactly the paper's standing regularity for (5); the skew term adds nothing since $J$ is
a constant matrix.)

**Assumption (M) — normalisability.** $M_0:=\int e^{-U_0}<\infty$ and $M:=\int e^{-U}<\infty$.
Write $\pi_0:=e^{-U_0}dx/M_0$ and $\pi:=e^{-U}dx/M$.

### 0.1 Two elementary skew identities used everywhere

For $J^\top=-J$ and any $v\in\mathbb R^d$, any symmetric $H\in\mathbb R^{d\times d}$:

$$\langle v,Jv\rangle=0, \tag{0.2}$$
$$\operatorname{tr}(JH)=\operatorname{tr}\big((JH)^\top\big)=\operatorname{tr}(H J^\top)=-\operatorname{tr}(JH)\ \Rightarrow\ \operatorname{tr}(JH)=0. \tag{0.3}$$

---

### 0.2 Why $\pi\propto e^{-U}$ survives the skew perturbation (needed before anything else)

The stationary Fokker–Planck operator for (S) is
$\mathcal L^\ast\rho=-\nabla\!\cdot\!(b_J\rho)+\Delta(\sigma^2\rho)$.
Substituting $\rho=e^{-U}$ and using $\sigma^2\rho=\gamma e^{-U}=e^{-U_0}$ and
$b_J\rho=e^{-U_0}(J-I)\nabla U_0$:

$$\mathcal L^\ast e^{-U}
=-\nabla\!\cdot\!\big(e^{-U_0}(J-I)\nabla U_0\big)+\Delta e^{-U_0}.$$

Since $\Delta e^{-U_0}=-\nabla\!\cdot\!\big(e^{-U_0}\nabla U_0\big)$, the two $-I$ contributions cancel and

$$\mathcal L^\ast e^{-U}=-\nabla\!\cdot\!\big(e^{-U_0}J\nabla U_0\big)
=e^{-U_0}\underbrace{\langle\nabla U_0,J\nabla U_0\rangle}_{=0\ \text{by }(0.2)}
-e^{-U_0}\underbrace{\operatorname{tr}\!\big(J\nabla^2U_0\big)}_{=0\ \text{by }(0.3)}=0. \tag{0.4}$$

**Where the hypotheses bite.** (0.4) uses (i) $U_0\in C^2$ (for $\nabla^2U_0$), (ii) $J$ **skew**
(both identities), (iii) $J$ **constant**. For a state-dependent $J(x)$, skew pointwise, one gets
instead $\nabla\!\cdot\!(J(x)\nabla U_0)=\langle\operatorname{div}J,\nabla U_0\rangle$ with
$(\operatorname{div}J)_j:=\sum_i\partial_iJ_{ij}$, so invariance requires the extra condition
$\langle\operatorname{div}J,\nabla U_0\rangle\equiv0$ — for instance every column of $J$
divergence-free. **This is a genuine extra hypothesis, absent in the constant-$J$ case.**

*Numerically verified (sympy, exact symbolic zero):* $\mathcal L^\ast\pi=0$ for (T2) with $d=2$,
$\nu=5$, $\Sigma=\begin{psmallmatrix}2&0.6\\0.6&1\end{psmallmatrix}$, $J=\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$
(non-commuting with $\Sigma$), and for (T1) with symbolic $\iota,\beta$.

### 0.3 $L^2(\pi)$ decomposition (context; it is what motivates the whole extension)

For $f,g\in C_c^\infty$, writing $\mathcal L=\mathcal S+\mathcal A$ with
$\mathcal Sf=\gamma(\Delta f-\langle\nabla U_0,\nabla f\rangle)$ (the paper's anchored generator) and
$\mathcal Af=\gamma\langle J\nabla U_0,\nabla f\rangle$:

$$\int(\mathcal Sf)g\,d\pi=-\tfrac1M\!\int\langle\nabla f,\nabla g\rangle e^{-U_0}dx \quad(\text{symmetric}),$$
$$\int(\mathcal Af)g\,d\pi=\tfrac1M\!\int\langle J\nabla U_0,\nabla f\rangle\,g\,e^{-U_0}dx
=-\tfrac1M\!\int f\,\langle J\nabla U_0,\nabla g\rangle e^{-U_0}dx=-\!\int f(\mathcal Ag)\,d\pi,$$

the second line because $\nabla\!\cdot\!(e^{-U_0}J\nabla U_0)=0$ by (0.4). Hence

* $\mathcal S=$ paper's generator (symmetric ⇒ Lemma 5's reversibility is exactly what is destroyed);
* $\mathcal A^\ast=-\mathcal A$, so the **Dirichlet form is unchanged**:
  $\mathcal E(f)=-\int f\mathcal Lf\,d\pi=\int e^{U-U_0}\|\nabla f\|^2d\pi$ — i.e. **Lemma 6 holds verbatim for every skew $J$**;
* consequently **Proposition 7 (the $\chi^2$ bound $\chi^2(\mu_t\|\pi)\le\chi^2(\mu_0\|\pi)e^{-2at/C_P}$) holds verbatim**:
  for non-symmetric $\mathcal L$ the density ratio $h_t=d\mu_t/d\pi$ evolves under $\mathcal L^\dagger=\mathcal S-\mathcal A$,
  so $\frac{d}{dt}\|h_t-1\|^2_{L^2(\pi)}=2\langle h_t-1,\mathcal L^\dagger(h_t-1)\rangle=-2\mathcal E(h_t-1)$, unchanged.
  The **acceleration** from $J$ therefore does *not* show up in $\mathcal E$; it shows up in the
  $L^2(\pi)$ spectral gap of the non-normal operator $\mathcal L$, which dominates that of $\mathcal S$
  (Hwang–Hwang–Sheu 1993/2005; Lelièvre–Nier–Pavliotis 2013). This is the standard picture and is
  the reason the extension is worth making.

---

## 1. The time change: Lemma 10′ and Theorem 11′

### 1.1 The clock

Define the (pathwise) **inverse clock**

$$A(s):=\int_0^s e^{(U_0-U)(Z_r)}\,dr=\int_0^s \gamma(Z_r)^{-1}\,dr ,\qquad s\ge0, \tag{1.1}$$

and, exactly as in the paper,

$$\ell(t):=\inf\Big\{s>0:\ \int_0^s e^{(U_0-U)(Z_r)}dr>t\Big\}=A^{-1}(t). \tag{1.2}$$

### Lemma 10′ (skew analogue of Lemma 10)

*Assume (A), (M) and Assumption 9′ (13′) of §2 below. Then almost surely:*

1. *$Z$ is non-explosive and positive Harris recurrent with unique invariant probability measure $\pi_0$;*
2. *$A$ is finite, $C^1$ and strictly increasing on $[0,\infty)$, $A(0)=0$, $A'(s)=\gamma(Z_s)^{-1}>0$, and $A(\infty)=\infty$;*
3. *$\ell=A^{-1}:[0,\infty)\to[0,\infty)$ is a $C^1$, strictly increasing bijection with $\ell(0)=0$ and*
$$\frac{d\ell(t)}{dt}=e^{(U-U_0)(Z_{\ell(t)})}=\gamma(Z_{\ell(t)}). \tag{1.3}$$

**Proof.**

*(1) Foster–Lyapunov.* Take $V(x)=\|x\|^2$, so $\nabla V=2x$, $\Delta V=2d$. Then

$$\mathcal L_0V(x)=2d+2\big\langle (J-I)\nabla U_0(x),\,x\big\rangle
=2\Big[d-\langle x,\nabla U_0(x)\rangle+\langle x,J\nabla U_0(x)\rangle\Big], \tag{1.4}$$

using $\langle J\nabla U_0,x\rangle=x^\top J\nabla U_0=\langle x,J\nabla U_0\rangle$. Under (13′),
$\mathcal L_0V\le-2c_0<0$ on $\{\|x\|^2\ge K\}$; on the compact set $C=\{\|x\|^2<K\}$, $\mathcal L_0V$ is
continuous hence bounded by some $b<\infty$. Thus $\mathcal L_0V\le-2c_0+(b+2c_0)\mathbf 1_C$ with
$V\ge0$, $V(x)\to\infty$. Khasminskii's non-explosion test gives $\zeta_Z=\infty$ a.s.; the
Meyn–Tweedie continuous-time drift criterion (CD2/CD3) then gives positive Harris recurrence,
**provided compact sets are petite**. Petiteness holds here because the diffusion coefficient of
(R) is the constant nondegenerate $\sqrt2 I$ and the drift $(J-I)\nabla U_0$ is locally bounded, so
the transition kernels have strictly positive continuous densities; hence (R) is
Lebesgue-irreducible and all compacts are petite. Positive Harris recurrence gives a unique
invariant probability measure; by §0.2 (with $J$ replaced by $J$, $\gamma\equiv1$, i.e.
$\mathcal L_0^\ast e^{-U_0}=-\nabla\!\cdot\!(e^{-U_0}J\nabla U_0)=0$) the measure $e^{-U_0}dx$ is invariant,
so it must be the (finite) one, i.e. $\pi_0$. □

*(2)* On $[0,s]$ the map $r\mapsto\gamma(Z_r)^{-1}$ is continuous (composition of continuous maps;
uses non-explosion of $Z$ from (1)), hence bounded there; so $A(s)<\infty$ and, by the fundamental
theorem of calculus, $A\in C^1$ with $A'(s)=\gamma(Z_s)^{-1}>0$. For $A(\infty)=\infty$: Harris
recurrence gives $\int_0^\infty\mathbf 1_{\{\|Z_s\|\le1\}}ds=\infty$ a.s., and by continuity and
compactness $\kappa:=\inf_{\|x\|\le1}\gamma(x)^{-1}>0$, so
$A(\infty)\ge\kappa\int_0^\infty\mathbf 1_{\{\|Z_s\|\le1\}}ds=\infty$. □

*(3)* Inverse function theorem for the $C^1$ strictly increasing bijection $A$; differentiating
$A(\ell(t))=t$ gives $\gamma(Z_{\ell(t)})^{-1}\ell'(t)=1$. □

**Two separate finiteness statements, do not conflate them.**

* $\ell(t)<\infty$ for **all** $t$ $\iff$ $A(\infty)=\infty$. This is what recurrence buys.
  If $A(\infty)=A_\infty<\infty$ (transient $Z$ escaping fast enough that $\int^\infty\gamma^{-1}(Z_r)dr$
  converges), then $X$ has finite lifetime $\zeta=A_\infty$: **the time-changed process explodes**.
* $\ell(t)\to\infty$ as $t\to\infty$ $\iff$ $A(s)<\infty$ for all finite $s$, which follows from
  non-explosion of $Z$ alone (no recurrence needed).

### Theorem 11′ (skew analogue of Theorem 11)

*Assume (A), (M) and (13′). Set $X_t:=Z_{\ell(t)}$. Then there is a $d$-dimensional Brownian
motion $W$, with respect to the time-changed filtration $\mathcal G_t:=\mathcal F_{\ell(t)}$, such that
$(X,W)$ is a weak solution of (S) on $[0,\infty)$ (no explosion). Under (A) pathwise uniqueness
holds, so $X$ is the unique strong solution and (S) is well posed. Moreover $\pi$ is invariant
for $X$.*

**Proof.** Let $\mathcal F_s:=\sigma\big(Z_0,\widetilde W_r,\,r\le s\big)$ (augmented). Write out (R) at
the random time $\ell(t)$:

$$X_t=Z_{\ell(t)}=X_0+\int_0^{\ell(t)}(J-I)\nabla U_0(Z_s)\,ds+\sqrt2\,\widetilde W_{\ell(t)}. \tag{1.5}$$

**Step 1 (drift — a pathwise change of variables).** By Lemma 10′(3), $\ell:[0,t]\to[0,\ell(t)]$ is
a $C^1$ increasing bijection, so the substitution $s=\ell(u)$, $ds=\ell'(u)du$ is legitimate for the
$\omega$-wise Lebesgue integral:

$$\int_0^{\ell(t)}(J-I)\nabla U_0(Z_s)ds
=\int_0^t (J-I)\nabla U_0(Z_{\ell(u)})\,\ell'(u)\,du
\stackrel{(1.3)}{=}\int_0^t\gamma(X_u)(J-I)\nabla U_0(X_u)\,du=\int_0^t b_J(X_u)\,du. \tag{1.6}$$

**This is the only place the drift appears, and the computation is valid for an arbitrary locally
bounded measurable drift field.** In particular the skew part $J\nabla U_0$ needs no special
treatment; the fact that $J^\top=-J$ is *not used here at all*.

**Step 2 ($\ell(t)$ is a stopping time; the time-changed martingale).** $A(s)$ is a continuous
functional of $Z|_{[0,s]}$, hence $\mathcal F_s$-measurable, and
$$\{\ell(t)\le s\}=\{A(s)\ge t\}\in\mathcal F_s,$$
so each $\ell(t)$ is an $(\mathcal F_s)$-stopping time, a.s. finite (Lemma 10′), and
$t\mapsto\ell(t)$ is a.s. continuous and strictly increasing with $\ell(0)=0$. Put
$\mathcal G_t:=\mathcal F_{\ell(t)}$ (right-continuous by continuity of $\ell$) and

$$M_t:=\widetilde W_{\ell(t)}.$$

By the optional-sampling / time-change theorem for continuous local martingales
(Revuz–Yor, Prop. V.1.5; Ethier–Kurtz Ch. 6 — localise along $\tau_n=\inf\{t:\|M_t\|\ge n\}$, which
is needed because $\widetilde W$ is not uniformly integrable), $M$ is a continuous
$(\mathcal G_t)$-local martingale with

$$\langle M^i,M^j\rangle_t=\langle \widetilde W^i,\widetilde W^j\rangle_{\ell(t)}=\delta_{ij}\,\ell(t). \tag{1.7}$$

**Step 3 (Lévy's characterisation).** $\ell'(s)=\gamma(X_s)$ is continuous, $(\mathcal G_s)$-adapted and
strictly positive, so $\ell'^{-1/2}$ is locally bounded and

$$W_t:=\int_0^t \ell'(s)^{-1/2}\,dM_s=\int_0^t e^{-(U-U_0)(X_s)/2}\,dM_s \tag{1.8}$$

is a well-defined continuous $(\mathcal G_t)$-local martingale with

$$\langle W^i,W^j\rangle_t=\int_0^t\ell'(s)^{-1}d\langle M^i,M^j\rangle_s
\stackrel{(1.7)}{=}\delta_{ij}\int_0^t\frac{\ell'(s)}{\ell'(s)}ds=\delta_{ij}\,t.$$

By Lévy's theorem $W$ is a standard $d$-dimensional $(\mathcal G_t)$-Brownian motion. Inverting (1.8)
(the integrands $\ell'^{\pm1/2}$ are locally bounded and mutually reciprocal),

$$\sqrt2\,\widetilde W_{\ell(t)}=\sqrt2\,M_t=\sqrt2\int_0^t\ell'(s)^{1/2}dW_s
=\sqrt2\int_0^t e^{(U-U_0)(X_s)/2}\,dW_s=\sqrt2\int_0^t\sigma(X_s)dW_s. \tag{1.9}$$

**Step 4.** Substituting (1.6) and (1.9) into (1.5),

$$X_t=X_0+\int_0^t\gamma(X_s)(J-I)\nabla U_0(X_s)\,ds+\sqrt2\int_0^t\gamma(X_s)^{1/2}dW_s,$$

which is (S). Non-explosion of $X$ on $[0,\infty)$ is exactly $\ell(t)<\infty\ \forall t$ (Lemma 10′).

**Step 5 (uniqueness).** Under (A), $b_J$ and $\sigma$ are locally Lipschitz, so pathwise uniqueness
holds up to explosion; since the constructed solution does not explode and pathwise uniqueness
forces all solutions to coincide up to the minimum of their explosion times, **no** solution
explodes and pathwise uniqueness is global. Yamada–Watanabe then upgrades weak existence +
pathwise uniqueness to strong existence and uniqueness in law. *(Alternative route, no Lipschitz
needed: the martingale problem for $\mathcal L_0$ is well posed by Stroock–Varadhan — constant
nondegenerate diffusion coefficient, locally bounded measurable drift, plus non-explosion from
(13′) — and by the time-change theorem for martingale problems (Ethier–Kurtz Thm 6.1.4) the
martingale problem for $\mathcal L=\gamma\mathcal L_0$ is then well posed too.)*

**Invariance of $\pi$.** Direct: §0.2. Conceptually: if $\mu_0$ is invariant for the generator
$\mathcal L_0$ and $\mathcal L=\gamma\mathcal L_0$ with $\gamma>0$, then $\gamma^{-1}\mu_0$ is invariant for
$\mathcal L$ (whenever normalisable), because
$\int\mathcal Lf\,\gamma^{-1}d\mu_0=\int\mathcal L_0f\,d\mu_0=0$. With $\mu_0=e^{-U_0}dx$ and
$\gamma^{-1}=e^{U_0-U}$ this is $\gamma^{-1}\mu_0=e^{-U}dx$. Normalisability is Assumption (M) —
**this is where $\int e^{-U}<\infty$ is genuinely needed and it is not implied by (13′).** $\square$

### Remark 1.1 (the general statement; $J$ is irrelevant to the time change)

Steps 1–4 used only: $\gamma>0$ continuous, $Z$ non-explosive, $A(\infty)=\infty$. Hence:

> **Lemma (time change).** Let $\gamma>0$ be continuous, $F:\mathbb R^d\to\mathbb R^d$ locally bounded
> measurable, and let $Z$ be a non-explosive solution of $dZ=F(Z)ds+\sqrt2\,d\widetilde W$. Let
> $A(s)=\int_0^s\gamma^{-1}(Z_r)dr$ and suppose $A(\infty)=\infty$; put $\ell=A^{-1}$. Then
> $X_t:=Z_{\ell(t)}$ is a weak solution of $dX=\gamma(X)F(X)dt+\sqrt2\,\gamma(X)^{1/2}dW$, with
> $W$ given by (1.8). If $\mu_0$ is invariant for $Z$ then $\gamma^{-1}\mu_0$ (if finite) is
> invariant for $X$.

The paper's Theorem 11 is $F=-\nabla U_0$; Theorem 11′ is $F=(J-I)\nabla U_0$. **The skew
perturbation is completely transparent to the time-change machinery.** The *only* place $J$
enters the analysis is the recurrence hypothesis (13′), i.e. §2.

### Remark 1.2 (why $\sqrt2$ and $\gamma^{1/2}$ must be matched)

The construction forces the pairing $\big(\text{drift }\gamma F,\ \text{diffusion }\gamma^{1/2}\big)$:
the drift picks up the factor $\ell'=\gamma$ once (Lebesgue rescaling) and the martingale picks it
up as $\sqrt{\ell'}=\gamma^{1/2}$ (quadratic-variation rescaling). Any other pairing is *not* a time
change of (R). This is exactly the paper's structural choice $b=-\nabla U_0e^{U-U_0}$,
$\sigma=e^{(U-U_0)/2}$, and it is what makes the whole heavy-tail construction work; the skew term
must therefore be inserted **inside** the same $\gamma$ factor, i.e. as
$\gamma(x)(J-I)\nabla U_0(x)$ and **not** as $\gamma(x)(-\nabla U_0(x))+J\nabla U_0(x)$ or
$\gamma(x)(-\nabla U_0(x))+J\nabla U(x)$. Those alternatives are also $\pi$-invariant for suitable
$J$-fields but are *not* time changes of (R), and Theorems 11′/15′ then both fail.

---

## 2. Assumption 9′: the corrected non-explosion / recurrence condition

### 2.1 Derivation

The paper's (13), $d-\langle x,\nabla U_0(x)\rangle\le-c_0$ for $\|x\|^2\ge K$, is precisely
$\tfrac12\mathcal L_0^{\mathrm{rev}}\|x\|^2\le-c_0$ for the reversible reference generator
$\mathcal L_0^{\mathrm{rev}}=\Delta-\langle\nabla U_0,\nabla\cdot\rangle$. With the skew drift,
computation (1.4) gives $\tfrac12\mathcal L_0\|x\|^2=d-\langle x,\nabla U_0\rangle+\langle x,J\nabla U_0\rangle$.
Hence:

> **Assumption 9′ (13′).** There exist $c_0>0$, $K\ge0$ such that
> $$d-\langle x,\nabla U_0(x)\rangle+\langle x,J\nabla U_0(x)\rangle\ \le\ -c_0
> \qquad\text{for all }\|x\|^2\ge K. \tag{13′}$$

**Yes, the skew drift does change the condition**, and it changes it by exactly the term
$h(x):=\langle x,J\nabla U_0(x)\rangle=-\langle Jx,\nabla U_0(x)\rangle$.

Everything Lemma 10′ needs from (13′) is: non-explosion of $Z$, and Harris recurrence
(occupation time of a ball infinite). Both come from the Foster–Lyapunov argument in the proof of
Lemma 10′(1). Note that (13′) constrains **$U_0$ only** (no $e^{U-U_0}$ factor), unlike the paper's
Assumption 1/(8) which constrains the $X$-dynamics — the reason is that (13′) is a statement about
the *reference* process $Z$.

### 2.2 When does $h(x)=\langle x,J\nabla U_0(x)\rangle$ vanish? A complete characterisation

Since $J$ is skew, $\{e^{tJ}\}_{t\in\mathbb R}$ is a one-parameter group of **rotations**
($e^{tJ}\in SO(d)$). Then $\frac{d}{dt}U_0(e^{tJ}x)=\langle\nabla U_0(e^{tJ}x),Je^{tJ}x\rangle$, so:

$$\boxed{\ h\equiv0\ \iff\ U_0\circ e^{tJ}=U_0\ \ \forall t\in\mathbb R\ }
\qquad\text{($U_0$ is invariant under the rotation group generated by $J$).} \tag{2.1}$$

Consequences.

* **$U_0$ radial ⟹ $h\equiv0$ for every skew $J$.** In particular **target (T1)**,
  $U_0(x)=\beta\log(1+\|x\|^2)$: $\nabla U_0=\frac{2\beta x}{1+\|x\|^2}$, so
  $h(x)=\frac{2\beta}{1+\|x\|^2}\langle x,Jx\rangle=0$ by (0.2). **For (T1) the condition (13′) is
  literally identical to the paper's (13), namely $\beta>d/2$, for every skew $J$ and every
  magnitude of $J$.** *(Verified numerically: $\max|h|\le2.1\times10^{-16}$ over 2000 random points.)*
* **Quadratic-type $U_0$**, $U_0=\beta\log q$ with $q=1+\frac1\nu(x-\mu)^\top\Sigma^{-1}(x-\mu)$
  (**target (T2)**): $e^{tJ}$-invariance requires $e^{tJ}$ to preserve both $\Sigma^{-1}$ and $\mu$, i.e.
  $$[J,\Sigma^{-1}]=0\quad(\iff[J,\Sigma]=0)\qquad\text{and}\qquad J\mu=0 .$$
  If $[J,\Sigma]=0$ but $J\mu\ne0$, then $h(x)=-\frac{2\beta}{\nu q}\langle x,J\Sigma^{-1}\mu\rangle=O(1/\|x\|)$:
  nonzero but **asymptotically harmless**, so (13′) still reduces to the reversible condition.
* Isotropic $\Sigma=\sigma^2I$: $[J,\Sigma]=0$ automatically, so (with $J\mu=0$, e.g. $\mu=0$) $h\equiv0$.

*Verified numerically:* $\max_x|U_0(e^{tJ}x)-U_0(x)|$ and $\max_x|h(x)|$ are simultaneously
$\approx10^{-14}$ for (T1) and for isotropic (T2), and simultaneously $\approx3$ for anisotropic (T2).

### 2.3 The extra term is never *uniformly* helpful (a sharp obstruction)

> **Proposition 2.1.** Let $U_0\in C^1$ and $J^\top=-J$ constant. For every $0\le r<R<\infty$,
> $$\int_{\{r\le\|x\|\le R\}}\langle x,J\nabla U_0(x)\rangle\,e^{-U_0(x)}\,dx=0. \tag{2.2}$$
> Consequently, if $h=\langle x,J\nabla U_0\rangle\le0$ everywhere on some annulus/exterior region
> $\{\|x\|\ge r\}$, then $h\equiv0$ there.

**Proof.** $h\,e^{-U_0}=-\langle x,J\nabla e^{-U_0}\rangle=-\langle J^\top x,\nabla e^{-U_0}\rangle
=\langle Jx,\nabla\phi\rangle$ with $\phi:=e^{-U_0}$. By the divergence theorem on the annulus
$\mathcal A_{r,R}$, and using $\nabla\!\cdot\!(Jx)=\operatorname{tr}(J)=0$,

$$\int_{\mathcal A_{r,R}}\langle Jx,\nabla\phi\rangle
=\int_{\partial\mathcal A_{r,R}}\phi\,\langle Jx,n\rangle\,dS-\int_{\mathcal A_{r,R}}\phi\,\nabla\!\cdot\!(Jx)
=\int_{\partial\mathcal A_{r,R}}\phi\,\langle Jx,n\rangle\,dS .$$

On each boundary sphere the outward normal is $n=\pm x/\|x\|$, so $\langle Jx,n\rangle=\pm\langle Jx,x\rangle/\|x\|=0$
by (0.2). Both boundary terms vanish identically. $\square$

**Interpretation.** The skew correction has exactly zero $e^{-U_0}$-weighted mass on *every* spherical
shell — no integrability hypothesis, no decay hypothesis needed (the boundary terms vanish for
geometric reasons, not by decay). So the skew term can only *redistribute* the drift condition, never
uniformly improve it: on any tail region it is either identically $0$ or strictly positive somewhere.
**In the Foster–Lyapunov sense with $V=\|x\|^2$, an irreversible perturbation is at best neutral and
generically harmful.** *(Verified numerically: (2.2) holds to $\le2\times10^{-15}$ on the annuli
$[0,3],[3,12],[0,40],[7.5,25]$ for anisotropic (T2), where $h\not\equiv0$.)*

### 2.4 Exactly how harmful: the anisotropic Student-$t$ (T2)

Take (T2): $q(x)=1+\frac1\nu y^\top\Sigma^{-1}y$ with $y=x-\mu$, $U_0=\beta\log q$,
$\nabla U_0=\frac{2\beta}{\nu q}\Sigma^{-1}y$. Then

$$d-\langle x,\nabla U_0\rangle+\langle x,J\nabla U_0\rangle
= d+\frac{2\beta}{\nu q}\Big[\,y^\top\big(S-\Sigma^{-1}\big)y+\langle\mu,(J-I)\Sigma^{-1}y\rangle\Big],
\qquad S:=\operatorname{sym}\!\big(J\Sigma^{-1}\big)=\tfrac12\big[J,\Sigma^{-1}\big]. \tag{2.3}$$

($S$ is symmetric: $S^\top=\tfrac12(\Sigma^{-1}J^\top-J^\top\Sigma^{-1})=\tfrac12(J\Sigma^{-1}-\Sigma^{-1}J)=S$.)
The $\mu$-term is $O(1/\|y\|)$. Writing $y=Ru$, $\|u\|=1$, and letting $R\to\infty$:

$$\limsup_{\|x\|\to\infty}\Big[d-\langle x,\nabla U_0\rangle+\langle x,J\nabla U_0\rangle\Big]
= d-2\beta\big(1-\rho_J\big),\qquad
\boxed{\ \rho_J:=\max_{u\ne0}\frac{u^\top S u}{u^\top\Sigma^{-1}u}=\lambda_{\max}\!\big(\Sigma^{1/2}S\,\Sigma^{1/2}\big).\ } \tag{2.4}$$

Hence

$$\textbf{(13′) holds for some }c_0>0\iff \beta\,(1-\rho_J)>\frac d2 . \tag{2.5}$$

With $J=0$ this is the paper's $\beta>d/2$, i.e. (for $\beta=\frac{d+\nu}2-1$) $\nu>2$, matching Example 1.
With $J\ne0$:

> **Proposition 2.2.** $\rho_J\ge0$ always, with $\rho_J=0$ iff $[J,\Sigma]=0$.

**Proof.** $\operatorname{tr}\!\big(\Sigma^{1/2}S\Sigma^{1/2}\big)=\operatorname{tr}(S\Sigma)
=\tfrac12\operatorname{tr}(J-\Sigma^{-1}J\Sigma)=\tfrac12(\operatorname{tr}J-\operatorname{tr}J)=0$.
A symmetric matrix with zero trace has $\lambda_{\max}\ge0$, with $\lambda_{\max}=0$ forcing all
eigenvalues $\le0$ and summing to $0$, hence all $=0$, hence $\Sigma^{1/2}S\Sigma^{1/2}=0$, hence $S=0$. $\square$

So for (T2) the skew term is **neutral iff $J$ commutes with $\Sigma$, and strictly harmful otherwise**,
tightening the admissible $\beta$ from $\beta>d/2$ to $\beta>\frac{d}{2(1-\rho_J)}$; equivalently, with
$\beta=\frac{d+\nu}2-1$,

$$\rho_J<\frac{\nu-2}{d+\nu-2}. \tag{2.6}$$

Since $S$ is linear in $J$, $\rho_{\theta J}=\theta\rho_J$ for $\theta>0$: there is a **critical
skew magnitude** $\theta^\star=\big(1-\tfrac{d}{2\beta}\big)/\rho_{J}$ beyond which (13′) fails.

*Numerically verified* for $d=2$, $\nu=5$, $\beta=2.5$, $\mu=(1,-0.5)$,
$\Sigma=\begin{psmallmatrix}2&0.6\\0.6&1\end{psmallmatrix}$, $J=\theta\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$:
$\rho_{J}/\theta=0.6098780366$, $\theta^\star=0.9838032590$, and the predicted $\limsup$ (2.4)
matches $\sup_{\|x-\mu\|=10^6}$ of the left side of (13′) to 5–6 decimals at every
$\theta\in\{0,0.5,0.9,0.98,1,1.5\}$ (e.g. $\theta=1$: predicted $+0.04939018$, empirical $+0.04939562$).

### 2.5 The harm is (largely) a Lyapunov-function artifact — the $J$-free repair

$V=\|x\|^2$ is not sacred. The general term to kill is $\langle J\nabla U_0,\nabla V\rangle$, and it
vanishes **pointwise for every skew $J$** as soon as $\nabla V\parallel\nabla U_0$:

> **Proposition 2.3 (anchored Lyapunov functions).** Let $\Psi\in C^2(\mathbb R)$ be increasing and
> $V:=\Psi\circ U_0$. Then for every constant skew $J$,
> $$\mathcal L_0V=\Psi''(U_0)\|\nabla U_0\|^2+\Psi'(U_0)\big(\Delta U_0-\|\nabla U_0\|^2\big), \tag{2.7}$$
> which **does not depend on $J$ at all**.

**Proof.** $\nabla V=\Psi'\nabla U_0$; $\Delta V=\Psi''\|\nabla U_0\|^2+\Psi'\Delta U_0$; and
$\langle(J-I)\nabla U_0,\nabla V\rangle=\Psi'\big(\langle J\nabla U_0,\nabla U_0\rangle-\|\nabla U_0\|^2\big)
=-\Psi'\|\nabla U_0\|^2$ by (0.2). $\square$

> **Assumption 9″ ($J$-free form).** There exist an increasing $\Psi\in C^2$ with $V=\Psi\circ U_0\ge0$,
> $V(x)\to\infty$ as $\|x\|\to\infty$, and $c_0>0$, $K\ge0$, with $\mathcal L_0V\le-c_0$ on $\{\|x\|^2\ge K\}$.

Assumption 9″ implies everything Lemma 10′ needs, for **every** constant skew $J$ and every magnitude.
(If one only wants recurrence + non-explosion, $\mathcal L_0V\le0$ off a compact suffices — Khasminskii.)

* **(T1)**: $U_0=\beta\log(1+\|x\|^2)$, $\Psi(u)=e^{u/\beta}-1$ gives $V=\|x\|^2$, so 9″ **is** (13) **is**
  (13′). The three coincide; nothing is lost.
* **(T2)**: $\Psi(u)=\nu(e^{u/\beta}-1)$ gives $V=V_\Sigma(x):=y^\top\Sigma^{-1}y$, and (2.7) evaluates to
  $$\mathcal L_0V_\Sigma=2\operatorname{tr}(\Sigma^{-1})-\frac{4\beta}{\nu q}\big\|\Sigma^{-1}y\big\|^2
  \ \xrightarrow[\ \|x\|\to\infty\ ]{\ \sup\ }\ 2\operatorname{tr}(\Sigma^{-1})-4\beta\,\lambda_{\min}(\Sigma^{-1}),$$
  giving the **$J$-free** condition
  $$\beta>\frac{\operatorname{tr}(\Sigma^{-1})}{2\,\lambda_{\min}(\Sigma^{-1})}. \tag{2.8}$$

**Honest comparison.** $\frac{\operatorname{tr}(\Sigma^{-1})}{2\lambda_{\min}(\Sigma^{-1})}\ge\frac d2$
with equality iff $\Sigma\propto I$. So $V_\Sigma$ trades a $J$-dependence for an
**anisotropy price**; neither (2.5) nor (2.8) dominates the other, and one should take whichever
holds. For the numerical example above with $\theta=1$: (2.5) **fails**
($\beta(1-\rho_J)=0.9753<1=d/2$) while (2.8) **holds** ($\beta=2.5>2.086303$), so $Z$ is in fact
still positive Harris recurrent and Theorem 11′ still applies. **The $\theta=1$ "harm" was purely a
Lyapunov artifact.** *(Verified: $\max|\langle J\nabla U_0,\nabla V_\Sigma\rangle|\le10^{-14}$ and
$\sup_{\|x-\mu\|=10^6}\mathcal L_0V_\Sigma=-0.72545751$, numerically identical for $\theta=0,1,5$ and
equal to $2\operatorname{tr}(\Sigma^{-1})-4\beta\lambda_{\min}(\Sigma^{-1})$.)*

There is a one-parameter family $V=q^p$, $p\in(0,1+\beta)$ (i.e. $\Psi(u)=e^{pu/\beta}$), all $J$-free;
$p=1$ gives $V_\Sigma$. Optimising over $p$ (and over $\Psi$ generally) is an open direction.

### 2.6 Summary of the answer to "does the skew drift change the condition?"

1. **Yes**, literally: with $V=\|x\|^2$ the condition becomes (13′), the extra term being
   $h(x)=\langle x,J\nabla U_0(x)\rangle$.
2. $h\equiv0$ **iff $U_0$ is invariant under the rotation group $e^{tJ}$**. This covers the paper's
   own heavy-tail family (T1) (radial $U_0$) and isotropic/commuting (T2) — for these, **the paper's
   Assumption 9 is unchanged and the skew term is free.**
3. $h$ can never uniformly help (Prop. 2.1: zero $e^{-U_0}$-mass on every shell), and for anisotropic
   (T2) with $[J,\Sigma]\ne0$ it is strictly harmful, with the sharp threshold (2.5)–(2.6) governed by
   $\rho_J=\lambda_{\max}(\Sigma^{1/2}\tfrac12[J,\Sigma^{-1}]\Sigma^{1/2})$.
4. The harm is at least partly removable by switching to an **anchored Lyapunov function**
   $V=\Psi\circ U_0$, for which the drift condition is $J$-free (Prop. 2.3). Design rule: **pick the
   Lyapunov function so that its level sets are level sets of $U_0$, or pick $J$ so that $e^{tJ}$
   preserves $U_0$.**
5. (13′) certifies **only** non-explosion and recurrence — the two things Lemma 10′ needs. It says
   nothing about the *acceleration* $J$ is meant to deliver, which lives in the spectral gap
   (§0.3). A Foster–Lyapunov condition insensitive to $J$ is therefore the *right* outcome, not a
   weakness.

---

## 3. Theorem 15′: exact equivalence of the two discretisations

### 3.1 The two schemes

Fix $\eta>0$, $x_0\in\mathbb R^d$, and a sequence $(\xi_k)_{k\ge1}\subset\mathbb R^d$
(in practice i.i.d. $\mathcal N(0,I_d)$; for the theorem below, arbitrary).

**(SAL-EM)** — Euler–Maruyama for (S), the analogue of (16):
$$x_{k+1}=x_k+\eta\,\underbrace{e^{(U-U_0)(x_k)}(J-I)\nabla U_0(x_k)}_{b_J(x_k)}
+\sqrt{2\eta}\,\underbrace{e^{(U-U_0)(x_k)/2}}_{\sigma(x_k)}\,\xi_{k+1}. \tag{3.1}$$

**(SAL-RTC)** — random time-change scheme, the analogue of (33)–(34): $z_0=x_0$, $\ell_0=0$,
$$\ell_{k+1}=\ell_k+\eta\,e^{(U-U_0)(z_k)},\qquad \Delta\ell_k:=\ell_{k+1}-\ell_k, \tag{3.2}$$
$$z_{k+1}=z_k+\Delta\ell_k\,(J-I)\nabla U_0(z_k)+\sqrt{2\,\Delta\ell_k}\ \xi_{k+1}. \tag{3.3}$$

### Theorem 15′

*Let $\gamma=e^{U-U_0}$ be finite and strictly positive and $\nabla U_0$ be defined on the visited
points. Then for every $\eta>0$, every $x_0$, every matrix $J\in\mathbb R^{d\times d}$ (skewness not
required) and **every** realisation of $(\xi_k)_{k\ge1}$ (synchronous coupling), the two recursions
generate the identical sequence:*
$$z_k=x_k\quad\text{for all }k\ge0,\qquad\text{and}\qquad \ell_k=\eta\sum_{j=0}^{k-1}\gamma(x_j). \tag{3.4}$$
*The two schemes are the same map; (3.4) is a deterministic algebraic identity, not a distributional one.*

**Proof.** Induction on $k$. $z_0=x_0$ by construction. Suppose $z_k=x_k=:v$. By (3.2),
$$\Delta\ell_k=\eta\,\gamma(z_k)=\eta\,\gamma(v)>0 .$$
Substituting into (3.3) and using $\Delta\ell_k>0$ so that
$\sqrt{2\Delta\ell_k}=\sqrt{2\eta\,\gamma(v)}=\sqrt{2\eta}\,\gamma(v)^{1/2}=\sqrt{2\eta}\,\sigma(v)$:
$$z_{k+1}=v+\eta\,\gamma(v)(J-I)\nabla U_0(v)+\sqrt{2\eta}\,\sigma(v)\,\xi_{k+1}
=v+\eta\,b_J(v)+\sqrt{2\eta}\,\sigma(v)\,\xi_{k+1}=x_{k+1}. $$
The clock formula follows by telescoping (3.2) with $z_j=x_j$. $\square$

**Exactly which facts were used.** Only three: (i) $\gamma>0$ (so the square root is real and
$\sqrt{\gamma}=\sigma$); (ii) the clock increment is evaluated **explicitly at $z_k$** (the same point
at which the drift and the diffusion coefficient are evaluated); (iii) the **same** $\xi_{k+1}$ is
used. Nothing about $J$, nothing about $U,U_0$ beyond definedness, no assumption from §1 or §2,
no step-size restriction.

### 3.2 Where the identity is exact — and where it is not

**(E1) Exact regardless of $J$.** The proof never uses $J^\top=-J$; (3.4) holds for arbitrary
$J\in\mathbb R^{d\times d}$, indeed with $(J-I)\nabla U_0$ replaced by an arbitrary field $F$.
So **the Theorem-15 equivalence survives the loss of reversibility**, and there is no "irreversible
correction term" to add to Algorithm 2. *(Verified: bitwise identical for $J$ skew, $J=0$, and
$J=\begin{psmallmatrix}0.3&-1\\2&0.1\end{psmallmatrix}$ non-skew.)*

**(E2) Exact at equal iteration index, not at equal time.** $x_k$ targets $X_{k\eta}$ (physical
clock) while $z_k$ targets $Z_{\ell_k}$ (reference clock). The identity pairs index $k$ with the pair
$(k\eta,\ell_k)$. In particular **$\ell_k\ne\ell(k\eta)$** in general: the exact clock solves the ODE
$\ell'=\gamma(Z_\ell)$, whereas (3.2) is one explicit-Euler step of it with the integrand frozen at
$z_k$. *(Numerically, e.g. $\eta=10^{-2}$, $n=5000$: $\ell_n\approx 68$ against $n\eta=50$; the ratio
fluctuates around $\mathbb E_\pi[\gamma]=\mathbb E_\pi[q]=1+\tfrac{d}{\nu-2}=5/3$ for (T2) — MC check $1.6658\pm0.0007$.)*

**(E3) Neither scheme equals the diffusion.** (3.4) is an equivalence of two *algorithms*. Both
still carry the Euler bias; under Assumption 12 the paper's Theorem 14 bound
$\mathbb E\|X_{\eta k}-x_k\|^2\le C^2\eta$ and
$W_2(\nu_k,\pi)\le\sqrt2e^{-(m-\alpha)k\eta}W_2(\nu_0,\pi)+\sqrt2C\eta^{1/2}$
is inherited **verbatim** by (SAL-RTC), with no separate analysis.

> **How Assumption 12 degrades under $J$ (target (T2)).** Here $b_J(x)=\frac{2\beta}{\nu}(J-I)\Sigma^{-1}(x-\mu)$
> is still linear, so with $v=x-y$ and $S=\operatorname{sym}(J\Sigma^{-1})=\tfrac12[J,\Sigma^{-1}]$ as in (2.3),
> $$\langle b_J(x)-b_J(y),x-y\rangle=\tfrac{2\beta}{\nu}\,v^\top\!\big(S-\Sigma^{-1}\big)v
> \ \le\ -m\|v\|^2,\qquad m=\tfrac{2\beta}{\nu}\,\lambda_{\min}\!\big(\Sigma^{-1}-S\big).$$
> **The skew part does *not* drop out** (it does only when $[J,\Sigma]=0$, i.e. $S=0$, recovering the
> paper's $m=\frac{2\beta}{\nu}\lambda_{\min}(\Sigma^{-1})=\beta c_2$). Moreover
> $$m>0\iff \Sigma^{-1}-S\succ0\iff \lambda_{\max}\!\big(\Sigma^{1/2}S\Sigma^{1/2}\big)<1\iff \boxed{\rho_J<1},$$
> **the same $\rho_J$ as in (2.4)**. So one single spectral quantity controls both the recurrence
> condition (13′) and the contractivity hypothesis (18). Meanwhile (19) degrades,
> $L=\frac{2\beta}{\nu}\|(J-I)\Sigma^{-1}\|_2$, growing like $\|J\|$, while $\alpha$ in (20) is
> **untouched** since $\sigma=q^{1/2}$ does not involve $J$. Hence $0<\alpha<m$ is a genuine new cap
> on $\|J\|$, and $\eta_{\max}$ in (24) shrinks like $\|J\|^{-2}$.
> *(Verified numerically: $m_{\rm pred}$ matches $-\max_v\langle b_J(x)-b_J(y),v\rangle/\|v\|^2$ to all
> printed digits for $\theta\in\{0,0.5,0.9,1,1.6,1.6397,1.7\}$, and $m$ changes sign exactly at
> $\theta=1/\rho_1=1.6396721$, i.e. at $\rho_J=1$.)*

Note the two caps differ: $\rho_J<1$ (i.e. $\theta<1.6397$ in the running example) for Assumption 12,
versus $\beta(1-\rho_J)>d/2$ (i.e. $\theta<0.9838$) for (13′). The second is the *removable* one
(§2.5); the first is intrinsic — it is about the actual contraction of the drift field, not about a
choice of Lyapunov function.

**(E4) Fails for any non-explicit / higher-order clock.** If (3.2) is replaced by an implicit rule
$\ell_{k+1}=\ell_k+\eta\gamma(z_{k+1})$, a trapezoidal rule
$\ell_{k+1}=\ell_k+\frac\eta2(\gamma(z_k)+\gamma(\tilde z_{k+1}))$, a midpoint rule, or an RK step, the
two schemes differ by $O(\eta^2)$ per step. *(Verified: trapezoidal clock, $\eta=10^{-2}$, $n=3000$,
same seed ⟹ $\max_k\|x_k-z_k\|=1.5\times10^{-2}$.)* Such a clock is a *better* ODE solver but breaks
the equivalence; if one wants Algorithm 2 to be provably the same as Algorithm 1, the clock must
stay explicit-Euler.

**(E5) Fails if the two occurrences of $\gamma$ are different evaluations.** The clock in (3.2) and
the coefficients in (3.3) must use the **same numerical value** of $e^{(U-U_0)(z_k)}$ and the same
$\nabla U_0(z_k)$. This is a real constraint for the paper's **Gaussian-smoothing** Algorithms 1–2:
there, $U_0$ is replaced by a Monte-Carlo estimate $\tilde U_0(x_k)=f(x_k)+\frac1N\sum_ig(x_k+\mu\xi_{i,k})$
and $\nabla U_0$ by $\nabla\tilde U_0(x_k)$. Remark 22's equivalence then requires the **same**
smoothing draws $\{\xi_{i,k},\hat\xi_{i,k}\}$ in both algorithms and in both places. Redrawing for
the clock destroys the identity. *(Verified: perturbing $\gamma$ in the clock alone by a relative
$10^{-3}$ Gaussian gives $\max_k\|x_k-z_k\|=5.3$ after 3000 steps at $\eta=10^{-2}$ — the schemes
separate at $O(1)$, not $O(10^{-3})$, because the recursion amplifies.)*

**(E6) Without synchronous coupling: identity in law, not pathwise.** If the two algorithms draw
independent $\xi$'s, (3.4) fails pathwise but the two are still **the same Markov chain**: given
$x_k=z_k=v$, $\Delta\ell_k=\eta\gamma(v)$ is $\mathcal F_k$-measurable, so both transitions are
$$Q(v,\cdot)=\mathcal N\!\Big(v+\eta\,b_J(v),\ 2\eta\,\gamma(v)\,I_d\Big).$$
This is the correct weaker statement to quote when the noise streams are not shared.

**(E7) Floating point.** The identity is exact in exact arithmetic. In IEEE-754 double precision it
is **bitwise identical (0 ulp)** provided both codes form the increment the same way, i.e.
`dl = eta*gamma(x)` and then `sqrt(2*dl)`. Coding (3.2)–(3.3) *literally*, with an accumulator
$\ell_k$ and $\Delta\ell_k:=\ell_{k+1}-\ell_k$, injects a cancellation error of relative size
$O(\varepsilon_{\rm mach}\,\ell_k/\Delta\ell_k)=O(\varepsilon_{\rm mach}\,k)$ into $\Delta\ell_k$, and the
trajectories separate slowly. *(Verified below: $\sim10^{-12}$ absolute after $2\times10^4$ steps,
growing with $k$; versus exactly $0$ ulp with the increment-first form.)*

**(E8) Step-size clipping.** In the tails $\Delta\ell_k=\eta\gamma(z_k)$ is unbounded — for (T1)
$\gamma=(1+\|x\|^2)^{\iota-\beta}$, for (T2) $\gamma=q\asymp\|x\|^2$ — so (SAL-RTC) is an Euler scheme
for (R) with an **unbounded** effective step. A safeguard $\Delta\ell_k\leftarrow\min(\eta\gamma(z_k),\Delta_{\max})$
breaks (3.4) *unless* the same clipping is applied to (SAL-EM) as the state-dependent step
$\eta_k:=\min(\eta,\Delta_{\max}/\gamma(x_k))$, in which case the identity is restored (Theorem 15′
holds verbatim with $\eta\to\eta_k$, since only $\mathcal F_k$-measurability of $\eta_k$ is used).

### 3.3 What the equivalence *means*

(SAL-RTC) is exactly **Euler–Maruyama for the irreversible reference SDE (R) with a random,
predictable, state-dependent step size $\Delta\ell_k=\eta\,e^{(U-U_0)(z_k)}>0$.** Since
$\Delta\ell_k\in\mathcal F_k$ and $\xi_{k+1}\perp\mathcal F_k$, the increment $\sqrt{2\Delta\ell_k}\,\xi_{k+1}$
has exactly the conditional law $\mathcal N(0,2\Delta\ell_k I_d)$ of $\sqrt2(\widetilde W_{\ell_{k+1}}-\widetilde W_{\ell_k})$.
(Note this is *not* the same as time-changing an a priori fixed Brownian path — in the continuous
construction $\ell(t)$ is a stopping time of $\widetilde W$; in the scheme $\ell_k$ is a function of
$\xi_1,\dots,\xi_k$ and the fresh $\xi_{k+1}$ supplies the increment. The two agree in law, which
is all that is needed.)

So the (skew-)anchored algorithm **is** "irreversible reference Langevin run on a $\gamma$-adaptive
clock": long reference-steps in the tails where $\gamma$ is large, short steps near the mode. Because
the clock is monotone ($\Delta\ell_k>0$ always), no time reversal can occur. Both algorithms cost the
same per step: one evaluation of $\nabla U_0$ and one of $e^{U-U_0}$.

---

## 4. Numerical verification

Scripts: `skew_time_change.py`, `skew_time_change2.py`, `part1_fixed.py`, `asm12.py` (all in
`/tmp/claude-0/-home-user/a27bb4e7-3347-59a6-87c3-52bea1592b16/scratchpad/code/`).
Model (T2) throughout: $d=2$, $\nu=5$, $\beta=\frac{d+\nu}2-1=2.5$, $\mu=(1,-0.5)$,
$\Sigma=\begin{psmallmatrix}2&0.6\\0.6&1\end{psmallmatrix}$, $J=\theta\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$
(**not** commuting with $\Sigma$).

### 4.1 Task 4 — pathwise identity of the two discretisations

Both schemes implemented with the *generic* $\gamma=\exp(U-U_0)$ (not the closed form $q$),
shared `np.random.default_rng(1234)`:

| $J$ | $\eta$ | $n$ | (SAL-EM) vs (SAL-RTC), increment form | vs literal $\Delta\ell_k=\ell_{k+1}-\ell_k$ | $\ell_n$ (vs $n\eta$) |
|---|---|---|---|---|---|
| $\theta=1$ skew | $10^{-3}$ | $20000$ | **bitwise identical, 0 ulp** | $1.13\times10^{-12}$ | $29.42$ ($20$) |
| $\theta=1$ skew | $10^{-2}$ | $5000$  | **bitwise identical, 0 ulp** | $3.95\times10^{-13}$ | $67.92$ ($50$) |
| $\theta=1$ skew | $5\!\times\!10^{-2}$ | $2000$ | **bitwise identical, 0 ulp** | $2.66\times10^{-13}$ | $152.15$ ($100$) |
| $J=0$ (paper) | all three | | **bitwise identical, 0 ulp** | $\le1.4\times10^{-12}$ | |
| $J$ arbitrary (not skew) | all three | | **bitwise identical, 0 ulp** | $\le1.5\times10^{-12}$ | |

Negative controls (same seed, $\eta=10^{-2}$, $n=3000$): trapezoidal clock ⟹ $1.5\times10^{-2}$;
independently resampled $\gamma$ in the clock (rel. $10^{-3}$) ⟹ $5.3$. **This confirms Theorem 15′
and localises exactly where exactness is lost (E4, E5, E7).**

### 4.2 Task 1 — the time change actually produces the SAL-SDE

**(a) Lévy characterisation of $W$ in (1.8).** One long path of (R) with $\theta=0.9$, $h=5\times10^{-4}$,
$4\times10^5$ steps; $dt_i:=h/\gamma(Z_{s_i})$, $dW_i:=\gamma(Z_{s_i})^{-1/2}\,d\widetilde W_i$, $t_{\rm end}=128.109$:

* $\langle W^1,W^1\rangle_t/t-1=8.9\times10^{-4}$, $\langle W^2,W^2\rangle_t/t-1=3.4\times10^{-3}$;
* $\langle W^1,W^2\rangle_t/t=-1.4\times10^{-3}$;
* standardised increments $dW^i/\sqrt{dt_i}$: KS vs $\mathcal N(0,1)$ gives $p=0.17$ and $p=0.52$;
  sample sd $1.0004$, $0.9981$.

**(b) The invariant-measure signature (a sharp, burn-in-free test).** For (T2),
$e^{-U_0}\propto q^{-\beta}$ is exactly the Student-$t$ law $t_{\nu_0}(\mu,\tfrac{\nu}{\nu_0}\Sigma)$
with $\nu_0=2\beta-d=3$ — a **different** distribution from $\pi=t_5(\mu,\Sigma)$
(verified: density ratio constant to $7\times10^{-16}$). Under $t_\nu(\mu,\Sigma)$ one has
$q-1=\tfrac d\nu F_{d,\nu}$ exactly, giving an exact KS test.

Start $Z_0\sim\pi=t_5(\mu,\Sigma)$, integrate (R) with $\theta=0.9$, $h=2\times10^{-3}$, $n=20000$
paths, form $A(s)$ by trapezoid, invert to $\ell(T)$ with $T=3$ and interpolate $X_T=Z_{\ell(T)}$
(coverage $99.98\%$):

| sample | KS vs $\pi=t_5(\mu,\Sigma)$ | KS vs $\pi_0=t_3(\mu,\tfrac53\Sigma)$ |
|---|---|---|
| $X_T=Z_{\ell(T)}$ (time-changed) | $D=0.0030$, **$p=0.99$** | $D=0.187$, $p=0$ |
| $Z_s$, $s=0$ (control) | $D=0.0088$, $p=0.088$ | $D=0.189$, $p=0$ |
| $Z_s$, $s=3$ (control) | $D=0.131$, $p=3\times10^{-298}$ | $D=0.068$, $p=6\times10^{-81}$ |
| $Z_s$, $s=30$ (control) | $D=0.175$, $p=0$ | $D=0.014$, $p=7\times10^{-4}$ |

Also $\mathbb E[X_T]=(1.0018,-0.4995)$ vs $\mu=(1,-0.5)$; $\operatorname{Cov}[X_T]=\begin{psmallmatrix}3.290&0.967\\0.967&1.601\end{psmallmatrix}$
vs $\tfrac{\nu}{\nu-2}\Sigma=\begin{psmallmatrix}3.333&1.000\\1.000&1.667\end{psmallmatrix}$.

**Reading.** Without the time change, $Z_s$ relaxes away from $\pi$ toward $\pi_0$ (residual
$D=0.014$ at $s=30$ is Euler bias at $h=2\times10^{-3}$). With the time change, $Z_{\ell(T)}$ stays
distributed as $\pi$ to within sampling error. This is exactly Theorem 11′ + invariance of $\pi$
for (S), in the presence of a nonzero skew drift.

### 4.3 Task 2 — the drift-condition analysis

* $\mathcal L^\ast\pi=0$: **exact symbolic zero** (sympy) for (T2) with $\theta=1$ and for (T1) with symbolic $\iota,\beta$.
* (T1): $\max_x|\langle x,J\nabla U_0\rangle|\le2.1\times10^{-16}$ over 2000 points.
* $S=\tfrac12[J,\Sigma^{-1}]$ symmetric, $\operatorname{tr}(S\Sigma)=0.0$ (Prop. 2.2).
* $\rho_J/\theta=0.6098780366$, $\theta^\star=0.9838032590$; predicted $\limsup$ $d-2\beta(1-\rho_J)$ vs
  empirical $\sup_{\|x-\mu\|=10^6}$ agree to 5–6 decimals for $\theta\in\{0,0.5,0.9,0.98,1,1.5\}$;
  the sign flip occurs exactly across $\theta^\star$.
* Annulus identity (2.2): $\le2\times10^{-15}$ on four annuli (quadrature tolerance $10^{-12}$).
* $\langle J\nabla U_0,\nabla V_\Sigma\rangle$: $\le9\times10^{-15}$ for $\theta\in\{0.5,1,5\}$;
  $\sup_{\|x-\mu\|=10^6}\mathcal L_0V_\Sigma=-0.72545751$ **identical** for $\theta=0,1,5$ and equal to
  $2\operatorname{tr}(\Sigma^{-1})-4\beta\lambda_{\min}(\Sigma^{-1})$.
* $e^{tJ}$-invariance characterisation (2.1): $\max|U_0(e^{tJ}x)-U_0(x)|$ and $\max|h|$ are both
  $\approx10^{-14}$ for (T1) and isotropic (T2), and both $\approx3$ for anisotropic (T2).

### 4.4 Assumption 12 constants under the skew perturbation (target (T2))

$m=\frac{2\beta}\nu\lambda_{\min}(\Sigma^{-1}-S)$ predicted vs. $-\max_{v\ne0}\frac{\langle b_J(x)-b_J(y),v\rangle}{\|v\|^2}$
measured over $2\times10^5$ random $v$ ($J=\theta\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$):

| $\theta$ | $\rho_J$ | $m$ predicted | $m$ measured | $L$ |
|---|---|---|---|---|
| $0$ | $0$ | $0.438399$ | $0.438399$ | $1.390869$ |
| $0.5$ | $0.304939$ | $0.382188$ | $0.382188$ | $1.555039$ |
| $0.9$ | $0.548890$ | $0.273926$ | $0.273926$ | $1.871223$ |
| $1.0$ | $0.609878$ | $0.241137$ | $0.241137$ | $1.966986$ |
| $1.6$ | $0.975805$ | $0.016076$ | $0.016076$ | $2.624286$ |
| $1.6397$ | $1.000017$ | $-0.000011$ | $-0.000011$ | $2.671271$ |
| $1.7$ | $1.036793$ | $-0.024648$ | $-0.024648$ | $2.743222$ |

$m$ changes sign exactly at $\rho_J=1$, i.e. $\theta=1/\rho_1=1.6396721$, confirming
$m>0\iff\rho_J<1$. Also $\mathbb E_\pi[\gamma]=\mathbb E_\pi[q]$: MC $1.66580\pm0.00069$ against the
exact $1+\frac{d}{\nu-2}=1.66667$.

---

## 5. Open questions

1. Does a **$J$-dependent** Lyapunov function give a $c_0$ that *improves* with $\|J\|$
   (quantifying acceleration at the Foster–Lyapunov level, not just spectrally)? For linear drift
   this is the Lelièvre–Nier–Pavliotis picture; for the anchored reference $\nabla U_0=\beta\nabla q/q$
   it is open.
2. Optimise $\Psi$ in Prop. 2.3 ($V=q^p$ family) to get the best $J$-free constant for (T2); is
   $\inf_p$ ever better than both (2.5) and (2.8)?
3. Does the skew drift improve or degrade the paper's Theorem-2 tail rates ($r\ge0$ / $r>0$
   dichotomy) under Assumption 1's *anchored* condition (8) rather than (13)?
4. For (T2) the frozen-$q$ dynamics is an exact-solvable linear SDE; an **exponential integrator**
   $z_{k+1}=\mu+e^{\Delta\ell_kA}(z_k-\mu)+\text{noise}$, $A=-\tfrac{2\beta}\nu(I-J)\Sigma^{-1}$, would
   remove the $\Delta\ell_k$-blow-up in the tails (E8) — but it **breaks** Theorem 15′. Is there an
   integrator that is both stable in the tails and exactly clock-equivalent?
5. State-dependent $J(x)$: §0.2 shows invariance needs $\langle\operatorname{div}J,\nabla U_0\rangle\equiv0$.
   Theorem 11′ and Theorem 15′ then still hold verbatim (neither uses skewness). Is there a
   $J(x)$ tuned to $\nabla U_0$ that measurably accelerates heavy-tail mixing?
6. Optimal $\theta$ in practice: (2.6) caps $\theta$ only through a *sufficient* condition, and §2.5
   shows the cap is removable; the real trade-off is spectral gap vs. Euler stability ($L$ grows
   like $\|J\|$, shrinking $\eta_{\max}$ in (24)).
