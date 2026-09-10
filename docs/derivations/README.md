# Long-form derivations

Working notes, kept because they carry the full algebra that
[`../THEORY.md`](../THEORY.md) only summarises.

* `invariance.md` — the generator computation, the $\pi$-divergence-free
  property of the added field, the $L^2(\pi)$ symmetric/antisymmetric split, and
  the proof that a radial target is inert to a constant skew $J$.
* `time-change.md` — the skew analogues of the paper's Lemma 10, Theorem 11 and
  Theorem 15, the corrected non-explosion condition, and the characterisation
  $\langle x, J\nabla U_0\rangle \equiv 0 \iff U_0\circ e^{tJ} = U_0$.
* `lyapunov.md` — the drift condition for the Euclidean and the
  geometry-adapted Lyapunov functions, and the structural lemma
  $\mathcal L_J(\Phi\circ U_0) = \mathcal L(\Phi\circ U_0)$.

**Status.** These are derivations, not peer-reviewed proofs, and they were not
put through an independent adversarial review pass — that stage was cut for
compute.  Treat the algebra as a draft to check, not as settled.

What *is* independently checked is narrower and lives in `tests/`: the
time-change equivalence to $10^{-15}$, invariance of the target under every
sampler started at stationarity, the exact second-moment prediction against a
150 000-particle simulation, the mean-square stepsize limit against actual
blow-up, and the inertness of an isotropic target to $J$.  `THEORY.md` is
written to claim only what those checks and the exact computations support.
