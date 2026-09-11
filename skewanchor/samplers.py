r"""Samplers compared in the experiments.

Every sampler advances an *ensemble* of ``n`` independent particles, matching
the protocol of Section 6 of the paper: the empirical distribution of the
ensemble at iteration ``k`` is compared with the exact target.

The algorithms
--------------
``ula``
    Unadjusted Langevin, ``x <- x - eta grad U(x) + sqrt(2 eta) xi``.
``skew_ula``
    ``x <- x + eta (J - I) grad U(x) + sqrt(2 eta) xi``.  The added field
    ``J grad U`` is pi-divergence free, so pi is still invariant for the SDE.
``anchored``
    Euler-Maruyama for the anchored Langevin SDE, paper Eq. (16).
``skew_anchored``
    This work: ``x <- x + eta e^{(U-U0)(x)} (J - I) grad U0(x)
                     + sqrt(2 eta) e^{(U-U0)(x)/2} xi``.
    ``anchored`` is the special case ``J = 0``.
``time_changed_skew_anchored``
    The random-time-change form, paper Eqs. (33)-(34) with the skew drift.
    Pathwise identical to ``skew_anchored`` under synchronous coupling
    (paper Theorem 15); kept as an independent implementation used to
    cross-check the main one.
``mala``
    Metropolis-adjusted Langevin on ``U`` -- an asymptotically exact baseline.
``underdamped``
    Kinetic (underdamped) Langevin on ``U`` -- the standard *other* way of
    accelerating by breaking reversibility, included so the skew perturbation
    is not only compared with reversible methods.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Result",
    "simulate",
    "ula_step",
    "skew_ula_step",
    "skew_anchored_step",
    "generic_skew_anchored_step",
    "field_anchored_step",
    "warmup_schedule",
    "stream_anchored_step",
    "make_step",
    "run",
    "run_time_changed",
    "SAMPLERS",
]


class Result:
    """Container for one ensemble run."""

    def __init__(self, iters, records, final_state, diverged_at, n_steps):
        self.iters = np.asarray(iters)
        self.records = records
        self.final_state = final_state
        self.diverged_at = diverged_at
        self.n_steps = n_steps

    @property
    def diverged(self):
        return self.diverged_at is not None

    def column(self, key):
        return np.array([r[key] for r in self.records], dtype=np.float64)


def simulate(step, x0, n_steps, rng, record_at=None, recorder=None,
             blowup_threshold=1e10):
    """Advance ``x0`` for ``n_steps`` iterations, recording along the way.

    A run that diverges is *not* silently dropped: the iteration at which it
    happened is reported in ``Result.diverged_at`` and the records collected up
    to that point are returned.

    Divergence means a non-finite state *or* one exceeding
    ``blowup_threshold``.  The magnitude test matters: a mean-square-unstable
    scheme grows geometrically but can sit at, say, ``1e29`` for the whole run
    without ever reaching infinity, and a finite-but-absurd state would
    otherwise be reported as a successful run.  A Student-t ensemble of a few
    thousand points does not come close to ``1e10``, so the test cannot fire on
    a legitimate heavy tail.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    record_at = None if record_at is None else set(int(k) for k in record_at)
    iters, records = [], []
    if recorder is not None and (record_at is None or 0 in record_at):
        iters.append(0)
        records.append(recorder(x, 0))
    diverged_at = None
    for k in range(1, n_steps + 1):
        x = step(x, rng)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > blowup_threshold:
            diverged_at = k
            break
        if recorder is not None and (record_at is None or k in record_at):
            iters.append(k)
            records.append(recorder(x, k))
    return Result(iters, records, x, diverged_at, n_steps)


# --------------------------------------------------------------- step rules


def ula_step(target, eta):
    sq = np.sqrt(2.0 * eta)

    def step(x, rng):
        return x - eta * target.grad_U(x) + sq * rng.standard_normal(x.shape)

    return step


def skew_ula_step(target, eta, J):
    sq = np.sqrt(2.0 * eta)
    J = np.zeros((target.d, target.d)) if J is None else np.asarray(J, dtype=np.float64)
    M = (J - np.eye(target.d)).T  # so that x @ M == ((J - I) x^T)^T rowwise

    def step(x, rng):
        return x + eta * (target.grad_U(x) @ M) + sq * rng.standard_normal(x.shape)

    return step


def skew_anchored_step(target, eta, J=None):
    r"""Euler-Maruyama for  dX = e^{U-U0}(J - I) grad U0 dt + sqrt(2) e^{(U-U0)/2} dW."""
    d = target.d
    J = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
    M = ((J - np.eye(d)) @ target.Sigma_inv).T
    coef = 2.0 * target.beta / target.nu
    s = target.s
    sq = np.sqrt(2.0 * eta)

    def step(x, rng):
        z = x - target.mu
        q = 1.0 + np.einsum("ni,ij,nj->n", z, target.Sigma_inv, z) / target.nu
        drift = (coef * q ** (s - 1.0))[:, None] * (z @ M)
        noise = (q ** (0.5 * s))[:, None] * rng.standard_normal(x.shape)
        return x + eta * drift + sq * noise

    return step


def generic_skew_anchored_step(target, eta, J=None, schedule=None):
    """Euler-Maruyama for any target exposing ``anchor_scale`` and ``grad_U0``.

    Slower than :func:`skew_anchored_step` (which hard-codes the log-quadratic
    algebra) but works for composite potentials such as the heavy-tailed MCP
    target in :mod:`skewanchor.nonsmooth`.
    """
    d = target.d
    J = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
    I = np.eye(d)
    sq = np.sqrt(2.0 * eta)
    counter = {"k": 0}

    def step(x, rng):
        s_k = 1.0 if schedule is None else float(schedule(counter["k"]))
        counter["k"] += 1
        scale = target.anchor_scale(x)
        drift = scale[:, None] * (target.grad_U0(x) @ (s_k * J - I).T)
        noise = np.sqrt(scale)[:, None] * rng.standard_normal(x.shape)
        return x + eta * drift + sq * noise

    return step


def warmup_schedule(target, eta, n_relax=3.0, shape="smooth"):
    r"""Warm-up of the skew field over ``n_relax`` stiff relaxation times.

    Starting from a prior that is much wider than the target, the skew drift
    converts the excess spread along the *stiff* axis into a transient excursion
    along the soft one, of size roughly ``||J|| x (initial stiff spread)``: a
    hump in the convergence curve before it falls.  The rotation buys nothing
    during that phase -- the stiff direction is contracting on its own, a
    hundred times faster -- so holding it back until the contraction is done
    removes the hump at no cost, and usually converges sooner as well.

    One stiff relaxation takes ``1 / (eta c lambda_max(Sigma^{-1}))``
    iterations with ``c = 2 beta / nu``.  Returns a callable ``k -> [0, 1]``.

    ``shape`` is ``smooth`` (a smoothstep, the default and the best of the three
    at every length tested), ``linear``, or ``exponential``.

    ``n_relax = 3`` is a good default everywhere.  A constant field does better
    still at ``5`` -- the hump vanishes completely and convergence is unchanged
    -- but a tilted field can converge in fewer iterations than a five-relaxation
    ramp takes, so there the shorter ramp is worth more.
    """
    c = 2.0 * target.beta / target.nu
    lam_max = 1.0 / float(np.min(target.Sigma_evals))
    k_relax = 1.0 / (eta * c * lam_max)
    K = max(1.0, n_relax * k_relax)
    if shape == "linear":
        return lambda k: min(1.0, k / K)
    if shape == "exponential":
        return lambda k: 1.0 - np.exp(-3.0 * k / K)
    if shape == "smooth":
        def smooth(k):
            u = min(1.0, k / K)
            return u * u * (3.0 - 2.0 * u)
        return smooth
    raise ValueError(f"unknown ramp shape {shape!r}")


def _radial_psi(field, x):
    """psi values of a RadialModulated field, or a constant, or None."""
    from .skewfield import ConstantSkew, RadialModulated
    if field is None:
        return None, None
    if isinstance(field, ConstantSkew):
        return None, field.J
    if isinstance(field, RadialModulated):
        return field.psi(x), field.J0
    return NotImplemented, None


def field_anchored_step(target, eta, field=None, integrator="euler", n_bins=512,
                        schedule=None):
    r"""Anchored step with a state-dependent skew field and a choice of integrator.

    ``integrator``:

    ``euler``   the explicit Euler-Maruyama step of the paper, extended to
                ``J(x)``.  Works for any field and any target.
    ``cayley``  the drift's linear part is advanced by the Cayley transform
                ``(I + eta B/2)^{-1}(I - eta B/2)``.  That map has unit modulus
                on the skew part, so the rotation is no longer amplified each
                step -- which is what the explicit scheme gets wrong.
    ``expm``    the drift's linear part is advanced exactly by
                ``exp(-eta B)``.  For a state-dependent field ``B`` depends on
                ``x`` only through the scalar ``psi``, so the exponential is
                computed once per ``psi`` bin.

    ``cayley`` and ``expm`` need the log-quadratic structure that makes the
    anchored drift linear (``s = 1``), with a constant or radially modulated
    field; they raise otherwise.

    ``schedule`` is an optional callable ``k -> [0, 1]`` scaling the skew part at
    iteration ``k`` -- see :func:`warmup_schedule`.  A scaled skew field is still
    skew, so the target is preserved at every iteration and therefore by the
    time-inhomogeneous chain as well.  **The returned step is stateful when a
    schedule is given** (it counts iterations), so build a fresh one per run.
    """
    from scipy.linalg import expm as _expm

    d = target.d
    sq = np.sqrt(2.0 * eta)
    coef = 2.0 * target.beta / target.nu
    A = target.Sigma_inv
    s_exp = target.s

    if integrator == "euler":
        counter = {"k": 0}

        def step(x, rng):
            s_k = 1.0 if schedule is None else float(schedule(counter["k"]))
            counter["k"] += 1
            g = target.grad_U0(x)
            rot = g if field is None else field.apply(x, g)
            drift = target.anchor_scale(x)[:, None] * ((0.0 if field is None else s_k) * rot - g)
            noise = target.sigma(x)[:, None] * rng.standard_normal(x.shape)
            return x + eta * drift + sq * noise
        return step

    if abs(s_exp - 1.0) > 1e-12:
        raise ValueError("cayley/expm need the canonical anchor beta = iota - 1 (s = 1)")
    psi_probe, J0 = _radial_psi(field, np.zeros((1, d)))
    if psi_probe is NotImplemented:
        raise ValueError("cayley/expm support constant or radially modulated fields only")

    def B_of_psi(psi):
        Jl = np.zeros((d, d)) if J0 is None else psi * J0
        return coef * (np.eye(d) - Jl) @ A

    def _prop(psi):
        B = B_of_psi(psi)
        if integrator == "cayley":
            return np.linalg.solve(np.eye(d) + 0.5 * eta * B, np.eye(d) - 0.5 * eta * B)
        return _expm(-eta * B)

    if field is None or field.is_constant:
        full = 1.0 if J0 is not None else 0.0
        M_full = _prop(full)
        counter = {"k": 0}

        def step(x, rng):
            if schedule is None:
                M = M_full
            else:
                M = _prop(full * float(schedule(counter["k"])))
                counter["k"] += 1
            noise = target.sigma(x)[:, None] * rng.standard_normal(x.shape)
            return x @ M.T + sq * noise
        return step

    # radially modulated: bin on psi and build one propagator per bin
    counter = {"k": 0}

    def step(x, rng):
        psi = field.psi(x)
        if schedule is not None:
            psi = psi * float(schedule(counter["k"]))
            counter["k"] += 1
        lo, hi = float(psi.min()), float(psi.max())
        if hi - lo < 1e-14:
            centres = np.array([lo])
            idx = np.zeros(x.shape[0], dtype=int)
        else:
            edges = np.linspace(lo, hi, n_bins + 1)
            centres = 0.5 * (edges[:-1] + edges[1:])
            idx = np.clip(np.searchsorted(edges, psi) - 1, 0, n_bins - 1)
        used = np.unique(idx)
        Ms = np.empty((len(used), d, d))
        for k, u in enumerate(used):
            B = B_of_psi(centres[u])
            if integrator == "cayley":
                Ms[k] = np.linalg.solve(np.eye(d) + 0.5 * eta * B, np.eye(d) - 0.5 * eta * B)
            else:
                Ms[k] = _expm(-eta * B)
        remap = np.searchsorted(used, idx)
        out = np.einsum("nij,nj->ni", Ms[remap], x)
        noise = target.sigma(x)[:, None] * rng.standard_normal(x.shape)
        return out + sq * noise

    return step


def stream_anchored_step(target, eta, field, integrator="expm", reference=None,
                         schedule=None):
    r"""Anchored step whose skew part is a state-dependent stream field (``d = 2``).

    The drift is ``b_anchored(x) + c(x)`` with
    ``c(x) = e^{U(x)} J_0 grad Phi(x)`` -- the complete two-dimensional family of
    target-preserving perturbations.

    The field is split into the constant part (whose drift is linear, and which
    the chosen integrator advances exactly) and the nonlinear remainder, taken
    explicitly.  ``reference`` is the constant-field member used for the split;
    it defaults to the ``a = b = 0`` member of the same family when the field
    was built by :meth:`StreamField2D.quadrupole`.
    """
    from scipy.linalg import expm as _expm

    if target.d != 2:
        raise ValueError("stream fields are implemented for d = 2")
    if abs(target.s - 1.0) > 1e-12:
        raise ValueError("stream step needs the canonical anchor beta = iota - 1")

    d = 2
    coef = 2.0 * target.beta / target.nu
    A = target.Sigma_inv
    J0 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    sq = np.sqrt(2.0 * eta)
    delta = getattr(field, "delta", 0.0) if reference is None else reference

    def _prop(scale):
        B = coef * (np.eye(d) - scale * delta * J0) @ A
        if integrator == "euler":
            return np.eye(d) - eta * B
        if integrator == "cayley":
            return np.linalg.solve(np.eye(d) + 0.5 * eta * B, np.eye(d) - 0.5 * eta * B)
        if integrator == "expm":
            return _expm(-eta * B)
        raise ValueError(integrator)

    M_full = _prop(1.0)
    base = field.constant_member()
    counter = {"k": 0}

    def step(x, rng):
        if schedule is None:
            s_k, M = 1.0, M_full
        else:
            s_k = float(schedule(counter["k"]))
            counter["k"] += 1
            M = _prop(s_k)
        extra = s_k * (field.drift(x) - base.drift(x))
        y = x + eta * extra
        return y @ M.T + sq * (target.sigma(x))[:, None] * rng.standard_normal(x.shape)

    return step


def mala_step(target, eta):
    """Metropolis-adjusted Langevin.  Vectorised over independent chains."""
    sq = np.sqrt(2.0 * eta)

    def logq(a, b):
        """log density of proposing ``b`` from ``a``."""
        mean = a - eta * target.grad_U(a)
        return -np.sum((b - mean) ** 2, axis=1) / (4.0 * eta)

    def step(x, rng):
        prop = x - eta * target.grad_U(x) + sq * rng.standard_normal(x.shape)
        log_acc = (target.U(x) - target.U(prop)) + (logq(prop, x) - logq(x, prop))
        accept = np.log(rng.random(x.shape[0])) < log_acc
        out = np.where(accept[:, None], prop, x)
        return out

    return step


def underdamped_step(target, eta, gamma=2.0, state=None):
    """Kinetic Langevin, OBABO splitting.

    The velocity is carried in a mutable ``state`` dict so the public interface
    stays ``step(x, rng) -> x``.
    """
    c = np.exp(-gamma * eta / 2.0)
    sc = np.sqrt(1.0 - c * c)

    def step(x, rng):
        v = state["v"]
        v = c * v + sc * rng.standard_normal(x.shape)          # O
        v = v - 0.5 * eta * target.grad_U(x)                    # B
        x = x + eta * v                                         # A
        v = v - 0.5 * eta * target.grad_U(x)                    # B
        v = c * v + sc * rng.standard_normal(x.shape)          # O
        state["v"] = v
        return x

    return step


# ------------------------------------------------------------------- facade


SAMPLERS = ("ula", "skew_ula", "anchored", "skew_anchored", "mala", "underdamped")


def make_step(name, target, eta, J=None, rng=None, gamma=2.0, n=None, schedule=None):
    if name == "ula":
        return ula_step(target, eta), None
    if name == "skew_ula":
        if J is None:
            raise ValueError("skew_ula needs J")
        return skew_ula_step(target, eta, J), None
    fast = hasattr(target, "Sigma_inv") and getattr(target, "s", None) is not None \
        and type(target).__name__ == "LogQuadraticTarget"
    if name == "anchored":
        if fast:
            return skew_anchored_step(target, eta, None), None
        return generic_skew_anchored_step(target, eta, None), None
    if name == "skew_anchored":
        if fast and schedule is None:
            return skew_anchored_step(target, eta, J), None
        if fast:
            from .skewfield import ConstantSkew
            return field_anchored_step(target, eta, ConstantSkew(J), "euler",
                                       schedule=schedule), None
        return generic_skew_anchored_step(target, eta, J, schedule=schedule), None
    if name == "mala":
        return mala_step(target, eta), None
    if name == "underdamped":
        if n is None:
            raise ValueError("underdamped needs the ensemble size n")
        state = {"v": (rng if rng is not None else np.random.default_rng()).standard_normal((n, target.d))}
        return underdamped_step(target, eta, gamma, state), state
    raise ValueError(f"unknown sampler {name!r}")


def run(name, target, x0, eta, n_steps, rng, J=None, record_at=None, recorder=None,
        gamma=2.0, schedule=None):
    step, _ = make_step(name, target, eta, J=J, rng=rng, gamma=gamma, n=x0.shape[0],
                        schedule=schedule)
    return simulate(step, x0, n_steps, rng, record_at=record_at, recorder=recorder)


def run_time_changed(target, x0, eta, n_steps, rng, J=None, record_at=None, recorder=None):
    r"""Random-time-change implementation (paper Eqs. 33-34) with the skew drift.

    ``l_{k+1} = l_k + eta e^{(U - U0)(z_k)}`` and
    ``z_{k+1} = z_k + dl_k (J - I) grad U0(z_k) + sqrt(2 dl_k) xi_{k+1}``.
    Under synchronous coupling this reproduces ``skew_anchored`` exactly, which
    :mod:`tests.test_equivalence` checks to machine precision.
    """
    d = target.d
    J = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
    Jm = (J - np.eye(d)).T

    def step(z, rng):
        dl = eta * target.anchor_scale(z)          # e^{(U - U0)(z)} eta
        g = target.grad_U0(z)
        return z + dl[:, None] * (g @ Jm) + np.sqrt(2.0 * dl)[:, None] * rng.standard_normal(z.shape)

    return simulate(step, x0, n_steps, rng, record_at=record_at, recorder=recorder)
