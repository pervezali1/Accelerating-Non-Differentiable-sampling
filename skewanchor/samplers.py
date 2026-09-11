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


def generic_skew_anchored_step(target, eta, J=None):
    """Euler-Maruyama for any target exposing ``anchor_scale`` and ``grad_U0``.

    Slower than :func:`skew_anchored_step` (which hard-codes the log-quadratic
    algebra) but works for composite potentials such as the heavy-tailed MCP
    target in :mod:`skewanchor.nonsmooth`.
    """
    d = target.d
    J = np.zeros((d, d)) if J is None else np.asarray(J, dtype=np.float64)
    Jm = (J - np.eye(d)).T
    sq = np.sqrt(2.0 * eta)

    def step(x, rng):
        scale = target.anchor_scale(x)
        drift = scale[:, None] * (target.grad_U0(x) @ Jm)
        noise = np.sqrt(scale)[:, None] * rng.standard_normal(x.shape)
        return x + eta * drift + sq * noise

    return step


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


def field_anchored_step(target, eta, field=None, integrator="euler", n_bins=512):
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
    """
    from scipy.linalg import expm as _expm

    d = target.d
    sq = np.sqrt(2.0 * eta)
    coef = 2.0 * target.beta / target.nu
    A = target.Sigma_inv
    s_exp = target.s

    if integrator == "euler":
        def step(x, rng):
            g = target.grad_U0(x)
            rot = g if field is None else field.apply(x, g)
            drift = target.anchor_scale(x)[:, None] * ((0.0 if field is None else 1.0) * rot - g)
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

    if field is None or field.is_constant:
        B = B_of_psi(1.0 if J0 is not None else 0.0)
        if integrator == "cayley":
            M = np.linalg.solve(np.eye(d) + 0.5 * eta * B, np.eye(d) - 0.5 * eta * B)
        else:
            M = _expm(-eta * B)

        def step(x, rng):
            noise = target.sigma(x)[:, None] * rng.standard_normal(x.shape)
            return x @ M.T + sq * noise
        return step

    # radially modulated: bin on psi and build one propagator per bin
    def step(x, rng):
        psi = field.psi(x)
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


def stream_anchored_step(target, eta, field, integrator="expm", reference=None):
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
    B = coef * (np.eye(d) - delta * J0) @ A
    if integrator == "euler":
        M = np.eye(d) - eta * B
    elif integrator == "cayley":
        M = np.linalg.solve(np.eye(d) + 0.5 * eta * B, np.eye(d) - 0.5 * eta * B)
    elif integrator == "expm":
        M = _expm(-eta * B)
    else:
        raise ValueError(integrator)
    base = field.constant_member()

    def step(x, rng):
        extra = field.drift(x) - base.drift(x)
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


def make_step(name, target, eta, J=None, rng=None, gamma=2.0, n=None):
    if name == "ula":
        return ula_step(target, eta), None
    if name == "skew_ula":
        if J is None:
            raise ValueError("skew_ula needs J")
        return skew_ula_step(target, eta, J), None
    fast = hasattr(target, "Sigma_inv") and getattr(target, "s", None) is not None \
        and type(target).__name__ == "LogQuadraticTarget"
    if name == "anchored":
        return (skew_anchored_step if fast else generic_skew_anchored_step)(target, eta, None), None
    if name == "skew_anchored":
        return (skew_anchored_step if fast else generic_skew_anchored_step)(target, eta, J), None
    if name == "mala":
        return mala_step(target, eta), None
    if name == "underdamped":
        if n is None:
            raise ValueError("underdamped needs the ensemble size n")
        state = {"v": (rng if rng is not None else np.random.default_rng()).standard_normal((n, target.d))}
        return underdamped_step(target, eta, gamma, state), state
    raise ValueError(f"unknown sampler {name!r}")


def run(name, target, x0, eta, n_steps, rng, J=None, record_at=None, recorder=None, gamma=2.0):
    step, _ = make_step(name, target, eta, J=J, rng=rng, gamma=gamma, n=x0.shape[0])
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
