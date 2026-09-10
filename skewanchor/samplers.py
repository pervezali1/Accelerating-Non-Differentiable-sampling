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


def simulate(step, x0, n_steps, rng, record_at=None, recorder=None):
    """Advance ``x0`` for ``n_steps`` iterations, recording along the way.

    A run that produces a non-finite state is *not* silently dropped: the
    iteration at which it happened is reported in ``Result.diverged_at`` and the
    records collected up to that point are returned.
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
        if not np.all(np.isfinite(x)):
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
