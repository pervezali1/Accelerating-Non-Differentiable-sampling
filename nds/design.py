"""Designing the skew field from the geometry, instead of drawing it at random.

Near its bulk the target looks like ``U(w) ~ (w - m)^T H (w - m) / 2``, and the
unadjusted dynamics ``dw = -(D + J) grad U dt + sqrt(2 D) dB`` relaxes at the
rate set by the smallest real part in the spectrum of ``M_J = (D + J) H``.  With
``D = I`` and ``J = 0`` that rate is ``lambda_min(H)``: the slowest direction of
the posterior sets the pace, and since the step size of an explicit scheme is
capped by ``lambda_max(H)``, the number of iterations needed scales with the
condition number of ``H``.

A skew ``J`` cannot change ``trace(M_J) = trace(H)`` -- the diagonal of ``J`` is
zero -- so no choice can push the *mean* rate above ``trace(H) / d``.  What it
can do is stop one direction from being the slowest, and the construction below
does that in the cheapest useful way:

* work in the eigenbasis of ``H``, and couple the ``k``-th smallest eigendirection
  with the ``k``-th largest in a 2x2 block;
* inside a block with eigenvalues ``a < b``, the coupling ``[[0, s], [-s, 0]]``
  gives ``[[a, s b], [-s a, b]]``, whose two eigenvalues are real and collide at
  the block mean ``(a + b) / 2`` when ``s = (b - a) / (2 sqrt(a b))``.

Pairing smallest with largest maximises the smallest block mean over all
pairings, so the relaxation rate goes from ``lambda_min(H)`` to roughly the
median eigenvalue.  Staying exactly at the collision point matters for a
discrete scheme: beyond it the eigenvalues become complex, and an explicit step
then has to shrink like the inverse square of the imaginary part, which costs
more than the extra real part gains.
"""

from __future__ import annotations

import numpy as np

from .sampler import Geometry, run_chain, stable_step_size
from .skew import ConstantSkew, ZeroSkew


def paired_skew(H: np.ndarray, amplitude: float = 1.0) -> tuple:
    """Skew ``J`` coupling slow to fast eigendirections of ``H``, and a report.

    ``amplitude`` scales every block coupling in units of the collision value,
    so ``1.0`` is the design point and ``0.0`` returns ``J = 0``.
    """
    H = np.asarray(H, float)
    H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(H)
    if lam.min() <= 0:
        raise ValueError("H must be positive definite")
    d = len(lam)
    S = np.zeros((d, d))
    block_rates = []
    for k in range(d // 2):
        i, j = k, d - 1 - k
        a, b = lam[i], lam[j]
        s = amplitude * (b - a) / (2.0 * np.sqrt(a * b))
        S[i, j], S[j, i] = s, -s
        block_rates.append(0.5 * (a + b))
    if d % 2:
        block_rates.append(lam[d // 2])
    J = V @ S @ V.T
    J = 0.5 * (J - J.T)  # kill round-off in the antisymmetry
    report = spectral_report(H, J)
    report["block_rates"] = np.array(block_rates)
    report["amplitude"] = float(amplitude)
    return J, report


def spectral_report(H: np.ndarray, J: np.ndarray | None = None,
                    D: np.ndarray | None = None) -> dict:
    """Rates and step-size limits of the linearised dynamics ``M = (D + J) H``."""
    d = len(H)
    J = np.zeros((d, d)) if J is None else J
    D = np.eye(d) if D is None else np.asarray(D, float)
    lam = np.linalg.eigvalsh(0.5 * (H + H.T))
    mu = np.linalg.eigvals((D + J) @ H)
    return {
        "eigenvalues_H": lam,
        "reversible_rate": float(lam.min()),
        "mean_rate_bound": float(lam.mean()),
        "irreversible_rate": float(mu.real.min()),
        "max_real_rate": float(mu.real.max()),
        "max_abs_rate": float(np.abs(mu).max()),
        "condition_number": float(lam.max() / lam.min()),
        "speedup": float(mu.real.min() / lam.min()),
    }


def explicit_step_size(
    H: np.ndarray, J: np.ndarray | None = None, safety: float = 0.25,
    D: np.ndarray | None = None,
) -> float:
    """Step size for an explicit unadjusted step, matched across fields.

    The rule is ``h = 2 safety / max_i Re mu_i`` for ``mu`` the spectrum of
    ``(D + J) H``.  Two things make it the right rule for comparing fields.
    First, it is the classical explicit-Euler optimum for a reversible chain up
    to the ``safety`` factor, so ``J = 0`` runs at its own best step size rather
    than at a step size chosen for someone else.  Second, the unadjusted chain
    inflates the stationary variance of mode ``i`` by ``2 / (2 - h mu_i)``, so
    fixing ``h max Re mu`` fixes the worst-case inflation -- at ``safety = 0.25``
    it is at most ``4 / 3`` -- and every field is compared at the same
    discretisation bias even though each runs at its own step size.
    """
    d = len(H)
    J = np.zeros((d, d)) if J is None else J
    D = np.eye(d) if D is None else np.asarray(D, float)
    mu = np.linalg.eigvals((D + J) @ np.asarray(H, float))
    return float(2.0 * safety / mu.real.max())


def variance_inflation(H: np.ndarray, J: np.ndarray | None, h: float,
                       D: np.ndarray | None = None) -> float:
    """Worst-case stationary variance inflation of the unadjusted chain."""
    d = len(H)
    J = np.zeros((d, d)) if J is None else J
    D = np.eye(d) if D is None else np.asarray(D, float)
    mu = np.linalg.eigvals((D + J) @ np.asarray(H, float)).real
    return float(np.max(2.0 / (2.0 - h * mu)))


def discrete_decay(H: np.ndarray, J: np.ndarray | None, h: float,
                   D: np.ndarray | None = None) -> float:
    """Worst per-iteration contraction factor of the explicit step at ``h``.

    Values below one contract; a value above one means the step is unstable for
    that field, which is the check the runner applies before using ``h``.
    """
    d = len(H)
    J = np.zeros((d, d)) if J is None else J
    D = np.eye(d) if D is None else np.asarray(D, float)
    mu = np.linalg.eigvals((D + J) @ H)
    return float(np.abs(1.0 - h * mu).max())


def _slow_coordinates(H: np.ndarray, n_slow: int) -> tuple:
    """Standardised projections onto the slowest directions of ``H``.

    Returns a matrix ``P`` of shape ``(n_slow, d)`` such that ``P (w - m)`` has
    unit variance in each component under the Gaussian ``N(m, H^{-1})``, taken
    along the eigendirections with the smallest eigenvalues -- the ones the
    reversible dynamics is slowest to explore.
    """
    lam, V = np.linalg.eigh(0.5 * (H + H.T))
    k = int(min(n_slow, len(lam)))
    return (V[:, :k] * np.sqrt(lam[:k])).T, lam[:k]


def ensemble_discrepancy(W: np.ndarray, m: np.ndarray, P: np.ndarray) -> float:
    """How far an ensemble is from the target spread along the slow directions.

    Each standardised slow coordinate should have mean zero and unit variance at
    stationarity, so ``(log var)^2 + mean^2`` is zero for an equilibrated
    ensemble, large for one that has not spread out yet, and large again for one
    that has overshot into the tails.  It needs no reference posterior and no
    held-out data: ``m`` and ``P`` come from the warm-up.
    """
    U = P @ (W - np.asarray(m).reshape(-1, 1))
    var = np.maximum(U.var(axis=1, ddof=1), 1e-12)
    return float(np.mean(np.log(var) ** 2 + U.mean(axis=1) ** 2))


def calibrate_amplitude(
    target,
    H: np.ndarray,
    m: np.ndarray,
    make_field=None,
    amplitudes=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8),
    safety: float = 0.15,
    n_iter: int = 300,
    n_checkpoints: int = 15,
    n_walkers: int = 16,
    n_slow: int = 4,
    seed: int = 0,
    guard_iter: int = 150,
    fd_eps: float = 1e-2,
) -> dict:
    """Pick the rotation amplitude from a short unadjusted pilot.

    The collision amplitude comes from a quadratic model of the bulk, so on a
    posterior that is far from quadratic -- a near-separable logistic likelihood,
    say -- it can be too aggressive: the rotation turns a large gradient along a
    stiff direction into a large velocity along a flat one and the walker
    overshoots into the tail instead of settling.  The pilot scores each
    amplitude by :func:`ensemble_discrepancy` along the slowest directions,
    averaged over checkpoints so that a chain which equilibrates sooner scores
    better, and penalises overshoot as much as sluggishness.  The potential
    ``U`` is *not* a usable score here: the directions being accelerated are the
    flat ones, which barely move it.

    Everything the score uses -- the warm-up Hessian estimate and mean -- is
    already used by the design itself: no gradients, no held-out data, no
    reference posterior.  ``0.0`` sits in the ladder, so the procedure is free
    to conclude that no rotation beats some rotation.

    ``make_field(amplitude)`` returns the field to score and the constant matrix
    whose spectrum sets the step size; it defaults to the constant field, and
    each field being compared should be calibrated with its own factory, since
    a field that switches itself off in the tails tolerates a larger amplitude
    than one that does not.
    """
    if make_field is None:

        def make_field(amplitude):
            J, _ = paired_skew(H, amplitude=amplitude)
            return ConstantSkew(J, 1.0), J

    geometry = Geometry.identity(len(H))
    P, _ = _slow_coordinates(H, n_slow)
    segment = max(n_iter // n_checkpoints, 1)
    rows = []
    for amplitude in amplitudes:
        field, J = (ZeroSkew(), None) if amplitude == 0.0 else make_field(amplitude)
        h = stable_step_size(
            target, field, explicit_step_size(H, J, safety=safety),
            n_iter=guard_iter, n_walkers=8, seed=seed + 3,
            geometry=geometry, fd_eps=fd_eps,
        )["step_size"]
        W = np.zeros((len(H), n_walkers))
        scores = []
        for step in range(n_checkpoints):
            res = run_chain(
                target, field, h, segment, n_walkers=n_walkers, seed=seed + 13 + step,
                geometry=geometry, fd_eps=fd_eps, metropolis=False, W0=W,
            )
            W = res.final_state
            if not np.isfinite(W).all():
                scores.append(np.inf)
                break
            scores.append(ensemble_discrepancy(W, m, P))
        rows.append({
            "amplitude": float(amplitude),
            "step_size": h,
            "score": float(np.mean(scores)),
            "final_score": float(scores[-1]),
        })
    finite = [row for row in rows if np.isfinite(row["score"])]
    best = min(finite or rows, key=lambda row: row["score"])
    return {"amplitude": best["amplitude"], "ladder": rows}
