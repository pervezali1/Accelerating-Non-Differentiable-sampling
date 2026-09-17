r"""The constraint set and the state-dependent skew-symmetric field ``J_s(w)``.

In three dimensions the field is the hat map of the offset ``q = w - center``,

.. math::
    J_s(q) = s \begin{pmatrix} 0 & -q_3 & q_2 \\ q_3 & 0 & -q_1 \\
                              -q_2 & q_1 & 0 \end{pmatrix},
    \qquad J_s(q)\,v = s\, (q \times v).

The telescope posterior has ``D = 11 > 3``, so the ``D``-dimensional field is
built by embedding one such block per coordinate triple ``(i, j, k)`` and adding
them:

.. math::
    J[i,j] \mathrel{-}= c\,q_k,\quad J[i,k] \mathrel{+}= c\,q_j,\quad
    J[j,k] \mathrel{-}= c\,q_i,

with the transposed entries carrying the opposite sign and
``c = s/\sqrt{m}`` for ``m`` triples, so that adding blocks does not inflate
``\|J\|`` with ``m``.

Three properties make this usable, and all three survive the sum because each is
linear in the blocks.

**Skew symmetry.** Each block is skew by construction, so ``J' = -J``.

**Tangency,** ``q'J_s(w) = 0``.  For a single block this is
``q'(q \times \cdot) = 0``; summed over blocks each term still annihilates its
own three components of ``q`` and touches no others.  On the boundary of a ball
centred at ``center`` the outward normal is parallel to ``q``, so the
non-reversible drift ``-a J_s \nabla U_0`` is *tangent* to the boundary.  That
is what makes a plain projection legitimate here: the skew term never pushes
against the constraint, so it needs no oblique reflection and no boundary
correction.

**Divergence freedom,** ``\operatorname{div} J_s(w) = 0``, meaning
``\sum_j \partial_j J[i,j] = 0`` for every ``i``.  This is stronger than "each
block is divergence free": entry ``J[i,j]`` of the block on ``(i,j,k)`` is
proportional to ``q_k`` with ``k \notin \{i,j\}``, so ``\partial_j J[i,j] = 0``
*termwise*, before any cancellation.  Consequently the drift needs no
``\operatorname{div} J_s`` correction term, and none is added.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "generate_cyclic_triples",
    "construct_J",
    "J_matvec",
    "project_to_centered_ball",
    "distance_to_center",
    "check_skew_symmetry",
    "check_tangency",
    "check_divergence_free",
    "verify_field_properties",
    "natural_alpha_scale",
    "field_operator_norm",
    "TRIPLES_D11",
]

#: the cover the specification names for ``D = 11``; :func:`generate_cyclic_triples`
#: reproduces it exactly and this constant exists so a test can say so
TRIPLES_D11: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (2, 3, 4),
    (4, 5, 6),
    (6, 7, 8),
    (8, 9, 10),
    (10, 0, 1),
)


def generate_cyclic_triples(D: int) -> tuple[tuple[int, int, int], ...]:
    r"""Overlapping cyclic triples ``(i, i+1, i+2) mod D`` for ``i = 0, 2, 4, ...``.

    Stepping by two makes consecutive triples share one coordinate, which is what
    couples the blocks into a single connected chain rather than a set of
    independent rotations; stepping by three would give a block-diagonal field
    whose blocks never talk to each other.  Every coordinate is covered, and
    triples that repeat the same *set* of indices are dropped (which only ever
    happens at ``D = 3``, where the cycle comes back round to itself).

    At ``D = 11`` this returns exactly the specified cover
    ``(0,1,2), (2,3,4), (4,5,6), (6,7,8), (8,9,10), (10,0,1)``.
    """
    if D < 3:
        raise ValueError(f"the cross-product construction needs D >= 3, got {D}")
    triples: list[tuple[int, int, int]] = []
    seen: set[frozenset[int]] = set()
    for i in range(0, D, 2):
        triple = (i % D, (i + 1) % D, (i + 2) % D)
        if len(set(triple)) != 3:
            raise ValueError(f"degenerate triple {triple} at D={D}")
        key = frozenset(triple)
        if key in seen:
            continue
        seen.add(key)
        triples.append(triple)
    covered = {idx for triple in triples for idx in triple}
    missing = set(range(D)) - covered
    if missing:
        raise ValueError(f"triples do not cover coordinates {sorted(missing)}")
    return tuple(triples)


def construct_J(
    w: np.ndarray,
    center: np.ndarray,
    triples: tuple[tuple[int, int, int], ...],
    s: float = 1.0,
) -> np.ndarray:
    r"""The ``D x D`` skew field ``J_s(w)`` at ``w``, from ``q = w - center``.

    ``scale = s / sqrt(len(triples))`` keeps the operator norm from growing like
    the number of blocks: the blocks act on overlapping but largely different
    coordinate planes, so their contributions add roughly in quadrature.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    center = np.asarray(center, dtype=np.float64).ravel()
    if w.shape != center.shape:
        raise ValueError(f"w has shape {w.shape} but center has {center.shape}")
    D = w.size
    q = w - center
    J = np.zeros((D, D), dtype=np.float64)
    scale = s / np.sqrt(len(triples))
    for i, j, k in triples:
        J[i, j] -= scale * q[k]
        J[i, k] += scale * q[j]
        J[j, i] += scale * q[k]
        J[j, k] -= scale * q[i]
        J[k, i] -= scale * q[j]
        J[k, j] += scale * q[i]
    return J


def J_matvec(
    q: np.ndarray,
    v: np.ndarray,
    triples: tuple[tuple[int, int, int], ...],
    s: float = 1.0,
) -> np.ndarray:
    r"""``J_s(w) v`` without forming the matrix, as a sum of cross products.

    Block ``(i,j,k)`` contributes ``scale * (q_{ijk} x v_{ijk})`` to the three
    components ``i, j, k``.  Mathematically identical to
    ``construct_J(...) @ v``; kept because it is ``O(m)`` rather than ``O(D^2)``
    and because agreement between the two is a useful test of the sign
    convention.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    out = np.zeros_like(q)
    scale = s / np.sqrt(len(triples))
    for i, j, k in triples:
        idx = [i, j, k]
        out[idx] += scale * np.cross(q[idx], v[idx])
    return out


def distance_to_center(w: np.ndarray, center: np.ndarray) -> np.ndarray:
    r"""``||w - center||_2``, for a single state or a ``(n, D)`` stack of them."""
    w = np.asarray(w, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).ravel()
    return np.linalg.norm(np.atleast_2d(w) - center, axis=1) if w.ndim > 1 else float(
        np.linalg.norm(w - center)
    )


def project_to_centered_ball(
    w: np.ndarray, center: np.ndarray, R: float
) -> tuple[np.ndarray, bool]:
    r"""Euclidean projection onto ``{v : ||v - center|| <= R}``.

    With ``q = w - center``, the projection is ``w`` itself when ``||q|| <= R``
    and ``center + R q / ||q||`` otherwise.  Returns the projected point and
    whether the projection was active, so the sampler can record how often the
    constraint binds.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    center = np.asarray(center, dtype=np.float64).ravel()
    q = w - center
    norm = float(np.linalg.norm(q))
    if norm <= R or norm == 0.0:
        return w, False
    return center + (R / norm) * q, True


def field_operator_norm(
    w: np.ndarray,
    center: np.ndarray,
    triples: tuple[tuple[int, int, int], ...],
    s: float = 1.0,
) -> float:
    r"""``||J_s(w)||_2``, the spectral norm of the field at one state."""
    return float(np.linalg.norm(construct_J(w, center, triples, s), 2))


def natural_alpha_scale(
    radius: float, n_triples: int, s: float = 1.0
) -> float:
    r"""The ``alpha`` at which the rotation is comparable to the identity.

    This is the scale that makes ``alpha`` interpretable, and it is not ``1``.
    The field is **linear in the offset** ``q = w - center``: a single block
    contributes a hat map of ``q`` restricted to three coordinates, so
    ``||J_s(w)|| \approx s\,\|q\|/\sqrt{m}`` with ``m`` the number of blocks.
    A posterior that sits a distance ``R`` from its own centre therefore sees a
    perturbation of size ``\alpha s R/\sqrt{m}``, and

    .. math:: \alpha^\star = \sqrt{m} / (s R)

    is where ``\|\alpha J_s\| \approx 1``, i.e. where the non-reversible drift
    is the same size as the reversible one.  On this problem ``R \approx 0.19``
    and ``m = 6``, so ``\alpha^\star \approx 13``: at ``\alpha`` of order one the
    rotation is a few-percent perturbation and *cannot* change anything
    measurable, which is a statement about the geometry of the construction and
    not about non-reversibility.  Sweeps should be read in units of
    ``\alpha^\star``.
    """
    if radius <= 0 or s == 0:
        return float("inf")
    return float(np.sqrt(n_triples) / (abs(s) * radius))


# ------------------------------------------------------------------ validation
def check_skew_symmetry(J: np.ndarray, tol: float = 1e-10) -> float:
    r"""Assert ``||J + J'|| < tol`` and return the residual."""
    residual = float(np.linalg.norm(J + J.T))
    if residual >= tol:
        raise AssertionError(f"J is not skew-symmetric: ||J + J'|| = {residual:.3e}")
    return residual


def check_tangency(
    J: np.ndarray, q: np.ndarray, tol: float = 1e-9
) -> float:
    r"""Assert ``||q' J|| < tol`` and return the residual.

    The tolerance is scaled by ``||q|| ||J||`` because the identity is exact in
    real arithmetic and only limited by rounding, which grows with the size of
    the operands.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    residual = float(np.linalg.norm(q @ J))
    scale = max(1.0, float(np.linalg.norm(q)) * float(np.linalg.norm(J)))
    if residual >= tol * scale:
        raise AssertionError(
            f"tangency q'J = 0 violated: ||q'J|| = {residual:.3e} "
            f"(tolerance {tol * scale:.3e})"
        )
    return residual


def check_divergence_free(
    w: np.ndarray,
    center: np.ndarray,
    triples: tuple[tuple[int, int, int], ...],
    s: float = 1.0,
    eps: float = 1e-5,
    tol: float = 1e-7,
) -> float:
    r"""Assert ``sum_j d J[i,j] / d w_j = 0`` by central differences.

    The identity is exact and termwise -- ``J[i,j]`` depends on ``q_k`` only,
    with ``k`` outside ``{i,j}`` -- so this is a check on the implementation
    rather than on the mathematics, and it is what licenses leaving the
    ``div J`` correction out of the drift.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    D = w.size
    div = np.zeros(D)
    for j in range(D):
        step = np.zeros(D)
        step[j] = eps
        J_plus = construct_J(w + step, center, triples, s)
        J_minus = construct_J(w - step, center, triples, s)
        div += (J_plus[:, j] - J_minus[:, j]) / (2.0 * eps)
    residual = float(np.linalg.norm(div))
    if residual >= tol:
        raise AssertionError(f"div J is not zero: ||div J|| = {residual:.3e}")
    return residual


def verify_field_properties(
    center: np.ndarray,
    triples: tuple[tuple[int, int, int], ...],
    s: float = 1.0,
    n_points: int = 16,
    seed: int = 0,
    radius: float = 3.0,
) -> dict[str, float]:
    r"""Check skewness, tangency, divergence freedom and ``J_matvec`` agreement.

    Evaluated at ``n_points`` random states around ``center``.  Returns the worst
    residual of each identity, so a caller can log them rather than only learn
    that nothing raised.
    """
    center = np.asarray(center, dtype=np.float64).ravel()
    D = center.size
    rng = np.random.default_rng(seed)
    worst = {"skew": 0.0, "tangency": 0.0, "divergence": 0.0, "matvec": 0.0}
    for _ in range(n_points):
        q = rng.normal(scale=radius, size=D)
        w = center + q
        J = construct_J(w, center, triples, s)
        worst["skew"] = max(worst["skew"], check_skew_symmetry(J))
        worst["tangency"] = max(worst["tangency"], check_tangency(J, q))
        worst["divergence"] = max(
            worst["divergence"], check_divergence_free(w, center, triples, s)
        )
        v = rng.normal(size=D)
        worst["matvec"] = max(
            worst["matvec"], float(np.abs(J @ v - J_matvec(q, v, triples, s)).max())
        )
    if worst["matvec"] >= 1e-10:
        raise AssertionError(
            f"J_matvec disagrees with construct_J by {worst['matvec']:.3e}"
        )
    return worst
