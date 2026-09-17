"""Manuscript-faithful construction of the state-dependent matrix ``J(w)``.

For a coordinate triple ``(a, b, c)`` put

    ``k1 = grad_psi[a]``, ``k2 = grad_psi[b]``, ``k3 = grad_psi[c]``

and use the cross-product (hat) matrix of ``k = (k1, k2, k3)``

.. math::

    [k]_\\times = \\begin{pmatrix} 0 & -k_3 & k_2 \\\\
                                   k_3 & 0 & -k_1 \\\\
                                   -k_2 & k_1 & 0 \\end{pmatrix},

inserted into rows and columns ``(a, b, c)`` of a ``d x d`` zero matrix.  The
final matrix is ``J(w) = swirl_scale * sum_of_blocks``.

Three structural properties -- all verified numerically in
:mod:`pnral.diagnostics` and in the test-suite -- make the construction
admissible.

**Skew symmetry.**  ``[k]_x^T = -[k]_x`` blockwise, hence ``J^T = -J``.

**Divergence freeness.**  ``grad_psi[i] = -p w_i (w_i^2 + eps^2)^{p/2-1}``
depends on ``w_i`` only, so ``g`` is separable and, for the block above,

    ``(div J)_a = d/dw_b (k3) + d/dw_c (-k2)
                = d_b d_c psi - d_c d_b psi = 0``,

and likewise for every other column.  The identity is exact (each term
vanishes individually because of separability), so **no ``div J`` correction
term is added to the Langevin drift**.

**Boundary compatibility.**  The outward normal of ``{g = Lambda}`` is
``n = grad_g / ||grad_g||`` and ``grad_g = -grad_psi``.  Restricted to a
triple, the block applied to the normal is proportional to ``k x k = 0``, so
``J(w) n(w) = 0`` exactly, and therefore
``n^T J grad_U0 = -(J n)^T grad_U0 = 0``: the non-reversible probability
current is tangent to the boundary and cannot push mass through it.

Scaling is a *constant*: ``swirl_scale`` for the disjoint construction and
``swirl_scale / sqrt(number_of_triples)`` for the overlapping extension.  No
state-dependent normalisation of ``J`` is ever applied -- that would destroy
divergence freeness.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from .constraint import grad_psi_constraint

__all__ = [
    "cross_product_block",
    "disjoint_triples",
    "overlapping_triples",
    "construct_J_from_triples",
    "construct_J_disjoint",
    "construct_J_overlapping",
    "make_J_builder",
    "operator_norm_J",
    "check_skew_symmetry",
    "finite_difference_divergence",
    "check_divergence_free",
    "check_boundary_tangency",
]

Triple = Tuple[int, int, int]

#: The manuscript's primary triples for ``d = 11``.
DISJOINT_TRIPLES_D11: List[Triple] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

#: The optional overlapping extension for ``d = 11`` (all coordinates swirl).
OVERLAPPING_TRIPLES_D11: List[Triple] = [
    (0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8), (8, 9, 10), (10, 0, 1),
]


def cross_product_block(k1: float, k2: float, k3: float) -> np.ndarray:
    """The ``3 x 3`` cross-product matrix of ``(k1, k2, k3)``."""
    return np.array([[0.0, -k3, k2],
                     [k3, 0.0, -k1],
                     [-k2, k1, 0.0]], dtype=np.float64)


def disjoint_triples(d: int) -> List[Triple]:
    """Consecutive disjoint triples ``(0,1,2), (3,4,5), ...``.

    For ``d = 11`` this returns ``[(0,1,2), (3,4,5), (6,7,8)]``; coordinates
    9 and 10 then receive reversible drift and diffusion but no direct
    non-reversible swirl.
    """
    return [(i, i + 1, i + 2) for i in range(0, d - d % 3, 3)]


def overlapping_triples(d: int) -> List[Triple]:
    """Chained, wrapping triples so that every coordinate participates.

    For ``d = 11`` this reproduces the specification exactly::

        [(0,1,2), (2,3,4), (4,5,6), (6,7,8), (8,9,10), (10,0,1)]
    """
    if d < 5:
        raise ValueError("overlapping triples require d >= 5")
    return [(s, (s + 1) % d, (s + 2) % d) for s in range(0, d, 2)]


def construct_J_from_triples(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    swirl_scale: float,
    triples: Sequence[Triple],
    block_scale: float = 1.0,
) -> np.ndarray:
    """Assemble ``J(w) = swirl_scale * block_scale * sum_t [k^{(t)}]_x``.

    ``block_scale`` is the *constant* ``1`` (disjoint) or
    ``1/sqrt(len(triples))`` (overlapping); it never depends on ``w``.
    """
    w = np.asarray(w, dtype=np.float64)
    d = w.shape[0]
    grad_psi = grad_psi_constraint(w, p_constraint, epsilon_constraint)

    J = np.zeros((d, d), dtype=np.float64)
    for (a, b, c) in triples:
        block = cross_product_block(grad_psi[a], grad_psi[b], grad_psi[c])
        index = np.array([a, b, c])
        # Additive insertion: overlapping triples accumulate on shared entries.
        J[np.ix_(index, index)] += block
    return (swirl_scale * block_scale) * J


def construct_J_disjoint(w: np.ndarray, p_constraint: float,
                         epsilon_constraint: float,
                         swirl_scale: float = 1.0) -> np.ndarray:
    """Primary construction: disjoint triples, constant scaling ``swirl_scale``."""
    w = np.asarray(w, dtype=np.float64)
    return construct_J_from_triples(
        w, p_constraint, epsilon_constraint, swirl_scale,
        disjoint_triples(w.shape[0]), block_scale=1.0)


def construct_J_overlapping(w: np.ndarray, p_constraint: float,
                            epsilon_constraint: float,
                            swirl_scale: float = 1.0) -> np.ndarray:
    """Sensitivity extension: overlapping triples, scaling ``1/sqrt(n_triples)``."""
    w = np.asarray(w, dtype=np.float64)
    triples = overlapping_triples(w.shape[0])
    return construct_J_from_triples(
        w, p_constraint, epsilon_constraint, swirl_scale, triples,
        block_scale=1.0 / np.sqrt(len(triples)))


def make_J_builder(mode: str, d: int, p_constraint: float,
                   epsilon_constraint: float,
                   swirl_scale: float) -> Tuple[Callable[[np.ndarray], np.ndarray],
                                                List[Triple], float]:
    """Return ``(builder, triples, block_scale)`` for ``mode``.

    ``mode`` is ``"disjoint"`` (primary experiment) or ``"overlapping"``
    (labelled extension).  The returned closure takes ``w`` only, so the
    sampler cannot accidentally alter the constraint parameters mid-run.
    """
    if mode == "disjoint":
        triples = disjoint_triples(d)
        block_scale = 1.0
    elif mode == "overlapping":
        triples = overlapping_triples(d)
        block_scale = 1.0 / float(np.sqrt(len(triples)))
    else:
        raise ValueError(f"unknown triples mode {mode!r}")

    def builder(w: np.ndarray) -> np.ndarray:
        return construct_J_from_triples(w, p_constraint, epsilon_constraint,
                                        swirl_scale, triples, block_scale)

    return builder, triples, block_scale


def operator_norm_J(J: np.ndarray, triples: Sequence[Triple] | None = None,
                    grad_psi: np.ndarray | None = None,
                    block_scale: float = 1.0,
                    swirl_scale: float = 1.0) -> float:
    """Spectral norm of ``J``.

    For *disjoint* triples the blocks act on orthogonal subspaces and the
    singular values of ``[k]_x`` are ``(||k||, ||k||, 0)``, so the operator
    norm is ``swirl_scale * max_t ||k^{(t)}||``.  That shortcut is used when
    the triples are disjoint (it is exact and far cheaper than an SVD);
    otherwise the singular values are computed directly.
    """
    if triples is not None and grad_psi is not None:
        flat = [i for t in triples for i in t]
        if len(set(flat)) == len(flat):          # disjoint -> closed form
            norms = [float(np.linalg.norm(grad_psi[list(t)])) for t in triples]
            return float(swirl_scale * block_scale * max(norms))
    return float(np.linalg.norm(J, ord=2))


# ----------------------------------------------------------------------
# Numerical verification helpers
# ----------------------------------------------------------------------
def check_skew_symmetry(J: np.ndarray, tolerance: float = 1e-12) -> float:
    """Return ``max |J + J^T|`` and raise if it exceeds ``tolerance``."""
    residual = float(np.max(np.abs(J + J.T)))
    if residual > tolerance:
        raise AssertionError(f"J is not skew-symmetric: max|J + J^T| = {residual}")
    return residual


def finite_difference_divergence(
    builder: Callable[[np.ndarray], np.ndarray],
    w: np.ndarray,
    step: float = 1e-5,
) -> np.ndarray:
    """Centred finite-difference estimate of ``(div J)_j = sum_i d_i J_ij``."""
    w = np.asarray(w, dtype=np.float64)
    d = w.shape[0]
    divergence = np.zeros(d, dtype=np.float64)
    for i in range(d):
        forward = w.copy()
        backward = w.copy()
        forward[i] += step
        backward[i] -= step
        derivative = (builder(forward) - builder(backward)) / (2.0 * step)
        divergence += derivative[i, :]          # row i contributes d_i J_ij
    return divergence


def check_divergence_free(builder: Callable[[np.ndarray], np.ndarray],
                          w: np.ndarray, step: float = 1e-5,
                          tolerance: float = 1e-8) -> float:
    """Return ``max |div J|`` from finite differences and raise if too large."""
    residual = float(np.max(np.abs(finite_difference_divergence(builder, w, step))))
    if residual > tolerance:
        raise AssertionError(f"div J is not zero: max|div J| = {residual}")
    return residual


def check_boundary_tangency(J: np.ndarray, normal: np.ndarray,
                            tolerance: float = 1e-10) -> float:
    """Return ``||J n||_inf`` and raise if the current is not tangent."""
    residual = float(np.max(np.abs(J @ normal)))
    if residual > tolerance:
        raise AssertionError(f"J n != 0 on the boundary: ||J n||_inf = {residual}")
    return residual
