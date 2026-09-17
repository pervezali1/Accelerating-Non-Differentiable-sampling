"""Divergence-free skew-symmetric matrix field ``J(w)``.

Construction
------------
With ``d = 9`` the coordinates are split into the disjoint triples

    (0, 1, 2), (3, 4, 5), (6, 7, 8).

For a triple ``(a, b, c)`` set ``k = (k1, k2, k3) = grad_psi[(a, b, c)]`` and
insert the cross-product (hat) matrix

    [[  0, -k3,  k2],
     [ k3,   0, -k1],
     [-k2,  k1,   0]]

into rows and columns ``(a, b, c)`` of a ``d x d`` zero matrix.  ``J(w)`` is
the sum of these disjoint blocks, i.e. a block-diagonal matrix (up to the
coordinate permutation) whose blocks are ``[k]_x``.

Three structural properties hold and are checked numerically:

1. **Skew-symmetry.**  ``[k]_x^T = -[k]_x`` blockwise, hence ``J^T = -J``.

2. **Divergence-free.**  ``div(J)_i = sum_j d J_ij / d w_j``.  Within a block,
   row ``a`` contributes ``d(-grad_psi[c])/dw_b + d(grad_psi[b])/dw_c
   = -d^2 psi / (dw_b dw_c) + d^2 psi / (dw_c dw_b) = 0``.  (For the separable
   ``psi`` used here the mixed partials vanish individually as well.)
   Because ``div J = 0`` **no ``div(J)`` correction is added to the drift.**

3. **Tangency.**  ``[k]_x k = k x k = 0``, so ``J(w) @ grad_psi(w) = 0`` and
   therefore ``J(w) @ grad_g(w) = 0`` and ``J(w) @ normal(w) = 0``.  In
   particular the non-reversible drift is tangential to every level set of the
   constraint, boundary included.

Operator norm
-------------
``[k]_x^T [k]_x = ||k||^2 I - k k^T`` has eigenvalues ``||k||^2, ||k||^2, 0``,
so the singular values of a block are ``(||k||, ||k||, 0)``.  ``J`` is block
diagonal, hence

    ||J(w)||_2 = max over blocks of ||k_block||_2 .

This closed form is used in the sampler (it costs no SVD) and is verified
against ``numpy.linalg.norm(J, 2)`` in the tests.

Swirl multiplier
----------------
A *constant* multiplier ``s`` may be applied, ``J -> s * J``.  Scaling by a
constant preserves skew-symmetry, the divergence-free property and tangency.
State-dependent normalisation (e.g. dividing by ``||J(w)||``) is **never**
performed: ``div(J/||J||) != 0`` in general, which would destroy the invariant
measure.  Only the product ``alpha * s`` controls the perturbation strength,
so the primary experiment fixes ``s = 1``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from constraint import grad_psi


def coordinate_triples(d: int) -> Tuple[Tuple[int, int, int], ...]:
    """Disjoint consecutive coordinate triples ``(0,1,2), (3,4,5), ...``.

    If ``d`` is not a multiple of three the leftover coordinates are simply
    omitted; the corresponding rows and columns of ``J`` stay zero, which
    preserves every structural property above.
    """
    return tuple(
        (i, i + 1, i + 2) for i in range(0, d - d % 3, 3)
    )


def construct_J(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    swirl: float = 1.0,
    triples: Sequence[Tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Build ``J(w)`` from ``grad_psi(w)`` on disjoint coordinate triples.

    Parameters
    ----------
    w
        State, shape ``(d,)``.
    p_constraint, epsilon_constraint
        Constraint parameters entering ``grad_psi = -grad_g``.
    swirl
        Constant multiplier ``s``.  Must not depend on ``w``.
    triples
        Coordinate partition; defaults to :func:`coordinate_triples`.

    Returns
    -------
    ndarray, shape ``(d, d)``
        Skew-symmetric, divergence-free matrix with ``J @ grad_psi(w) = 0``.
    """
    w = np.asarray(w, dtype=float)
    d = w.size
    if triples is None:
        triples = coordinate_triples(d)

    gradient_psi = grad_psi(w, p_constraint, epsilon_constraint)
    J = np.zeros((d, d), dtype=float)

    for (a, b, c) in triples:
        k1 = gradient_psi[a]
        k2 = gradient_psi[b]
        k3 = gradient_psi[c]
        # [[ 0, -k3,  k2],
        #  [ k3,  0, -k1],
        #  [-k2, k1,   0]]
        J[a, b] = -k3
        J[a, c] = k2
        J[b, a] = k3
        J[b, c] = -k1
        J[c, a] = -k2
        J[c, b] = k1

    if swirl != 1.0:
        J *= swirl
    return J


def operator_norm_J(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    swirl: float = 1.0,
    triples: Sequence[Tuple[int, int, int]] | None = None,
) -> float:
    """Closed-form spectral norm ``||J(w)||_2`` (no SVD required).

    Equals ``s * max_block ||grad_psi[(a,b,c)]||_2`` by the singular-value
    identity documented in the module docstring.
    """
    w = np.asarray(w, dtype=float)
    if triples is None:
        triples = coordinate_triples(w.size)
    if not triples:
        return 0.0
    gradient_psi = grad_psi(w, p_constraint, epsilon_constraint)
    return float(swirl) * max(
        float(np.linalg.norm(gradient_psi[list(t)])) for t in triples
    )


def skew_symmetry_error(J: np.ndarray) -> float:
    """Maximum absolute entry of ``J + J^T`` (zero for a skew matrix)."""
    return float(np.abs(J + J.T).max())


def divergence_J(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    h: float = 1e-5,
    swirl: float = 1.0,
    triples: Sequence[Tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Centred finite-difference divergence ``div(J)_i = sum_j dJ_ij/dw_j``.

    Returns a vector of length ``d``; the test reports its maximum absolute
    component, which must be ~0.
    """
    w = np.asarray(w, dtype=float)
    d = w.size
    divergence = np.zeros(d, dtype=float)
    for j in range(d):
        plus, minus = w.copy(), w.copy()
        plus[j] += h
        minus[j] -= h
        J_plus = construct_J(plus, p_constraint, epsilon_constraint, swirl, triples)
        J_minus = construct_J(minus, p_constraint, epsilon_constraint, swirl, triples)
        divergence += (J_plus[:, j] - J_minus[:, j]) / (2.0 * h)
    return divergence


def tangency_residual(
    w: np.ndarray,
    p_constraint: float,
    epsilon_constraint: float,
    swirl: float = 1.0,
    triples: Sequence[Tuple[int, int, int]] | None = None,
) -> float:
    """``|| J(w) @ normal(w) ||`` with ``normal = grad_g / ||grad_g||``."""
    from constraint import unit_normal  # local import avoids a cycle at import time

    J = construct_J(w, p_constraint, epsilon_constraint, swirl, triples)
    normal = unit_normal(w, p_constraint, epsilon_constraint)
    return float(np.linalg.norm(J @ normal))
