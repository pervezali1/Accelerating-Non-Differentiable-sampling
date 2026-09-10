"""Skew-symmetric acceleration of anchored Langevin algorithms on heavy tails.

Extends the anchored Langevin dynamics of Gurbuzbalaban, Nguyen, Zhang and Zhu
(arXiv:2509.19455) with a constant skew-symmetric matrix ``J``, which makes the
dynamics non-reversible while leaving the target invariant.
"""

from . import analysis, metrics, runner, samplers, skew, targets  # noqa: F401

__all__ = ["analysis", "metrics", "runner", "samplers", "skew", "targets"]
__version__ = "0.1.0"
