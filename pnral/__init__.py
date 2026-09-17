"""Projected Non-Reversible Anchored Langevin (PNRAL) on the MAGIC data set.

Modules
-------
``config``                configuration dataclasses and JSON (de)serialisation
``data``                  loading, encoding, splitting and standardising MAGIC
``target``                the non-smooth target potential U
``anchor``                the smooth anchor U0, grad U0 and log a / a
``constraint``            the smoothed l_p functional g and the pilot rule
``nonreversible_matrix``  the skew-symmetric, divergence-free J(w)
``projection``            Euclidean projection onto K
``sampler``               the PNRAL update, calibration and initialisation
``diagnostics``           ESS / R-hat / checks and the MCMC figures
``prediction``            posterior-averaged predictive metrics and figures
``experiment``            the end-to-end driver and command-line interface
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import (anchor, config, constraint, data, diagnostics,
               nonreversible_matrix, prediction, projection, sampler, target)

__all__ = [
    "anchor", "config", "constraint", "data", "diagnostics",
    "nonreversible_matrix", "prediction", "projection", "sampler", "target",
    "__version__",
]
