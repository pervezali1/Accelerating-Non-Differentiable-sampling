"""Derivative-free irreversible sampling experiments.

The package is deliberately small:

* :mod:`nds.data` -- the four binary-classification benchmarks,
* :mod:`nds.target` -- the logistic posterior and its central-difference surrogate,
* :mod:`nds.skew` -- the skew-symmetric fields ``J`` and their divergences,
* :mod:`nds.sampler` -- Metropolis-corrected irreversible steps, warm-up, calibration,
* :mod:`nds.reference` -- the gradient-based gold-standard posterior,
* :mod:`nds.metrics` -- effective sample size and convergence bookkeeping.
"""

__all__ = ["data", "metrics", "reference", "sampler", "skew", "target"]
