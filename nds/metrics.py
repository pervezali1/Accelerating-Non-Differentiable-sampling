"""Convergence diagnostics for the ensemble runs."""

from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of a one-dimensional trace, via FFT."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n]
    if acf[0] <= 0:
        return np.zeros(n)
    return acf / acf[0]


def ess(traces: np.ndarray) -> float:
    """Effective sample size of ``(n_iter, m)`` walker traces.

    Geyer's initial-positive-sequence estimator is applied per walker and the
    results are summed, which is the right thing here because the walkers are
    independent chains.
    """
    traces = np.atleast_2d(np.asarray(traces, float))
    if traces.ndim == 1:
        traces = traces[:, None]
    total = 0.0
    for m in range(traces.shape[1]):
        rho = autocorrelation(traces[:, m])
        n = len(rho)
        # sum pairs until the pair sum goes negative (Geyer, 1992)
        s = 0.0
        for k in range(1, n - 1, 2):
            pair = rho[k] + rho[k + 1]
            if pair < 0:
                break
            s += pair
        total += n / max(1.0 + 2.0 * s, 1.0)
    return float(total)


def iterations_to_reach(curve: np.ndarray, reference: float, tolerance: float) -> int:
    """First iteration at which ``curve`` stays within ``tolerance`` of ``reference``.

    Returns ``-1`` if the curve never settles inside the band.
    """
    inside = np.abs(np.asarray(curve, float) - reference) <= tolerance
    if not inside.any():
        return -1
    # the first index from which everything that follows is inside the band
    tail = np.flatnonzero(~inside)
    return int(tail[-1] + 1) if len(tail) and tail[-1] + 1 < len(inside) else (
        0 if inside.all() else -1
    )
