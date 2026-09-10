r"""Metrics for comparing an ensemble against a heavy-tailed target.

The primary metric follows the paper: the 2-Wasserstein distance, computed from
the *exact* quantile function of the target (Section 6.1/6.4).  In dimension
``d > 1`` we use the sliced 2-Wasserstein distance,
``SW2^2 = (1/L) sum_l W2^2(proj_l nu, proj_l pi)`` -- the paper takes ``L = d``
axis-aligned projections.

Because every projection of a multivariate Student-t is again a univariate
Student-t, the reference quantiles are available in closed form, so no sampling
noise enters through the reference side.

Heavy tails make the empirical 2-Wasserstein distance a noisy and downward-slow
estimator.  Two guards are provided:

* :func:`w2_reference_floor` computes the same statistic on *exact* i.i.d.
  draws from the target.  Every convergence curve bottoms out at this floor;
  reporting it stops us from reading estimator noise as a real difference.
* Robust companions (:func:`energy_distance`, :func:`mmd2_imq`,
  :func:`ks_statistic`) are bounded or light-tailed functionals and are used to
  confirm anything the Wasserstein curves suggest.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "w2_1d_to_exact",
    "w2_squared_1d_to_exact",
    "w2_squared_1d_two_sample",
    "w2_squared_1d_midpoint",
    "sliced_w2_midpoint",
    "sliced_w2_two_sample",
    "axis_directions",
    "random_directions",
    "sliced_w2",
    "axis_sliced_w2",
    "w2_reference_floor",
    "energy_distance",
    "mmd2_imq",
    "ks_statistic",
    "moment_errors",
    "slow_direction_error",
    "iact_geyer",
    "ess",
]


# ----------------------------------------------------------- 1-d Wasserstein


def _cell_quadrature(n, n_quad=8, n_tail=48, tail_power=8):
    """Quadrature nodes/weights for each cell ``[(j-1)/n, j/n]`` of the uniform
    grid on ``(0, 1)``.

    Interior cells get plain Gauss-Legendre in ``u``.  The two extreme cells are
    where a heavy-tailed reference quantile blows up (``F^{-1}(u) ~ u^{-1/nu}``
    as ``u -> 0``), so they are integrated in the *tail probability* ``w`` with
    the substitution ``w = (1/n) v^p``: the integrand becomes
    ``v^{p(1-2/nu)-1}``, which is smooth for ``p`` large enough.  The upper cell
    is evaluated through the inverse survival function so that ``1 - w`` is never
    formed in floating point (it would round to exactly 1 and give ``+inf``).

    Returns ``(u_mid, w_mid, v_tail, w_tail)`` where ``u_mid``/``w_mid`` cover
    cells ``1 .. n-2`` and ``v_tail``/``w_tail`` are the shared tail-probability
    nodes for cells ``0`` and ``n-1``.
    """
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_quad)
    off = 0.5 * (gl_x + 1.0)
    wgt = 0.5 * gl_w / n

    j = np.arange(1, n - 1)
    u_mid = (j[:, None] + off[None, :]) / n
    w_mid = np.tile(wgt, (n - 2, 1))

    tx, tw = np.polynomial.legendre.leggauss(n_tail)
    v = 0.5 * (tx + 1.0)
    vw = 0.5 * tw
    a = 1.0 / n
    p = tail_power
    v_tail = a * v ** p                      # tail probability nodes in (0, 1/n)
    w_tail = a * p * v ** (p - 1) * vw       # matching weights
    return u_mid, w_mid, v_tail, w_tail


_QUAD_CACHE = {}


def _quad(n):
    if n not in _QUAD_CACHE:
        _QUAD_CACHE[n] = _cell_quadrature(n)
    return _QUAD_CACHE[n]


def w2_squared_1d_to_exact(x, ppf, isf=None):
    r"""Exact ``W_2^2`` between the empirical law of ``x`` and a continuous law.

    ``W_2^2 = int_0^1 (F_n^{-1}(u) - F^{-1}(u))^2 du``, with ``F_n^{-1}`` the
    piecewise-constant empirical quantile function.

    ``ppf`` is the reference quantile function; ``isf`` is its inverse survival
    function (``isf(w) == ppf(1-w)``, evaluated stably).  When ``isf`` is not
    given it is emulated as ``ppf(1-w)``, which loses the far upper tail.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 samples")
    if isf is None:
        isf = lambda w: ppf(1.0 - w)
    u_mid, w_mid, v_tail, w_tail = _quad(n)

    total = float(np.sum(w_mid * (x[1:-1, None] - ppf(u_mid)) ** 2))
    total += float(np.sum(w_tail * (x[0] - ppf(v_tail)) ** 2))
    total += float(np.sum(w_tail * (x[-1] - isf(v_tail)) ** 2))
    return total


def w2_1d_to_exact(x, ppf, isf=None):
    return float(np.sqrt(max(w2_squared_1d_to_exact(x, ppf, isf), 0.0)))


def w2_squared_1d_midpoint(x, ppf):
    """Naive midpoint quantile estimator ``(1/n) sum_j (x_(j) - F^{-1}((j-0.5)/n))^2``.

    This is the estimator most implementations reach for, and it *truncates* the
    tail contribution: for a Student-t with small ``nu`` it can report a value
    several times smaller than the true ``W_2^2``.  Kept so our numbers can be
    compared with the ones in the paper, and so the size of the gap can be shown.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = x.size
    u = (np.arange(n) + 0.5) / n
    return float(np.mean((x - ppf(u)) ** 2))


def sliced_w2_midpoint(samples, target, directions=None):
    samples = np.asarray(samples, dtype=np.float64)
    directions = axis_directions(target.d) if directions is None else directions
    total = 0.0
    for theta in directions:
        total += w2_squared_1d_midpoint(
            samples @ theta, lambda u, th=theta: target.projected_ppf(u, th)
        )
    return float(np.sqrt(total / len(directions)))


def w2_squared_1d_two_sample(x, y):
    """``W_2^2`` between two equal-size empirical samples on the line.

    Free of the tail-quadrature issue above; used as a cross-check of the
    quantile-based estimator.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    if x.size != y.size:
        n = min(x.size, y.size)
        u = (np.arange(n) + 0.5) / n
        x = np.quantile(x, u)
        y = np.quantile(y, u)
    return float(np.mean((x - y) ** 2))


# ------------------------------------------------------------- sliced W2


def sliced_w2(samples, target, directions):
    """Sliced 2-Wasserstein distance against exact projected quantiles."""
    samples = np.asarray(samples, dtype=np.float64)
    total = 0.0
    for theta in directions:
        proj = samples @ theta
        total += w2_squared_1d_to_exact(
            proj,
            lambda u, th=theta: target.projected_ppf(u, th),
            lambda w, th=theta: target.projected_isf(w, th),
        )
    return float(np.sqrt(total / len(directions)))


def axis_directions(d):
    return list(np.eye(d))


def random_directions(d, L, rng):
    v = rng.standard_normal((L, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return list(v)


def axis_sliced_w2(samples, target):
    """The paper's choice: ``L = d`` axis-aligned projections."""
    return sliced_w2(samples, target, axis_directions(target.d))


def w2_reference_floor(target, n, rng, n_rep=20, directions=None):
    """Value of the sliced-W2 statistic on *exact* draws -- the noise floor.

    Returns ``(mean, std)`` over ``n_rep`` independent exact samples of size
    ``n``.  Convergence curves cannot go below this.
    """
    directions = axis_directions(target.d) if directions is None else directions
    vals = [sliced_w2(target.sample(n, rng), target, directions) for _ in range(n_rep)]
    return float(np.mean(vals)), float(np.std(vals))


# ------------------------------------------------------- robust companions


def sliced_w2_two_sample(x, y, directions):
    """Sliced ``W_2`` between two empirical samples, for targets whose projected
    quantiles are not available in closed form (exact reference draws instead)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = 0.0
    for theta in directions:
        total += w2_squared_1d_two_sample(x @ theta, y @ theta)
    return float(np.sqrt(total / len(directions)))


def energy_distance(x, y):
    """Squared energy distance ``2 E|X-Y| - E|X-X'| - E|Y-Y'|`` (unbiased-ish)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def mean_dist(a, b, same):
        # chunked to keep memory bounded for n ~ 5000
        total, count = 0.0, 0
        step = 1024
        for i in range(0, a.shape[0], step):
            block = np.linalg.norm(a[i : i + step, None, :] - b[None, :, :], axis=2)
            if same:
                m = block.shape[0]
                idx = np.arange(i, min(i + step, a.shape[0]))
                block[np.arange(m), idx] = 0.0
            total += block.sum()
            count += block.size - (block.shape[0] if same else 0)
        return total / count

    return float(2.0 * mean_dist(x, y, False) - mean_dist(x, x, True) - mean_dist(y, y, True))


def mmd2_imq(x, y, c=1.0, beta=-0.5):
    """Squared MMD with the inverse multiquadric kernel ``(c^2+|u|^2)^beta``.

    Bounded kernel, so this is finite and well behaved for heavy tails.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def k_mean(a, b, same):
        total, count = 0.0, 0
        step = 1024
        for i in range(0, a.shape[0], step):
            sq = np.sum((a[i : i + step, None, :] - b[None, :, :]) ** 2, axis=2)
            block = (c * c + sq) ** beta
            if same:
                m = block.shape[0]
                idx = np.arange(i, min(i + step, a.shape[0]))
                block[np.arange(m), idx] = 0.0
            total += block.sum()
            count += block.size - (block.shape[0] if same else 0)
        return total / count

    return float(k_mean(x, x, True) + k_mean(y, y, True) - 2.0 * k_mean(x, y, False))


def ks_statistic(samples, target):
    """Largest per-coordinate Kolmogorov-Smirnov distance to the exact marginal."""
    samples = np.asarray(samples, dtype=np.float64)
    out = []
    for i in range(target.d):
        scale = np.sqrt(target.Sigma[i, i])
        cdf = lambda v, i=i, s=scale: stats.t.cdf(v, df=target.nu, loc=target.mu[i], scale=s)
        out.append(stats.ks_1samp(samples[:, i], cdf).statistic)
    return float(np.max(out))


# ------------------------------------------------------------ moment errors


def moment_errors(samples, target):
    """Relative errors of the mean and of the covariance (Frobenius)."""
    samples = np.asarray(samples, dtype=np.float64)
    out = {}
    if target.nu > 1:
        out["mean_err"] = float(np.linalg.norm(samples.mean(axis=0) - target.mean()))
    if target.nu > 2:
        C = target.cov()
        out["cov_rel_err"] = float(
            np.linalg.norm(np.cov(samples, rowvar=False).reshape(C.shape) - C) / np.linalg.norm(C)
        )
    return out


def slow_direction_error(samples, target):
    """Relative error of the variance along the slowest direction of ``Sigma``.

    The slow direction (largest eigenvalue of ``Sigma``) is the one a reversible
    sampler explores last, so this is the cleanest low-variance signal of
    acceleration.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if target.nu <= 2:
        return np.nan
    v = target.Sigma_evecs[:, int(np.argmax(target.Sigma_evals))]
    proj = samples @ v
    C = target.cov()
    truth = float(v @ C @ v)
    return float(abs(proj.var() - truth) / truth)


# ------------------------------------------------ single-chain efficiency


def iact_geyer(chain, max_lag=None):
    """Integrated autocorrelation time, Geyer initial-positive-sequence."""
    x = np.asarray(chain, dtype=np.float64).ravel()
    n = x.size
    x = x - x.mean()
    max_lag = min(n - 1, 2000 if max_lag is None else max_lag)
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conjugate(f))[:n].real / n
    if acov[0] <= 0:
        return np.nan
    rho = acov / acov[0]
    # Geyer: sum consecutive pairs while they stay positive
    total, k = 0.0, 1
    while k + 1 < max_lag:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        total += pair
        k += 2
    return float(1.0 + 2.0 * total)


def ess(chain):
    tau = iact_geyer(chain)
    n = np.asarray(chain).size
    return float(n / tau) if np.isfinite(tau) and tau > 0 else np.nan
