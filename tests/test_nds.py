"""Correctness checks for the derivative-free irreversible sampler.

Run with ``python3 tests/test_nds.py`` or ``python3 -m pytest tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.design import (  # noqa: E402
    discrete_decay,
    explicit_step_size,
    paired_skew,
    variance_inflation,
)
from nds.reference import reference_posterior  # noqa: E402
from nds.sampler import Geometry, calibrate_step_size, run_chain  # noqa: E402
from nds.skew import (  # noqa: E402
    ConstantSkew,
    DirectionalSkew,
    LocalizedSkew,
    ZeroSkew,
    cyclic_skew,
    random_skew,
    whiten,
)
from nds.target import LogisticPosterior  # noqa: E402


def _toy(n: int = 120, d: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))])
    w = rng.normal(size=d)
    p = 1.0 / (1.0 + np.exp(-(X @ w)))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_skew_matrices_are_skew():
    for A in (random_skew(7, 3), cyclic_skew(7)):
        assert np.allclose(A, -A.T, atol=1e-12)
        assert abs(np.linalg.norm(A, ord=2) - 1.0) < 1e-10
    L = np.tril(np.ones((5, 5))) + np.eye(5)
    W = whiten(random_skew(5, 1), L)
    assert np.allclose(W, -W.T, atol=1e-12)


def test_fd_grad_matches_analytic_gradient():
    X, y = _toy()
    target = LogisticPosterior(X, y)
    rng = np.random.default_rng(1)
    W = rng.normal(size=(target.d, 5)) * 0.6
    fd = target.fd_grad(W, eps=1e-3)
    exact = target.grad(W)
    assert np.abs(fd - exact).max() / np.abs(exact).max() < 1e-5


def test_divergence_matches_finite_differences():
    d, rho, alpha, eps = 5, 1.4, 0.8, 1e-6
    rng = np.random.default_rng(2)
    B = rng.normal(size=(d, d))
    D = B @ B.T + np.eye(d)
    metric = np.linalg.inv(D)
    A = whiten(random_skew(d, 4), np.linalg.cholesky(D))
    for profile in ("grow", "decay"):
        field = LocalizedSkew(A, alpha, rho=rho, profile=profile, metric=metric)
        w = rng.normal(size=(d, 1))
        numeric = np.zeros((d, 1))
        for j in range(d):
            wp, wm = w.copy(), w.copy()
            wp[j] += eps
            wm[j] -= eps
            numeric[:, 0] += alpha * (field.scale(wp)[0, 0] - field.scale(wm)[0, 0]) / (2 * eps) * A[:, j]
        assert np.abs(numeric - field.divergence(w)).max() < 1e-7
    assert np.allclose(
        LocalizedSkew(A, alpha, rho=rho, metric=metric, drop_correction=True).divergence(
            np.ones((d, 1))
        ),
        0.0,
    )
    assert np.allclose(ConstantSkew(A, alpha).divergence(np.ones((d, 1))), 0.0)


def test_directional_divergence_matches_finite_differences():
    d, eps, alpha = 6, 1e-6, 0.9
    rng = np.random.default_rng(7)
    B = rng.normal(size=(d, d))
    D = B @ B.T + np.eye(d)
    L = np.linalg.cholesky(D)
    A = whiten(random_skew(d, 8), L)
    direction = np.linalg.solve(L.T, np.eye(d)[:, 0])
    center = rng.normal(size=d) * 3.0
    field = DirectionalSkew(A, alpha, direction=direction, length_scale=1.3, center=center)
    w = center.reshape(d, 1) + rng.normal(size=(d, 1)) * 0.5
    numeric = np.zeros((d, 1))
    for j in range(d):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        numeric[:, 0] += alpha * (field.scale(wp)[0, 0] - field.scale(wm)[0, 0]) / (2 * eps) * A[:, j]
    assert np.abs(numeric - field.divergence(w)).max() < 1e-6
    # the profile is centred on the bulk, so the correction is not negligible there
    assert np.linalg.norm(field.divergence(center.reshape(d, 1))) > 0.1


def _spread_spd(d: int = 12, kappa: float = 500.0, seed: int = 11) -> np.ndarray:
    """A positive definite matrix with a log-spread spectrum."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    lam = np.logspace(0, np.log10(kappa), d)
    return Q @ np.diag(lam) @ Q.T


def test_paired_skew_removes_the_slowest_direction():
    H = _spread_spd()
    J, report = paired_skew(H)
    assert np.allclose(J, -J.T, atol=1e-10)
    lam = np.linalg.eigvalsh(H)
    # the achieved rate is the smallest block mean, well above lambda_min
    assert abs(report["irreversible_rate"] - report["block_rates"].min()) < 1e-6 * lam.max()
    assert report["irreversible_rate"] > 5.0 * lam.min()
    # and it cannot exceed the mean eigenvalue, which no skew term can beat
    assert report["irreversible_rate"] <= report["mean_rate_bound"] * (1 + 1e-9)
    # the spectrum stays real, which is what keeps the explicit step size
    mu = np.linalg.eigvals((np.eye(len(H)) + J) @ H)
    assert np.abs(mu.imag).max() < 1e-6 * lam.max()
    # zero amplitude is the reversible baseline
    J0, report0 = paired_skew(H, amplitude=0.0)
    assert np.allclose(J0, 0.0)
    assert abs(report0["irreversible_rate"] - lam.min()) < 1e-8 * lam.max()


def test_step_size_rule_matches_the_inflation_bound():
    H = _spread_spd()
    J, _ = paired_skew(H)
    for safety in (0.1, 0.25, 0.5):
        for field in (None, J):
            h = explicit_step_size(H, field, safety=safety)
            assert discrete_decay(H, field, h) < 1.0
            bound = 2.0 / (2.0 - 2.0 * safety)
            assert variance_inflation(H, field, h) <= bound * (1 + 1e-9)


def test_designed_field_converges_faster_unadjusted():
    """End-to-end: on an anisotropic posterior the designed rotation wins."""
    rng = np.random.default_rng(3)
    n, d = 400, 8
    scales = np.logspace(0, -1.5, d - 1)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1)) * scales])
    w_true = rng.normal(size=d)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X @ w_true)))).astype(float)
    target = LogisticPosterior(X, y, prior_scale=2.0)

    ref = reference_posterior(target, X, y, n_iter=6000, n_chains=8, seed=0)
    H = np.linalg.pinv(ref["posterior_cov"])
    metric = H
    J, _ = paired_skew(H)
    errors = {}
    for key, field in (("zero", ZeroSkew()), ("designed", ConstantSkew(J, 1.0))):
        h = explicit_step_size(H, None if key == "zero" else J, safety=0.25)
        res = run_chain(
            target, field, h, 400, n_walkers=16, seed=5,
            geometry=Geometry.identity(d), metropolis=False,
            ref_mean=ref["posterior_mean"], ref_metric=metric,
        )
        errors[key] = float(res.mean_error[-1].mean())
    assert errors["designed"] < 0.6 * errors["zero"], errors


def test_every_variant_samples_the_same_posterior():
    """The Metropolis correction must make all four variants agree with a long
    reference run, whether or not the divergence term is included."""
    X, y = _toy(n=200, d=4, seed=5)
    target = LogisticPosterior(X, y)
    ref = reference_posterior(target, X, y, n_iter=6000, n_chains=8, seed=0)
    geo = Geometry.from_matrix(ref["laplace_cov"])
    metric = np.linalg.pinv(ref["posterior_cov"])
    A = whiten(random_skew(target.d, 0), geo.L)
    rho = float(np.sqrt(max(ref["posterior_mean"] @ (metric @ ref["posterior_mean"]), 1e-9)))
    fields = [
        ZeroSkew(),
        ConstantSkew(A, 0.5),
        LocalizedSkew(A, 0.5, rho=rho, metric=metric),
        LocalizedSkew(A, 0.5, rho=rho, metric=metric, drop_correction=True),
    ]
    for field in fields:
        h = calibrate_step_size(
            target, field, n_iter=150, n_walkers=6, geometry=geo, seed=1
        )["step_size"]
        res = run_chain(
            target, field, h, 4000, n_walkers=16, seed=2, geometry=geo,
            W0=np.repeat(ref["posterior_mean"][:, None], 16, axis=1),
            ref_mean=ref["posterior_mean"], ref_metric=metric,
        )
        # error is measured in units of reference posterior standard deviations
        assert res.mean_error[-1].mean() < 0.5, (field.label, res.mean_error[-1].mean())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
