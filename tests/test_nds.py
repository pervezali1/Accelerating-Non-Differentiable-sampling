"""Correctness checks for the derivative-free irreversible sampler.

Run with ``python3 tests/test_nds.py`` or ``python3 -m pytest tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored_constrained import (  # noqa: E402
    outward_tilt_direction,
    tilted_axial_field,
    Ball as PaperBall,
    LassoLogistic as ConstrainedLasso,
    SmoothedLpBall,
    ball_axial_field,
    reference_constrained_rwm as constrained_lasso_reference,
    run_anchored_srnsgld,
    sublevel_axial_field,
)
from nds.constrained import (  # noqa: E402
    Ball,
    BlockHatField,
    ConstantField,
    FramedField,
    MinibatchLogistic,
    ZeroField,
    curvature_step_size,
    eigen_frame,
    field_from_rho,
    reference_constrained_rwm,
    run_constrained_sgld,
    superdiagonal_skew,
)
from nds.anchored import (  # noqa: E402
    LassoLogistic,
    reference_posterior_rwm,
    run_anchored_chain,
)
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
from nds.target import LogisticPosterior, _sigmoid  # noqa: E402


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


def test_anchor_bounds_the_target_and_its_clock():
    X, y = _toy(n=150, d=5, seed=4)
    target = LassoLogistic(X, y, penalty=5.0, delta=0.05)
    rng = np.random.default_rng(9)
    W = rng.normal(size=(target.d, 6))
    U, U0 = target.potential(W), target.anchor_potential(W)
    assert np.all(U <= U0 + 1e-12)                      # the anchor majorises
    assert np.allclose(target.log_weight(W), U - U0)    # and the likelihood cancels
    clock = target.clock(W)
    assert np.all(clock > 0) and np.all(clock <= 1.0 + 1e-12)
    assert clock.min() >= target.clock_floor - 1e-12
    # the anchor is smooth, so its difference quotient matches its true gradient
    Z = target.linear(W)
    exact = target.X.T @ (_sigmoid(Z) - target.y[:, None]) + target.weights * W / np.sqrt(
        W**2 + target.delta**2
    )
    fd = target.anchor_fd_grad(W, eps=1e-3)
    assert np.abs(fd - exact).max() / np.abs(exact).max() < 1e-4


def test_anchored_chain_hits_the_metropolis_reference():
    """The clock is what makes the smooth-anchor drift target the kinked law."""
    X, y = _toy(n=200, d=4, seed=6)
    target = LassoLogistic(X, y, penalty=5.0, delta=0.05)
    ref = reference_posterior_rwm(target, X, y, n_iter=30000, n_warmup=5000, seed=0)
    metric = np.linalg.pinv(ref["posterior_cov"])
    h = explicit_step_size(metric, None, safety=0.15)
    res = run_anchored_chain(
        target, ZeroSkew(), h, 20000, n_walkers=16, seed=3,
        geometry=Geometry.identity(target.d), ref_mean=ref["posterior_mean"],
        ref_metric=metric, scheme="anchored",
    )
    assert np.isfinite(res.potential[-1]).all()
    assert res.mean_error[-1].mean() < 0.4, res.mean_error[-1].mean()
    assert 0.0 < res.meta["mean_clock"] <= 1.0


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


# --------------------------------------------------------------------------
# The constrained d = 9 construction: a constant tridiagonal J_a, a
# block-diagonal state-dependent J_s of 3x3 hat maps, and a ball to reflect in.
# --------------------------------------------------------------------------
def _numeric_divergence(field, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """``Gamma_i = sum_j d_j J_ij``, by central differences of the matrix."""
    d = len(x)
    out = np.zeros(d)
    for j in range(d):
        xp, xm = x.astype(float).copy(), x.astype(float).copy()
        xp[j] += eps
        xm[j] -= eps
        out += (field.matrix(xp)[:, j] - field.matrix(xm)[:, j]) / (2.0 * eps)
    return out


def test_constrained_fields_are_skew_and_divergence_free():
    """The reason for the block form: no correction term to estimate."""
    rng = np.random.default_rng(0)
    d = 9
    x = rng.normal(size=d)
    V = eigen_frame(np.diag(np.linspace(1.0, 9.0, d)))
    fields = [
        ConstantField(d, 0.7),
        BlockHatField(d, 0.7),
        FramedField(BlockHatField(d, 0.7), V),
        FramedField(ConstantField(d, 0.7), V),
    ]
    for field in fields:
        J = field.matrix(x)
        assert np.abs(J + J.T).max() < 1e-12, field.label
        assert np.abs(_numeric_divergence(field, x)).max() < 1e-8, field.label
        assert np.abs(field.divergence(x[:, None])).max() == 0.0, field.label
        g = rng.normal(size=(d, 1))
        assert np.abs(field.apply(x[:, None], g) - J @ g).max() < 1e-12, field.label

    # the superdiagonal pattern, and its operator norm 2 a cos(pi / (d + 1))
    J = superdiagonal_skew(d, 0.5)
    assert np.allclose(np.diag(J, 1), 0.5) and np.allclose(np.diag(J, -1), -0.5)
    assert np.abs(J).sum() == 2 * (d - 1) * 0.5
    sv = np.linalg.svd(J, compute_uv=False)[0]
    assert abs(sv - 2 * 0.5 * np.cos(np.pi / (d + 1))) < 1e-10


def test_block_hat_field_is_tangential_and_needs_no_oblique_reflection():
    """``J_s(x) x = 0`` makes skew reflection on a centred ball plain projection."""
    rng = np.random.default_rng(1)
    d, r = 9, 2.0
    ball = Ball(r)
    V = eigen_frame(np.diag(np.linspace(1.0, 9.0, d)))
    for field in (BlockHatField(d, 0.9), FramedField(BlockHatField(d, 0.9), V)):
        X = rng.normal(size=(d, 64))
        assert np.abs(field.apply(X, X)).max() < 1e-12, field.label
        Y = X * (1.02 * r / np.sqrt((X * X).sum(axis=0)))  # just outside
        oblique, out, missed = ball.reflect(Y, field)
        assert out.all() and missed == 0
        assert np.abs(oblique - ball.project(Y)).max() < 1e-12, field.label

    # the constant field is *not* tangential, so its reflection is oblique, but
    # it still points inward: n . (I + J) n = 1 because n . J n = 0
    field = ConstantField(d, 0.9)
    X = rng.normal(size=(d, 64))
    N = X / np.sqrt((X * X).sum(axis=0))
    gamma = N + field.apply(r * N, N)
    assert np.abs((N * gamma).sum(axis=0) - 1.0).max() < 1e-12
    Y = N * (1.02 * r)
    oblique, out, missed = ball.reflect(Y, field)
    assert out.all() and missed == 0
    assert np.abs(np.sqrt((oblique * oblique).sum(axis=0)) - r).max() < 1e-10
    assert np.abs(oblique - ball.project(Y)).max() > 1e-6  # genuinely oblique


def test_field_amplitudes_are_matched_by_operator_norm():
    """``rho`` has to mean the same rotation for both fields, or the comparison
    between them is a comparison of amplitudes."""
    rng = np.random.default_rng(2)
    d, r = 9, 2.0
    for rho in (0.5, 2.0):
        assert abs(np.linalg.svd(field_from_rho("constant", d, rho, r).matrix(np.zeros(d)),
                                 compute_uv=False)[0] - rho) < 1e-10
        field = field_from_rho("state", d, rho, r)
        X = rng.normal(size=(d, 400))
        X *= r / np.sqrt((X * X).sum(axis=0))
        norms = [np.linalg.svd(field.matrix(X[:, i]), compute_uv=False)[0] for i in range(400)]
        assert abs(np.mean(norms) / rho - 1.0) < 0.05, (rho, np.mean(norms))


def _constrained_toy(n: int = 200, d: int = 3, seed: int = 7, radius: float = 0.7):
    """A toy whose unconstrained mode is outside ``K_r``, so the ball binds."""
    X, y = _toy(n=n, d=d, seed=seed)
    return MinibatchLogistic(X, y), X, y, Ball(radius)


def test_reflected_chains_all_hit_the_constrained_reference():
    """Whatever ``J`` is, the invariant law has to stay the constrained posterior.

    The reference is a random-walk Metropolis chain that rejects every proposal
    outside the ball, which is exact.  The remaining error is the projected
    Euler scheme's boundary bias, shared by all three fields.
    """
    model, X, y, ball = _constrained_toy()
    h, geom = curvature_step_size(model, ball, safety=0.05)
    ref = reference_constrained_rwm(
        model, ball, X, y, n_iter=60000, n_warmup=10000, seed=0,
        start=np.asarray(geom["map"], float),
    )
    metric = np.linalg.pinv(ref["posterior_cov"])
    assert 0.1 < ref["acceptance"] < 0.5
    errors = {}
    for key in ("zero", "constant", "state"):
        field = field_from_rho(key, model.d, 1.0, ball.radius)
        out = run_constrained_sgld(
            model, field, ball, X, y, n_iter=20000, n_walkers=16, step_size=h,
            batch_size=model.n, seed=2, start_radius=ball.radius,
            reference_mean=ref["posterior_mean"], reference_metric=metric,
        )
        assert out["missed_reflections"] == 0, key
        assert out["boundary_rate"] > 0.05, key  # the constraint is doing something
        errors[key] = out["mean_error"][-1]
        # error in units of posterior standard deviations
        assert errors[key] < 0.6, (key, errors[key])
    assert max(errors.values()) - min(errors.values()) < 0.2, errors


def test_projection_bias_shrinks_with_the_step_size():
    """The shared error floor is discretisation, not a wrong invariant law."""
    model, X, y, ball = _constrained_toy()
    h, geom = curvature_step_size(model, ball, safety=0.05)
    ref = reference_constrained_rwm(
        model, ball, X, y, n_iter=60000, n_warmup=10000, seed=0,
        start=np.asarray(geom["map"], float),
    )
    metric = np.linalg.pinv(ref["posterior_cov"])
    errs = []
    for div in (1, 4):
        out = run_constrained_sgld(
            model, ZeroField(model.d), ball, X, y, n_iter=20000 * div, n_walkers=16,
            step_size=h / div, batch_size=model.n, seed=2, start_radius=ball.radius,
            reference_mean=ref["posterior_mean"], reference_metric=metric,
        )
        errs.append(out["mean_error"][-1])
    # O(sqrt(h)) would predict a factor of two; anything clearly below one is
    # enough to rule out a biased invariant law
    assert errs[1] < 0.75 * errs[0], errs


# --------------------------------------------------------------------------
# The paper's own fields and constraint sets, with a lasso target sampled by
# anchored Langevin: nds/anchored_constrained.py.
# --------------------------------------------------------------------------
def test_paper_fields_satisfy_the_three_assumptions():
    """Skew, divergence free, and ``J n = 0`` on the respective boundary."""
    rng = np.random.default_rng(0)
    d = 9
    ball, lp = PaperBall(2.0, squared=True), SmoothedLpBall(2.4, 0.2, 4.0)
    fields = {
        "state": (ball_axial_field(d, [5.0, 5.0, 5.0]), ball),
        "sublevel": (sublevel_axial_field(d, [2.0, 7.0, 2.0], 2.4, 0.2), lp),
    }
    for key, (field, domain) in fields.items():
        x = rng.normal(size=d)
        J = field.matrix(x)
        assert np.abs(J + J.T).max() < 1e-12, key
        assert np.abs(_numeric_divergence(field, x)).max() < 1e-7, key
        assert np.abs(field.divergence(x[:, None])).max() == 0.0, key
        g = rng.normal(size=(d, 1))
        assert np.abs(field.apply(x[:, None], g) - J @ g).max() < 1e-12, key
        # J n = 0 on the boundary, which is what turns the oblique boundary
        # condition into the plain Neumann one
        X = rng.normal(size=(d, 32))
        X = domain._shrink(X * 3.0) if hasattr(domain, "_shrink") else X * (
            domain.radius / np.sqrt((X * X).sum(axis=0))
        )
        assert np.abs(field.apply(X, domain.normal(X))).max() < 1e-10, key

    # the constant field is the counterexample: J_a n is not zero
    J_a = ConstantField(d, 2.0)
    X = rng.normal(size=(d, 32))
    assert np.abs(J_a.apply(X, ball.normal(X))).max() > 0.1

    # the paper writes J_s(x) w = s (x cross w); check the sign convention
    f3 = ball_axial_field(3, 5.0)
    x3, w3 = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(f3.matrix(x3) @ w3, 5.0 * np.cross(x3, w3))


def test_skew_projection_returns_to_the_constraint_set():
    """Both domains, all three fields, at overshoots a Langevin step can make."""
    rng = np.random.default_rng(1)
    d = 9
    ball, lp = PaperBall(2.0, squared=True), SmoothedLpBall(2.4, 0.2, 4.0)
    cases = [
        (ball, ball_axial_field(d, [5.0, 5.0, 5.0])),
        (ball, ConstantField(d, 2.0)),
        (ball, ZeroField(d)),
        (lp, sublevel_axial_field(d, [2.0, 7.0, 2.0], 2.4, 0.2)),
        (lp, ConstantField(d, 2.0)),
        (lp, ZeroField(d)),
    ]
    for domain, field in cases:
        Z = rng.normal(size=(d, 128))
        inside = (
            domain._shrink(Z * 3.0) if hasattr(domain, "_shrink")
            else Z * (domain.radius / np.sqrt((Z * Z).sum(axis=0)))
        )
        for over in (1.002, 1.01):
            Y = inside * over
            out, mask, failed = domain.retract(Y, field)
            assert mask.all(), (domain.key, field.key, over)
            assert failed == 0, (domain.key, field.key, over, failed)
            assert domain.contains(out).all(), (domain.key, field.key, over)
            if field.key in ("state", "sublevel", "zero"):
                # J n = 0, so the skew projection is the plain one
                plain, _, _ = domain.retract(Y, ZeroField(d))
                assert np.abs(out - plain).max() < 1e-10, (domain.key, field.key)

    # gamma always points inward: n . (I + J) n = 1 for any skew J
    N = rng.normal(size=(d, 64))
    N /= np.sqrt((N * N).sum(axis=0))
    for field in (ConstantField(d, 2.0), ball_axial_field(d, [5.0, 5.0, 5.0])):
        gamma = N + field.apply(ball.radius * N, N)
        assert np.abs((N * gamma).sum(axis=0) - 1.0).max() < 1e-12, field.key


def test_lasso_anchor_majorises_and_its_clock_is_bounded():
    X, y = _toy(n=150, d=3, seed=8)
    target = ConstrainedLasso(X, y, lam=10.0, delta=0.1)
    rng = np.random.default_rng(2)
    W = rng.normal(size=(3, 64))
    U, U0 = target.potential(W), target.anchor_potential(W)
    assert (U0 >= U - 1e-10).all()
    clock = target.clock(W)
    assert np.allclose(clock, np.exp(U - U0))
    assert (clock <= 1.0 + 1e-12).all()
    assert (clock >= target.clock_floor - 1e-12).all()
    assert abs(target.clock_floor - np.exp(-10.0 * 3 * 0.1)) < 1e-12

    # the drift follows the anchor, so its gradient is the one that must be right
    idx = np.arange(target.n)
    G = target.anchor_grad(W[:, :4], idx)
    eps = 1e-6
    for j in range(3):
        plus, minus = W[:, :4].copy(), W[:, :4].copy()
        plus[j] += eps
        minus[j] -= eps
        fd = (target.anchor_potential(plus) - target.anchor_potential(minus)) / (2 * eps)
        assert np.abs(G[j] - fd).max() < 1e-3 * max(1.0, np.abs(fd).max())


def test_anchored_constrained_chains_hit_the_exact_lasso_reference():
    """The invariant law has to be the kinked, constrained posterior for every ``J``.

    The reference rejects proposals outside ``K`` and uses the exact ``|x|_1``,
    so it carries no discretisation, projection, clock or mini-batch bias.  The
    residual error is the projected scheme's, and it has to be the same for all
    three fields -- a field that changed the invariant law would stand out.
    """
    X, y = _toy(n=200, d=3, seed=4)
    data = {"X_train": X, "y_train": y, "X_test": X, "y_test": y}
    lam = 10.0
    target = ConstrainedLasso(X, y, lam=lam, delta=0.5 / lam)
    domain = PaperBall(0.5, squared=True)
    ref = constrained_lasso_reference(
        target, domain, data, n_iter=40000, n_warmup=10000, seed=0
    )
    assert 0.1 < ref["acceptance"] < 0.5
    metric = np.linalg.pinv(ref["posterior_cov"])
    errors = {}
    for field in (ZeroField(3), ConstantField(3, 1.0), ball_axial_field(3, 1.0)):
        out = run_anchored_srnsgld(
            target, field, domain, data, n_iter=20000, n_walkers=16, step_size=2e-4,
            batch_size=len(y), seed=2, start_radius=0.3,
            reference_mean=ref["posterior_mean"], reference_metric=metric,
        )
        assert out["failed_retractions"] == 0, field.key
        assert out["boundary_rate"] > 0.05, field.key  # the constraint is active
        assert 0.0 < out["clock"].mean() <= 1.0, field.key
        errors[field.key] = out["mean_error"]
        assert out["mean_error"] < 0.9, (field.key, out["mean_error"])
    assert max(errors.values()) - min(errors.values()) < 0.3, errors


def test_assumption_two_with_a_radial_h_freezes_the_radial_drift():
    r"""The degeneracy that makes the paper's ``J_s`` powerless on a ball.

    Its axial vector is ``k = s x``, so ``J_s(x) x = 0`` *everywhere*, not only
    on the boundary, and then ``x . (I + J) grad U = x . grad U`` identically:
    the field cannot change ``d|x|^2/dt`` at any point, for any amplitude.  The
    same degeneracy hits ``J_g`` exactly at ``p = 2``, where ``grad g`` is
    parallel to ``x``, and not for other ``p``.
    """
    rng = np.random.default_rng(3)
    d = 9
    X, G = rng.normal(size=(d, 128)), rng.normal(size=(d, 128))

    frozen = [
        ball_axial_field(d, [5.0, 5.0, 5.0]),
        sublevel_axial_field(d, [5.0, 5.0, 5.0], 2.0, 0.2),  # p = 2: same degeneracy
    ]
    for field in frozen:
        assert np.abs((X * field.apply(X, G)).sum(axis=0)).max() < 1e-10, field.key
        assert np.abs(field.apply(X, X)).max() < 1e-10, field.key

    free = [
        ConstantField(d, 2.0),
        sublevel_axial_field(d, [5.0, 5.0, 5.0], 2.4, 0.2),  # p != 2: not parallel to x
    ]
    for field in free:
        assert np.abs((X * field.apply(X, G)).sum(axis=0)).max() > 1.0, field.key

    # and J_g still leaves g itself alone, which is the assumption it is built for
    lp = SmoothedLpBall(2.4, 0.2, 4.0)
    Jg = sublevel_axial_field(d, [5.0, 5.0, 5.0], 2.4, 0.2)
    assert np.abs((lp.grad_g(X) * Jg.apply(X, G)).sum(axis=0)).max() < 1e-9


def test_tilted_field_keeps_the_assumptions_and_unfreezes_the_radius():
    """The paper's own recipe with a non-radial ``h``: same assumptions, free radius."""
    rng = np.random.default_rng(4)
    d = 9
    ball, lp = PaperBall(2.0, squared=True), SmoothedLpBall(2.4, 0.2, 4.0)
    u = rng.normal(size=d)
    for domain, p, eps in ((ball, 2.0, 0.0), (lp, 2.4, 0.2)):
        for tilt in (0.0, 2.0):
            field = tilted_axial_field(d, [1.0, 1.0, 1.0], domain, u, tilt, p=p, eps=eps)
            x = rng.normal(size=d)
            J = field.matrix(x)
            assert np.abs(J + J.T).max() < 1e-12, (domain.key, tilt)
            assert np.abs(_numeric_divergence(field, x)).max() < 1e-6, (domain.key, tilt)
            # Assumption 2 survives the tilt, because (level - g) vanishes there
            B = rng.normal(size=(d, 64))
            B = domain._shrink(B * 3.0) if hasattr(domain, "_shrink") else B * (
                domain.radius / np.sqrt((B * B).sum(axis=0))
            )
            assert np.abs(field.apply(B, domain.normal(B))).max() < 1e-9, (domain.key, tilt)
            # but inside, the tilt is exactly what frees the radial motion
            Y = 0.5 * B
            G = rng.normal(size=(d, 64))
            radial = np.abs((Y * field.apply(Y, G)).sum(axis=0)).max()
            if domain.key == "ball" and tilt == 0.0:
                assert radial < 1e-10
            elif tilt > 0.0:
                assert radial > 0.1, (domain.key, tilt, radial)


def test_outward_tilt_direction_beats_every_direction_tried():
    """The closed form is the maximiser, not a heuristic."""
    X, y = _toy(n=200, d=9, seed=11)
    target = ConstrainedLasso(X, y, lam=10.0, delta=0.05)
    domain = PaperBall(2.0, squared=True)
    u_star = outward_tilt_direction(target, domain, n_walkers=500, seed=0)
    assert abs(np.linalg.norm(u_star) - 1.0) < 1e-12

    rng = np.random.default_rng(0)
    W = domain.uniform(target.d, 500, rng, radius=1.0)
    G = target.anchor_grad(W, np.arange(target.n))
    c = np.concatenate([
        np.cross(G[3 * b : 3 * b + 3].T, W[3 * b : 3 * b + 3].T).mean(axis=0)
        for b in range(target.d // 3)
    ])

    def push(u):
        return -2.0 * float(np.dot(u / np.linalg.norm(u), c))

    best_random = max(push(rng.normal(size=target.d)) for _ in range(300))
    assert push(u_star) > best_random
    assert push(u_star) > 0.0 > push(-u_star)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
