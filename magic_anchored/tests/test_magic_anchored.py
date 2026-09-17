#!/usr/bin/env python3
"""Unit tests for every public function, runnable with pytest or directly.

    python3 tests/test_magic_anchored.py      # prints one line per test
    python3 -m pytest tests/ -q               # if pytest is installed

The tests that need the real data file are skipped when it is absent, so the
mathematical tests -- which are the ones that could be silently wrong -- always
run.  Each identity is checked against an independent computation rather than
against itself: gradients against central differences, the embedded field
against explicit cross products, the clock against the difference of the two
potentials it is defined as, ESS against an AR(1) series of known correlation.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import geometry as G  # noqa: E402
import target as T  # noqa: E402
from config import (  # noqa: E402
    ConstraintSpec,
    DataConfig,
    FieldConfig,
    PilotConfig,
    SamplerConfig,
    TargetConfig,
)
from sampler import (  # noqa: E402
    calibrate_delta,
    calibrate_step_size,
    choose_radius_from_pilot,
    find_smooth_map,
    initial_point,
    run_constrained_sampler,
    run_unconstrained_pilot,
)

DATA_PATH = os.path.join(os.path.dirname(HERE), "..", "data", "magic04.data")


def _toy(n: int = 400, d: int = 10, seed: int = 0):
    """A small separable-ish logistic problem with an intercept column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X = np.column_stack([np.ones(n), X])
    w_true = np.zeros(d + 1)
    w_true[:4] = [0.3, 1.0, -0.7, 0.5]
    y = (rng.random(n) < T.sigmoid(X @ w_true)).astype(np.float64)
    return X, y


# ----------------------------------------------------------------- geometry
def test_cyclic_triples_match_the_specified_cover_at_d11():
    assert G.generate_cyclic_triples(11) == G.TRIPLES_D11


def test_cyclic_triples_cover_every_coordinate_for_many_dimensions():
    for D in range(3, 40):
        triples = G.generate_cyclic_triples(D)
        covered = {i for t in triples for i in t}
        assert covered == set(range(D)), D
        for t in triples:
            assert len(set(t)) == 3, (D, t)
            assert all(0 <= i < D for i in t), (D, t)
    for bad in (0, 1, 2, -3):
        try:
            G.generate_cyclic_triples(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"D={bad} should have been rejected")


def test_three_dimensional_block_is_the_hat_map():
    """At ``D = 3`` the construction must reduce to ``J v = s (q x v)`` exactly."""
    rng = np.random.default_rng(1)
    center = rng.normal(size=3)
    q = rng.normal(size=3)
    w = center + q
    s = 1.7
    J = G.construct_J(w, center, ((0, 1, 2),), s)
    expected = s * np.array(
        [[0.0, -q[2], q[1]], [q[2], 0.0, -q[0]], [-q[1], q[0], 0.0]]
    )
    assert np.allclose(J, expected), np.abs(J - expected).max()
    v = rng.normal(size=3)
    assert np.allclose(J @ v, s * np.cross(q, v))


def test_field_is_skew_tangential_and_divergence_free():
    for D in (3, 6, 9, 11, 14):
        triples = G.generate_cyclic_triples(D)
        center = np.random.default_rng(D).normal(size=D)
        worst = G.verify_field_properties(center, triples, s=1.3, n_points=8, seed=D)
        assert worst["skew"] < 1e-11, (D, worst)
        assert worst["tangency"] < 1e-9, (D, worst)
        assert worst["divergence"] < 1e-7, (D, worst)


def test_field_uses_the_one_over_sqrt_m_scaling_and_is_linear_in_s_and_q():
    """``J`` is ``s/sqrt(m)`` times the plain sum of blocks, and linear in ``q``.

    Checked against the unnormalised sum of blocks built here independently, so
    the factor is verified rather than assumed.  The linearity in ``q`` is the
    property that makes :func:`geometry.natural_alpha_scale` the right way to
    read ``alpha``: a posterior close to its own centre sees a small rotation
    however large ``alpha`` looks.
    """
    rng = np.random.default_rng(3)
    for D in (6, 11, 21):
        triples = G.generate_cyclic_triples(D)
        m = len(triples)
        center = rng.normal(size=D)
        q = rng.normal(size=D)
        s = 2.3
        J = G.construct_J(center + q, center, triples, s)

        plain = np.zeros((D, D))
        for i, j, k in triples:
            block = np.zeros((D, D))
            block[i, j], block[i, k] = -q[k], q[j]
            block[j, i], block[j, k] = q[k], -q[i]
            block[k, i], block[k, j] = -q[j], q[i]
            plain += block
        assert np.allclose(J, s / np.sqrt(m) * plain), D

        # linear in q and in s, both exactly
        assert np.allclose(
            G.construct_J(center + 3.0 * q, center, triples, s), 3.0 * J
        )
        assert np.allclose(
            G.construct_J(center + q, center, triples, 2.0 * s), 2.0 * J
        )
        # and zero at the centre, so the field switches itself off there
        assert np.allclose(G.construct_J(center, center, triples, s), 0.0)


def test_projection_lands_on_the_sphere_and_leaves_the_interior_alone():
    rng = np.random.default_rng(4)
    center = rng.normal(size=7)
    R = 0.9
    inside = center + 0.3 * rng.normal(size=7) / np.linalg.norm(rng.normal(size=7))
    p, hit = G.project_to_centered_ball(inside, center, R)
    assert not hit and np.allclose(p, inside)
    outside = center + 5.0 * rng.normal(size=7)
    p, hit = G.project_to_centered_ball(outside, center, R)
    assert hit
    assert abs(np.linalg.norm(p - center) - R) < 1e-12
    # the projection is radial: it must not rotate the offset
    q_in, q_out = outside - center, p - center
    cosine = q_in @ q_out / (np.linalg.norm(q_in) * np.linalg.norm(q_out))
    assert abs(cosine - 1.0) < 1e-12
    # and the centre itself is a fixed point
    p, hit = G.project_to_centered_ball(center, center, R)
    assert not hit and np.allclose(p, center)


def test_natural_alpha_scale_is_where_the_rotation_matches_the_gradient():
    R, m, s = 0.19216, 6, 1.0
    a_star = G.natural_alpha_scale(R, m, s)
    center = np.zeros(11)
    triples = G.generate_cyclic_triples(11)
    # a state on the boundary in a direction the blocks see
    rng = np.random.default_rng(5)
    q = rng.normal(size=11)
    q *= R / np.linalg.norm(q)
    norm = G.field_operator_norm(center + q, center, triples, s)
    assert 0.2 < a_star * norm < 5.0, (a_star, norm)


# ------------------------------------------------------------------- target
def test_potentials_differ_only_by_the_penalty_and_the_anchor_majorises():
    X, y = _toy()
    lam, sigma0, delta = 3.0, 10.0, 0.02
    rng = np.random.default_rng(6)
    for _ in range(20):
        w = rng.normal(scale=0.8, size=X.shape[1])
        U = T.target_potential(w, X, y, lam, sigma0)
        U0 = T.anchor_potential(w, X, y, lam, sigma0, delta)
        assert U0 >= U - 1e-9, (U, U0)
        # the likelihood and the intercept prior cancel in the difference
        analytic = T.log_anchor_coefficient(w, lam, delta)
        assert abs(analytic - (U - U0)) < 1e-8 * max(1.0, abs(U))
        T.check_anchor_bounds(np.array([analytic]), X.shape[1], lam, delta)


def test_anchor_gradient_matches_central_differences():
    X, y = _toy(n=200, d=6, seed=7)
    lam, sigma0, delta = 2.5, 5.0, 0.03
    rng = np.random.default_rng(8)
    for _ in range(5):
        w = rng.normal(scale=0.6, size=X.shape[1])
        g = T.grad_anchor(w, X, y, lam, sigma0, delta)
        assert g.shape == (X.shape[1],)
        eps = 1e-6
        fd = np.empty_like(g)
        for j in range(w.size):
            step = np.zeros_like(w)
            step[j] = eps
            fd[j] = (
                T.anchor_potential(w + step, X, y, lam, sigma0, delta)
                - T.anchor_potential(w - step, X, y, lam, sigma0, delta)
            ) / (2 * eps)
        assert np.abs(g - fd).max() < 1e-5 * max(1.0, np.abs(fd).max())


def test_intercept_is_excluded_from_the_l1_penalty():
    """Moving ``beta_0`` must not change either penalty, or the clock."""
    X, y = _toy(n=120, d=5, seed=9)
    lam, sigma0, delta = 4.0, 3.0, 0.05
    rng = np.random.default_rng(10)
    w = rng.normal(size=X.shape[1])
    shifted = w.copy()
    shifted[0] += 2.75
    # the clock depends on beta_1..beta_p only
    assert T.log_anchor_coefficient(w, lam, delta) == T.log_anchor_coefficient(
        shifted, lam, delta
    )
    # and the penalty part of the gradient has a zero intercept entry: the whole
    # of grad[0] must be the likelihood term plus beta_0 / sigma0^2
    g = T.grad_anchor(w, X, y, lam, sigma0, delta)
    expected0 = T.grad_target_likelihood(w, X, y)[0] + w[0] / sigma0**2
    assert abs(g[0] - expected0) < 1e-12
    # doubling lambda must leave grad[0] untouched and change every other entry
    g2 = T.grad_anchor(w, X, y, 2 * lam, sigma0, delta)
    assert abs(g2[0] - g[0]) < 1e-12
    assert np.abs(g2[1:] - g[1:]).min() > 0


def test_clock_bounds_hold_at_extreme_states():
    """The bound must hold where it is tight: all coefficients exactly at zero."""
    lam, delta, D = 7.0, 0.04, 11
    p = D - 1
    log_a_min, a_min = T.anchor_bounds(D, lam, delta)
    assert np.isclose(log_a_min, -lam * p * delta)
    at_kink = np.zeros(D)
    assert np.isclose(T.log_anchor_coefficient(at_kink, lam, delta), log_a_min)
    assert np.isclose(T.anchor_coefficient(at_kink, lam, delta), a_min)
    # far from the kink the gap per term is delta^2 / (2|beta|), so the clock is
    # close to 1 but not equal to it -- the exact value is worth asserting
    far = np.full(D, 50.0)
    expected = np.exp(lam * p * (50.0 - np.sqrt(2500.0 + delta**2)))
    assert np.isclose(T.anchor_coefficient(far, lam, delta), expected)
    assert 1.0 - 1e-2 < expected < 1.0
    T.check_anchor_bounds(
        np.array([T.log_anchor_coefficient(at_kink, lam, delta),
                  T.log_anchor_coefficient(far, lam, delta)]), D, lam, delta
    )
    # a violated bound must be caught, not ignored
    try:
        T.check_anchor_bounds(np.array([1.0]), D, lam, delta)
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a positive log a should have been rejected")


def test_tempered_option_divides_the_likelihood_and_is_off_by_default():
    X, y = _toy(n=300, d=4, seed=11)
    lam, sigma0 = 1.0, 10.0
    w = np.full(X.shape[1], 0.2)
    plain = T.target_potential(w, X, y, lam, sigma0)
    tempered = T.target_potential(w, X, y, lam, sigma0, tempered=True)
    penalty = lam * np.abs(w[1:]).sum() + 0.5 * w[0] ** 2 / sigma0**2
    assert np.isclose(
        (plain - penalty) / X.shape[0], tempered - penalty, rtol=1e-12
    )
    assert TargetConfig().tempered is False


def test_hessian_matches_finite_differences_of_the_gradient():
    X, y = _toy(n=150, d=5, seed=12)
    lam, sigma0, delta = 2.0, 4.0, 0.06
    w = np.random.default_rng(13).normal(scale=0.4, size=X.shape[1])
    H = T.anchor_hessian(w, X, y, lam, sigma0, delta)
    eps = 1e-6
    fd = np.empty_like(H)
    for j in range(w.size):
        step = np.zeros_like(w)
        step[j] = eps
        fd[:, j] = (
            T.grad_anchor(w + step, X, y, lam, sigma0, delta)
            - T.grad_anchor(w - step, X, y, lam, sigma0, delta)
        ) / (2 * eps)
    assert np.abs(H - fd).max() < 1e-4 * np.abs(H).max()
    assert np.allclose(H, H.T)


# ------------------------------------------------------------------ sampler
def test_map_is_a_stationary_point_of_the_anchor():
    X, y = _toy(n=500, d=8, seed=14)
    tgt = T.LogisticTarget(X, y, 5.0, 10.0, 0.02)
    w_map, res = find_smooth_map(tgt)
    assert w_map.shape == (X.shape[1],)
    assert np.linalg.norm(tgt.grad_U0(w_map)) < 1e-3
    # and it is a minimum, not just a stationary point
    rng = np.random.default_rng(15)
    for _ in range(10):
        perturbed = w_map + 1e-3 * rng.normal(size=w_map.size)
        assert tgt.U0(perturbed) >= tgt.U0(w_map) - 1e-9


def test_step_size_calibration_scales_like_one_over_the_curvature():
    X, y = _toy(n=300, d=6, seed=16)
    tgt = T.LogisticTarget(X, y, 3.0, 10.0, 0.02)
    w_map, _ = find_smooth_map(tgt)
    h1, L1 = calibrate_step_size(tgt, w_map, 0.25)
    h2, L2 = calibrate_step_size(tgt, w_map, 0.5)
    assert np.isclose(L1, L2)
    assert np.isclose(h2, 2 * h1)
    assert np.isclose(h1 * L1, 0.25)


def test_delta_calibration_finds_an_interior_optimum():
    X, y = _toy(n=400, d=8, seed=17)
    best, rows = calibrate_delta(X, y, 20.0, 10.0,
                                 candidates=np.geomspace(1e-4, 1e-1, 9))
    assert any(np.isclose(r["delta"], best) for r in rows)
    steps = [r["effective_step"] for r in rows]
    assert max(steps) == max(r["effective_step"] for r in rows if r["delta"] == best)
    # the optimum is interior: both ends are worse, which is the trade-off
    assert steps[0] < max(steps) and steps[-1] < max(steps)


def test_initial_point_is_inside_and_independent_of_alpha():
    center = np.arange(11, dtype=float)
    R = 0.5
    for seed in (0, 1, 7):
        w0 = initial_point(center, R, 1e-2, seed)
        assert np.linalg.norm(w0 - center) <= R + 1e-12
        # the same seed gives the same start, whatever alpha will be used
        assert np.array_equal(w0, initial_point(center, R, 1e-2, seed))
    assert not np.array_equal(
        initial_point(center, R, 1e-2, 0), initial_point(center, R, 1e-2, 1)
    )
    # a huge perturbation still lands inside, via the projection
    w0 = initial_point(center, R, 1e3, 0)
    assert np.linalg.norm(w0 - center) <= R + 1e-12


def test_constrained_chain_stays_inside_and_records_everything():
    X, y = _toy(n=400, d=10, seed=18)
    tgt = T.LogisticTarget(X, y, 8.0, 10.0, 0.02)
    w_map, _ = find_smooth_map(tgt)
    h, _ = calibrate_step_size(tgt, w_map, 0.25)
    spec = ConstraintSpec(center=tuple(w_map), radius=0.25, source="test")
    n_iter = 600
    for alpha in (0.0, 25.0):
        cfg = SamplerConfig(alpha=alpha, h=h, n_iter=n_iter, burn_in=100,
                            thin=5, seed=3)
        r = run_constrained_sampler(tgt, spec, cfg, FieldConfig(s=1.0))
        assert r.trajectory.shape == (n_iter + 1, X.shape[1])
        assert r.radius_trace.max() <= spec.radius + 1e-9
        assert r.samples.shape[0] == cfg.n_retained
        assert np.isfinite(r.trajectory).all()
        assert r.log_a.shape == r.a.shape == (n_iter,)
        assert (r.a > 0).all() and (r.a <= 1 + 1e-12).all()
        assert r.n_grad_evaluations == n_iter
        assert r.drift_norm.shape == (n_iter,)
        if alpha == 0.0:
            assert np.all(r.nonreversible_drift_norm == 0.0)
        else:
            assert r.nonreversible_drift_norm.max() > 0.0
        # the split is consistent: ||b|| <= ||rev|| + ||nonrev||
        assert np.all(
            r.drift_norm <= r.reversible_drift_norm + r.nonreversible_drift_norm + 1e-9
        )


def test_alpha_zero_and_nonzero_share_the_same_start():
    """Validation check 8 in miniature: the comparison must be paired."""
    X, y = _toy(n=200, d=5, seed=19)
    tgt = T.LogisticTarget(X, y, 4.0, 10.0, 0.02)
    w_map, _ = find_smooth_map(tgt)
    spec = ConstraintSpec(center=tuple(w_map), radius=0.3, source="test")
    h, _ = calibrate_step_size(tgt, w_map, 0.25)
    runs = [
        run_constrained_sampler(
            tgt, spec,
            SamplerConfig(alpha=a, h=h, n_iter=50, burn_in=0, thin=1, seed=11),
        )
        for a in (0.0, 5.0)
    ]
    assert np.array_equal(runs[0].trajectory[0], runs[1].trajectory[0])
    assert np.array_equal(runs[0].center, runs[1].center)
    assert runs[0].radius == runs[1].radius


def test_pilot_is_unconstrained_and_sets_a_radius_that_contains_it():
    X, y = _toy(n=400, d=10, seed=20)
    tgt = T.LogisticTarget(X, y, 8.0, 10.0, 0.02)
    w_map, _ = find_smooth_map(tgt)
    h, _ = calibrate_step_size(tgt, w_map, 0.25)
    pcfg = PilotConfig(n_iter=2000, burn_in=500, quantile=0.999, inflation=1.10)
    pilot = run_unconstrained_pilot(tgt, w_map, h, pcfg)
    assert pilot.alpha == 0.0
    assert not pilot.projected_flag
    assert not pilot.projection_occurred.any()
    spec = choose_radius_from_pilot(pilot, w_map, pcfg)
    r = np.linalg.norm(pilot.trajectory[pcfg.burn_in :] - w_map, axis=1)
    assert np.isclose(spec.radius, 1.10 * np.quantile(r, 0.999))
    assert spec.pilot_exceedance_fraction <= 1e-3
    assert spec.dim == X.shape[1]


# -------------------------------------------------------------- diagnostics
def test_ess_and_rhat_behave_on_series_with_known_structure():
    import diagnostics as Dg

    rng = np.random.default_rng(21)
    n, rho = 4000, 0.8
    chains = np.empty((4, n, 2))
    for c in range(4):
        v = 0.0
        for t in range(n):
            v = rho * v + rng.normal() * np.sqrt(1 - rho**2)
            chains[c, t] = [v, rng.normal()]
    ess = Dg.effective_sample_size(chains)
    rhat = Dg.split_r_hat(chains)
    # the correlated coordinate has far fewer effective draws than the iid one
    assert ess[0] < 0.5 * ess[1]
    # and the iid coordinate is close to the nominal count
    assert 0.5 * 4 * n < ess[1] <= 1.3 * 4 * n
    assert np.all(rhat < 1.05)
    # offsetting one chain must be detected
    broken = chains.copy()
    broken[0, :, 0] += 12.0
    assert Dg.split_r_hat(broken)[0] > 1.5


def test_posterior_predictive_averages_probabilities_not_parameters():
    import diagnostics as Dg

    rng = np.random.default_rng(22)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    samples = rng.normal(scale=1.5, size=(500, 4))
    p_bar = Dg.posterior_predictive_probabilities(samples, X)
    direct = T.sigmoid(X @ samples.T).mean(axis=1)
    assert np.allclose(p_bar, direct)
    # batching must not change the answer
    assert np.allclose(
        p_bar, Dg.posterior_predictive_probabilities(samples, X, batch_size=7)
    )
    # and it is NOT the plug-in at the mean, because sigmoid is not affine
    plug_in = T.sigmoid(X @ samples.mean(axis=0))
    assert np.abs(p_bar - plug_in).max() > 1e-3
    assert np.all((p_bar > 0) & (p_bar < 1))


def test_predictive_metrics_are_computed_on_the_right_convention():
    import diagnostics as Dg

    y = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=float)
    p = np.array([0.1, 0.4, 0.9, 0.8, 0.6, 0.2, 0.55, 0.45])
    sc = Dg.evaluate_predictive_performance(y, p, alpha=1.0, n_samples=10)
    assert sc.accuracy == 1.0
    assert sc.sensitivity == 1.0 and sc.specificity == 1.0
    # rows are truth, columns prediction
    assert sc.confusion.tolist() == [[4, 0], [0, 4]]
    assert sc.roc_auc == 1.0
    # a perfectly wrong predictor
    bad = Dg.evaluate_predictive_performance(y, 1.0 - p)
    assert bad.accuracy == 0.0 and bad.roc_auc == 0.0
    # calibration bins sum to the sample size
    assert sc.calibration_counts.sum() == y.size


def test_chain_diagnostics_warn_when_the_chains_are_bad():
    import warnings

    import diagnostics as Dg

    X, y = _toy(n=200, d=5, seed=23)
    tgt = T.LogisticTarget(X, y, 4.0, 10.0, 0.02)
    w_map, _ = find_smooth_map(tgt)
    h, _ = calibrate_step_size(tgt, w_map, 0.25)
    # a radius so tight that the projection is active almost every step
    spec = ConstraintSpec(center=tuple(w_map), radius=1e-4, source="test")
    group = [
        run_constrained_sampler(
            tgt, spec,
            SamplerConfig(alpha=0.0, h=h, n_iter=400, burn_in=100, thin=2, seed=s),
        )
        for s in (1, 2)
    ]
    names = ["intercept"] + [f"x{j}" for j in range(X.shape[1] - 1)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = Dg.calculate_chain_diagnostics(group, names)
    assert d.projection_frequency > 0.10
    assert any("projection frequency" in str(c.message) for c in caught)
    assert d.warnings_raised


# -------------------------------------------------------- data (needs the file)
def test_data_loading_and_preprocessing_if_the_file_is_present():
    if not os.path.exists(DATA_PATH):
        print("  (skipped: magic04.data not found)", end="")
        return
    from data import load_magic_data, preprocess_data

    cfg = DataConfig(path=DATA_PATH)
    X_raw, y, names = load_magic_data(cfg=cfg)
    assert X_raw.shape == (19020, 10)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert y.sum() == 12332  # the 'g' rows
    ds = preprocess_data(X_raw, y, names, cfg)
    assert ds.dim == 11
    assert ds.X_train.dtype == np.float64
    assert np.allclose(ds.X_train[:, 0], 1.0)
    assert np.allclose(ds.X_test[:, 0], 1.0)
    # the scaler saw only the training half
    assert np.abs(ds.X_train[:, 1:].mean(axis=0)).max() < 1e-12
    assert np.abs(ds.X_train[:, 1:].std(axis=0) - 1.0).max() < 1e-12
    assert np.abs(ds.X_test[:, 1:].mean(axis=0)).max() > 1e-12  # would be 0 if leaked
    # stratification held the class balance
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 5e-3
    assert ds.n_train + ds.n_test == 19020
    assert ds.param_names[0] == "intercept" and len(ds.param_names) == 11


def main() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
