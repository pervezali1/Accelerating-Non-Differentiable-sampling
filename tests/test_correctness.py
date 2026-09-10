"""Correctness tests for the skew-anchored Langevin implementation.

Run with ``python -m pytest tests -q`` (or ``python tests/test_correctness.py``).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import analysis, metrics, samplers, skew  # noqa: E402
from skewanchor.targets import anisotropic_student_t, isotropic_polynomial  # noqa: E402


# --------------------------------------------------------------- structure


def test_paper_section_64_target_matches():
    """iota = 2, d = 1 reproduces pi(x) propto (1+x^2)^{-2} exactly."""
    t = isotropic_polynomial(1, 2.0)
    assert np.isclose(t.nu, 3.0)
    assert np.isclose(t.beta, 1.0)          # the paper's beta = 1 < iota
    assert np.isclose(t.s, 1.0)
    x = np.array([[0.0], [1.0], [2.0]])
    assert np.allclose(t.q(x), [1.0, 2.0, 5.0])
    assert np.allclose(t.U(x), 2.0 * np.log([1.0, 2.0, 5.0]))
    assert np.allclose(t.cov(), [[1.0]])    # variance is exactly 1


def test_gradients_match_finite_differences():
    t = anisotropic_student_t(3, 6.0, 20.0, rotate=True)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3))
    h = 1e-6
    for name, f, g in [("U", t.U, t.grad_U), ("U0", t.U0, t.grad_U0)]:
        num = np.zeros_like(x)
        for i in range(3):
            e = np.zeros(3)
            e[i] = h
            num[:, i] = (f(x + e) - f(x - e)) / (2 * h)
        assert np.allclose(num, g(x), rtol=1e-5, atol=1e-7), name


def test_anchored_drift_identity():
    """b(x) = -grad U0(x) e^{(U-U0)(x)} must equal the closed form used in the step."""
    t = anisotropic_student_t(4, 7.0, 30.0, rotate=True)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((7, 4)) * 2
    lhs = -t.grad_U0(x) * t.anchor_scale(x)[:, None]
    assert np.allclose(lhs, t.anchored_drift(x))
    assert np.allclose(t.anchored_drift(x), t.skew_anchored_drift(x, np.zeros((4, 4))))
    assert np.allclose(t.sigma(x) ** 2, t.anchor_scale(x))


def test_skew_constructions_are_skew():
    t = anisotropic_student_t(6, 8.0, 25.0, rotate=True)
    rng = np.random.default_rng(2)
    for kind in ("cyclic", "pairwise", "random", "eigen", "optimal"):
        J = skew.build(kind, t, 1.5, rng)
        assert skew.is_skew(J), kind


def test_trace_invariance_bounds_the_gap():
    """Tr(J A) = 0 forces the eigenvalues of (I-J)A to sum to Tr(A)."""
    t = anisotropic_student_t(5, 6.0, 40.0)
    A = t.Sigma_inv
    rng = np.random.default_rng(3)
    for _ in range(5):
        J = skew.random_skew(5, 3.0, rng)
        assert np.isclose(np.trace(J @ A), 0.0, atol=1e-10)
        assert np.isclose(np.sum(np.linalg.eigvals((np.eye(5) - J) @ A).real), np.trace(A))
        assert skew.spectral_abscissa(J, A) <= skew.gap_upper_bound(A) + 1e-8


# ------------------------------------------------------------- dynamics


def test_time_change_equivalence_is_exact():
    """Paper Theorem 15, extended to the skew drift: pathwise identical."""
    t = anisotropic_student_t(3, 6.0, 20.0, rotate=True)
    J = skew.cyclic(3, 1.7)
    x0 = np.random.default_rng(4).standard_normal((64, 3))
    a = samplers.run("skew_anchored", t, x0, 0.004, 120, np.random.default_rng(9), J=J)
    b = samplers.run_time_changed(t, x0, 0.004, 120, np.random.default_rng(9), J=J)
    assert np.max(np.abs(a.final_state - b.final_state)) < 1e-12


def test_target_is_invariant_for_every_dynamics():
    """Start at stationarity; moments must not drift beyond discretisation bias."""
    t = anisotropic_student_t(3, 8.0, 10.0)
    J = skew.cyclic(3, 1.0)
    rng = np.random.default_rng(5)
    x0 = t.sample(120_000, rng)
    truth = np.diag(t.cov())
    for method, kw in [("anchored", {}), ("skew_anchored", {"J": J}),
                       ("ula", {}), ("skew_ula", {"J": J}), ("mala", {})]:
        r = samplers.run(method, t, x0, 0.001, 300, np.random.default_rng(6), **kw)
        assert r.diverged_at is None, method
        got = r.final_state.var(axis=0)
        assert np.all(np.abs(got - truth) / truth < 0.12), (method, got, truth)


def test_isotropic_target_is_a_negative_control():
    """For Sigma = I the skew drift only rotates: radii are identical in law.

    With the same driving noise rotated by the same matrix, the skew-anchored
    chain is an exact rotation of the anchored chain, so ||x|| agrees pathwise.
    """
    t = isotropic_polynomial(3, 3.0)
    A = t.Sigma_inv
    for kind in ("cyclic", "random", "eigen"):
        J = skew.build(kind, t, 2.0, np.random.default_rng(7))
        assert np.isclose(skew.spectral_abscissa(J, A), skew.spectral_abscissa(skew.zero(3), A))
    rng = np.random.default_rng(8)
    x0 = rng.standard_normal((4000, 3))
    J = skew.cyclic(3, 1.0)
    a = samplers.run("anchored", t, x0, 0.002, 400, np.random.default_rng(12))
    b = samplers.run("skew_anchored", t, x0, 0.002, 400, np.random.default_rng(13), J=J)
    # distributions of the radius must agree far inside Monte Carlo error
    ra, rb = np.sort(np.linalg.norm(a.final_state, axis=1)), np.sort(np.linalg.norm(b.final_state, axis=1))
    assert np.abs(np.median(ra) - np.median(rb)) < 0.05 * np.median(ra)


# ------------------------------------------------ exact second-moment theory


def test_second_moment_recursion_matches_simulation():
    t = anisotropic_student_t(2, 5.0, 100.0)
    J = skew.cyclic(2, 2.0)
    eta = 0.004
    pred = analysis.stationary_covariance(t, J, eta)
    assert pred is not None
    rng = np.random.default_rng(14)
    x0 = rng.standard_normal((150_000, 2)) * 3
    r = samplers.run("skew_anchored", t, x0, eta, 3000, np.random.default_rng(15), J=J)
    emp = (r.final_state.T @ r.final_state) / r.final_state.shape[0]
    assert np.allclose(emp, pred, rtol=0.05, atol=0.01), (emp, pred)


def test_continuous_stationary_covariance_is_J_free():
    """B C + C B^T = 2(1 + Tr(A C)/nu) I holds for every skew J with C = nu/(nu-2) Sigma."""
    t = anisotropic_student_t(4, 9.0, 30.0, rotate=True)
    C = t.cov()
    rng = np.random.default_rng(16)
    for delta in (0.0, 1.0, 5.0):
        J = skew.random_skew(4, delta, rng) if delta else None
        B = analysis.drift_matrix(t, J)
        lhs = B @ C + C @ B.T
        rhs = 2.0 * (1.0 + np.trace(t.Sigma_inv @ C) / t.nu) * np.eye(4)
        assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10), delta


def test_mean_square_stepsize_limit_is_stricter_than_drift_limit():
    t = anisotropic_student_t(2, 5.0, 100.0)
    for delta in (0.0, 1.0, 3.0):
        J = skew.cyclic(2, delta) if delta else None
        assert analysis.max_stable_stepsize(t, J) < analysis.deterministic_stepsize_limit(t, J)


def test_stepsize_above_the_limit_actually_blows_up():
    t = anisotropic_student_t(2, 5.0, 100.0)
    J = skew.cyclic(2, 1.0)
    eta_max = analysis.max_stable_stepsize(t, J)
    rng = np.random.default_rng(17)
    x0 = rng.standard_normal((2000, 2))
    below = samplers.run("skew_anchored", t, x0, 0.4 * eta_max, 4000, np.random.default_rng(18), J=J)
    above = samplers.run("skew_anchored", t, x0, 1.6 * eta_max, 4000, np.random.default_rng(19), J=J)
    assert np.max(np.abs(below.final_state)) < 1e3
    assert np.max(np.abs(above.final_state)) > 1e4 or above.diverged_at is not None


# ----------------------------------------------------------------- metrics


def test_w2_estimator_is_finite_and_scales():
    t = isotropic_polynomial(1, 2.0)     # nu = 3, the paper's target
    rng = np.random.default_rng(20)
    floors = [metrics.w2_reference_floor(t, n, rng, n_rep=8)[0] for n in (1000, 16000)]
    assert all(np.isfinite(floors))
    assert floors[1] < floors[0]         # the floor must fall with n


def test_w2_is_zero_only_for_the_right_law():
    t = anisotropic_student_t(2, 8.0, 4.0)
    rng = np.random.default_rng(21)
    good = metrics.axis_sliced_w2(t.sample(8000, rng), t)
    bad = metrics.axis_sliced_w2(rng.standard_normal((8000, 2)), t)
    assert bad > 5 * good


def test_midpoint_estimator_understates_heavy_tails():
    t = isotropic_polynomial(1, 2.0)
    rng = np.random.default_rng(22)
    x = t.sample(5000, rng)
    assert metrics.sliced_w2_midpoint(x, t) < metrics.axis_sliced_w2(x, t)


if __name__ == "__main__":
    ns = dict(globals())
    failed = 0
    for name, fn in sorted(ns.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print("all passed" if not failed else f"{failed} failures")
    sys.exit(1 if failed else 0)
