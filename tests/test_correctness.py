"""Automated correctness tests for the constrained NR-Anchored-Langevin code.

The twelve checks of specification §12 are implemented here, one test function
each (plus a few supporting checks).  Run with ``pytest tests`` or directly as
``python tests/test_correctness.py``.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CHAIN_SEED_BASE,
    FD_SEED,
    ConstraintConfig,
    DataConfig,
    ProjectionConfig,
    SamplerConfig,
    TargetConfig,
)
from constraint import (  # noqa: E402
    boundary_point,
    g_value,
    grad_g,
    rescale_into_constraint,
    unit_normal,
)
from nonreversible_matrix import (  # noqa: E402
    construct_J,
    coordinate_triples,
    divergence_J,
    operator_norm_J,
    skew_symmetry_error,
    tangency_residual,
)
from projection import check_kkt, project_onto_K  # noqa: E402
from sampler import run_chain  # noqa: E402
from synthetic_data import generate_dataset  # noqa: E402
from target import LogisticTarget, WeightedL1Target  # noqa: E402

CONSTRAINT = ConstraintConfig()
P = CONSTRAINT.p_constraint
EPS = CONSTRAINT.epsilon_constraint
RNG = np.random.default_rng(FD_SEED)



def _executable_source(path: pathlib.Path) -> str:
    """Return the module source with every string literal and comment removed."""
    import io
    import tokenize

    pieces = []
    with open(path, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            pieces.append(token.string)
    return " ".join(pieces)


def _random_states(n: int = 12, scale: float = 1.5) -> np.ndarray:
    """Random probe states, including the origin and a near-boundary point."""
    rng = np.random.default_rng(FD_SEED + 1)
    states = [np.zeros(CONSTRAINT.d), np.full(CONSTRAINT.d, 1e-9)]
    states += [rng.normal(scale=scale, size=CONSTRAINT.d) for _ in range(n)]
    return np.asarray(states)


def _small_dataset_and_target():
    """A cheap logistic problem used by the sampler-level tests."""
    data_cfg = DataConfig(n_train=300, n_test=100)
    dataset = generate_dataset(data_cfg, CONSTRAINT)
    target = LogisticTarget.from_config(dataset.X_train, dataset.y_train, TargetConfig())
    return dataset, target


# ==========================================================================
# 1. g(0) = d * epsilon_constraint ** p_constraint
# ==========================================================================
def test_g_at_origin():
    value = float(g_value(np.zeros(CONSTRAINT.d), P, EPS))
    expected = CONSTRAINT.d * EPS ** P
    assert abs(value - expected) < 1e-14, (value, expected)
    assert abs(value - CONSTRAINT.g_at_origin) < 1e-14


# ==========================================================================
# 2. Lambda_constraint > g(0)
# ==========================================================================
def test_threshold_exceeds_origin():
    assert CONSTRAINT.Lambda_constraint > CONSTRAINT.g_at_origin
    # The gap is exactly radius_budget ** p_constraint by construction.
    gap = CONSTRAINT.Lambda_constraint - CONSTRAINT.g_at_origin
    assert abs(gap - CONSTRAINT.radius_budget ** P) < 1e-12
    # beta_true must sit strictly inside, with substantial slack.
    dataset = generate_dataset(DataConfig(n_train=50, n_test=10), CONSTRAINT)
    g_true = float(g_value(dataset.beta_true, P, EPS))
    assert g_true <= CONSTRAINT.interior_level + 1e-10
    assert CONSTRAINT.Lambda_constraint - g_true > 0.4 * gap


def test_rescaling_hits_the_midpoint_level():
    """If beta_raw is outside the safe region, bisection puts it on the midpoint."""
    big = np.full(CONSTRAINT.d, 3.0)          # g(big) far above the midpoint level
    assert float(g_value(big, P, EPS)) > CONSTRAINT.interior_level
    rescaled, t = rescale_into_constraint(big, CONSTRAINT)
    assert 0.0 < t < 1.0
    assert abs(float(g_value(rescaled, P, EPS)) - CONSTRAINT.interior_level) < 1e-10


# ==========================================================================
# 3. Every sampled state lies in K
# ==========================================================================
def test_every_state_is_feasible():
    _, target = _small_dataset_and_target()
    step_size = 0.2 / target.lipschitz_constant()
    # Deliberately tight constraint so that projections actually fire.
    tight = ConstraintConfig(d=9, p_constraint=1.5, epsilon_constraint=0.05, radius_budget=0.9)
    for alpha in (0.0, 0.5, 1.0):
        cfg = SamplerConfig(
            n_iterations=1500, burn_in=200, thin=5, step_size=step_size,
            alpha=alpha, seed=CHAIN_SEED_BASE,
        )
        chain = run_chain(target, np.zeros(CONSTRAINT.d), tight, cfg)
        values = g_value(chain.trace, P, EPS)
        assert values.max() <= tight.Lambda_constraint + 1e-8, values.max()
        # The tight radius must genuinely activate the projection.
        assert chain.diagnostics["projected"].mean() > 0.0


# ==========================================================================
# 4. J(w) is skew-symmetric
# ==========================================================================
def test_J_is_skew_symmetric():
    for w in _random_states():
        J = construct_J(w, P, EPS)
        assert skew_symmetry_error(J) == 0.0
        assert np.allclose(np.diag(J), 0.0)
    # A constant swirl multiplier preserves skew-symmetry.
    w = _random_states()[3]
    assert skew_symmetry_error(construct_J(w, P, EPS, swirl=2.5)) == 0.0


def test_J_block_structure_matches_specification():
    """Rows/columns (a,b,c) carry exactly [[0,-k3,k2],[k3,0,-k1],[-k2,k1,0]]."""
    w = _random_states()[4]
    J = construct_J(w, P, EPS)
    grad_psi_w = -grad_g(w, P, EPS)
    for (a, b, c) in coordinate_triples(CONSTRAINT.d):
        k1, k2, k3 = grad_psi_w[a], grad_psi_w[b], grad_psi_w[c]
        block = J[np.ix_([a, b, c], [a, b, c])]
        expected = np.array([[0.0, -k3, k2], [k3, 0.0, -k1], [-k2, k1, 0.0]])
        assert np.array_equal(block, expected)
    # Off-block entries are exactly zero (the triples are disjoint).
    mask = np.ones_like(J, dtype=bool)
    for (a, b, c) in coordinate_triples(CONSTRAINT.d):
        mask[np.ix_([a, b, c], [a, b, c])] = False
    assert np.all(J[mask] == 0.0)


# ==========================================================================
# 5. Finite-difference divergence of J is approximately zero
# ==========================================================================
def test_divergence_of_J_is_zero():
    worst = 0.0
    for w in _random_states(n=8):
        divergence = divergence_J(w, P, EPS, h=1e-5)
        worst = max(worst, float(np.abs(divergence).max()))
    assert worst < 1e-9, f"max |div J| = {worst}"


# ==========================================================================
# 6. J(w) @ normal(w) is approximately zero on boundary points
# ==========================================================================
def test_tangency_on_the_boundary():
    rng = np.random.default_rng(FD_SEED + 2)
    worst = 0.0
    for _ in range(10):
        direction = rng.normal(size=CONSTRAINT.d)
        w = boundary_point(direction, CONSTRAINT)
        assert abs(float(g_value(w, P, EPS)) - CONSTRAINT.Lambda_constraint) < 1e-9
        normal = unit_normal(w, P, EPS)
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-12
        worst = max(worst, tangency_residual(w, P, EPS))
    assert worst < 1e-12, f"max ||J n|| = {worst}"


def test_operator_norm_closed_form_matches_svd():
    for w in _random_states():
        J = construct_J(w, P, EPS, swirl=1.7)
        closed = operator_norm_J(w, P, EPS, swirl=1.7)
        assert abs(closed - float(np.linalg.norm(J, 2))) < 1e-10


# ==========================================================================
# 7. The anchor coefficient satisfies its theoretical bounds
# ==========================================================================
def test_anchor_bounds():
    _, target = _small_dataset_and_target()
    lower = -(target.d - 1) * target.lambda_lasso * target.delta_anchor
    assert abs(lower - (-8 * target.lambda_lasso * target.delta_anchor)) < 1e-12
    for w in _random_states(n=20, scale=2.0):
        log_a = target.log_a(w)
        assert lower - 1e-12 <= log_a <= 1e-12, log_a
        assert target.a_lower_bound - 1e-12 <= np.exp(log_a) <= 1.0 + 1e-12
        assert target.check_anchor_bounds(w)
    # The bound is attained at the origin (all slopes exactly zero).
    assert abs(target.log_a(np.zeros(target.d)) - lower) < 1e-12
    # log_a equals U - U0 exactly, and is NOT computed as exp(U)/exp(U0).
    for w in _random_states(n=5):
        assert abs(target.log_a(w) - (target.U(w) - target.U0(w))) < 1e-8

    # Same for the weighted-L1 validation target (all coordinates penalised).
    weighted = WeightedL1Target(omega=np.linspace(0.8, 2.4, 9), delta_anchor=0.05)
    for w in _random_states(n=5):
        assert weighted.check_anchor_bounds(w)
    assert abs(
        weighted.log_a(np.zeros(9)) - (-weighted.delta_anchor * weighted.omega.sum())
    ) < 1e-12


# ==========================================================================
# 8. The intercept is excluded from the L1 penalty
# ==========================================================================
def test_intercept_is_not_l1_penalised():
    _, target = _small_dataset_and_target()
    base = np.zeros(target.d)

    # Moving the intercept changes U only through the likelihood and the
    # Gaussian prior: the difference must match w0^2/(2 sigma^2) exactly.
    shifted = base.copy()
    shifted[0] = 1.3
    likelihood_change = target.negative_log_likelihood(shifted) - target.negative_log_likelihood(base)
    prior_change = shifted[0] ** 2 / (2.0 * target.sigma_intercept ** 2)
    assert abs((target.U(shifted) - target.U(base)) - (likelihood_change + prior_change)) < 1e-9
    assert abs((target.U0(shifted) - target.U0(base)) - (likelihood_change + prior_change)) < 1e-9

    # A slope of the same size DOES pick up the L1 term.
    slope = base.copy()
    slope[1] = 1.3
    slope_likelihood = target.negative_log_likelihood(slope) - target.negative_log_likelihood(base)
    penalty = target.U(slope) - target.U(base) - slope_likelihood
    assert abs(penalty - target.lambda_lasso * 1.3) < 1e-9

    # log_a ignores the intercept entirely.
    w = np.array([5.0] + [0.0] * (target.d - 1))
    assert abs(target.log_a(w) - target.log_a(np.zeros(target.d))) < 1e-14
    # The prior gradient at the intercept is w0 / sigma^2, with no lambda term.
    w = RNG.normal(size=target.d)
    assert abs(target.grad_prior(w)[0] - w[0] / target.sigma_intercept ** 2) < 1e-14


# ==========================================================================
# 9. The projection satisfies its KKT conditions
# ==========================================================================
def test_projection_kkt():
    rng = np.random.default_rng(FD_SEED + 3)
    n_projected = 0
    for _ in range(25):
        y = rng.normal(scale=3.0, size=CONSTRAINT.d)
        result = project_onto_K(y, CONSTRAINT)
        audit = check_kkt(result, y, CONSTRAINT)
        assert audit["satisfied"], audit
        assert result.g_value <= CONSTRAINT.Lambda_constraint + 1e-8
        if result.projected:
            n_projected += 1
            assert result.eta > 0.0
            assert abs(result.g_value - CONSTRAINT.Lambda_constraint) < 1e-8
            assert np.all(np.abs(result.z) <= np.abs(y) + 1e-12)
            assert np.all(np.sign(result.z) == np.sign(y))
    assert n_projected > 0, "the test never exercised an actual projection"


def test_projection_is_feasible_passthrough():
    interior = np.zeros(CONSTRAINT.d)
    result = project_onto_K(interior, CONSTRAINT)
    assert not result.projected
    assert result.distance == 0.0
    assert np.array_equal(result.z, interior)


def test_projection_matches_a_generic_optimiser():
    """Cross-check the KKT solver against SLSQP on the same QP."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(FD_SEED + 4)
    for _ in range(5):
        y = rng.normal(scale=3.0, size=CONSTRAINT.d)
        exact = project_onto_K(y, CONSTRAINT)
        if not exact.projected:
            continue
        reference = minimize(
            fun=lambda z: 0.5 * np.sum((z - y) ** 2),
            x0=y,
            jac=lambda z: z - y,
            constraints=[
                {
                    "type": "ineq",
                    "fun": lambda z: CONSTRAINT.Lambda_constraint - g_value(z, P, EPS),
                }
            ],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-14},
        )
        assert np.linalg.norm(exact.z - reference.x) < 1e-5
        # The exact solution can only be better than the numerical one.
        assert 0.5 * np.sum((exact.z - y) ** 2) <= reference.fun + 1e-9


def test_p_equals_two_shortcut_is_radial():
    """For p = 2 the projection is a radial rescaling onto a Euclidean ball."""
    ball = ConstraintConfig(d=9, p_constraint=2.0, epsilon_constraint=0.05, radius_budget=2.0)
    radius = np.sqrt(ball.Lambda_constraint - ball.d * ball.epsilon_constraint ** 2)
    rng = np.random.default_rng(FD_SEED + 5)
    y = rng.normal(scale=4.0, size=ball.d)
    result = project_onto_K(y, ball)
    assert result.projected
    assert abs(np.linalg.norm(result.z) - radius) < 1e-10
    # Radial: z is a positive multiple of y.
    ratios = result.z / y
    assert np.allclose(ratios, ratios[0])
    assert check_kkt(result, y, ball)["satisfied"]


def test_radial_scaling_is_not_the_projection_when_p_is_not_two():
    """Guards the specification's warning about naive radial scaling."""
    rng = np.random.default_rng(FD_SEED + 6)
    y = rng.normal(scale=3.0, size=CONSTRAINT.d)
    exact = project_onto_K(y, CONSTRAINT)
    assert exact.projected

    # Radial alternative: shrink y by the scalar t that puts g(t y) on the boundary.
    from scipy.optimize import brentq

    t = brentq(
        lambda s: float(g_value(s * y, P, EPS)) - CONSTRAINT.Lambda_constraint, 0.0, 1.0
    )
    radial = t * y
    assert abs(float(g_value(radial, P, EPS)) - CONSTRAINT.Lambda_constraint) < 1e-9
    # Both are feasible and on the boundary, but radial is strictly worse.
    assert np.sum((radial - y) ** 2) > np.sum((exact.z - y) ** 2) + 1e-9


# ==========================================================================
# 10. Results are reproducible for fixed seeds
# ==========================================================================
def test_reproducibility():
    _, target = _small_dataset_and_target()
    step_size = 0.2 / target.lipschitz_constant()
    w0 = np.zeros(CONSTRAINT.d)

    def run(seed: int):
        cfg = SamplerConfig(
            n_iterations=400, burn_in=100, thin=5, step_size=step_size,
            alpha=0.5, seed=seed,
        )
        return run_chain(target, w0, CONSTRAINT, cfg)

    first, second, other = run(123), run(123), run(124)
    assert np.array_equal(first.trace, second.trace)
    assert np.array_equal(first.samples, second.samples)
    assert not np.array_equal(first.trace, other.trace)

    # The data set is reproducible too.
    a = generate_dataset(DataConfig(n_train=100, n_test=50), CONSTRAINT)
    b = generate_dataset(DataConfig(n_train=100, n_test=50), CONSTRAINT)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_train, b.y_train)


# ==========================================================================
# 11. alpha = 0 removes only the non-reversible term
# ==========================================================================
def test_alpha_zero_is_the_reversible_baseline():
    _, target = _small_dataset_and_target()
    step_size = 0.2 / target.lipschitz_constant()
    w0 = np.zeros(CONSTRAINT.d)
    n_iterations, seed = 500, 4242

    cfg = SamplerConfig(
        n_iterations=n_iterations, burn_in=100, thin=5, step_size=step_size,
        alpha=0.0, seed=seed,
    )
    chain = run_chain(target, w0, CONSTRAINT, cfg)

    # (a) The recorded non-reversible drift is identically zero and the total
    #     drift equals the reversible drift.
    assert np.all(chain.diagnostics["nonreversible_drift_norm"] == 0.0)
    assert np.allclose(
        chain.diagnostics["drift_norm"], chain.diagnostics["reversible_drift_norm"],
        atol=0.0, rtol=0.0,
    )

    # (b) Independent reference implementation of plain projected anchored
    #     Langevin (no J anywhere).  It must reproduce the chain bit for bit.
    rng = np.random.default_rng(seed)
    w = project_onto_K(w0, CONSTRAINT, ProjectionConfig()).z
    reference_trace = np.empty((n_iterations, CONSTRAINT.d))
    for k in range(n_iterations):
        grad = target.grad_U0(w)
        a = np.exp(target.log_a(w))
        drift = -a * grad
        xi = rng.standard_normal(CONSTRAINT.d)
        proposal = w + step_size * drift + np.sqrt(2.0 * step_size * a) * xi
        reference_trace[k] = w
        w = project_onto_K(proposal, CONSTRAINT, ProjectionConfig()).z
    assert np.array_equal(chain.trace, reference_trace)

    # (c) The swirl multiplier is irrelevant when alpha = 0.
    cfg_swirl = SamplerConfig(
        n_iterations=n_iterations, burn_in=100, thin=5, step_size=step_size,
        alpha=0.0, swirl=7.3, seed=seed,
    )
    assert np.array_equal(run_chain(target, w0, CONSTRAINT, cfg_swirl).trace, chain.trace)


def test_only_the_product_alpha_times_swirl_matters():
    """J -> s J with alpha -> alpha/s leaves the dynamics unchanged."""
    _, target = _small_dataset_and_target()
    step_size = 0.2 / target.lipschitz_constant()
    w0 = np.zeros(CONSTRAINT.d)
    common = dict(n_iterations=300, burn_in=50, thin=5, step_size=step_size, seed=99)
    one = run_chain(target, w0, CONSTRAINT, SamplerConfig(alpha=0.5, swirl=1.0, **common))
    two = run_chain(target, w0, CONSTRAINT, SamplerConfig(alpha=0.25, swirl=2.0, **common))
    assert np.allclose(one.trace, two.trace, atol=1e-12, rtol=0.0)


# ==========================================================================
# 12. No state-dependent normalisation of J is performed
# ==========================================================================
def test_no_state_dependent_normalisation_of_J():
    # (a) Entries of J are the raw grad_psi components, un-normalised.
    for w in _random_states(n=6):
        J = construct_J(w, P, EPS)
        grad_psi_w = -grad_g(w, P, EPS)
        for (a, b, c) in coordinate_triples(CONSTRAINT.d):
            assert J[a, b] == -grad_psi_w[c]
            assert J[b, c] == -grad_psi_w[a]
            assert J[c, a] == -grad_psi_w[b]

    # (b) ||J(w)|| genuinely varies with w: it is not pinned to any constant.
    norms = [operator_norm_J(w, P, EPS) for w in _random_states(n=10, scale=2.0)]
    assert np.ptp(norms) > 1e-3, "||J|| looks state-independent, i.e. normalised"

    # (c) The swirl multiplier is a pure constant: J(w; s) == s * J(w; 1).
    w = _random_states()[5]
    assert np.allclose(construct_J(w, P, EPS, swirl=3.0), 3.0 * construct_J(w, P, EPS),
                       atol=0.0, rtol=1e-15)

    # (d) Source-level guard: no division of J by a norm anywhere in the
    #     executable code (docstrings and comments are stripped first, since
    #     they legitimately *discuss* why normalisation is forbidden).
    root = pathlib.Path(__file__).resolve().parents[1]
    pattern = re.compile(r"J\w*\s*/=|J\w*\s*/\s*[A-Za-z_(]|norm\s*\([^)]*\)\s*\*\s*J\w*")
    for name in ("nonreversible_matrix.py", "sampler.py"):
        code = _executable_source(root / name)
        assert not pattern.search(code), f"possible normalisation of J in {name}"


# ==========================================================================
# Supporting checks
# ==========================================================================
def test_gradients_match_finite_differences():
    _, target = _small_dataset_and_target()
    w = RNG.normal(scale=0.4, size=target.d)
    h = 1e-6
    numerical = np.zeros(target.d)
    for i in range(target.d):
        plus, minus = w.copy(), w.copy()
        plus[i] += h
        minus[i] -= h
        numerical[i] = (target.U0(plus) - target.U0(minus)) / (2.0 * h)
    analytic = target.grad_U0(w)
    assert np.max(np.abs(analytic - numerical)) / max(1.0, np.max(np.abs(analytic))) < 1e-6

    weighted = WeightedL1Target(omega=np.linspace(0.8, 2.4, 9), delta_anchor=0.05)
    x = RNG.normal(size=9)
    numerical = np.zeros(9)
    for i in range(9):
        plus, minus = x.copy(), x.copy()
        plus[i] += h
        minus[i] -= h
        numerical[i] = (weighted.U0(plus) - weighted.U0(minus)) / (2.0 * h)
    assert np.max(np.abs(weighted.grad_U0(x) - numerical)) < 1e-6


def test_softplus_is_numerically_stable():
    """np.logaddexp(0, z) must not overflow for large |z|."""
    _, target = _small_dataset_and_target()
    w = np.full(target.d, 50.0)
    assert np.isfinite(target.U(w))
    assert np.isfinite(target.U0(w))
    assert np.all(np.isfinite(target.grad_U0(w)))


def _main() -> int:
    """Run every test in this module without pytest."""
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS  {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
            except Exception as error:  # pragma: no cover
                failures += 1
                print(f"ERROR {name}: {type(error).__name__}: {error}")
    print(f"\n{'all tests passed' if failures == 0 else f'{failures} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
