"""Tests for state-dependent skew fields and the rotation-aware integrators."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skewanchor import samplers, skew, skewfield as sf  # noqa: E402
from skewanchor.targets import anisotropic_student_t  # noqa: E402


def _div_pi_c(target, drift, x, h=1e-4):
    """``div(pi c)/pi`` by central differences -- zero iff the target is preserved."""
    n, d = x.shape
    out = np.zeros(n)
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        pip, pim = np.exp(-target.U(x + e)), np.exp(-target.U(x - e))
        out += (pip * drift(x + e)[:, i] - pim * drift(x - e)[:, i]) / (2 * h)
    return out / np.exp(-target.U(x))


def test_naive_condition_holds_for_radial_and_fails_off_it():
    t = anisotropic_student_t(2, 5.0, 100.0)
    J0 = skew.lnp_optimal(t.Sigma_inv)
    x = t.sample(300, np.random.default_rng(0))
    for kind in sf.PROFILES:
        fld = sf.RadialModulated(J0, t, sf.make_profile(kind, 1.0))
        assert np.abs(sf.invariance_residual(fld, t, x)).max() < 1e-5, kind

    class Coordwise(sf.SkewField):          # psi = x_1^2: violates the condition
        def apply(self, x, v):
            x = np.atleast_2d(x)
            return (x[:, 0] ** 2)[:, None] * (np.atleast_2d(v) @ J0.T)

    assert np.abs(sf.invariance_residual(Coordwise(), t, x)).max() > 1.0


def test_correction_makes_every_skew_field_admissible():
    """The whole point of (COR): no condition on J once div J is subtracted."""
    t = anisotropic_student_t(2, 5.0, 100.0)
    J0 = skew.lnp_optimal(t.Sigma_inv)
    x = t.sample(200, np.random.default_rng(1))

    class Coordwise(sf.SkewField):
        def apply(self, x, v):
            x = np.atleast_2d(x)
            return (np.exp(-((x[:, 0] / 0.2) ** 2)))[:, None] * (np.atleast_2d(v) @ J0.T)

        def matrix_at(self, x):
            x = np.atleast_2d(x)
            return np.exp(-((x[:, 0] / 0.2) ** 2))[:, None, None] * J0[None]

    fld = Coordwise()

    def uncorrected(y):
        return t.anchor_scale(y)[:, None] * fld.apply(y, t.grad_U0(y))

    def corrected(y):
        return t.anchor_scale(y)[:, None] * (fld.apply(y, t.grad_U0(y)) - fld.divergence(y))

    bad = np.abs(_div_pi_c(t, uncorrected, x)).max()
    good = np.abs(_div_pi_c(t, corrected, x)).max()
    assert bad > 1.0, bad
    assert good < 1e-3 * bad, (good, bad)


def test_stream_field_reproduces_the_constant_field():
    t = anisotropic_student_t(2, 5.0, 100.0)
    x = np.random.default_rng(2).standard_normal((8, 2))
    delta = 4.95
    got = sf.StreamField2D.quadrupole(t, delta, 0.0, 0.0).drift(x)
    J = delta * np.array([[0.0, 1.0], [-1.0, 0.0]])
    want = t.anchor_scale(x)[:, None] * (t.grad_U0(x) @ J.T)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_stream_field_preserves_the_target_for_a_tilted_member():
    t = anisotropic_student_t(2, 5.0, 100.0)
    x = t.sample(200, np.random.default_rng(3))
    for a, b in [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.3)]:
        fld = sf.StreamField2D.quadrupole(t, 4.95, a, b)
        res = np.abs(_div_pi_c(t, fld.drift, x)).max()
        scale = np.abs(fld.drift(x)).max()
        assert res < 1e-4 * scale, (a, b, res, scale)


def test_curl_field_is_divergence_free_in_three_dimensions():
    t = anisotropic_student_t(3, 6.0, 100.0)
    G = np.diag([1.0, 2.0, 3.0])
    fld = sf.CurlSkew(lambda x: x @ G)
    x = t.sample(200, np.random.default_rng(4))
    assert np.abs(sf.invariance_residual(fld, t, x)).max() < 1e-6
    M = fld.matrix_at(x[:10])
    assert np.abs(M + np.transpose(M, (0, 2, 1))).max() < 1e-12


def test_field_step_matches_the_constant_path_exactly():
    t = anisotropic_student_t(2, 5.0, 100.0)
    J0 = skew.lnp_optimal(t.Sigma_inv)
    x0 = np.random.default_rng(5).standard_normal((300, 2))
    a = samplers.simulate(samplers.skew_anchored_step(t, 1e-3, J0), x0, 40,
                          np.random.default_rng(6))
    b = samplers.simulate(samplers.field_anchored_step(t, 1e-3, sf.ConstantSkew(J0), "euler"),
                          x0, 40, np.random.default_rng(6))
    assert np.abs(a.final_state - b.final_state).max() < 1e-12


def test_integrators_agree_as_the_stepsize_vanishes():
    """Euler, Cayley and expm are all consistent; they differ only at O(eta)."""
    t = anisotropic_student_t(2, 5.0, 100.0)
    J = skew.lnp_optimal(t.Sigma_inv)
    x0 = np.random.default_rng(7).standard_normal((400, 2))
    errs = []
    for eta in (4e-5, 1e-5):
        outs = [samplers.simulate(samplers.field_anchored_step(t, eta, sf.ConstantSkew(J), k),
                                  x0, int(round(4e-4 / eta)), np.random.default_rng(8)).final_state
                for k in ("euler", "cayley", "expm")]
        errs.append(max(np.abs(outs[0] - outs[1]).max(), np.abs(outs[1] - outs[2]).max()))
    assert errs[1] < 0.6 * errs[0], errs


def test_rotation_aware_integrators_preserve_the_target():
    t = anisotropic_student_t(2, 5.0, 100.0)
    J = 4.0 * skew.lnp_optimal(t.Sigma_inv)
    rng = np.random.default_rng(9)
    x0 = t.sample(120_000, rng)
    truth = np.diag(t.cov())
    for integ in ("cayley", "expm"):
        step = samplers.field_anchored_step(t, 3e-4, sf.ConstantSkew(J), integ)
        res = samplers.simulate(step, x0, 400, np.random.default_rng(10))
        assert res.diverged_at is None, integ
        got = res.final_state.var(axis=0)
        assert np.all(np.abs(got - truth) / truth < 0.15), (integ, got, truth)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
