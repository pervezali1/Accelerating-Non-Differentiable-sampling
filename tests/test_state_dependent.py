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


# --------------------------------------------------------------- warm-up ramp


def test_warmup_schedule_shape_and_length():
    t = anisotropic_student_t(2, 5.0, 100.0)
    eta = 3e-4
    c = 2 * t.beta / t.nu
    k_relax = 1.0 / (eta * c * (1.0 / t.Sigma_evals.min()))
    for shape in ("linear", "smooth", "exponential"):
        sch = samplers.warmup_schedule(t, eta, 3.0, shape=shape)
        assert sch(0) < 1e-9, shape                       # starts off
        assert sch(10 * k_relax) > 0.99, shape            # ends on
        vals = [sch(k) for k in range(0, int(4 * k_relax), 5)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), shape  # monotone
        assert all(0.0 <= v <= 1.0 + 1e-12 for v in vals), shape


def test_warmup_schedule_cap():
    """``max_iters`` shortens the ramp and never lengthens it."""
    t = anisotropic_student_t(2, 5.0, 100.0)
    eta = 3e-4
    c = 2 * t.beta / t.nu
    k_relax = 1.0 / (eta * c * (1.0 / t.Sigma_evals.min()))
    uncapped = samplers.warmup_schedule(t, eta, 10.0)
    capped = samplers.warmup_schedule(t, eta, 10.0, max_iters=k_relax)
    assert uncapped(k_relax) < 0.5                      # a tenth of the way in
    assert capped(k_relax) > 0.999                      # the cap ends it here
    # a cap longer than the ramp changes nothing
    loose = samplers.warmup_schedule(t, eta, 3.0, max_iters=100 * k_relax)
    plain = samplers.warmup_schedule(t, eta, 3.0)
    assert max(abs(loose(k) - plain(k)) for k in range(0, int(5 * k_relax), 7)) < 1e-12


def test_schedule_matches_the_unramped_step_once_warm():
    """After the ramp completes the two dynamics are the same map."""
    t = anisotropic_student_t(2, 5.0, 100.0)
    J = 4.95 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    eta = 3e-4
    sch = samplers.warmup_schedule(t, eta, 3.0)
    warm = samplers.field_anchored_step(t, eta, sf.ConstantSkew(J), "euler", schedule=sch)
    plain = samplers.field_anchored_step(t, eta, sf.ConstantSkew(J), "euler")
    x = np.random.default_rng(0).standard_normal((64, 2))
    burn = np.random.default_rng(1)
    for _ in range(400):                                   # run the ramp out
        warm(x, burn)
    a = warm(x, np.random.default_rng(2))
    b = plain(x, np.random.default_rng(2))
    assert np.abs(a - b).max() < 1e-12


def test_warmup_preserves_the_target():
    t = anisotropic_student_t(2, 5.0, 100.0)
    J = 4.95 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    eta = 3e-4
    rng = np.random.default_rng(3)
    x0 = t.sample(80_000, rng)
    step = samplers.field_anchored_step(t, eta, sf.ConstantSkew(J), "euler",
                                        schedule=samplers.warmup_schedule(t, eta, 3.0))
    res = samplers.simulate(step, x0, 600, np.random.default_rng(4))
    assert res.diverged_at is None
    got, truth = res.final_state.var(axis=0), np.diag(t.cov())
    assert np.all(np.abs(got - truth) / truth < 0.15), (got, truth)


def test_warmup_removes_the_transient_hump():
    """The hump is the rotation flinging the prior's stiff-direction excess
    along the soft axis; warming up removes it without slowing convergence."""
    from skewanchor import metrics

    t = anisotropic_student_t(2, 5.0, 100.0)
    J = 4.95 * np.array([[0.0, 1.0], [-1.0, 0.0]])
    eta = 3e-4
    rec = [1, 20, 40, 60, 80, 110, 150, 220, 320, 500, 800, 1400]

    def curve(schedule):
        x = np.random.default_rng(5).standard_normal((3000, 2)) * np.sqrt(10.0)
        step = samplers.field_anchored_step(t, eta, sf.ConstantSkew(J), "euler",
                                            schedule=schedule)
        rr = np.random.default_rng(9)
        out, k = [], 0
        for tk in rec:
            while k < tk:
                x = step(x, rr)
                k += 1
            out.append(metrics.axis_sliced_w2(x, t))
        return np.array(out)

    plain = curve(None)
    warm = curve(samplers.warmup_schedule(t, eta, 5.0))
    assert plain.max() / plain[0] > 2.5, plain.max() / plain[0]
    assert warm.max() / warm[0] < 1.1, warm.max() / warm[0]
    assert warm[-1] < 1.3 * plain[-1]          # and it still converges


def test_stream_quadrupole_rewrite_matches_the_stream_form():
    """c = delta e^{U-U0} J0 [F grad U0 - grad F]  is  e^U J0 grad Phi."""
    t = anisotropic_student_t(2, 5.0, 100.0)
    x = np.random.default_rng(0).standard_normal((8, 2))
    J0 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    S = t.Sigma_inv_half / np.sqrt(t.nu)
    for a in (0.0, -3.0, 0.5):
        got = sf.StreamField2D.quadrupole(t, 4.95, a, 0.0).drift(x)

        def Phi(y, a=a):
            u = y @ S
            q = 1.0 + np.einsum("ni,ni->n", u, u)
            return -4.95 * q ** (-t.beta) * (1.0 + a * (u[:, 0] ** 2 - u[:, 1] ** 2) / q)

        h = 1e-7
        g = np.zeros_like(x)
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            g[:, i] = (Phi(x + e) - Phi(x - e)) / (2 * h)
        want = np.exp(t.U(x))[:, None] * (g @ J0.T)
        assert np.abs(got - want).max() < 1e-5 * max(1.0, np.abs(want).max()), a


def test_stream_field_preserves_a_composite_target():
    """The stream form needs nothing from the potential, so it works for the
    heavy-tailed non-differentiable target too."""
    from skewanchor import nonsmooth

    C = nonsmooth.make(2, 5.0, 100.0)
    xs = C.sample(200, np.random.default_rng(1))
    for a in (0.0, -3.0):
        fld = sf.StreamField2D.quadrupole(C, 4.95, a, 0.0, geometry=C.base)
        res = np.abs(_div_pi_c(C, fld.drift, xs)).max()
        scale = np.abs(fld.drift(xs)).max()
        assert res < 1e-5 * scale, (a, res, scale)


def test_stream_euler_step_preserves_a_composite_target():
    from skewanchor import nonsmooth

    C = nonsmooth.make(2, 5.0, 100.0)
    rng = np.random.default_rng(2)
    C.cov(rng, 300_000)
    truth = np.diag(C.cov())
    x0 = C.sample(80_000, rng)
    for a in (0.0, -3.0):
        fld = sf.StreamField2D.quadrupole(C, 4.95, a, 0.0, geometry=C.base)
        step = samplers.stream_euler_step(C, 2e-4, fld)
        res = samplers.simulate(step, x0, 400, np.random.default_rng(3))
        assert res.diverged_at is None, a
        got = res.final_state.var(axis=0)
        assert np.all(np.abs(got - truth) / truth < 0.15), (a, got, truth)


# ------------------------------------------------- the cross-product field


def test_cross_product_field_matrix_and_algebra():
    """Js(x) = [[0,-s x3, s x2],[s x3,0,-s x1],[-s x2, s x1,0]] and Js v = s (x cross v)."""
    s = 1.7
    fld = sf.CrossProductSkew(s)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 3)) * 2
    v = rng.standard_normal((6, 3))
    M = fld.matrix_at(x)
    for k in range(x.shape[0]):
        want = np.array([[0.0, -s * x[k, 2], s * x[k, 1]],
                         [s * x[k, 2], 0.0, -s * x[k, 0]],
                         [-s * x[k, 1], s * x[k, 0], 0.0]])
        assert np.allclose(M[k], want), k
    assert np.abs(M + np.transpose(M, (0, 2, 1))).max() < 1e-12
    assert np.allclose(fld.apply(x, v), s * np.cross(x, v))


def test_cross_product_field_is_divergence_free():
    """Identically zero, so no correction term is needed anywhere."""
    fld = sf.CrossProductSkew(1.3)
    x = np.random.default_rng(1).standard_normal((8, 3)) * 3
    assert np.abs(fld.divergence(x)).max() == 0.0
    assert np.abs(sf.SkewField.divergence(fld, x)).max() < 1e-8   # finite differences agree


def test_cross_product_field_preserves_the_target_without_correction():
    t = anisotropic_student_t(3, 6.0, 100.0)
    fld = sf.CrossProductSkew(1.0)
    x = t.sample(300, np.random.default_rng(2))

    def added(y):                      # the added field alone, no correction
        return t.anchor_scale(y)[:, None] * fld.apply(y, t.grad_U0(y))

    res = np.abs(_div_pi_c(t, added, x)).max()
    assert res < 1e-5 * np.abs(added(x)).max(), res


def test_cross_product_drift_reduces_to_a_quadratic_field():
    """The q factors cancel: c = s (2 beta/nu) (x cross Sigma^{-1} x)."""
    t = anisotropic_student_t(3, 6.0, 100.0)
    s = 0.8
    fld = sf.CrossProductSkew(s)
    x = t.sample(200, np.random.default_rng(3))
    got = t.anchor_scale(x)[:, None] * fld.apply(x, t.grad_U0(x))
    want = s * (2 * t.beta / t.nu) * np.cross(x, x @ t.Sigma_inv)
    assert np.abs(got - want).max() < 1e-9 * max(1.0, np.abs(want).max())
    # tangent to spheres and to the level sets of the anchor
    assert np.abs(np.einsum("ni,ni->n", x, got)).max() < 1e-9 * np.abs(got).max()
    assert np.abs(np.einsum("ni,ni->n", t.grad_U0(x), got)).max() < 1e-9 * np.abs(got).max()


def test_cross_product_field_vanishes_on_an_isotropic_target():
    """x cross x = 0, so the field is automatically inert where no J can help."""
    from skewanchor.targets import isotropic_polynomial

    t = isotropic_polynomial(3, 4.0)
    fld = sf.CrossProductSkew(2.0)
    x = t.sample(200, np.random.default_rng(4))
    added = t.anchor_scale(x)[:, None] * fld.apply(x, t.grad_U0(x))
    assert np.abs(added).max() < 1e-12


def test_cross_product_sampler_preserves_the_target():
    t = anisotropic_student_t(3, 6.0, 100.0)
    rng = np.random.default_rng(5)
    x0 = t.sample(100_000, rng)
    step = samplers.field_anchored_step(t, 2e-5, sf.CrossProductSkew(1.0), "euler")
    res = samplers.simulate(step, x0, 600, np.random.default_rng(6))
    assert res.diverged_at is None
    got, truth = res.final_state.var(axis=0), np.diag(t.cov())
    assert np.all(np.abs(got - truth) / truth < 0.12), (got, truth)

if __name__ == "__main__":
    import traceback
    failed = ran = 0
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            ran += 1
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
                traceback.print_exc()
    print(f"{ran} run, " + ("all passed" if not failed else f"{failed} failures"))
    sys.exit(1 if failed else 0)
