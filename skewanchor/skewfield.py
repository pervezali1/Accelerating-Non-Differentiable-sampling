r"""State-dependent skew-symmetric fields ``J(x)``.

With a *constant* skew ``J`` the added drift ``e^{U-U_0} J grad U_0`` is
pi-divergence free because ``Tr(J H) = 0`` for symmetric ``H`` and
``<v, J v> = 0``.  Allowing ``J`` to depend on ``x`` adds one more term.  Writing
``rho_0 = e^{-U_0}`` and ``c(x) = e^{(U-U_0)(x)} J(x) grad U_0(x)``,

    div(pi c) ∝ div(rho_0 J grad U_0)
             = -rho_0 <grad U_0, J grad U_0>      (= 0, J skew)
             +  rho_0 <J, Hess U_0>_F             (= 0, skew vs symmetric)
             -  rho_0 <div J, grad U_0>,

where ``(div J)_j := sum_i d_i J_{ji}``.  So the target is preserved **iff**

    <div J(x), grad U_0(x)> = 0   for every x.                      (INV)

Two families satisfy (INV), and both are implemented here.

``RadialModulated`` -- ``J(x) = psi(q(x)) J_0`` with ``J_0`` constant skew.
    Then ``div J = psi'(q) J_0 grad q`` and ``grad U_0 ∝ grad q``, so (INV) holds
    for **every** profile ``psi``, by skew-symmetry again.  In ``d = 2`` this is
    not just an example: every skew field of the form ``psi(x) J_0`` satisfying
    (INV) has ``psi`` constant on the orbits of ``x -> J_0 A x``, and those
    orbits are exactly the level sets of ``q``.  So in two dimensions
    ``psi(q) J_0`` is the *complete* answer, and the freedom is one scalar
    profile of one variable.

``CurlSkew`` -- ``J(x) v = v x grad f(x)`` for a scalar potential ``f``
    (``d = 3``; the general-``d`` version is ``J = div A`` for a totally
    antisymmetric 3-tensor ``A``).  Here ``div J = 0`` *identically*, since it
    contracts an antisymmetric symbol with the symmetric Hessian of ``f``, so
    (INV) holds for **any target and any anchor** -- no structure required.
    This family does not exist in ``d = 2``, which is why the two-dimensional
    case is confined to radial modulation.

Note what neither family can escape: ``<grad U_0, J grad U_0> = 0`` always, so
the added drift is tangent to the level sets of the anchor.  These flows move
mass around the level sets of ``U_0``, never across them.

The correction that removes condition (INV) altogether
-----------------------------------------------------

(INV) is an artefact of insisting that the added drift be exactly
``e^{U-U_0} J grad U_0``.  Add one term and the condition disappears:

    c(x) = e^{(U-U_0)(x)} [ J(x) grad U_0(x) - (div J)(x) ]              (COR)

preserves ``pi`` for **every** skew matrix field ``J``, with no constraint at
all.  The proof is two divergences that cancel:

    div(rho_0 J grad U_0) = -rho_0 <div J, grad U_0>,
    div(rho_0 div J)      = -rho_0 <grad U_0, div J>,

the second because ``sum_ij d_i d_j J_ji = 0`` (a symmetric second derivative
contracted with an antisymmetric matrix).  Subtracting gives zero.  Equivalently
``pi c = div S`` for the antisymmetric matrix field ``S = -e^{-U_0} J``, and the
divergence of a divergence of an antisymmetric field vanishes identically.

Two consequences.  The corrected drift is **not** confined to the level sets of
``U_0``, so the whole class of flows opens up.  And in ``d = 2`` the complete
family has a one-line description: any ``pi``-divergence-free field is a skew
gradient, so

    c(x) = e^{U(x)} J_0 grad Phi(x)                                      (2D)

for an arbitrary scalar stream function ``Phi``, with ``Phi = -delta e^{-U_0}``
recovering the constant field of norm ``delta``.  :class:`StreamField2D`
implements that form directly.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SkewField",
    "ConstantSkew",
    "RadialModulated",
    "CurlSkew",
    "StreamField2D",
    "PROFILES",
    "make_profile",
    "invariance_residual",
]


class SkewField:
    """Base class.  ``apply(x, v)`` returns ``J(x) v`` row-wise."""

    is_constant = False

    def apply(self, x, v):
        raise NotImplementedError

    def matrix_at(self, x):
        """``(n, d, d)`` stack of ``J(x_i)``, for diagnostics and tests."""
        x = np.atleast_2d(x)
        n, d = x.shape
        out = np.empty((n, d, d))
        for j in range(d):
            e = np.zeros((n, d))
            e[:, j] = 1.0
            out[:, :, j] = self.apply(x, e)
        return out

    def norm_at(self, x):
        return np.array([np.linalg.norm(M, 2) for M in self.matrix_at(x)])

    def divergence(self, x, h=1e-5):
        """``(div J)_j = sum_i d_i J_{ji}``, by central differences by default."""
        x = np.atleast_2d(x)
        n, d = x.shape
        out = np.zeros((n, d))
        for i in range(d):
            e = np.zeros(d)
            e[i] = h
            out += (self.matrix_at(x + e)[:, :, i] - self.matrix_at(x - e)[:, :, i]) / (2 * h)
        return out


class ConstantSkew(SkewField):
    """``J(x) = J``.  The field used in the first round of experiments."""

    is_constant = True

    def __init__(self, J):
        self.J = np.asarray(J, dtype=np.float64)
        self.d = self.J.shape[0]

    def apply(self, x, v):
        return np.atleast_2d(v) @ self.J.T

    def matrix_at(self, x):
        return np.broadcast_to(self.J, (np.atleast_2d(x).shape[0],) + self.J.shape)

    def divergence(self, x, h=None):
        return np.zeros_like(np.atleast_2d(x))


class RadialModulated(SkewField):
    r"""``J(x) = psi(q(x)) J_0``: the rotation strength varies with the
    ellipsoidal radius ``q(x) = 1 + (x-mu)^T Sigma^{-1}(x-mu)/nu``.

    Preserves the target for every profile ``psi``; in ``d = 2`` this is the
    complete family.
    """

    def __init__(self, J0, target, profile):
        self.J0 = np.asarray(J0, dtype=np.float64)
        self.target = target
        self.profile = profile
        self.d = self.J0.shape[0]

    def psi(self, x):
        return self.profile(self.target.q(np.atleast_2d(x)))

    def apply(self, x, v):
        x = np.atleast_2d(x)
        return self.psi(x)[:, None] * (np.atleast_2d(v) @ self.J0.T)

    def matrix_at(self, x):
        return self.psi(x)[:, None, None] * self.J0[None, :, :]

    def divergence(self, x, h=1e-6):
        """``div J = J_0 grad psi``; ``psi'`` by a one-dimensional difference."""
        x = np.atleast_2d(x)
        q = self.target.q(x)
        dpsi = (self.profile(q + h) - self.profile(q - h)) / (2 * h)
        gq = 2.0 * (x - self.target.mu) @ self.target.Sigma_inv / self.target.nu
        return dpsi[:, None] * (gq @ self.J0.T)


class CurlSkew(SkewField):
    r"""``J(x) v = v x grad f(x)`` in ``d = 3``.

    Divergence free by construction, so it preserves *any* target.  ``grad_f``
    must map ``(n, 3) -> (n, 3)``.
    """

    def __init__(self, grad_f, d=3):
        if d != 3:
            raise ValueError("CurlSkew as written is the d = 3 cross-product form")
        self.grad_f = grad_f
        self.d = 3

    def apply(self, x, v):
        x = np.atleast_2d(x)
        return np.cross(np.atleast_2d(v), self.grad_f(x))

    def divergence(self, x, h=None):
        return np.zeros_like(np.atleast_2d(x))


class StreamField2D:
    r"""The complete two-dimensional family: ``c(x) = e^{U(x)} J_0 grad Phi(x)``.

    Parameterised by a stream function rather than a matrix field, because in
    ``d = 2`` every ``pi``-divergence-free drift is of this form.  ``grad_Phi``
    maps ``(n, 2) -> (n, 2)``; supply it analytically when you can.

    ``quadrupole`` builds the family used in the experiments,

        Phi = -delta q^{-beta} (1 + a (u1^2 - u2^2)/q + b (2 u1 u2)/q),
        u   = Sigma^{-1/2} x / sqrt(nu),   q = 1 + |u|^2,

    whose ``a = b = 0`` member is exactly the constant field of norm ``delta``.
    ``a`` and ``b`` tilt the rotation towards or away from the stiff axis; both
    are inadmissible as bare ``J(x)`` fields and become legal only through the
    stream form.
    """

    J0 = np.array([[0.0, 1.0], [-1.0, 0.0]])

    def __init__(self, target, grad_Phi, delta=0.0, a=0.0, b=0.0):
        if target.d != 2:
            raise ValueError("StreamField2D is two-dimensional by construction")
        self.target = target
        self.grad_Phi = grad_Phi
        self.delta = float(delta)
        self.a = float(a)
        self.b = float(b)
        self._drift = None
        self.geometry = None

    def drift(self, x):
        if getattr(self, "_drift", None) is not None:
            return self._drift(np.atleast_2d(x))
        x = np.atleast_2d(x)
        return (np.exp(self.target.U(x)))[:, None] * (self.grad_Phi(x) @ self.J0.T)

    def constant_member(self):
        """The ``a = b = 0`` member of the same family -- the constant field."""
        return StreamField2D.quadrupole(self.target, self.delta, 0.0, 0.0,
                                        geometry=getattr(self, "geometry", None))

    @classmethod
    def quadrupole(cls, target, delta, a=0.0, b=0.0, geometry=None):
        r"""The tilted family used in the experiments.

            Phi = -delta e^{-U0} (1 + a w1 + b w2),
            w1  = (u1^2 - u2^2)/q,   w2 = 2 u1 u2 / q,
            u   = Sigma^{-1/2} x / sqrt(nu),   q = 1 + |u|^2,

        whose ``a = b = 0`` member is exactly the constant field of norm
        ``delta``.  ``a`` tilts the rotation towards the soft axis (``a < 0``) or
        the stiff one (``a > 0``); ``b`` shears it.  Neither is admissible as a
        bare ``J(x)`` field -- both become legal through the stream form.

        Evaluated as

            c = delta e^{U-U0} J_0 [ (1 + a w1 + b w2) grad U0 - a grad w1 - b grad w2 ],

        which is ``e^U J_0 grad Phi`` rewritten so that only the bounded factor
        ``e^{U-U0}`` appears; ``e^{U}`` itself is ``q^{iota}`` and overflows
        long before the dynamics does.

        This form needs only ``anchor_scale`` and ``grad_U0`` from the target, so
        it works for a composite potential too -- pass the Student-t core as
        ``geometry`` to supply the quadratic form that ``w1`` and ``w2`` are
        built from.
        """
        geo = target if geometry is None else geometry
        S = geo.Sigma_inv_half / np.sqrt(geo.nu)

        def shape(x):
            """(F, grad F) with F = 1 + a w1 + b w2."""
            u = x @ S
            q = 1.0 + np.einsum("ni,ni->n", u, u)
            w1 = (u[:, 0] ** 2 - u[:, 1] ** 2) / q
            w2 = (2.0 * u[:, 0] * u[:, 1]) / q
            dw1 = (np.stack([2 * u[:, 0], -2 * u[:, 1]], 1) * q[:, None]
                   - (u[:, 0] ** 2 - u[:, 1] ** 2)[:, None] * 2 * u) / q[:, None] ** 2
            dw2 = (np.stack([2 * u[:, 1], 2 * u[:, 0]], 1) * q[:, None]
                   - (2 * u[:, 0] * u[:, 1])[:, None] * 2 * u) / q[:, None] ** 2
            return 1.0 + a * w1 + b * w2, (a * dw1 + b * dw2) @ S

        def drift(x):
            x = np.atleast_2d(x)
            F, dF = shape(x)
            v = F[:, None] * target.grad_U0(x) - dF
            return delta * target.anchor_scale(x)[:, None] * (v @ cls.J0.T)

        obj = cls(target, None, delta=delta, a=a, b=b)
        obj._drift = drift
        obj.geometry = geometry
        return obj


# ------------------------------------------------------------------ profiles


def make_profile(kind, delta, q0=2.0, p=1.0, qc=2.0, width=1.0):
    """Scalar profiles ``psi(q)``, all with amplitude ``delta``.

    ``const``   flat -- recovers a constant ``J``.
    ``decay``   ``delta (q0/(q0 + q - 1))^p`` -- strong in the bulk, weak in the tail.
    ``grow``    ``delta ((q0 + q - 1)/q0)^p`` -- the opposite.
    ``bulk``    ``delta exp(-((q-1)/q0)^2)`` -- concentrated near the mode.
    ``shell``   ``delta exp(-((q-qc)/width)^2)`` -- concentrated on one ellipsoid.
    ``invq``    ``delta / q``.
    """
    if kind == "const":
        return lambda q: np.full_like(np.asarray(q, dtype=np.float64), delta)
    if kind == "decay":
        return lambda q: delta * (q0 / (q0 + np.asarray(q) - 1.0)) ** p
    if kind == "grow":
        return lambda q: delta * ((q0 + np.asarray(q) - 1.0) / q0) ** p
    if kind == "bulk":
        return lambda q: delta * np.exp(-(((np.asarray(q) - 1.0) / q0) ** 2))
    if kind == "shell":
        return lambda q: delta * np.exp(-(((np.asarray(q) - qc) / width) ** 2))
    if kind == "invq":
        return lambda q: delta / np.asarray(q)
    raise ValueError(f"unknown profile {kind!r}")


PROFILES = ("const", "decay", "grow", "bulk", "shell", "invq")


# ------------------------------------------------------------- invariance


def invariance_residual(field, target, x, h=1e-5):
    r"""``<div J(x), grad U_0(x)>``, by central differences.

    Must be zero for the target to be preserved.  Used as a test, and as a
    guard when a new field is added.
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    div = np.zeros((n, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        Jp = field.matrix_at(x + e)
        Jm = field.matrix_at(x - e)
        div += (Jp[:, :, i] - Jm[:, :, i]) / (2 * h)
    return np.einsum("ni,ni->n", div, target.grad_U0(x))
