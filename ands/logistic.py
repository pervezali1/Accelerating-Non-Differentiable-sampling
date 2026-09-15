"""Unconstrained Bayesian logistic regression, anchored at a smoothed L1 penalty.

This is the target used for the *unconstrained* non-reversible anchored Langevin
dynamics (NALD)

    dX = -a(X) (I + J(X)) grad U_0(X) dt + sqrt(2 a(X)) dW,      a = e^{U - U_0},

which is the canonical stream choice ``psi = e^{-U_0}`` of the theory.  There is no
projection and no reflection, so the only condition left on ``J`` is that it be
skew-symmetric with ``div J = 0``.  The boundary condition ``J(x) n(x) = 0``, which under
a ball constraint rules the constant field out and the block field in, has no force here.

The potential keeps a genuinely non-differentiable piece, or the anchoring would be
vacuous (``U = U_0`` gives ``a = 1`` and ordinary Langevin).  Where the constrained study
imposed ``||w|| <= r`` as a hard wall, the unconstrained analogue charges for size with an
L1 penalty and a Gaussian prior:

    U(w)   = tau * mean_i softplus(-y_i <w, phi_i>) + lam ||w||_1 + (kappa/2) ||w||^2
    U_0(w) = tau * mean_i softplus(-y_i <w, phi_i>) + p_0(w; mu)  + (kappa/2) ||w||^2

with ``p_0`` the exact Gaussian smoothing of ``lam ||.||_1`` from ``penalties.py``.  Only
the penalty is anchored -- the log-likelihood is already smooth -- so

    Delta = U - U_0 = lam ||w||_1 - p_0(w)

carries no data and cannot underflow with ``n``.  Lasso is convex, so ``Delta <= 0`` and
``a = e^Delta`` lands in ``(0, 1]``.

The likelihood is averaged rather than summed, so ``tau`` alone sets how sharp the
posterior is; the gradient is taken over a minibatch, matching the stochastic-gradient
protocol of the constrained experiments.
"""

import numpy as np
import torch

from . import penalties as _pen


class UnconstrainedLogisticTarget:
    """Bayesian logistic regression on ``R^d`` with a smoothed-L1 anchor."""

    def __init__(self, ds, tau=200.0, lam=0.5, kappa=0.5, mu=0.05, pen=None,
                 dtype=torch.float64):
        self.Phi = torch.as_tensor(ds.Phi, dtype=dtype)
        self.y = torch.as_tensor(ds.y, dtype=dtype)
        self.Psi = self.y.unsqueeze(1) * self.Phi          # rows y_i phi_i
        self.n, self.d = self.Psi.shape
        self.tau, self.lam, self.kappa, self.mu = float(tau), float(lam), float(kappa), float(mu)
        self.pen = pen if pen is not None else _pen.lasso(self.lam)
        self.name = ds.key

    # -- potentials -----------------------------------------------------------------
    def _nll(self, w, Psi):
        return torch.nn.functional.softplus(-(w @ Psi.T)).mean(dim=1)

    def U(self, w):
        """The exact, non-differentiable potential."""
        return (self.tau * self._nll(w, self.Psi)
                + self.pen.raw(w).sum(dim=1)
                + 0.5 * self.kappa * (w * w).sum(dim=1))

    def U0(self, w):
        """The smooth anchor: same likelihood, Gaussian-smoothed penalty."""
        p0, _ = self.pen.smooth(w, self.mu)
        return (self.tau * self._nll(w, self.Psi)
                + p0.sum(dim=1)
                + 0.5 * self.kappa * (w * w).sum(dim=1))

    def delta(self, w):
        """``U - U_0``.  Data-free, and non-positive whenever the penalty is convex."""
        p0, _ = self.pen.smooth(w, self.mu)
        return self.pen.raw(w).sum(dim=1) - p0.sum(dim=1)

    # -- the pieces the sampler needs ------------------------------------------------
    @torch.no_grad()
    def anchored(self, w, idx=None):
        """``(Delta, grad U_0)`` at ``w``, the gradient over rows ``idx`` if given.

        ``idx=None`` uses every row, which is the full-gradient dynamics; passing a
        minibatch of indices gives the stochastic-gradient version.
        """
        Psi = self.Psi if idx is None else self.Psi[idx]
        p0, dp0 = self.pen.smooth(w, self.mu)
        # d/dw mean_i softplus(-<w, psi_i>) = -mean_i sigma(-<w, psi_i>) psi_i
        sig = torch.sigmoid(-(w @ Psi.T))
        grad = -(self.tau / Psi.shape[0]) * (sig @ Psi) + dp0 + self.kappa * w
        return self.pen.raw(w).sum(dim=1) - p0.sum(dim=1), grad

    @torch.no_grad()
    def accuracy(self, w, Psi=None):
        """Per-walker accuracy of the classifier ``sign(<w, phi>)``.

        A tie is scored 0.5, so the origin sits at exactly chance rather than at the
        majority-class rate.
        """
        Psi = self.Psi if Psi is None else Psi
        m = w @ Psi.T
        return ((m > 0).to(m.dtype) + 0.5 * (m == 0).to(m.dtype)).mean(dim=1)

    def eval_matrix(self, ds, dtype=torch.float64):
        """``Psi`` for a held-out split, to pass to :meth:`accuracy`."""
        Phi = torch.as_tensor(ds.Phi, dtype=dtype)
        return torch.as_tensor(ds.y, dtype=dtype).unsqueeze(1) * Phi


@torch.no_grad()
def uniform_ball(n, d, radius=1.0, generator=None, dtype=torch.float64):
    """``n`` draws uniform on the centred ball of the given radius -- the paper's start."""
    z = torch.randn(n, d, generator=generator, dtype=dtype)
    z = z / z.norm(dim=1, keepdim=True)
    u = torch.rand(n, 1, generator=generator, dtype=dtype) ** (1.0 / d)
    return radius * u * z


@torch.no_grad()
def run_nald(target, field, eta=1e-4, n_steps=1000, n_walkers=100, batch=30,
             track_every=10, eval_sets=(), radius=1.0, seed=0):
    """Euler-Maruyama for the unconstrained NALD, tracking accuracy as it goes.

    Returns ``(iters, accs, w)`` where ``accs[k]`` has shape ``(len(iters), n_walkers)``
    for the ``k``-th entry of ``eval_sets`` (each a ``Psi`` matrix), and ``w`` is the final
    ensemble -- accuracy alone is scale-invariant and so nearly blind to sampling bias, so
    the walkers are needed to score the law itself.
    """
    gen = torch.Generator().manual_seed(int(seed))
    w = uniform_ball(n_walkers, target.d, radius, generator=gen)
    s2 = np.sqrt(2.0 * eta)

    def rec():
        return [target.accuracy(w, P).clone() for P in eval_sets]

    iters, hist = [0], [rec()]
    for k in range(n_steps):
        idx = torch.randint(0, target.n, (batch,), generator=gen)
        Delta, g = target.anchored(w, idx)
        a = torch.exp(Delta).unsqueeze(1)
        # -a (I + J) grad U_0 + a div J ; div J = 0 for both admissible fields here
        drift = -a * (g + field.apply(w, g)) + a * field.divergence(w)
        w = w + eta * drift + s2 * torch.sqrt(a) * torch.randn(
            n_walkers, target.d, generator=gen, dtype=w.dtype)
        if (k + 1) % track_every == 0:
            iters.append(k + 1)
            hist.append(rec())
    return (np.array(iters),
            [torch.stack([h[k] for h in hist]).numpy() for k in range(len(eval_sets))],
            w)


@torch.no_grad()
def run_rwm(target, n_chains=400, n_steps=20000, scale=None, seed=1,
            target_rate=0.25, adapt_frac=0.3):
    """Random-walk Metropolis on the exact ``U`` -- the reference the curves climb toward.

    Uses ``U`` only through function values, so it is a legitimate reference for a
    potential the sampler is never allowed to differentiate.

    The proposal scale is adapted during the first ``adapt_frac`` of the run to hit
    ``target_rate``.  Without that, a sharp posterior (large ``tau``) drives the
    acceptance rate to ~1% and the "reference" stops moving -- which silently reports a
    ceiling *below* what the samplers reach, and makes every gap look negative.
    """
    gen = torch.Generator().manual_seed(int(seed))
    sc = (2.38 / np.sqrt(target.d)) if scale is None else float(scale)
    w = torch.zeros(n_chains, target.d, dtype=torch.float64)
    Uw = target.U(w)
    n_adapt = int(adapt_frac * n_steps) if scale is None else 0
    acc, seen = 0.0, 0
    for k in range(n_steps):
        p = w + sc * torch.randn(n_chains, target.d, generator=gen, dtype=w.dtype)
        Up = target.U(p)
        take = torch.log(torch.rand(n_chains, generator=gen, dtype=w.dtype)) < (Uw - Up)
        w = torch.where(take.unsqueeze(1), p, w)
        Uw = torch.where(take, Up, Uw)
        rate = float(take.to(w.dtype).mean())
        if k < n_adapt:
            # Robbins-Monro on log-scale; the 1/sqrt(k) gain makes it settle
            sc *= float(np.exp((rate - target_rate) / np.sqrt(k + 1)))
        else:
            acc += rate
            seen += 1
    return w, (acc / max(seen, 1))
