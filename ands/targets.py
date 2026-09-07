"""The non-differentiable target: a hinge-loss Gibbs posterior with an L1 prior,
restricted to a Euclidean ball.

For labels ``y_i in {-1,+1}`` and standardised features ``phi_i`` write
``psi_i = y_i phi_i``.  The potential is

    U(w) = tau * mean_i max(0, 1 - w . psi_i)  +  lam * ||w||_1,

so ``pi(w) \\propto e^{-U(w)} 1_K(w)`` on ``K = {||w||_2 <= R}``.  Both terms are
non-differentiable: the hinge has a kink at every active margin, the L1 prior a kink on
every coordinate axis.  ``U`` is only ever *evaluated* -- never differentiated -- which
is the whole point of the anchored scheme.

The anchor smooths each kink at scale ``delta``:

    max(0, z) -> (z + sqrt(z^2 + delta^2)) / 2,      |w| -> sqrt(w^2 + delta^2),

both of which dominate the function they replace, so ``U_0 >= U`` and the anchor
multiplier ``e^{Delta} = e^{U - U_0}`` lies in ``(0, 1]``.  Its worst case is bounded by
``tau*delta/2 + lam*d*delta`` independently of the sample size ``n`` -- which is why the
hinge term is an average rather than a sum.
"""

import numpy as np
import torch


class BallConstrainedGibbsSVM:
    def __init__(self, dataset, tau=50.0, lam=1.0, delta=0.02, R=1.0, dtype=torch.float64):
        self.data = dataset
        self.tau, self.lam, self.delta, self.R = float(tau), float(lam), float(delta), float(R)
        self.d = dataset.d
        # psi_i = y_i phi_i, so the margin is 1 - w . psi_i
        self.Psi = torch.tensor(dataset.y[:, None] * dataset.Phi, dtype=dtype)
        self.n = self.Psi.shape[0]

    # ---------------- exact potential (evaluated, never differentiated) -------------
    @torch.no_grad()
    def U(self, w):
        z = 1.0 - w @ self.Psi.T                                    # (N, n) margins
        hinge = torch.clamp(z, min=0.0).mean(dim=1)
        return self.tau * hinge + self.lam * w.abs().sum(dim=1)

    # ---------------- anchor and its gradient --------------------------------------
    @torch.no_grad()
    def U0(self, w):
        z = 1.0 - w @ self.Psi.T
        soft = 0.5 * (z + torch.sqrt(z * z + self.delta ** 2))
        l1 = torch.sqrt(w * w + self.delta ** 2).sum(dim=1)
        return self.tau * soft.mean(dim=1) + self.lam * l1

    @torch.no_grad()
    def grad_U0(self, w):
        z = 1.0 - w @ self.Psi.T
        dsoft = 0.5 * (1.0 + z / torch.sqrt(z * z + self.delta ** 2))    # (N, n)
        g_hinge = -(self.tau / self.n) * (dsoft @ self.Psi)
        g_l1 = self.lam * w / torch.sqrt(w * w + self.delta ** 2)
        return g_hinge + g_l1

    @torch.no_grad()
    def anchor_gap(self, w):
        """``Delta = U - U0 <= 0``.  ``e^{Delta}`` is the anchor's time change."""
        return self.U(w) - self.U0(w)

    @torch.no_grad()
    def anchored_step(self, w):
        """``(Delta, grad U_0)`` from a single pass over the margins.

        ``U``, ``U_0`` and ``grad U_0`` all read the same ``(N, n)`` margin matrix, and
        forming it is the whole cost of a step, so the sampler asks for them together.
        """
        z = 1.0 - w @ self.Psi.T                                    # (N, n)
        root = torch.sqrt(z * z + self.delta ** 2)
        hinge = torch.clamp(z, min=0.0).mean(dim=1)
        soft = (0.5 * (z + root)).mean(dim=1)
        aw = w.abs()
        rw = torch.sqrt(w * w + self.delta ** 2)
        Delta = self.tau * (hinge - soft) + self.lam * (aw - rw).sum(dim=1)

        dsoft = 0.5 * (1.0 + z / root)
        grad = -(self.tau / self.n) * (dsoft @ self.Psi) + self.lam * w / rw
        return Delta, grad

    # ---------------- the constraint ------------------------------------------------
    @torch.no_grad()
    def project(self, w):
        nrm = torch.norm(w, dim=1, keepdim=True)
        return w * torch.clamp(self.R / nrm, max=1.0)

    @torch.no_grad()
    def inside(self, w):
        return torch.norm(w, dim=1) <= self.R * (1 + 1e-12)

    # ---------------- downstream statistics -----------------------------------------
    # These read the posterior the way a practitioner would.  All three are computed on
    # the same rows the posterior was fitted to, so they are *in-sample*: the question
    # they answer is whether a sampling bias reaches a decision statistic, not how well
    # the classifier generalises.
    @torch.no_grad()
    def _score(self, w):
        """``(N, n)`` in {0, 1/2, 1}: draw k scores row i by ``sign(w_k . psi_i)``, with
        an exact zero margin counted as half.

        The tie matters in exactly one place and it is the place the learning curve
        starts: every chain is launched from ``w = 0``, where every margin is exactly
        zero. Counting a tie as a coin flip puts that start at chance, 0.5, which is
        what it is. For any ``w != 0`` on continuous features ties have probability
        zero, so nothing else in the notebook is affected."""
        m = w @ self.Psi.T
        return (m > 0).to(m.dtype) + 0.5 * (m == 0).to(m.dtype)

    @torch.no_grad()
    def _correct(self, w_samples):
        w = torch.as_tensor(np.atleast_2d(np.asarray(w_samples)), dtype=self.Psi.dtype)
        return self._score(w)

    def accuracy(self, w_samples):
        """Accuracy of the single plug-in classifier ``w_bar = E[w]``."""
        w = torch.as_tensor(np.atleast_2d(np.asarray(w_samples)),
                            dtype=self.Psi.dtype).mean(dim=0, keepdim=True)
        return float(self._score(w).mean())

    def accuracy_per_draw(self, w_samples):
        """``(N,)`` accuracy of each posterior draw taken on its own.

        Its *spread* is a posterior quantity in its own right, so a sampler can get the
        mean right and the spread wrong."""
        return self._correct(w_samples).mean(dim=1).numpy()

    def accuracy_predictive(self, w_samples):
        """Accuracy of the posterior-predictive (majority-vote) classifier.

        Row ``i`` is called correctly when more than half the posterior mass classifies
        it correctly, i.e. this is the Bayes rule under the sampled posterior rather
        than under any single ``w``."""
        return float((self._correct(w_samples).mean(dim=0) > 0.5).double().mean())
