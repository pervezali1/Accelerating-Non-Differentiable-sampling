"""Samplers: the projected anchored Langevin scheme under study, and an exact
random-walk Metropolis reference.

Anchored, state-dependent-J dynamics on ``K``:

    dX = e^{Delta(X)} [ -(I + J(X)) grad U_0(X) + div J(X) ] dt
         + sqrt(2) e^{Delta(X)/2} dW,      Delta = U - U_0,

discretised with stepsize ``eta`` and Euclidean projection onto ``K``:

    x <- P_K( x - eta e^{Delta} [ (I + J(x)) grad U_0(x) - div J(x) ]
              + sqrt(2 eta) e^{Delta/2} xi ).

The reference chain never touches ``U_0``: it is a Metropolis random walk against the
exact non-smooth ``U`` with proposals rejected outside ``K``, which is exactly
reversible for ``pi 1_K``, so the comparison is against the true target rather than
against the smoothed one.
"""

import numpy as np
import torch

from .skew import build_skew


def run_anchored_langevin(target, skew="none", s=0.0, eta=1e-4, n_steps=2000,
                          N=2000, seed=0, drop_correction=False, x0_scale=0.0,
                          ref=None, track_every=10, w1_coords=None):
    """Return ``(final_samples, trace)`` where ``trace`` is ``(iteration, W1 per
    tracked coordinate)`` against ``ref`` (or ``None`` when ``ref`` is not given)."""
    from scipy.stats import wasserstein_distance

    d = target.d
    field = build_skew(skew, d, s, radius=target.R, drop_correction=drop_correction)
    g = torch.Generator().manual_seed(seed)

    x = torch.randn(N, d, generator=g) * x0_scale if x0_scale > 0 else torch.zeros(N, d)
    x = target.project(x)

    coords = list(range(d)) if w1_coords is None else list(w1_coords)
    iters, W = [], []
    sqrt2eta = np.sqrt(2.0 * eta)

    for it in range(n_steps):
        Delta, gU0 = target.anchored_step(x)
        eD = torch.exp(Delta).unsqueeze(1)
        drift = eD * (gU0 + field.apply(x, gU0) - field.divergence(x))
        noise = torch.exp(0.5 * Delta).unsqueeze(1) * torch.randn(N, d, generator=g)
        x = target.project(x - eta * drift + sqrt2eta * noise)
        if ref is not None and ((it + 1) % track_every == 0 or it == 0):
            xn = x.numpy()
            iters.append(it + 1)
            W.append([wasserstein_distance(ref[:, i], xn[:, i]) for i in coords])

    trace = None if ref is None else (np.array(iters), np.array(W))
    return x.numpy(), trace


@torch.no_grad()
def run_rwm_reference(target, n_chains=2000, n_steps=20000, burn=None, step=None,
                      cov=None, seed=0, thin_keep=1):
    """Random-walk Metropolis on the exact ``pi 1_K``.

    Proposals use ``2.38^2/d * cov`` (Roberts--Gelman--Gilks scaling); a proposal
    outside ``K`` has ``U = inf`` and is rejected, so the chain is reversible for the
    constrained target with no projection bias.  Returns the pooled final states of all
    chains plus diagnostics.
    """
    d = target.d
    burn = n_steps // 2 if burn is None else burn
    g = torch.Generator().manual_seed(seed)

    if cov is None:
        cov = torch.eye(d, dtype=torch.float64) * (0.05 ** 2)
    L = torch.linalg.cholesky(cov)
    scale = 2.38 / np.sqrt(d) if step is None else step

    x = torch.zeros(n_chains, d, dtype=torch.float64)
    Ux = target.U(x)
    accepted = torch.zeros(n_chains, dtype=torch.float64)
    kept = []

    for it in range(n_steps):
        prop = x + scale * (torch.randn(n_chains, d, generator=g) @ L.T)
        ok = target.inside(prop)
        Up = torch.where(ok, target.U(prop), torch.full_like(Ux, float("inf")))
        acc = (torch.log(torch.rand(n_chains, generator=g)) < (Ux - Up)) & ok
        x = torch.where(acc.unsqueeze(1), prop, x)
        Ux = torch.where(acc, Up, Ux)
        if it >= burn:
            accepted += acc.double()
            if thin_keep > 1 and (it - burn) % (max(1, (n_steps - burn) // thin_keep)) == 0:
                kept.append(x.clone())

    diag = {"acc_rate": float(accepted.mean() / max(1, n_steps - burn)),
            "n_chains": n_chains, "n_steps": n_steps, "scale": float(scale)}
    samples = torch.cat(kept, dim=0).numpy() if kept else x.numpy()
    return samples, diag


@torch.no_grad()
def pilot_covariance(target, n_chains=400, n_steps=4000, seed=1):
    """Cheap pilot RWM used only to shape the reference chain's proposal."""
    x, _ = run_rwm_reference(target, n_chains=n_chains, n_steps=n_steps, seed=seed)
    C = np.cov(x.T)
    C = C + 1e-8 * np.eye(target.d)
    return torch.tensor(C)
