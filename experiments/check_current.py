#!/usr/bin/env python3
"""Is the anchored chain actually non-reversible?  Measure its stationary current.

A ``pi``-reversible diffusion has zero stationary probability current, so the
mean signed area it sweeps per step in any plane is zero.  An irreversible one
sweeps area at a rate set by the antisymmetric part of its generator, and that
rate reverses when the skew matrix does.  This script estimates

    0.5 * E[ v_i dv_j - v_j dv_i ]   per step,

in the standardised eigen-coordinates ``v`` of the reference Hessian, over the
slow-fast planes the design couples, with the walkers started at stationarity so
that the transient does not contribute.  Four fields are compared: ``J = 0``, the
designed rotation, its negation, and the gated state-dependent version.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nds.anchored import LassoLogistic  # noqa: E402
from nds.data import PRETTY_NAMES, load  # noqa: E402
from nds.design import explicit_step_size, paired_skew  # noqa: E402
from nds.sampler import Geometry, warm_up  # noqa: E402
from nds.skew import ConstantSkew, GatedSkew, ZeroSkew  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def stationary_current(target, field, geo, h, mean, cov, P, pairs,
                       n_iter: int, n_walkers: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)
    W = mean[:, None] + L @ rng.standard_normal((target.d, n_walkers))
    acc = np.zeros((len(pairs), n_walkers))
    for _ in range(n_iter):
        a = target.clock(W)
        G = target.anchor_fd_grad(W)
        drift = field.divergence(W) - (geo.D @ G + field.apply(W, G))
        Wn = W + h * a * drift + np.sqrt(2.0 * h * a) * (geo.L @ rng.standard_normal(W.shape))
        v, vn = P @ (W - mean[:, None]), P @ (Wn - mean[:, None])
        for k, (i, j) in enumerate(pairs):
            acc[k] += 0.5 * (v[i] * vn[j] - v[j] * vn[i])
        W = Wn
    per_step = acc / n_iter
    return per_step.mean(axis=1), per_step.std(axis=1, ddof=1) / np.sqrt(n_walkers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="titanic")
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--n-iter", type=int, default=4000)
    parser.add_argument("--n-walkers", type=int, default=48)
    parser.add_argument("--safety", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    ds = load(args.dataset, seed=0)
    target = LassoLogistic(ds.X_train, ds.y_train, penalty=args.penalty, delta=args.delta)
    ref = np.load(os.path.join(RESULTS, f"reference_lasso_{args.dataset}_p{args.penalty:g}.npz"))
    mean, cov = ref["posterior_mean"], ref["posterior_cov"]
    lam, V = np.linalg.eigh(np.linalg.pinv(cov))
    P = (V * np.sqrt(lam)).T
    pairs = [(k, len(lam) - 1 - k) for k in range(len(lam) // 2)]

    pilot = warm_up(target, n_iter=200, n_walkers=8, seed=0, n_rounds=2, shrinkage=0.05)
    H_hat, m_hat = pilot["geometry"].D_inv, pilot["mean_state"]
    r0 = float(np.sqrt(max(m_hat @ (H_hat @ m_hat), 1e-12)))
    J, _ = paired_skew(H_hat, amplitude=1.0)
    geo = Geometry.identity(target.d)
    h = explicit_step_size(H_hat, J, safety=args.safety)

    fields = {
        "zero": ZeroSkew(),
        "constant_J": ConstantSkew(J, 1.0),
        "constant_minus_J": ConstantSkew(-J, 1.0),
        "gated_J": GatedSkew(J, 1.0, radius=0.75 * r0, width=0.1 * r0,
                             metric=H_hat, center=m_hat),
    }

    lines = [
        f"{PRETTY_NAMES[args.dataset]}: stationary current of the anchored chain",
        f"lambda = {args.penalty:g}, delta = {args.delta:g}, step size {h:.4g}, "
        f"{args.n_iter} iterations, {args.n_walkers} walkers started at stationarity",
        "mean signed area per step in the coupled planes of the reference eigenbasis; "
        "zero means reversible",
        "",
        "field              " + "".join(f"{'pair ' + str(p):>24s}" for p in pairs[:3]) + "   max |z|",
    ]
    out = {}
    for name, field in fields.items():
        m, se = stationary_current(target, field, geo, h, mean, cov, P, pairs,
                                   args.n_iter, args.n_walkers, args.seed)
        z = float(np.max(np.abs(m) / np.maximum(se, 1e-12)))
        out[name] = {"current": m.tolist(), "standard_error": se.tolist(), "max_z": z}
        cells = "".join(f"{m[k]:+13.5f} +/-{se[k]:8.5f}" for k in range(min(3, len(pairs))))
        lines.append(f"{name:18s} {cells}   {z:7.1f}")
        print(lines[-1], flush=True)

    lines += [
        "",
        "Reading: J = 0 is reversible (current within noise of zero), the designed "
        "rotation is not,",
        "and negating it reverses the current, which is what a probability current does.",
    ]
    with open(os.path.join(RESULTS, "stationary_current.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS, "stationary_current.json"), "w") as handle:
        json.dump({"dataset": args.dataset, "step_size": h, "pairs": pairs,
                   "penalty": args.penalty, "delta": args.delta, "fields": out}, handle, indent=2)
    print("wrote results/stationary_current.txt")


if __name__ == "__main__":
    main()
