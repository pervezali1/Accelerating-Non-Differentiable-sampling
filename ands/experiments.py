"""The experiment grid, so the notebook stays narrative and the runs stay reproducible.

Everything is cached to ``results/`` by a hash of its settings: re-executing the
notebook is cheap, and a changed setting invalidates only what it touches.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch

from . import diagnostics as G
from . import samplers as S
from . import targets as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# One place for every number the notebook depends on.
CONFIG = {
    "tau": 100.0,        # Gibbs inverse temperature on the mean hinge loss
    "lam": 1.0,          # L1 prior strength
    "delta": 0.02,       # anchor smoothing scale
    "N": 2000,           # walkers in the Langevin ensemble
    "n_steps": 3000,
    "eta": 3e-4,
    "s_main": 4.0,       # skew strength for the head-to-head
    "sweep": [0.0, 1.0, 2.0, 4.0, 8.0],
    "ref_chains": 2500,
    "ref_steps": 6000,
    "track_every": 25,
    "seed": 106,
}

DATASETS = {
    "titanic": {"R": 1.4, "title": "Titanic (survival)"},
    "magic":   {"R": 1.9, "title": "MAGIC Gamma Telescope (gamma vs hadron)"},
}

METHOD_LABEL = {
    "none":  r"$J = 0$",
    "const": r"constant $J_a$",
    "axial": r"state-dependent $J_s$",
    "axial_nocorr": r"$J_s$, correction dropped",
}

COLOR = {
    r"$J = 0$": "#C44E52",
    r"constant $J_a$": "#4C72B0",
    r"state-dependent $J_s$": "#55A868",
    r"$J_s$, correction dropped": "#8172B2",
}


def _key(tag, **kw):
    blob = json.dumps(kw, sort_keys=True, default=str)
    return "{}_{}".format(tag, hashlib.sha1(blob.encode()).hexdigest()[:12])


def _cache(tag, builder, **kw):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, _key(tag, **kw) + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    out = builder()
    np.savez_compressed(path, **out)
    return out


def build_target(key, cfg=CONFIG):
    from . import data as D
    ds = D.load_titanic() if key == "titanic" else D.load_magic(2000)
    return T.BallConstrainedGibbsSVM(ds, tau=cfg["tau"], lam=cfg["lam"],
                                     delta=cfg["delta"], R=DATASETS[key]["R"])


def reference(key, cfg=CONFIG, verbose=True):
    """Two independent RWM reference samples: one to score against, one to set the
    floor of what 'indistinguishable' means at this sample size."""
    tgt = build_target(key, cfg)

    def build():
        t0 = time.time()
        cov = S.pilot_covariance(tgt, n_chains=300, n_steps=2500, seed=1)
        a, da = S.run_rwm_reference(tgt, n_chains=cfg["ref_chains"], n_steps=cfg["ref_steps"],
                                    cov=cov, seed=2)
        b, db = S.run_rwm_reference(tgt, n_chains=cfg["ref_chains"], n_steps=cfg["ref_steps"],
                                    cov=cov, seed=3)
        return {"ref": a, "ref_b": b, "cov": cov.numpy(),
                "acc": np.array([da["acc_rate"], db["acc_rate"]]),
                "secs": np.array([time.time() - t0])}

    out = _cache("ref_" + key, build, key=key, **{k: cfg[k] for k in
                 ("tau", "lam", "delta", "ref_chains", "ref_steps")}, R=DATASETS[key]["R"])
    ref, ref_b = out["ref"], out["ref_b"]
    floor = G.w1_floor(ref, ref_b, n=cfg["N"], reps=8)
    if verbose:
        print("{:8s} reference: {} samples, RWM acceptance {:.3f}, "
              "boundary mass {:.4f}, W1 floor {:.4f}".format(
                  key, len(ref), float(out["acc"].mean()),
                  G.boundary_mass(ref, tgt.R), floor.mean()))
    return {"target": tgt, "ref": ref, "ref_b": ref_b, "floor": floor,
            "acc": float(out["acc"].mean())}


def chain(key, skew, s, cfg=CONFIG, eta=None, n_steps=None, ref=None, track=True,
          drop_correction=False, seed=None):
    tgt = build_target(key, cfg)
    eta = cfg["eta"] if eta is None else eta
    n_steps = cfg["n_steps"] if n_steps is None else n_steps
    seed = cfg["seed"] if seed is None else seed

    def build():
        x, tr = S.run_anchored_langevin(
            tgt, skew, s, eta=eta, n_steps=n_steps, N=cfg["N"], seed=seed,
            drop_correction=drop_correction, ref=(ref if track else None),
            track_every=cfg["track_every"])
        out = {"x": x}
        if tr is not None:
            out["iters"], out["W1"] = tr
        return out

    return _cache("chain_" + key, build, key=key, skew=skew, s=s, eta=eta,
                  n_steps=n_steps, N=cfg["N"], seed=seed, track=track,
                  drop_correction=drop_correction,
                  **{k: cfg[k] for k in ("tau", "lam", "delta", "track_every")},
                  R=DATASETS[key]["R"])


def score(ref, x, R):
    return G.summarise(ref, x, R)
