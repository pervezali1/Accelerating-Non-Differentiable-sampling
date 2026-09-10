r"""Experiment driver: repeated ensemble runs with bootstrap confidence bands."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import analysis, metrics, samplers, skew

__all__ = [
    "make_prior",
    "log_schedule",
    "make_recorder",
    "run_once",
    "run_repeated",
    "bootstrap_ci",
    "save_json",
    "load_json",
]


def make_prior(kind, d, n, rng):
    """Initial ensemble.  ``normal10`` and ``uniform5`` are the paper's two priors."""
    if kind == "normal10":
        return rng.standard_normal((n, d)) * np.sqrt(10.0)
    if kind == "uniform5":
        return rng.uniform(-5.0, 5.0, size=(n, d))
    if kind == "target":
        raise ValueError("use the target's own sampler for a stationary start")
    raise ValueError(f"unknown prior {kind!r}")


def log_schedule(n_steps, n_points=40):
    """Iterations at which to record, log-spaced (plus 0 and ``n_steps``)."""
    pts = np.unique(np.round(np.geomspace(1, n_steps, n_points)).astype(int))
    return sorted(set([0] + pts.tolist() + [n_steps]))


def make_recorder(target, which=("w2", "w2_mid", "slow", "cov")):
    """Recorder computing the metric suite on the current ensemble."""

    def recorder(x, k):
        out = {}
        if "w2" in which:
            out["w2"] = metrics.axis_sliced_w2(x, target)
        if "w2_mid" in which:
            out["w2_mid"] = metrics.sliced_w2_midpoint(x, target)
        if "slow" in which:
            out["slow"] = metrics.slow_direction_error(x, target)
        if "cov" in which:
            me = metrics.moment_errors(x, target)
            out.update(me)
        return out

    return recorder


def run_once(target, method, eta, n, n_steps, seed, prior="normal10", J=None,
             record_at=None, which=("w2", "w2_mid", "slow", "cov"), gamma=2.0):
    """One ensemble run.  Returns a dict of iteration-indexed metric arrays."""
    rng_init = np.random.default_rng(seed)
    rng_run = np.random.default_rng(seed + 10_000_019)
    x0 = make_prior(prior, target.d, n, rng_init)
    record_at = log_schedule(n_steps) if record_at is None else record_at
    recorder = make_recorder(target, which)
    t0 = time.time()
    if method == "time_changed":
        res = samplers.run_time_changed(target, x0, eta, n_steps, rng_run, J=J,
                                        record_at=record_at, recorder=recorder)
    else:
        res = samplers.run(method, target, x0, eta, n_steps, rng_run, J=J,
                           record_at=record_at, recorder=recorder, gamma=gamma)
    out = {"iters": np.asarray(res.iters), "diverged_at": res.diverged_at,
           "seconds": time.time() - t0}
    for key in res.records[0]:
        out[key] = res.column(key)
    return out


def run_repeated(target, method, eta, n, n_steps, n_rep, prior="normal10", J=None,
                 base_seed=0, which=("w2", "w2_mid", "slow", "cov"), gamma=2.0,
                 record_at=None):
    """``n_rep`` independent replications, aligned on a common record schedule.

    Divergent replications are counted and reported, never silently dropped: the
    returned dict carries ``n_diverged`` and the aggregate statistics are taken
    over the replications that stayed finite.
    """
    record_at = log_schedule(n_steps) if record_at is None else record_at
    runs, n_div, div_iters = [], 0, []
    for r in range(n_rep):
        out = run_once(target, method, eta, n, n_steps, base_seed + 1000 * r,
                       prior=prior, J=J, record_at=record_at, which=which, gamma=gamma)
        if out["diverged_at"] is not None:
            n_div += 1
            div_iters.append(out["diverged_at"])
            continue
        runs.append(out)
    agg = {"method": method, "eta": float(eta), "n": int(n), "n_steps": int(n_steps),
           "n_rep": int(n_rep), "n_diverged": int(n_div), "prior": prior,
           "diverged_at": div_iters,
           "J_norm": 0.0 if J is None else float(np.linalg.norm(J, 2))}
    if not runs:
        agg["iters"] = []
        return agg
    agg["iters"] = runs[0]["iters"].tolist()
    agg["seconds"] = float(np.mean([r["seconds"] for r in runs]))
    for key in runs[0]:
        if key in ("iters", "diverged_at", "seconds"):
            continue
        stack = np.array([r[key] for r in runs], dtype=np.float64)
        agg[key] = {
            "mean": np.nanmean(stack, axis=0).tolist(),
            "median": np.nanmedian(stack, axis=0).tolist(),
            "lo": np.nanpercentile(stack, 2.5, axis=0).tolist(),
            "hi": np.nanpercentile(stack, 97.5, axis=0).tolist(),
            "raw": stack.tolist(),
        }
    return agg


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, default=default)
    return path


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
