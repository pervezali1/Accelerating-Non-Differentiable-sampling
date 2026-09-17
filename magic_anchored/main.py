#!/usr/bin/env python3
"""Run the whole experiment, from the raw data file to the figures.

    python3 main.py --data ../data/magic04.data --out-dir outputs

The defaults are the ones reported in ``outputs/summary.txt``.  ``--quick``
shrinks every chain so the full pipeline runs in well under a minute, which is
the right setting for checking that an environment works before committing to
the real run.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    FieldConfig,
    PilotConfig,
    TargetConfig,
)
from experiment import run_experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data/magic04.data",
                   help="path to the raw magic04.data file")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0)

    p.add_argument("--lambda-scale", type=float, default=0.01,
                   help="lambda = lambda_scale * n_train; the likelihood is a sum "
                        "over observations, so the penalty has to scale with n")
    p.add_argument("--lam", type=float, default=None,
                   help="an explicit lambda, overriding --lambda-scale")
    p.add_argument("--sigma0", type=float, default=10.0,
                   help="standard deviation of the weak Gaussian intercept prior")
    p.add_argument("--delta", type=float, default=None,
                   help="anchor smoothing; omitted means calibrate it (delta enters "
                        "U0 only, so it changes the algorithm and not the target)")
    p.add_argument("--tempered", action="store_true",
                   help="EXPLICIT TEMPERED POSTERIOR: divide the log-likelihood by "
                        "n_train.  This is a different target and is off by default")

    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 1.0, 2.0, 10.0, 50.0, 200.0],
                   help="the first five are the values named in the specification; "
                        "J_s is linear in q = w - w_center, so on this posterior "
                        "(R ~ 0.19, m = 6 blocks) they are all a <=3%% perturbation, "
                        "and the last three reach alpha* ~ 13 where the rotation "
                        "matches the gradient")
    p.add_argument("--s", type=float, default=1.0,
                   help="field amplitude; only the product alpha*s matters, so this "
                        "stays at 1 and alpha is the knob")
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--n-chains-sensitivity", type=int, default=2)
    p.add_argument("--n-iter", type=int, default=60_000)
    p.add_argument("--burn-in", type=int, default=10_000)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--base-seed", type=int, default=1_000)

    p.add_argument("--h", type=float, default=None,
                   help="step size; omitted means h = step_safety / lambda_max")
    p.add_argument("--step-safety", type=float, default=0.25)
    p.add_argument("--fixed-radius", type=float, default=None,
                   help="bypass the pilot and use this radius")
    p.add_argument("--pilot-iter", type=int, default=40_000)
    p.add_argument("--pilot-quantile", type=float, default=0.999)
    p.add_argument("--pilot-inflation", type=float, default=1.10)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="tiny chains, for checking the pipeline runs")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.n_iter, args.burn_in, args.thin = 4_000, 1_000, 4
        args.pilot_iter, args.n_chains, args.n_chains_sensitivity = 4_000, 2, 2
        args.alphas = [0.0, 1.0]

    data_cfg = DataConfig(path=args.data, test_size=args.test_size,
                          split_seed=args.split_seed)
    target_cfg = TargetConfig(lambda_=args.lam, lambda_scale=args.lambda_scale,
                              sigma0=args.sigma0, delta=args.delta,
                              tempered=args.tempered)
    field_cfg = FieldConfig(s=args.s)
    pilot_cfg = PilotConfig(n_iter=args.pilot_iter,
                            burn_in=max(args.pilot_iter // 4, 1),
                            quantile=args.pilot_quantile,
                            inflation=args.pilot_inflation)
    cfg = ExperimentConfig(
        alphas=tuple(args.alphas), n_chains=args.n_chains,
        n_chains_sensitivity=args.n_chains_sensitivity,
        base_seed=args.base_seed, h=args.h, step_safety=args.step_safety,
        fixed_radius=args.fixed_radius, n_iter=args.n_iter,
        burn_in=args.burn_in, thin=args.thin, out_dir=args.out_dir,
        make_figures=not args.no_figures,
    )
    if args.tempered:
        print("WARNING: --tempered is enabled; the log-likelihood is divided by "
              "n_train, which is a DIFFERENT target from the one this experiment "
              "otherwise reports.", flush=True)

    out = run_experiment(data_cfg, target_cfg, field_cfg, pilot_cfg, cfg)
    print(open(os.path.join(out.out_dir, "summary.txt")).read())
    return 0 if out.validation["n_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
