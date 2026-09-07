"""Distances to the reference sample, and the boundary statistic."""

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def w1_per_coord(ref, x):
    return np.array([wasserstein_distance(ref[:, i], x[:, i]) for i in range(ref.shape[1])])


def max_ks(ref, x):
    return float(max(ks_2samp(ref[:, i], x[:, i]).statistic for i in range(ref.shape[1])))


def boundary_mass(x, R, tol=0.999):
    """Fraction of the sample sitting on (numerically, within ``tol`` of) ``dK``.

    Projection deposits an atom there that the continuous target does not have; how big
    it is separates discretisation error from a wrong boundary condition."""
    return float(np.mean(np.linalg.norm(x, axis=1) > tol * R))


def w1_floor(ref, ref_b, n=None, reps=8, seed=0):
    """The resolution limit of the comparison: W1 between two independent draws of the
    same size from the reference, per coordinate.  A method at the floor is
    indistinguishable from the truth at this sample size."""
    rng = np.random.default_rng(seed)
    n = len(ref) if n is None else n
    out = []
    for _ in range(reps):
        a = ref[rng.choice(len(ref), n, replace=False)]
        b = ref_b[rng.choice(len(ref_b), n, replace=False)]
        out.append(w1_per_coord(a, b))
    return np.mean(out, axis=0)


def summarise(ref, x, R):
    return {"W1": w1_per_coord(ref, x), "W1_mean": float(w1_per_coord(ref, x).mean()),
            "maxKS": max_ks(ref, x), "boundary": boundary_mass(x, R)}
