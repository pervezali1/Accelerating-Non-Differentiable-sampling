"""Titanic and MAGIC Gamma Telescope, prepared for the constrained sampler.

Both are binary-classification tables, and both are turned into the same object: a
standardised design matrix ``Phi`` (with intercept) and labels ``y`` in ``{-1, +1}``.
Raw files are cached under ``data/`` so a run is reproducible without network access.
"""

import hashlib
import os
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SOURCES = {
    # The 891-row Titanic training split, mirrored by seaborn-data.
    "titanic": ("titanic.csv",
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"),
    # UCI "MAGIC Gamma Telescope" (magic04.data), mirrored by mikeizbicki/datasets
    # because archive.ics.uci.edu is not reachable from the sandbox.
    "magic":   ("magic04.data",
                "https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/magic04.data"),
}

MAGIC_COLUMNS = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
                 "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]


def _cached(name: str) -> str:
    fname, url = SOURCES[name]
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            blob = r.read()
        with open(path, "wb") as f:
            f.write(blob)
    return path


def file_digest(name: str) -> str:
    with open(_cached(name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


class Dataset:
    """Design matrix, labels, and the names needed to read a coordinate plot."""

    def __init__(self, key, title, Phi, y, feature_names):
        self.key, self.title = key, title
        self.Phi, self.y = Phi, y
        self.feature_names = feature_names

    @property
    def n(self):
        return self.Phi.shape[0]

    @property
    def d(self):
        return self.Phi.shape[1]

    def __repr__(self):
        return "Dataset({}, n={}, d={})".format(self.key, self.n, self.d)


def _standardise(X, add_intercept=True, names=None):
    """Zero mean, unit variance; constant columns are dropped.  The intercept is
    prepended afterwards so it stays exactly 1."""
    X = np.asarray(X, dtype=float)
    sd = X.std(axis=0)
    keep = sd > 1e-12
    X, sd = X[:, keep], sd[keep]
    names = [nm for nm, k in zip(names, keep) if k]
    X = (X - X.mean(axis=0)) / sd
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["intercept"] + names
    return X, names


def load_titanic(seed=0):
    """Survival ~ class, sex, age, family size, fare, port of embarkation."""
    df = pd.read_csv(_cached("titanic"))
    df.columns = [c.lower() for c in df.columns]
    df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    feats = pd.DataFrame({
        "pclass_1":  (df["pclass"] == 1).astype(float),
        "pclass_2":  (df["pclass"] == 2).astype(float),
        "female":    (df["sex"] == "female").astype(float),
        "age":       df["age"].astype(float),
        "sibsp":     df["sibsp"].astype(float),
        "parch":     df["parch"].astype(float),
        "log_fare":  np.log1p(df["fare"].astype(float)),
        "emb_C":     (df["embarked"] == "C").astype(float),
        "emb_Q":     (df["embarked"] == "Q").astype(float),
    })
    Phi, names = _standardise(feats.values, names=list(feats.columns))
    y = np.where(df["survived"].values == 1, 1.0, -1.0)
    return Dataset("titanic", "Titanic (survival)", Phi, y, names)


def load_magic(n_max=2000, seed=0):
    """MAGIC Gamma Telescope: gamma (signal, +1) vs hadron (background, -1).

    Sub-sampled, stratified, to ``n_max`` rows -- the full 19 020 rows would make the
    reference chain (which needs ~10^5 sweeps of the exact non-smooth potential)
    dominate the runtime without changing anything about the geometry being studied.
    """
    df = pd.read_csv(_cached("magic"), header=None, names=MAGIC_COLUMNS)
    y_all = np.where(df["class"].values == "g", 1.0, -1.0)
    X_all = df[MAGIC_COLUMNS[:-1]].values.astype(float)
    # the four heavy-tailed size/shape columns are log-scaled before standardising
    for j, nm in enumerate(MAGIC_COLUMNS[:-1]):
        if nm in ("fLength", "fWidth", "fSize", "fDist"):
            X_all[:, j] = np.log(X_all[:, j] + 1e-6)

    if n_max is not None and n_max < len(df):
        rng = np.random.default_rng(seed)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(y_all == c),
                       size=int(round(n_max * np.mean(y_all == c))), replace=False)
            for c in (1.0, -1.0)])
        rng.shuffle(idx)
        X_all, y_all = X_all[idx], y_all[idx]

    Phi, names = _standardise(X_all, names=MAGIC_COLUMNS[:-1])
    return Dataset("magic", "MAGIC Gamma Telescope (gamma vs hadron)", Phi, y_all, names)


def load_all(magic_n=2000, seed=0):
    return {"titanic": load_titanic(seed=seed), "magic": load_magic(n_max=magic_n, seed=seed)}
