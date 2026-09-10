"""Binary-classification datasets used by the sampling experiments.

Every loader returns a :class:`Dataset` whose design matrix is standardised and
carries a leading intercept column, so that an identity preconditioner is a
reasonable default for the samplers in :mod:`nds.sampler`.

Raw files are cached under ``data/``.  The canonical homes of these datasets are
the UCI repository and the Kaggle Titanic competition; the URLs below are public
mirrors, because UCI and OpenML are not reachable from every environment.  SHA256
digests of the cached files are recorded in ``data/checksums.txt`` on download so
that a mirror change is visible.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.environ.get(
    "NDS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)

MIRRORS = {
    # Kaggle Titanic training split (891 passengers), via the seaborn-data mirror.
    "titanic.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    # UCI MAGIC Gamma Telescope 2004 (19020 events, 10 features).
    "magic04.data": "https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/magic04.data",
    # UCI Spambase (4601 e-mails, 57 features).
    "spambase.data": "https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/spambase.data",
}


@dataclass
class Dataset:
    """A standardised binary-classification problem with a train/test split."""

    name: str
    X_train: np.ndarray  # (n_train, d), first column is the intercept
    y_train: np.ndarray  # (n_train,) in {0, 1}
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list

    @property
    def dim(self) -> int:
        return self.X_train.shape[1]

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Dataset({self.name}, d={self.dim}, "
            f"n_train={len(self.y_train)}, n_test={len(self.y_test)}, "
            f"train_pos={self.y_train.mean():.3f})"
        )


def _cached(fname: str) -> str:
    """Return the local path of ``fname``, downloading it from its mirror once."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        url = MIRRORS[fname]
        with urllib.request.urlopen(url) as response, open(path, "wb") as handle:
            handle.write(response.read())
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        with open(os.path.join(DATA_DIR, "checksums.txt"), "a") as handle:
            handle.write(f"{digest}  {fname}  {url}\n")
    return path


def _standardise(X: np.ndarray, n_train: int) -> np.ndarray:
    """Centre and scale by training-set moments, then prepend an intercept."""
    mean = X[:n_train].mean(axis=0)
    scale = X[:n_train].std(axis=0)
    scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale
    return np.column_stack([np.ones(len(Z)), Z])


def _split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Dataset:
    """Stratified train/test split, standardised with training-set moments."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        cut = int(round(test_fraction * len(idx)))
        test_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))

    order = np.concatenate([train_idx, test_idx])
    n_train = len(train_idx)
    Xs = _standardise(X[order].astype(np.float64), n_train)
    ys = y[order].astype(np.float64)
    return Dataset(
        name=name,
        X_train=np.ascontiguousarray(Xs[:n_train]),
        y_train=np.ascontiguousarray(ys[:n_train]),
        X_test=np.ascontiguousarray(Xs[n_train:]),
        y_test=np.ascontiguousarray(ys[n_train:]),
        feature_names=["intercept"] + list(feature_names),
    )


def load_titanic(seed: int = 0) -> Dataset:
    """Titanic survival: 891 passengers, 9 engineered features."""
    import pandas as pd

    df = pd.read_csv(_cached("titanic.csv"))
    y = df["survived"].to_numpy()
    age = df["age"].fillna(df["age"].median())
    fare = np.log1p(df["fare"].fillna(df["fare"].median()))
    embarked = df["embarked"].fillna("S")
    columns = {
        "female": (df["sex"] == "female").astype(float),
        "age": age.astype(float),
        "sibsp": df["sibsp"].astype(float),
        "parch": df["parch"].astype(float),
        "log_fare": fare.astype(float),
        "pclass_2": (df["pclass"] == 2).astype(float),
        "pclass_3": (df["pclass"] == 3).astype(float),
        "embarked_C": (embarked == "C").astype(float),
        "embarked_Q": (embarked == "Q").astype(float),
    }
    X = np.column_stack([v.to_numpy() for v in columns.values()])
    return _split("titanic", X, y, list(columns), seed=seed)


def load_magic(seed: int = 0) -> Dataset:
    """MAGIC Gamma Telescope: 19020 events, gamma (1) versus hadron (0)."""
    import pandas as pd

    names = [
        "fLength", "fWidth", "fSize", "fConc", "fConc1",
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class",
    ]
    df = pd.read_csv(_cached("magic04.data"), header=None, names=names)
    y = (df["class"] == "g").to_numpy().astype(float)
    X = df[names[:-1]].to_numpy(dtype=float)
    return _split("magic", X, y, names[:-1], seed=seed)


def load_breast_cancer(seed: int = 0) -> Dataset:
    """Breast Cancer Wisconsin (diagnostic): 569 biopsies, 30 features.

    Shipped with scikit-learn, so this loader needs no network access.  The
    label is flipped to the clinically conventional direction: 1 = malignant.
    """
    from sklearn.datasets import load_breast_cancer as _sk_load

    raw = _sk_load()
    y = 1.0 - raw.target.astype(float)  # scikit-learn codes malignant as 0
    return _split("breast_cancer", raw.data, y, list(raw.feature_names), seed=seed)


def load_spambase(seed: int = 0) -> Dataset:
    """Spambase: 4601 e-mails, 57 heavy-tailed frequency features (log1p-scaled)."""
    raw = np.loadtxt(_cached("spambase.data"), delimiter=",")
    X = np.log1p(raw[:, :-1])
    y = raw[:, -1]
    names = [f"f{i:02d}" for i in range(X.shape[1])]
    return _split("spambase", X, y, names, seed=seed)


LOADERS = {
    "titanic": load_titanic,
    "magic": load_magic,
    "breast_cancer": load_breast_cancer,
    "spambase": load_spambase,
}

PRETTY_NAMES = {
    "titanic": "Titanic (survival)",
    "magic": "MAGIC Gamma Telescope",
    "breast_cancer": "Breast Cancer Wisconsin",
    "spambase": "Spambase",
}


def load(name: str, seed: int = 0) -> Dataset:
    return LOADERS[name](seed=seed)
