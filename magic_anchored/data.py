"""Loading and preprocessing the MAGIC Gamma Telescope data.

The raw file ``magic04.data`` is comma-separated with ten numerical predictors
and a class label in the last column, ``g`` for gamma and ``h`` for hadron.
Everything downstream assumes float64 and a design matrix whose *first* column
is the intercept, because the intercept is treated differently by the target:
it carries a weak Gaussian prior instead of the L1 penalty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DataConfig

__all__ = ["FEATURE_NAMES", "Dataset", "load_magic_data", "preprocess_data"]

#: the ten predictors, in file order, from the UCI description
FEATURE_NAMES: tuple[str, ...] = (
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
)


@dataclass(frozen=True)
class Dataset:
    """Standardised design matrices with an intercept column prepended."""

    X_train: np.ndarray  # (n_train, D)
    y_train: np.ndarray  # (n_train,) in {0, 1}
    X_test: np.ndarray  # (n_test, D)
    y_test: np.ndarray  # (n_test,)
    #: names of the ``D`` parameters, ``("intercept", <feature names>)``
    param_names: tuple[str, ...]
    scaler: StandardScaler

    @property
    def dim(self) -> int:
        """``D``, the parameter dimension (predictors plus intercept)."""
        return self.X_train.shape[1]

    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.X_test.shape[0]

    def summary(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "dim": self.dim,
            "train_positive_rate": float(self.y_train.mean()),
            "test_positive_rate": float(self.y_test.mean()),
            "param_names": list(self.param_names),
        }


def load_magic_data(
    path: str | None = None, cfg: DataConfig | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read ``magic04.data`` and return ``(X_raw, y, feature_names)``.

    ``y`` is 1 for the positive label (``g``, gamma) and 0 for the negative one
    (``h``, hadron).  The label column is read as text and compared after
    stripping, so trailing whitespace in the file is harmless; any label that is
    neither of the two configured strings is an error rather than a silent zero.
    """
    cfg = cfg or DataConfig()
    path = path or cfg.path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MAGIC data not found at {path!r}.  The file is the UCI "
            "'magic04.data', 19020 comma-separated rows of ten predictors and a "
            "g/h label."
        )
    raw = np.genfromtxt(path, delimiter=",", dtype=str, encoding="utf-8")
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"expected a 2-D comma-separated table, got {raw.shape}")

    X_raw = raw[:, :-1].astype(np.float64)
    labels = np.char.strip(raw[:, -1])
    positive = labels == cfg.positive_label
    negative = labels == cfg.negative_label
    unknown = ~(positive | negative)
    if unknown.any():
        bad = sorted(set(labels[unknown].tolist()))
        raise ValueError(
            f"unexpected class labels {bad!r}; expected only "
            f"{cfg.positive_label!r} and {cfg.negative_label!r}"
        )
    y = positive.astype(np.float64)

    n_features = X_raw.shape[1]
    names = (
        FEATURE_NAMES
        if n_features == len(FEATURE_NAMES)
        else tuple(f"x{j}" for j in range(n_features))
    )
    if not np.isfinite(X_raw).all():
        raise ValueError("predictors contain non-finite values")
    return X_raw, y, names


def preprocess_data(
    X_raw: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...] | None = None,
    cfg: DataConfig | None = None,
) -> Dataset:
    """Stratified split, scaler fitted on the *training* predictors only, intercept.

    The order matters and is the usual one: split first, then fit
    :class:`~sklearn.preprocessing.StandardScaler` on the training predictors and
    apply it to both halves, so no test information reaches the scaler.  The
    intercept column of ones is prepended *after* scaling, which keeps it out of
    the standardisation and puts ``beta_0`` at index 0 where the target expects
    it.
    """
    cfg = cfg or DataConfig()
    X_raw = np.ascontiguousarray(X_raw, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if X_raw.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X_raw.shape[0]} rows but y has {y.shape[0]}")

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=y,
    )
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw).astype(np.float64, copy=False)
    X_te = scaler.transform(X_te_raw).astype(np.float64, copy=False)

    X_train = np.column_stack([np.ones(X_tr.shape[0], dtype=np.float64), X_tr])
    X_test = np.column_stack([np.ones(X_te.shape[0], dtype=np.float64), X_te])

    names = feature_names or tuple(f"x{j}" for j in range(X_tr.shape[1]))
    param_names = ("intercept",) + tuple(names)
    if len(param_names) != X_train.shape[1]:
        raise ValueError(
            f"{len(param_names)} parameter names for {X_train.shape[1]} columns"
        )
    return Dataset(
        X_train=np.ascontiguousarray(X_train),
        y_train=np.ascontiguousarray(y_train),
        X_test=np.ascontiguousarray(X_test),
        y_test=np.ascontiguousarray(y_test),
        param_names=param_names,
        scaler=scaler,
    )


def load_and_preprocess(cfg: DataConfig | None = None) -> Dataset:
    """:func:`load_magic_data` followed by :func:`preprocess_data`."""
    cfg = cfg or DataConfig()
    X_raw, y, names = load_magic_data(cfg=cfg)
    return preprocess_data(X_raw, y, names, cfg)
