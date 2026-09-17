"""Loading and preprocessing of the MAGIC Gamma Telescope data set.

Pipeline (section 1 of the specification)

1. read ``magic04.data`` (10 numerical predictors + one ``g``/``h`` label);
2. encode ``g -> 1`` (gamma / signal) and ``h -> 0`` (hadron / background);
3. stratified 80/20 train/test split;
4. ``StandardScaler`` fitted on the *training* predictors only and applied to
   both splits;
5. an intercept column of ones appended **after** standardisation.

The parameter vector is therefore ``w = (intercept, beta_1, ..., beta_10)``
and ``d`` is read off the processed design matrix -- it is never hard coded.
Everything is ``float64``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__all__ = [
    "MAGIC_FEATURE_NAMES",
    "MAGIC_COLUMN_NAMES",
    "ProcessedData",
    "load_raw_magic",
    "encode_labels",
    "preprocess",
    "make_synthetic_magic_like",
    "save_preprocessing",
]

#: The ten numerical predictors, in the order used by the UCI file.
MAGIC_FEATURE_NAMES: List[str] = [
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
]

#: Predictors plus the class column of ``magic04.data``.
MAGIC_COLUMN_NAMES: List[str] = MAGIC_FEATURE_NAMES + ["class"]

#: Label encoding fixed by the specification.
LABEL_ENCODING = {"g": 1, "h": 0}

_MISSING_FILE_MESSAGE = """\
Could not find the MAGIC Gamma Telescope file at:

    {path}

This experiment deliberately never substitutes data silently.  Download
``magic04.data`` from the UCI Machine Learning Repository

    https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

and place it at the path above (or pass --data-path).  For a structural
smoke-test of the *code* only -- not for any scientific claim -- pass
--synthetic, which generates a clearly-labelled surrogate with the same
shape (19020 x 10 predictors, g/h labels).
"""


@dataclass
class ProcessedData:
    """Design matrices and labels ready for the sampler.

    Attributes
    ----------
    X_train, X_test:
        ``float64`` design matrices with the intercept column **first**,
        shape ``(n, d)`` with ``d = 1 + n_features``.
    y_train, y_test:
        ``float64`` label vectors in ``{0.0, 1.0}`` (``g -> 1``, ``h -> 0``).
    parameter_names:
        ``["intercept", "fLength", ...]`` aligned with the columns of ``X``.
    scaler:
        the :class:`~sklearn.preprocessing.StandardScaler` fitted on the
        training predictors only.
    is_synthetic:
        ``True`` only for the opt-in surrogate data set.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    parameter_names: List[str]
    feature_names: List[str]
    scaler: StandardScaler
    is_synthetic: bool = False
    source_path: Optional[str] = None

    @property
    def d(self) -> int:
        """Dimension of the parameter vector, read from the design matrix."""
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    def summary(self) -> dict:
        """Small JSON-serialisable description of the processed data."""
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "d": self.d,
            "parameter_names": list(self.parameter_names),
            "train_positive_fraction": float(self.y_train.mean()),
            "test_positive_fraction": float(self.y_test.mean()),
            "is_synthetic": bool(self.is_synthetic),
            "source_path": self.source_path,
            "dtype": str(self.X_train.dtype),
        }


def load_raw_magic(path: str | Path) -> pd.DataFrame:
    """Read ``magic04.data`` into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Location of the comma-separated, header-less UCI file.

    Raises
    ------
    FileNotFoundError
        If the file is absent.  The message explains where to obtain it;
        no fallback data set is ever used implicitly.
    ValueError
        If the file does not have the expected 11 columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_MISSING_FILE_MESSAGE.format(path=path.resolve()))

    frame = pd.read_csv(path, header=None, names=MAGIC_COLUMN_NAMES)
    if frame.shape[1] != len(MAGIC_COLUMN_NAMES):
        raise ValueError(
            f"expected {len(MAGIC_COLUMN_NAMES)} columns in {path}, "
            f"found {frame.shape[1]}"
        )
    frame[MAGIC_FEATURE_NAMES] = frame[MAGIC_FEATURE_NAMES].astype(np.float64)
    frame["class"] = frame["class"].astype(str).str.strip()
    return frame


def encode_labels(raw_labels: "pd.Series | np.ndarray") -> np.ndarray:
    """Encode the MAGIC class column as ``g -> 1`` and ``h -> 0``.

    The mapping is validated: any symbol outside ``{"g", "h"}`` raises, so a
    silently mis-encoded response cannot propagate into the posterior.
    """
    series = pd.Series(np.asarray(raw_labels).ravel()).astype(str).str.strip()
    unknown = sorted(set(series.unique()) - set(LABEL_ENCODING))
    if unknown:
        raise ValueError(f"unexpected class symbols {unknown}; expected 'g'/'h'")
    return series.map(LABEL_ENCODING).to_numpy(dtype=np.float64)


def _add_intercept(matrix: np.ndarray) -> np.ndarray:
    """Prepend a column of ones; the intercept is coordinate 0 of ``w``."""
    ones = np.ones((matrix.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, matrix]).astype(np.float64, copy=False)


def preprocess(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    split_seed: int = 20240917,
    is_synthetic: bool = False,
    source_path: Optional[str] = None,
) -> ProcessedData:
    """Encode, split, standardise and add the intercept column.

    The scaler is fitted on the training predictors only -- the test split
    never influences the preprocessing, the posterior or the step size.
    """
    predictors = frame[MAGIC_FEATURE_NAMES].to_numpy(dtype=np.float64)
    labels = encode_labels(frame["class"])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        predictors,
        labels,
        test_size=test_size,
        random_state=split_seed,
        stratify=labels,          # stratified 80/20 split
        shuffle=True,
    )

    scaler = StandardScaler().fit(X_train_raw)          # training rows only
    X_train = _add_intercept(scaler.transform(X_train_raw))
    X_test = _add_intercept(scaler.transform(X_test_raw))

    return ProcessedData(
        X_train=np.ascontiguousarray(X_train, dtype=np.float64),
        X_test=np.ascontiguousarray(X_test, dtype=np.float64),
        y_train=np.ascontiguousarray(y_train, dtype=np.float64),
        y_test=np.ascontiguousarray(y_test, dtype=np.float64),
        parameter_names=["intercept"] + list(MAGIC_FEATURE_NAMES),
        feature_names=list(MAGIC_FEATURE_NAMES),
        scaler=scaler,
        is_synthetic=is_synthetic,
        source_path=source_path,
    )


def make_synthetic_magic_like(
    n_rows: int = 19020,
    seed: int = 7,
    positive_fraction: float = 0.648,
) -> pd.DataFrame:
    """Generate a **surrogate** data set with the shape of ``magic04.data``.

    This exists solely so that the code path can be exercised end-to-end when
    the real UCI file is unavailable.  It is *not* the MAGIC data set and no
    scientific conclusion may be drawn from it; every artefact produced from
    it is tagged ``synthetic``.

    The surrogate mimics the coarse structure of the real file: ten strictly
    positive, right-skewed predictors, a class-imbalanced ``g``/``h`` label
    (about 65% gamma) and a moderately separable signal.
    """
    rng = np.random.default_rng(seed)
    n_gamma = int(round(n_rows * positive_fraction))
    n_hadron = n_rows - n_gamma

    # Log-normal predictors with class dependent location -> skewed, positive.
    base_mu = np.array([3.3, 2.6, 1.0, -1.2, -1.5, 1.5, 1.6, 1.2, 2.3, 5.4])
    shift = np.array([-0.25, -0.35, -0.10, 0.22, 0.20, 0.05, -0.15, 0.05, -0.95, 0.05])
    sigma = np.array([0.55, 0.60, 0.35, 0.45, 0.45, 0.70, 0.65, 0.60, 0.85, 0.30])

    def _draw(count: int, mu: np.ndarray) -> np.ndarray:
        latent = rng.standard_normal((count, mu.size))
        # A shared factor induces realistic correlation between predictors.
        latent += 0.45 * rng.standard_normal((count, 1))
        return np.exp(mu + sigma * latent)

    gamma = _draw(n_gamma, base_mu + shift)
    hadron = _draw(n_hadron, base_mu)
    predictors = np.vstack([gamma, hadron])
    labels = np.array(["g"] * n_gamma + ["h"] * n_hadron, dtype=object)

    order = rng.permutation(n_rows)
    frame = pd.DataFrame(predictors[order], columns=MAGIC_FEATURE_NAMES)
    frame["class"] = labels[order]
    return frame


def save_preprocessing(data: ProcessedData, output_dir: str | Path) -> dict:
    """Persist the fitted scaler (joblib) and its parameters (JSON).

    Returns the JSON-serialisable dictionary that was written.
    """
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.scaler, output_dir / "standard_scaler.joblib")

    payload = {
        "feature_names": list(data.feature_names),
        "parameter_names": list(data.parameter_names),
        "scaler_mean": data.scaler.mean_.tolist(),
        "scaler_scale": data.scaler.scale_.tolist(),
        "scaler_var": data.scaler.var_.tolist(),
        "n_features_in": int(data.scaler.n_features_in_),
        "label_encoding": LABEL_ENCODING,
        "intercept_column": 0,
        **data.summary(),
    }
    with (output_dir / "preprocessing.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def load_dataset(
    data_path: str | Path,
    test_size: float = 0.20,
    split_seed: int = 20240917,
    use_synthetic: bool = False,
    synthetic_n_rows: int = 19020,
    synthetic_seed: int = 7,
) -> ProcessedData:
    """Convenience wrapper: read (or synthesise) and preprocess in one call."""
    if use_synthetic:
        frame = make_synthetic_magic_like(synthetic_n_rows, synthetic_seed)
        return preprocess(frame, test_size, split_seed,
                          is_synthetic=True, source_path="<synthetic surrogate>")
    frame = load_raw_magic(data_path)
    return preprocess(frame, test_size, split_seed,
                      is_synthetic=False, source_path=str(data_path))
