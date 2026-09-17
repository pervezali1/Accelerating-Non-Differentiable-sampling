"""Section 1 and automatic check 14: preprocessing and label encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pnral.data import (MAGIC_FEATURE_NAMES, encode_labels, load_raw_magic,
                        make_synthetic_magic_like, preprocess)


def test_label_encoding_is_g_to_one_and_h_to_zero():
    labels = pd.Series(["g", "h", "g", " h "])
    encoded = encode_labels(labels)
    assert encoded.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert encoded.dtype == np.float64


def test_unknown_class_symbol_raises():
    with pytest.raises(ValueError, match="unexpected class symbols"):
        encode_labels(pd.Series(["g", "x"]))


def test_missing_file_message_points_at_uci(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        load_raw_magic(tmp_path / "magic04.data")
    assert "archive.ics.uci.edu" in str(info.value)
    assert "--synthetic" in str(info.value)


def test_dimension_is_eleven_and_intercept_is_first(data):
    assert data.d == 11 == 1 + len(MAGIC_FEATURE_NAMES)
    assert np.all(data.X_train[:, 0] == 1.0)
    assert np.all(data.X_test[:, 0] == 1.0)
    assert data.X_train.dtype == np.float64 and data.y_train.dtype == np.float64


def test_scaler_is_fitted_on_training_rows_only(data):
    # standardised training predictors have zero mean / unit variance
    predictors = data.X_train[:, 1:]
    assert np.allclose(predictors.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(predictors.std(axis=0), 1.0, atol=1e-10)
    # the test split is transformed, not re-fitted: its moments differ
    assert not np.allclose(data.X_test[:, 1:].mean(axis=0), 0.0, atol=1e-10)


def test_split_is_stratified():
    frame = make_synthetic_magic_like(4000, seed=3)
    processed = preprocess(frame, test_size=0.2, split_seed=1)
    assert processed.n_test == 800
    assert abs(processed.y_train.mean() - processed.y_test.mean()) < 0.02
