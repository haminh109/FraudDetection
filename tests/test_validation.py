import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.validation import (
    validate_binary_target,
    validate_dataframe,
    validate_feature_matrix,
    validate_model_artifact,
    validate_train_validation_inputs,
)


def test_validate_dataframe_rejects_missing_required_columns():
    df = pd.DataFrame({"feature_a": [1, 2]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(
            df,
            dataset_name="train dataframe",
            required_columns=["feature_a", "isFraud"],
        )


def test_validate_binary_target_rejects_non_binary_values():
    with pytest.raises(ValueError, match="must be binary"):
        validate_binary_target([0, 1, 2], dataset_name="train", target_name="isFraud")


def test_validate_feature_matrix_rejects_non_finite_numeric_values():
    df = pd.DataFrame({"feature_a": [1.0, np.inf]})

    with pytest.raises(ValueError, match="non-finite numeric values"):
        validate_feature_matrix(df, dataset_name="X_train")


def test_validate_train_validation_inputs_rejects_column_mismatch():
    X_train = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})
    X_val = pd.DataFrame({"feature_a": [1.0], "feature_c": [2.0]})

    with pytest.raises(ValueError, match="identical column order"):
        validate_train_validation_inputs(X_train, X_val, [0], [1])


def test_validate_model_artifact_accepts_probability_model():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)

    artifact = {
        "model_name": "demo_model",
        "model": model,
        "threshold": 0.5,
    }

    validated = validate_model_artifact(artifact)

    assert validated["model_name"] == "demo_model"
