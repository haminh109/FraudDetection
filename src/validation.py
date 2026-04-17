from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_MODEL_ARTIFACT_KEYS = {"model_name", "model", "threshold"}


def validate_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    required_columns: Iterable[str] | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{dataset_name} must be a pandas DataFrame.")

    if not allow_empty and df.empty:
        raise ValueError(f"{dataset_name} must not be empty.")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            f"{dataset_name} contains duplicate columns: {duplicate_columns}"
        )

    if required_columns:
        missing_columns = sorted(set(required_columns) - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing_columns}"
            )

    return df


def validate_binary_target(
    target: pd.Series | np.ndarray | list[Any],
    dataset_name: str = "dataset",
    target_name: str = "target",
) -> pd.Series:
    series = pd.Series(target, copy=False)

    if series.empty:
        raise ValueError(f"{dataset_name} {target_name} must not be empty.")

    if series.isna().any():
        raise ValueError(f"{dataset_name} {target_name} contains missing values.")

    unique_values = set(pd.unique(series))
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name} {target_name} must be binary with values 0/1."
        )

    return series


def validate_feature_matrix(
    df: pd.DataFrame,
    dataset_name: str = "feature matrix",
    require_finite_numeric: bool = True,
) -> pd.DataFrame:
    validate_dataframe(df=df, dataset_name=dataset_name)

    if require_finite_numeric:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and not np.isfinite(numeric_df.to_numpy()).all():
            raise ValueError(
                f"{dataset_name} contains non-finite numeric values."
            )

    return df


def validate_train_validation_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series | np.ndarray | list[Any],
    y_val: pd.Series | np.ndarray | list[Any],
) -> None:
    validate_feature_matrix(X_train, dataset_name="X_train")
    validate_feature_matrix(X_val, dataset_name="X_val")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("X_train and X_val must have identical column order.")

    y_train_series = validate_binary_target(
        y_train,
        dataset_name="train",
        target_name="y",
    )
    y_val_series = validate_binary_target(
        y_val,
        dataset_name="validation",
        target_name="y",
    )

    if len(X_train) != len(y_train_series):
        raise ValueError("X_train and y_train must have the same number of rows.")

    if len(X_val) != len(y_val_series):
        raise ValueError("X_val and y_val must have the same number of rows.")


def validate_model_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise TypeError("Model artifact must be a dictionary.")

    missing_keys = REQUIRED_MODEL_ARTIFACT_KEYS - set(artifact.keys())
    if missing_keys:
        raise ValueError(f"Model artifact missing keys: {sorted(missing_keys)}")

    threshold = float(artifact["threshold"])
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Model artifact threshold must be between 0 and 1.")

    model = artifact["model"]
    if not hasattr(model, "predict_proba") and not hasattr(
        model,
        "decision_function",
    ):
        raise ValueError(
            "Model artifact must provide predict_proba or decision_function."
        )

    return artifact
