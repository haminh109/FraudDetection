import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.inference import build_output, get_probabilities, prepare_features


def test_prepare_features_aligns_expected_columns_and_drops_extras():
    artifact = {
        "model_name": "demo_model",
        "model": object(),
        "threshold": 0.4,
        "feature_name_mapping": {"feature raw": "feature_raw"},
        "feature_names": ["feature_raw", "feature_b"],
    }
    df = pd.DataFrame(
        {
            "feature raw": [1.5],
            "unexpected": [999],
        }
    )

    prepared = prepare_features(df, artifact)

    assert list(prepared.columns) == ["feature_raw", "feature_b"]
    assert prepared.loc[0, "feature_raw"] == np.float32(1.5)
    assert prepared.loc[0, "feature_b"] == np.float32(0.0)


def test_get_probabilities_supports_predict_proba_models():
    X = pd.DataFrame({"feature_a": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)

    probabilities = get_probabilities(model, X)

    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_get_probabilities_supports_decision_function_models():
    X = pd.DataFrame({"feature_a": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([0, 0, 1, 1])
    model = LinearSVC().fit(X, y)

    probabilities = get_probabilities(model, X)

    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_build_output_adds_prediction_columns():
    df = pd.DataFrame({"feature_a": [1.0, 2.0]})
    probabilities = np.array([0.2, 0.9])

    output = build_output(
        df_input=df,
        probabilities=probabilities,
        threshold=0.5,
        model_name="demo_model",
    )

    assert output["prediction"].tolist() == [0, 1]
    assert output["model_name"].tolist() == ["demo_model", "demo_model"]
