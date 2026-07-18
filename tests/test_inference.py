"""Kiểm thử đầu ra hồi quy của dịch vụ inference."""

from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict_batch import predict_batch  # noqa: E402
from src.inference.predictor import (  # noqa: E402
    PREDICTION_OUTPUT_COLUMNS,
    PredictionService,
)


class DummyRegressor:
    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        values = np.array([-1.0, 12.345, 22.0], dtype=float)
        return values[: len(dataframe)]


def build_service() -> PredictionService:
    service = PredictionService.__new__(PredictionService)
    service.input_features = ["studytime"]
    service.model = DummyRegressor()
    service.best_model = "dummy_regressor"
    service.model_version = "1.0.0"
    service.best_scenario = "A_with_G1_G2"
    return service


def test_prediction_output_contains_only_regression_fields() -> None:
    service = build_service()
    dataframe = pd.DataFrame({"studytime": [1, 2, 3]})

    result = service.predict_frame(dataframe)

    assert tuple(result.columns) == PREDICTION_OUTPUT_COLUMNS
    assert tuple(result.columns) == (
        "predicted_g3",
        "model",
        "model_version",
        "scenario",
    )
    assert result["predicted_g3"].tolist() == [0.0, 12.35, 20.0]


def test_single_prediction_returns_numeric_g3(monkeypatch) -> None:
    service = build_service()
    batch_module = importlib.import_module("src.inference.predict_batch")
    monkeypatch.setattr(
        batch_module,
        "get_prediction_service",
        lambda _mode: service,
    )

    result = predict_batch(
        pd.DataFrame({"studytime": [1, 2]}),
        "with_g1_g2",
        include_input=False,
    )

    assert list(result.columns) == [
        "predicted_g3",
        "model",
        "model_version",
        "scenario",
    ]


def test_predict_single_returns_only_regression_result(monkeypatch) -> None:
    service = build_service()
    single_module = importlib.import_module("src.inference.predict_single")
    monkeypatch.setattr(
        single_module,
        "get_prediction_service",
        lambda _mode: service,
    )

    result = single_module.predict_single(
        {"studytime": 2},
        "with_g1_g2",
    )

    assert result == {
        "predicted_g3": 0.0,
        "model": "dummy_regressor",
        "model_version": "1.0.0",
        "scenario": "A_with_G1_G2",
    }
