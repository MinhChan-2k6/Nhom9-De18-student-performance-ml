"""
Tính toán và đóng gói các chỉ số đánh giá mô hình hồi quy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config.settings import FEATURE_SET_RAW
from src.utils.helpers import as_finite_1d_array


@dataclass(frozen=True)
class RegressionMetrics:

    mae: float
    rmse: float
    r2: float

    def to_dict(self, ndigits: int = 6) -> dict[str, float]:

        return {
            "mae": round(self.mae, ndigits),
            "rmse": round(self.rmse, ndigits),
            "r2": round(self.r2, ndigits),
        }


@dataclass
class EvaluationResult:

    scenario: str
    model: str
    metrics: RegressionMetrics
    estimator: Pipeline
    feature_set: str = FEATURE_SET_RAW
    tuning: dict[str, Any] | None = None

    @property
    def mae(self) -> float:
        return self.metrics.mae

    @property
    def rmse(self) -> float:
        return self.metrics.rmse

    @property
    def r2(self) -> float:
        return self.metrics.r2

    def to_dict(self) -> dict[str, Any]:

        payload: dict[str, Any] = {
            "scenario": self.scenario,
            "model": self.model,
            "feature_set": self.feature_set,
            **self.metrics.to_dict(),
        }

        if self.tuning is not None:
            payload["tuning"] = self.tuning

        return payload


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> RegressionMetrics:

    true_values = as_finite_1d_array(y_true, name="Giá trị G3 thực tế")
    predicted_values = as_finite_1d_array(y_pred, name="Giá trị G3 dự đoán")

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "Số lượng giá trị thực và dự đoán không bằng nhau: "
            f"{true_values.size} != {predicted_values.size}."
        )

    return RegressionMetrics(
        mae=float(mean_absolute_error(true_values, predicted_values)),
        rmse=float(mean_squared_error(true_values, predicted_values) ** 0.5),
        r2=float(r2_score(true_values, predicted_values)),
    )


def predict_and_evaluate(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, RegressionMetrics]:

    predictions = as_finite_1d_array(
        estimator.predict(X),
        name="Kết quả dụ đoán của model",
    )
    metrics = calculate_metrics(y, predictions)
    return predictions, metrics


def evaluate_pipeline(
    estimator: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scenario_name: str,
    model_name: str,
    feature_set: str = FEATURE_SET_RAW,
    tuning: dict[str, Any] | None = None,
) -> EvaluationResult:

    _, metrics = predict_and_evaluate(
        estimator=estimator,
        X=X_valid,
        y=y_valid,
    )

    return EvaluationResult(
        scenario=scenario_name,
        model=model_name,
        metrics=metrics,
        estimator=estimator,
        feature_set=feature_set,
        tuning=tuning,
    )
