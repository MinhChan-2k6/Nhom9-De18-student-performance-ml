"""Tính toán và đóng gói các chỉ số đánh giá mô hình hồi quy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RegressionMetrics:
    """Ba chỉ số chính dùng để đánh giá bài toán hồi quy."""

    mae: float
    rmse: float
    r2: float

    def to_dict(self, ndigits: int = 6) -> dict[str, float]:
        """Chuyển metric thành dictionary để lưu JSON."""

        return {
            "mae": round(self.mae, ndigits),
            "rmse": round(self.rmse, ndigits),
            "r2": round(self.r2, ndigits),
        }


@dataclass(frozen=True)
class PassFailMetrics:
    """Chỉ số khi quy đổi điểm G3 thành Đạt/Trượt."""

    threshold: float
    accuracy: float
    f1: float
    confusion_matrix: list[list[int]]

    def to_dict(self, ndigits: int = 6) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, ndigits),
            "f1": round(self.f1, ndigits),
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class EvaluationResult:
    """Kết quả của một model trong một scenario trên validation set."""

    scenario: str
    model: str
    metrics: RegressionMetrics
    estimator: Pipeline

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
        """Dữ liệu gọn để ghi vào metrics.json."""

        return {
            "scenario": self.scenario,
            "model": self.model,
            **self.metrics.to_dict(),
        }


def _as_1d_array(values: pd.Series | np.ndarray) -> np.ndarray:
    """Chuẩn hóa dữ liệu đầu vào về mảng một chiều."""

    array = np.asarray(values, dtype=float).reshape(-1)

    if array.size == 0:
        raise ValueError("Không thể đánh giá trên tập dữ liệu rỗng.")

    if not np.isfinite(array).all():
        raise ValueError("Dữ liệu đánh giá chứa NaN hoặc giá trị vô hạn.")

    return array


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> RegressionMetrics:
    """Tính MAE, RMSE và R² sau khi kiểm tra dữ liệu đầu vào."""

    true_values = _as_1d_array(y_true)
    predicted_values = _as_1d_array(y_pred)

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


def calculate_pass_fail_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    threshold: float = 10.0,
) -> PassFailMetrics:
    """Quy đổi điểm sang Đạt/Trượt và tính Accuracy, F1."""

    true_values = _as_1d_array(y_true)
    predicted_values = _as_1d_array(y_pred)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "Số lượng giá trị thực và dự đoán không bằng nhau: "
            f"{true_values.size} != {predicted_values.size}."
        )

    true_labels = (true_values >= threshold).astype(int)
    predicted_labels = (predicted_values >= threshold).astype(int)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    return PassFailMetrics(
        threshold=threshold,
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        f1=float(f1_score(true_labels, predicted_labels, zero_division=0)),
        confusion_matrix=matrix.astype(int).tolist(),
    )



def predict_and_evaluate(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, RegressionMetrics]:
    """Dự đoán bằng estimator và trả về cả prediction lẫn metric."""

    predictions = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    metrics = calculate_metrics(y, predictions)
    return predictions, metrics


def evaluate_pipeline(
    estimator: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scenario_name: str,
    model_name: str,
) -> EvaluationResult:
    """Đánh giá một pipeline trên validation set."""

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
    )
