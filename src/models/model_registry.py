"""Danh sách model và tiện ích tạo pipeline huấn luyện."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from src.models.baseline_linear import build_linear_regression
from src.models.gradient_boosting import build_gradient_boosting
from src.models.random_forest import build_random_forest


def build_candidate_models() -> dict[str, Any]:
    """Tạo các model cơ sở được dùng để so sánh."""

    return {
        "linear_regression": build_linear_regression(),
        "random_forest": build_random_forest(),
        "gradient_boosting": build_gradient_boosting(),
    }


def build_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
    feature_engineer: TransformerMixin | None = None,
) -> Pipeline:
    """Gộp feature engineering, tiền xử lý và model thành Pipeline."""

    steps: list[tuple[str, Any]] = []

    if feature_engineer is not None:
        steps.append(("feature_engineering", feature_engineer))

    steps.extend(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return Pipeline(steps=steps)
