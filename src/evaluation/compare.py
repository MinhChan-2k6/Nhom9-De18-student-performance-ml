"""
So sánh kết quả và lựa chọn mô hình tốt nhất.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.settings import (
    FEATURE_SET_ENGINEERED,
    FEATURE_SET_RAW,
    FEATURE_SETS,
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.evaluation.metrics import EvaluationResult


def is_better_result(
    candidate: EvaluationResult,
    current_best: EvaluationResult | None,
) -> bool:
    if current_best is None:
        return True

    if candidate.rmse < current_best.rmse:
        return True

    if np.isclose(candidate.rmse, current_best.rmse):
        if candidate.mae < current_best.mae:
            return True

        if np.isclose(candidate.mae, current_best.mae):
            return candidate.r2 > current_best.r2

    return False


def select_best_result(
    results: Iterable[EvaluationResult],
) -> EvaluationResult:
    best_result: EvaluationResult | None = None

    for result in results:
        if is_better_result(result, best_result):
            best_result = result

    if best_result is None:
        raise ValueError("Danh sách kết quả đánh giá đang rỗng.")

    return best_result


def results_to_dataframe(
    results: Iterable[EvaluationResult],
) -> pd.DataFrame:
    records = [result.to_dict() for result in results]

    if not records:
        raise ValueError("Không có kết quả để tạo bảng so sánh.")

    dataframe = pd.DataFrame.from_records(records)
    return dataframe.sort_values(
        by=["scenario", "feature_set", "rmse", "mae", "r2"],
        ascending=[True, True, True, True, False],
        ignore_index=True,
    )


def summarize_feature_set_results(
    results: Iterable[EvaluationResult],
) -> dict[str, dict[str, object]]:
    """Tóm tắt model tốt nhất của từng feature set trên validation."""

    result_list = list(results)
    summary: dict[str, dict[str, object]] = {}

    for feature_set in FEATURE_SETS:
        candidates = [
            result
            for result in result_list
            if result.feature_set == feature_set
        ]
        if not candidates:
            continue

        best_result = select_best_result(candidates)
        summary[feature_set] = {
            "best_model": best_result.model,
            **best_result.metrics.to_dict(),
        }

    return summary


def build_result_label(
    scenario: str,
    model: str,
    feature_set: str = FEATURE_SET_RAW,
) -> str:
    scenario_label = (
        "Có G1, G2"
        if scenario == SCENARIO_WITH_PREVIOUS_GRADES
        else (
            "Không G1, G2"
            if scenario == SCENARIO_WITHOUT_PREVIOUS_GRADES
            else scenario
        )
    )

    model_label = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "random_forest_grid_search": "RF Grid Search",
    }.get(model, model.replace("_", " ").title())

    feature_set_label = {
        FEATURE_SET_RAW: "Đặc trưng gốc",
        FEATURE_SET_ENGINEERED: "Có feature engineering",
    }.get(feature_set, feature_set)

    return f"{scenario_label} | {feature_set_label} | {model_label}"
