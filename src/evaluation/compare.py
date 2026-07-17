"""So sánh kết quả và lựa chọn mô hình tốt nhất."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.settings import (
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.evaluation.metrics import EvaluationResult


def is_better_result(
    candidate: EvaluationResult,
    current_best: EvaluationResult | None,
) -> bool:
    """So sánh theo RMSE, sau đó dùng MAE và R² để phá hòa."""

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
    """Chọn kết quả tốt nhất và báo lỗi nếu danh sách rỗng."""

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
    """Chuyển danh sách kết quả thành bảng để vẽ và xuất báo cáo."""

    records = [result.to_dict() for result in results]

    if not records:
        raise ValueError("Không có kết quả để tạo bảng so sánh.")

    dataframe = pd.DataFrame.from_records(records)
    return dataframe.sort_values(
        by=["scenario", "rmse", "mae", "r2"],
        ascending=[True, True, True, False],
        ignore_index=True,
    )


def build_result_label(scenario: str, model: str) -> str:
    """Tạo nhãn ngắn, dễ đọc cho bảng và biểu đồ."""

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

    return f"{scenario_label} | {model_label}"
