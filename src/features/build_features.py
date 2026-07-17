from __future__ import annotations

import pandas as pd

from src.config.settings import (
    EARLY_PREDICTION_EXCLUDED_COLUMNS,
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
    TARGET_COLUMN,
)


def get_scenario_features(
    dataframe: pd.DataFrame,
    scenario_name: str,
) -> list[str]:
    """Trả về danh sách cột đầu vào tương ứng với từng scenario.

    - Scenario A: sử dụng toàn bộ đặc trưng, bao gồm G1 và G2.
    - Scenario B: loại G1 và G2 để phục vụ dự báo sớm.
    """

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} trong dữ liệu."
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    if scenario_name == SCENARIO_WITH_PREVIOUS_GRADES:
        return feature_columns

    if scenario_name == SCENARIO_WITHOUT_PREVIOUS_GRADES:
        return [
            column
            for column in feature_columns
            if column not in EARLY_PREDICTION_EXCLUDED_COLUMNS
        ]

    raise ValueError(f"Scenario không hợp lệ: {scenario_name}")


def split_features_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Tách DataFrame thành tập đặc trưng X và biến mục tiêu y."""

    missing_features = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Dữ liệu thiếu các cột đặc trưng: "
            + ", ".join(missing_features)
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} trong dữ liệu."
        )

    X = dataframe.loc[:, feature_columns].copy()
    y = dataframe.loc[:, TARGET_COLUMN].copy()

    return X, y
