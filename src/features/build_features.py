"""Xây dựng đặc trưng cho hai kịch bản dự đoán điểm G3."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.settings import (
    EARLY_PREDICTION_EXCLUDED_COLUMNS,
    FEATURE_SET_ENGINEERED,
    FEATURE_SET_RAW,
    FEATURE_SETS,
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
    TARGET_COLUMN,
)

COMMON_ENGINEERED_FEATURES: tuple[str, ...] = (
    "parent_education_mean",
    "alcohol_consumption_mean",
    "social_activity_score",
    "support_count",
    "absence_log1p",
)

PREVIOUS_GRADE_ENGINEERED_FEATURES: tuple[str, ...] = (
    "previous_grade_mean",
    "grade_trend",
)

COMMON_ENGINEERING_SOURCE_COLUMNS: tuple[str, ...] = (
    "Medu",
    "Fedu",
    "Dalc",
    "Walc",
    "freetime",
    "goout",
    "schoolsup",
    "famsup",
    "absences",
)


class StudentFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        scenario_name: str,
        feature_set: str = FEATURE_SET_ENGINEERED,
    ) -> None:
        self.scenario_name = scenario_name
        self.feature_set = feature_set

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> "StudentFeatureEngineer":
        del y
        input_frame = _ensure_dataframe(X)
        validate_feature_set(self.feature_set)
        validate_scenario(self.scenario_name)

        # Gọi transform một lần để phát hiện sớm cột nguồn bị thiếu.
        transformed = self.transform(input_frame)
        self.feature_names_in_ = np.asarray(input_frame.columns, dtype=object)
        self.n_features_in_ = input_frame.shape[1]
        self.feature_names_out_ = np.asarray(
            transformed.columns,
            dtype=object,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        input_frame = _ensure_dataframe(X)
        validate_feature_set(self.feature_set)
        validate_scenario(self.scenario_name)

        if self.feature_set == FEATURE_SET_RAW:
            return input_frame.copy()

        return add_engineered_features(
            input_frame,
            scenario_name=self.scenario_name,
        )

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        if input_features is not None:
            raw_names = [str(name) for name in input_features]
        elif hasattr(self, "feature_names_in_"):
            raw_names = [str(name) for name in self.feature_names_in_]
        else:
            raise ValueError("Transformer chưa được fit và thiếu input_features.")

        if self.feature_set == FEATURE_SET_RAW:
            return np.asarray(raw_names, dtype=object)

        return np.asarray(
            [*raw_names, *get_engineered_feature_names(self.scenario_name)],
            dtype=object,
        )


def _ensure_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Feature engineering yêu cầu đầu vào là pandas DataFrame."
        )
    return dataframe


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    missing_columns = [
        column for column in columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{context} thiếu các cột nguồn: " + ", ".join(missing_columns)
        )


def _to_numeric(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(dataframe[column], errors="coerce")


def _yes_no_to_number(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map({"yes": 1.0, "no": 0.0})


def validate_scenario(scenario_name: str) -> None:
    if scenario_name not in {
        SCENARIO_WITH_PREVIOUS_GRADES,
        SCENARIO_WITHOUT_PREVIOUS_GRADES,
    }:
        raise ValueError(f"Scenario không hợp lệ: {scenario_name}")


def validate_feature_set(feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        valid_values = ", ".join(FEATURE_SETS)
        raise ValueError(
            f"Feature set không hợp lệ: {feature_set}. "
            f"Giá trị hợp lệ: {valid_values}."
        )


def get_engineered_feature_names(scenario_name: str) -> list[str]:

    validate_scenario(scenario_name)
    feature_names = list(COMMON_ENGINEERED_FEATURES)

    if scenario_name == SCENARIO_WITH_PREVIOUS_GRADES:
        feature_names.extend(PREVIOUS_GRADE_ENGINEERED_FEATURES)

    return feature_names


def add_engineered_features(
    dataframe: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """Tạo các đặc trưng có ý nghĩa nghiệp vụ, không dùng G3.

    Các đặc trưng chung:
    - Trung bình học vấn cha mẹ.
    - Trung bình mức dùng đồ uống có cồn trong tuần/cuối tuần.
    - Mức hoạt động xã hội từ thời gian rảnh và số lần đi chơi.
    - Số loại hỗ trợ học tập từ nhà trường/gia đình.
    - log1p(absences) để giảm độ lệch của số buổi vắng.

    Scenario có G1/G2 còn có:
    - Trung bình điểm hai kỳ trước.
    - Xu hướng điểm G2 - G1.
    """

    input_frame = _ensure_dataframe(dataframe)
    validate_scenario(scenario_name)
    _require_columns(
        input_frame,
        COMMON_ENGINEERING_SOURCE_COLUMNS,
        context="Feature engineering",
    )

    result = input_frame.copy()

    result["parent_education_mean"] = (
        _to_numeric(result, "Medu") + _to_numeric(result, "Fedu")
    ) / 2.0
    result["alcohol_consumption_mean"] = (
        _to_numeric(result, "Dalc") + _to_numeric(result, "Walc")
    ) / 2.0
    result["social_activity_score"] = (
        _to_numeric(result, "freetime") + _to_numeric(result, "goout")
    )
    result["support_count"] = (
        _yes_no_to_number(result["schoolsup"])
        + _yes_no_to_number(result["famsup"])
    )

    absences = _to_numeric(result, "absences").clip(lower=0)
    result["absence_log1p"] = np.log1p(absences)

    if scenario_name == SCENARIO_WITH_PREVIOUS_GRADES:
        _require_columns(
            result,
            ("G1", "G2"),
            context="Feature engineering cho scenario có G1/G2",
        )
        g1 = _to_numeric(result, "G1")
        g2 = _to_numeric(result, "G2")
        result["previous_grade_mean"] = (g1 + g2) / 2.0
        result["grade_trend"] = g2 - g1

    return result


def build_feature_engineer(
    scenario_name: str,
    feature_set: str,
) -> StudentFeatureEngineer:

    validate_scenario(scenario_name)
    validate_feature_set(feature_set)
    return StudentFeatureEngineer(
        scenario_name=scenario_name,
        feature_set=feature_set,
    )


def get_scenario_features(
    dataframe: pd.DataFrame,
    scenario_name: str,
) -> list[str]:

    validate_scenario(scenario_name)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} trong dữ liệu."
        )

    feature_columns = [
        column for column in dataframe.columns if column != TARGET_COLUMN
    ]

    if scenario_name == SCENARIO_WITH_PREVIOUS_GRADES:
        return feature_columns

    return [
        column
        for column in feature_columns
        if column not in EARLY_PREDICTION_EXCLUDED_COLUMNS
    ]


def get_model_feature_columns(
    dataframe: pd.DataFrame,
    scenario_name: str,
    feature_set: str,
) -> list[str]:
    raw_feature_columns = get_scenario_features(dataframe, scenario_name)
    raw_frame = dataframe.loc[:, raw_feature_columns]
    feature_engineer = build_feature_engineer(scenario_name, feature_set)
    transformed_frame = feature_engineer.fit_transform(raw_frame)
    return transformed_frame.columns.tolist()


def split_features_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [
        column for column in feature_columns if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Dữ liệu thiếu các cột đặc trưng: " + ", ".join(missing_features)
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} trong dữ liệu."
        )

    X = dataframe.loc[:, feature_columns].copy()
    y = dataframe.loc[:, TARGET_COLUMN].copy()

    return X, y
