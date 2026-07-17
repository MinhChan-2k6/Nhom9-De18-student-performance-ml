"""Dịch vụ nạp model và thực hiện dự đoán điểm G3."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from src.config.paths import get_scenario_artifact_paths
from src.config.settings import (
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.utils.io import load_joblib, load_json

PredictionMode = Literal["with_g1_g2", "early_prediction"]

MODE_TO_SCENARIO: dict[PredictionMode, str] = {
    "with_g1_g2": SCENARIO_WITH_PREVIOUS_GRADES,
    "early_prediction": SCENARIO_WITHOUT_PREVIOUS_GRADES,
}

PREDICTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "predicted_g3",
    "converted_result",
    "model",
    "scenario",
)


class PredictionService:
    """Nạp đúng pipeline theo chế độ và cung cấp API dự đoán dùng chung."""

    def __init__(self, prediction_mode: PredictionMode) -> None:
        if prediction_mode not in MODE_TO_SCENARIO:
            valid_modes = ", ".join(MODE_TO_SCENARIO)
            raise ValueError(
                f"Chế độ dự đoán không hợp lệ: {prediction_mode}. "
                f"Các chế độ hợp lệ: {valid_modes}."
            )

        self.prediction_mode = prediction_mode
        self.scenario = MODE_TO_SCENARIO[prediction_mode]
        self.paths = get_scenario_artifact_paths(self.scenario)

        self.model = load_joblib(self.paths.model)
        self.metrics = load_json(self.paths.metrics)

        if not isinstance(self.metrics, dict):
            raise ValueError(
                f"Nội dung metrics phải là object JSON: {self.paths.metrics}"
            )

        self.best_scenario = self.metrics.get("best_scenario", self.scenario)
        self.best_model = self.metrics.get("best_model", "unknown")
        self.input_features = self.metrics.get("input_features", [])

        if self.best_scenario != self.scenario:
            raise ValueError(
                "Scenario trong metrics.json không khớp với thư mục model."
            )

        if not isinstance(self.input_features, list) or not self.input_features:
            raise ValueError(
                f"{self.paths.metrics} chưa có danh sách input_features hợp lệ."
            )

        if len(self.input_features) != len(set(self.input_features)):
            raise ValueError(
                f"Danh sách input_features bị trùng trong {self.paths.metrics}."
            )

    def _prepare_input_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Kiểm tra và sắp xếp DataFrame đúng thứ tự đặc trưng của model."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Dữ liệu dự đoán phải là pandas.DataFrame.")

        if dataframe.empty:
            raise ValueError("Dữ liệu dự đoán không có dòng nào.")

        duplicate_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()
        if duplicate_columns:
            raise ValueError(
                "Dữ liệu có tên cột bị trùng: "
                + ", ".join(map(str, duplicate_columns))
            )

        missing_features = [
            feature
            for feature in self.input_features
            if feature not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(
                "Thiếu đặc trưng bắt buộc: " + ", ".join(missing_features)
            )

        input_frame = dataframe.loc[:, self.input_features].copy()
        missing_value_counts = input_frame.isna().sum()
        columns_with_missing_values = missing_value_counts[
            missing_value_counts > 0
        ]

        if not columns_with_missing_values.empty:
            details = ", ".join(
                f"{column} ({int(count)} ô)"
                for column, count in columns_with_missing_values.items()
            )
            raise ValueError(
                "Các đặc trưng bắt buộc đang có dữ liệu thiếu: " + details
            )

        return input_frame

    def predict_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Dự đoán nhiều dòng và trả về riêng các cột kết quả."""

        input_frame = self._prepare_input_frame(dataframe)
        raw_scores = np.asarray(
            self.model.predict(input_frame),
            dtype=float,
        ).reshape(-1)

        if raw_scores.size != len(input_frame):
            raise RuntimeError(
                "Số lượng kết quả dự đoán không khớp số dòng đầu vào."
            )

        if not np.isfinite(raw_scores).all():
            raise ValueError("Model trả về điểm dự đoán không hợp lệ.")

        clipped_scores = np.clip(raw_scores, 0.0, 20.0)
        rounded_scores = np.round(clipped_scores, 2)

        return pd.DataFrame(
            {
                "predicted_g3": rounded_scores,
                "converted_result": np.where(
                    clipped_scores >= 10.0,
                    "Đạt",
                    "Trượt",
                ),
                "model": self.best_model,
                "scenario": self.best_scenario,
            },
            index=dataframe.index,
        )

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Dự đoán một học sinh từ mapping tên đặc trưng và giá trị."""

        if not isinstance(features, Mapping):
            raise TypeError("features phải là một mapping tên cột và giá trị.")

        result = self.predict_frame(pd.DataFrame([dict(features)])).iloc[0]

        return {
            "predicted_g3": float(result["predicted_g3"]),
            "converted_result": str(result["converted_result"]),
            "model": str(result["model"]),
            "scenario": str(result["scenario"]),
        }


@lru_cache(maxsize=len(MODE_TO_SCENARIO))
def get_prediction_service(
    prediction_mode: PredictionMode,
) -> PredictionService:
    """Cache riêng một service cho mỗi chế độ dự đoán."""

    return PredictionService(prediction_mode=prediction_mode)
