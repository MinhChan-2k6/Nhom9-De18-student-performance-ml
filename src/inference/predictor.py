"""Dịch vụ nạp model hồi quy và dự đoán điểm G3."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Mapping

import pandas as pd

from src.config.paths import get_scenario_artifact_paths
from src.config.settings import (
    MODEL_VERSION,
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.utils.helpers import normalize_grade_scores, validate_dataframe
from src.utils.io import load_joblib, load_json

PredictionMode = Literal["with_g1_g2", "early_prediction"]

MODE_TO_SCENARIO: dict[PredictionMode, str] = {
    "with_g1_g2": SCENARIO_WITH_PREVIOUS_GRADES,
    "early_prediction": SCENARIO_WITHOUT_PREVIOUS_GRADES,
}

PREDICTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "predicted_g3",
    "model",
    "model_version",
    "scenario",
)


class PredictionService:
    """Nạp đúng pipeline theo chế độ vaf cung cấp API dự đoán dùng chung."""

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
        self.model_version = str(
            self.metrics.get("model_version", MODEL_VERSION)
        )
        self.input_features = self.metrics.get("input_features", [])

        if self.metrics.get("task") not in (None, "regression"):
            raise ValueError("Artifact hiện tại không phải model hồi quy.")

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

        return validate_dataframe(
            dataframe,
            self.input_features,
            context="Dữ liệu dự đoán",
        )

    def predict_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Dự đoán nhiều dòng và trả về các cột kết quả hồi quy."""

        input_frame = self._prepare_input_frame(dataframe)
        predicted_scores = normalize_grade_scores(
            self.model.predict(input_frame)
        )

        if predicted_scores.size != len(input_frame):
            raise RuntimeError(
                "Số lượng kết quả dự đoán không khớp số dòng đầu vào."
            )

        return pd.DataFrame(
            {
                "predicted_g3": predicted_scores,
                "model": self.best_model,
                "model_version": self.model_version,
                "scenario": self.best_scenario,
            },
            index=dataframe.index,
        )

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Dự đoán điểm G3 của một học sinh từ mapping đặc trưng."""

        if not isinstance(features, Mapping):
            raise TypeError("features phải là một mapping tên cột và giá trị.")

        result = self.predict_frame(pd.DataFrame([dict(features)])).iloc[0]

        return {
            "predicted_g3": float(result["predicted_g3"]),
            "model": str(result["model"]),
            "model_version": str(result["model_version"]),
            "scenario": str(result["scenario"]),
        }


@lru_cache(maxsize=len(MODE_TO_SCENARIO))
def get_prediction_service(
    prediction_mode: PredictionMode,
) -> PredictionService:
    """Cache riêng một service cho mỗi chế độ dự đoans"""

    return PredictionService(prediction_mode=prediction_mode)
