from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.config.paths import get_scenario_artifact_paths
from src.config.settings import (
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)

PredictionMode = Literal["with_g1_g2", "early_prediction"]

MODE_TO_SCENARIO: dict[str, str] = {
    "with_g1_g2": SCENARIO_WITH_PREVIOUS_GRADES,
    "early_prediction": SCENARIO_WITHOUT_PREVIOUS_GRADES,
}


class PredictionService:
    """Load đúng pipeline theo chế độ và dự đoán một học sinh."""

    def __init__(self, prediction_mode: PredictionMode) -> None:
        if prediction_mode not in MODE_TO_SCENARIO:
            raise ValueError(f"Chế độ dự đoán không hợp lệ: {prediction_mode}")

        self.prediction_mode = prediction_mode
        self.scenario = MODE_TO_SCENARIO[prediction_mode]
        self.paths = get_scenario_artifact_paths(self.scenario)

        if not self.paths.model.exists() or self.paths.model.stat().st_size == 0:
            raise FileNotFoundError(
                f"Không tìm thấy model cho scenario {self.scenario} tại "
                f"{self.paths.model}. Hãy chạy python -m src.training.train."
            )

        self.model = joblib.load(self.paths.model)
        self.metrics = self._load_metrics()
        self.best_scenario = self.metrics.get("best_scenario", self.scenario)
        self.best_model = self.metrics.get("best_model", "unknown")
        self.input_features = self.metrics.get("input_features", [])

        if self.best_scenario != self.scenario:
            raise ValueError(
                "Scenario trong metrics.json không khớp với thư mục model."
            )

        if not self.input_features:
            raise ValueError(
                f"{self.paths.metrics} chưa có danh sách input_features."
            )

    def _load_metrics(self) -> dict[str, Any]:
        if not self.paths.metrics.exists() or self.paths.metrics.stat().st_size == 0:
            raise FileNotFoundError(
                f"Không tìm thấy metrics tại {self.paths.metrics}."
            )

        try:
            return json.loads(self.paths.metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"File {self.paths.metrics} không đúng định dạng JSON."
            ) from exc

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        missing_features = [
            feature for feature in self.input_features if feature not in features
        ]
        if missing_features:
            raise ValueError(
                "Thiếu đặc trưng bắt buộc: " + ", ".join(missing_features)
            )

        ordered_features = {
            feature: features[feature] for feature in self.input_features
        }
        input_frame = pd.DataFrame([ordered_features])

        raw_score = float(self.model.predict(input_frame)[0])
        predicted_g3 = float(np.clip(raw_score, 0.0, 20.0))
        converted_result = "Đạt" if predicted_g3 >= 10.0 else "Trượt"

        return {
            "predicted_g3": round(predicted_g3, 2),
            "converted_result": converted_result,
            "model": self.best_model,
            "scenario": self.best_scenario,
        }


@lru_cache(maxsize=2)
def get_prediction_service(
    prediction_mode: PredictionMode,
) -> PredictionService:
    """Cache riêng một service cho mỗi chế độ dự đoán."""

    return PredictionService(prediction_mode=prediction_mode)
