from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_regression_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


class PredictionService:
    """Load the trained pipeline once and expose single-record prediction."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            raise FileNotFoundError(
                "Không tìm thấy model hợp lệ tại artifacts/best_regression_model.pkl."
            )

        self.model = joblib.load(MODEL_PATH)
        self.metrics = self._load_metrics()
        self.best_scenario = self.metrics.get("best_scenario")
        self.best_model = self.metrics.get("best_model", "unknown")
        self.input_features = self.metrics.get("input_features", [])

        if not self.input_features:
            raise ValueError("metrics.json chưa có danh sách input_features.")

    @staticmethod
    def _load_metrics() -> dict[str, Any]:
        if not METRICS_PATH.exists() or METRICS_PATH.stat().st_size == 0:
            raise FileNotFoundError("Không tìm thấy artifacts/metrics.json.")

        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("metrics.json không đúng định dạng JSON.") from exc

    def predict(self, features: dict[str, Any], prediction_mode: str) -> dict[str, Any]:
        if self.best_scenario == "A_with_G1_G2" and prediction_mode != "with_g1_g2":
            raise ValueError(
                "Model hiện tại chỉ hỗ trợ chế độ có G1 và G2. "
                "Cần huấn luyện và lưu riêng model Scenario B cho dự báo sớm."
            )

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


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """Return one cached model service per API process."""

    return PredictionService()
