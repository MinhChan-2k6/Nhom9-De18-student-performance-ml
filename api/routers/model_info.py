from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.config.paths import get_scenario_artifact_paths
from src.config.settings import (
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.utils.io import load_json

router = APIRouter(
    prefix="/model-info",
    tags=["Model Info"],
)


@router.get(
    "",
    summary="Xem thông tin các mô hình đang sử dụng",
)
def get_model_info() -> list[dict[str, Any]]:
    scenarios = [
        {
            "prediction_mode": "with_g1_g2",
            "scenario": SCENARIO_WITH_PREVIOUS_GRADES,
        },
        {
            "prediction_mode": "early_prediction",
            "scenario": SCENARIO_WITHOUT_PREVIOUS_GRADES,
        },
    ]

    result: list[dict[str, Any]] = []

    try:
        for item in scenarios:
            paths = get_scenario_artifact_paths(item["scenario"])
            metrics = load_json(paths.metrics)

            result.append(
                {
                    "prediction_mode": item["prediction_mode"],
                    "scenario": item["scenario"],
                    "best_model": metrics.get(
                        "best_model",
                        "unknown",
                    ),
                    "model_version": metrics.get(
                        "model_version",
                        "1.0.0",
                    ),
                    "input_features": metrics.get(
                        "input_features",
                        [],
                    ),
                    "metrics": metrics,
                }
            )

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy file thông tin model: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể đọc thông tin model: {exc}",
        ) from exc