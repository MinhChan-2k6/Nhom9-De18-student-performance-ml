<<<<<<< Updated upstream
from __future__ import annotations

import logging
=======
ưfrom __future__ import annotations

import logging
from typing import Any
>>>>>>> Stashed changes

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.crud import create_prediction_history
from api.database import get_db
from api.schemas import PredictRequest, PredictResponse
from src.inference.predictor import PredictionService, get_prediction_service


LOGGER = logging.getLogger(__name__)


# ==================================================
# GIÁ TRỊ MẶC ĐỊNH CHO CÁC ĐẶC TRƯNG TÙY CHỌN
# ==================================================
DEFAULT_FEATURES: dict[str, Any] = {
    "school": "gp",
    "address": "u",
    "famsize": "gt3",
    "Pstatus": "t",
    "Medu": 3,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 2,
    "health": 4,
}


# ==================================================
# METRICS DỰ PHÒNG
#
# Bình thường endpoint lấy trực tiếp test_metrics từ metrics.json
# đã được PredictionService nạp. Các giá trị dưới đây chỉ dùng khi
# service giả lập trong test hoặc artifact cũ chưa có test_metrics.
# ==================================================
FALLBACK_SCENARIO_METRICS: dict[str, dict[str, float]] = {
    "A_with_G1_G2": {
        "mae": 0.956767,
        "rmse": 1.389162,
    },
    "B_without_G1_G2": {
        "mae": 2.122980,
        "rmse": 2.775914,
    },
}


router = APIRouter(
    tags=["Prediction"],
)


<<<<<<< Updated upstream
=======
def _get_reference_metrics(
    service: PredictionService,
    scenario: str,
) -> tuple[float, float]:
    """Lấy MAE/RMSE của đúng model đang được dùng để dự đoán."""

    metrics_payload = getattr(service, "metrics", None)
    if isinstance(metrics_payload, dict):
        test_metrics = metrics_payload.get("test_metrics")
        if isinstance(test_metrics, dict):
            try:
                mae = float(test_metrics["mae"])
                rmse = float(test_metrics["rmse"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if mae >= 0 and rmse >= 0:
                    return mae, rmse

    fallback_metrics = FALLBACK_SCENARIO_METRICS.get(scenario)
    if fallback_metrics is None:
        raise ValueError(
            f"Không tìm thấy metrics cho scenario: {scenario}"
        )

    return (
        float(fallback_metrics["mae"]),
        float(fallback_metrics["rmse"]),
    )


>>>>>>> Stashed changes
def _save_prediction_history_safely(
    db: Session,
    *,
    payload: PredictRequest,
<<<<<<< Updated upstream
    features: dict,
    result: dict,
) -> tuple[bool, int | None]:
=======
    features: dict[str, Any],
    result: dict[str, Any],
) -> tuple[bool, int | None]:
    """
    Lưu lịch sử nhưng không làm hỏng kết quả dự đoán nếu SQL Server lỗi.
    """
>>>>>>> Stashed changes

    try:
        created_history = create_prediction_history(
            db,
            full_name=payload.student.full_name,
            subject=features.get("subject"),
            prediction_mode=payload.prediction_mode,
<<<<<<< Updated upstream
            scenario=result["scenario"],
            predicted_g3=result["predicted_g3"],
            model_name=result["model"],
            model_version=result["model_version"],
=======
            scenario=str(result["scenario"]),
            predicted_g3=float(result["predicted_g3"]),
            model_name=str(result["model"]),
            model_version=str(result["model_version"]),
>>>>>>> Stashed changes
            input_features=features,
        )
        return True, int(created_history["prediction_id"])
    except SQLAlchemyError as exc:
        LOGGER.warning(
            "Dự đoán thành công, nhưng lưu lịch sử thất bại: %s",
            exc.__class__.__name__,
        )
        return False, None
    except Exception as exc:
        LOGGER.warning(
            "Dự đoán thành công, nhưng lưu lịch sử thất bại: %s",
            exc.__class__.__name__,
        )
        return False, None


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict_student(
    payload: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    """
    Dự đoán điểm G3 bằng mô hình hồi quy.

    Việc dự đoán và việc lưu lịch sử được tách biệt: nếu SQL Server
    tạm thời không hoạt động, API vẫn trả kết quả model thành công.
    """

    features = {
        **DEFAULT_FEATURES,
        **payload.features.model_dump(exclude_none=True),
    }

    if payload.prediction_mode == "with_g1_g2":
        if features.get("G1") is None or features.get("G2") is None:
            raise HTTPException(
<<<<<<< Updated upstream
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
=======
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
>>>>>>> Stashed changes
                detail="Mode with_g1_g2 requires both G1 and G2.",
            )

    # Phần dự đoán: lỗi model/artifact vẫn trả HTTP phù hợp.
    try:
        service = get_prediction_service(payload.prediction_mode)
        raw_result = service.predict(features=features)

        predicted_g3 = float(raw_result["predicted_g3"])
        model_name = str(raw_result["model"])
        model_version = str(raw_result["model_version"])
        scenario = str(raw_result["scenario"])

        mae, rmse = _get_reference_metrics(service, scenario)
        prediction_lower = max(0.0, predicted_g3 - rmse)
        prediction_upper = min(20.0, predicted_g3 + rmse)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
<<<<<<< Updated upstream
    except ValueError as exc:
=======
    except (KeyError, TypeError, ValueError) as exc:
>>>>>>> Stashed changes
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not run prediction: {exc}",
        ) from exc

    normalized_result: dict[str, Any] = {
        "predicted_g3": predicted_g3,
        "model": model_name,
        "model_version": model_version,
        "scenario": scenario,
    }

    # Phần database: lỗi lưu không được biến dự đoán thành HTTP 500.
    history_saved, history_id = _save_prediction_history_safely(
        db,
        payload=payload,
        features=features,
        result=normalized_result,
    )

    return PredictResponse(
        full_name=payload.student.full_name,
        predicted_g3=round(predicted_g3, 2),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        prediction_lower=round(prediction_lower, 2),
        prediction_upper=round(prediction_upper, 2),
        model=model_name,
        model_version=model_version,
        scenario=scenario,
<<<<<<< Updated upstream
=======
        history_saved=history_saved,
        history_id=history_id,
>>>>>>> Stashed changes
    )
