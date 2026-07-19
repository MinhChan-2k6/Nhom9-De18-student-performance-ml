from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.crud import create_prediction_history
from api.database import get_db
from api.schemas import PredictRequest, PredictResponse
from src.inference.predictor import get_prediction_service


# ==================================================
# GIÁ TRỊ MẶC ĐỊNH CHO CÁC ĐẶC TRƯNG TÙY CHỌN
# ==================================================
DEFAULT_FEATURES = {
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
# SAI SỐ THAM KHẢO CỦA TỪNG KỊCH BẢN
#
# Các giá trị được lấy từ kết quả đánh giá model.
# Khoảng dự đoán tham khảo được tính:
#
# predicted_g3 ± RMSE
#
# Đây KHÔNG phải khoảng tin cậy 95%.
# ==================================================
SCENARIO_METRICS = {
    "A_with_G1_G2": {
        "mae": 1.359425,
        "rmse": 1.886236,
    },
    "B_without_G1_G2": {
        "mae": 1.132185,
        "rmse": 1.900339,
    },
}


router = APIRouter(
    tags=["Prediction"],
)


def _save_prediction_history_safely(
    db: Session,
    *,
    payload: PredictRequest,
    features: dict,
    result: dict,
) -> tuple[bool, int | None]:

    try:
        created_history = create_prediction_history(
            db,
            full_name=payload.student.full_name,
            subject=features.get("subject"),
            prediction_mode=payload.prediction_mode,
            scenario=result["scenario"],
            predicted_g3=result["predicted_g3"],
            model_name=result["model"],
            model_version=result["model_version"],
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
    Dự đoán điểm G3 của học sinh bằng mô hình hồi quy.

    Kết quả bao gồm:
    - Điểm G3 dự đoán.
    - MAE tham khảo.
    - RMSE tham khảo.
    - Khoảng dự đoán tham khảo dựa trên RMSE.

    Hệ thống không phân loại Đạt hoặc Trượt.
    """

    # Chỉ lấy những trường người dùng thực sự nhập.
    features = payload.features.model_dump(
        exclude_none=True,
    )

    # Bổ sung giá trị mặc định cho các trường tùy chọn.
    features = {
        **DEFAULT_FEATURES,
        **features,
    }

    # Chế độ có G1/G2 bắt buộc phải nhập đủ hai điểm.
    if payload.prediction_mode == "with_g1_g2":
        if features.get("G1") is None or features.get("G2") is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Mode with_g1_g2 requires both G1 and G2.",
            )

    try:
        # Lấy prediction service theo chế độ dự đoán.
        service = get_prediction_service(
            payload.prediction_mode
        )

        # Thực hiện dự đoán hồi quy.
        result = service.predict(
            features=features
        )

        predicted_g3 = float(
            result["predicted_g3"]
        )

        scenario = str(
            result["scenario"]
        )

        model_name = str(
            result["model"]
        )

        # ==========================================
        # LẤY SAI SỐ THAM KHẢO
        # ==========================================
        scenario_metrics = SCENARIO_METRICS.get(
            scenario
        )

        if scenario_metrics is None:
            raise ValueError(
                f"Không tìm thấy metrics cho scenario: {scenario}"
            )

        mae = float(
            scenario_metrics["mae"]
        )

        rmse = float(
            scenario_metrics["rmse"]
        )

        # ==========================================
        # TÍNH KHOẢNG DỰ ĐOÁN THAM KHẢO
        #
        # Giới hạn trong thang điểm 0 - 20.
        # ==========================================
        prediction_lower = max(
            0.0,
            predicted_g3 - rmse,
        )

        prediction_upper = min(
            20.0,
            predicted_g3 + rmse,
        )

        # ==========================================
        # LƯU LỊCH SỬ VÀO SQL SERVER
        # ==========================================
        create_prediction_history(
            db,
            full_name=payload.student.full_name,
            subject=features.get("subject"),
            prediction_mode=payload.prediction_mode,
            scenario=scenario,
            predicted_g3=predicted_g3,
            model_name=model_name,
            model_version=None,
            input_features=features,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not run prediction: {exc}",
        ) from exc

    # ==========================================
    # TRẢ KẾT QUẢ HỒI QUY
    # ==========================================
    return PredictResponse(
        full_name=payload.student.full_name,
        predicted_g3=round(
            predicted_g3,
            2,
        ),
        mae=round(
            mae,
            4,
        ),
        rmse=round(
            rmse,
            4,
        ),
        prediction_lower=round(
            prediction_lower,
            2,
        ),
        prediction_upper=round(
            prediction_upper,
            2,
        ),
        model=model_name,
        scenario=scenario,
    )
