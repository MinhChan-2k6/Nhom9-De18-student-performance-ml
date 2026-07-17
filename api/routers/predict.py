from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.crud import create_prediction_history
from api.database import get_db
from api.schemas import PredictRequest, PredictResponse
from src.inference.predictor import get_prediction_service

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
router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict_student(
    payload: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictResponse:
    features = payload.features.model_dump(exclude_none=True)
    features = {
    **DEFAULT_FEATURES,
    **features,
}

    if payload.prediction_mode == "with_g1_g2":
        if features.get("G1") is None or features.get("G2") is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Chế độ có G1 và G2 yêu cầu nhập đủ hai điểm.",
            )

    try:
        service = get_prediction_service(payload.prediction_mode)
        result = service.predict(features=features)
        predicted_result = (
    "Đạt"
    if float(result["predicted_g3"]) >= 10
    else "Trượt"
)

        create_prediction_history(
            db,
            full_name=payload.student.full_name,
            subject=features.get("subject"),
            prediction_mode=payload.prediction_mode,
            scenario=result["scenario"],
            predicted_g3=result["predicted_g3"],
            predicted_result=predicted_result,
            model_name=result["model"],
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
            detail=f"Không thể dự đoán hoặc lưu lịch sử: {exc}",
        ) from exc
        
    return PredictResponse(
    full_name=payload.student.full_name,
    predicted_g3=result["predicted_g3"],
    converted_result=predicted_result,
    model=result["model"],
    scenario=result["scenario"],
)

   