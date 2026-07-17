<<<<<<< HEAD
=======
<<<<<<< Updated upstream
=======
>>>>>>> origin/MinhChan
from fastapi import APIRouter, HTTPException, status

from api.schemas import PredictRequest, PredictResponse
from src.inference.predictor import get_prediction_service

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict_student(payload: PredictRequest) -> PredictResponse:
    features = payload.features.model_dump(exclude_none=True)

    if payload.prediction_mode == "with_g1_g2":
        if features.get("G1") is None or features.get("G2") is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Chế độ có G1 và G2 yêu cầu nhập đủ hai điểm.",
            )

    try:
<<<<<<< HEAD
        service = get_prediction_service()
        result = service.predict(
            features=features,
            prediction_mode=payload.prediction_mode,
        )
=======
        service = get_prediction_service(payload.prediction_mode)
        result = service.predict(features=features)
>>>>>>> origin/MinhChan
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
            detail="Không thể thực hiện dự đoán.",
        ) from exc

    return PredictResponse(
        full_name=payload.student.full_name,
        **result,
    )
<<<<<<< HEAD
=======
>>>>>>> Stashed changes
>>>>>>> origin/MinhChan
