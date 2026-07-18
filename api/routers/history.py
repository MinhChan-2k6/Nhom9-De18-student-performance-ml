from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.crud import (
    get_prediction_by_id,
    get_prediction_history,
)
from api.database import get_db
from api.schemas import PredictionHistoryResponse


router = APIRouter(
    prefix="/history",
    tags=["History"],
)


def format_history_row(
    row: dict[str, Any],
) -> PredictionHistoryResponse:
    """
    Chuyển dữ liệu lịch sử từ SQL Server
    thành response của API.

    Bài toán là thuần hồi quy nên chỉ trả về
    điểm G3 dự đoán, không có Đạt/Trượt.
    """

    input_features = None
    input_json = row.get("input_json")

    if input_json:
        try:
            input_features = json.loads(input_json)

        except (json.JSONDecodeError, TypeError):
            input_features = None

    return PredictionHistoryResponse(
        prediction_id=row["prediction_id"],
        student_code=row.get("student_code"),
        full_name=row.get("full_name"),
        subject=row.get("subject"),
        prediction_mode=row["prediction_mode"],
        scenario=row.get("scenario"),
        predicted_g3=float(row["predicted_g3"]),
        model_name=row["model_name"],
        model_version=row.get("model_version"),
        input_features=input_features,
        created_at=row["created_at"],
    )


@router.get(
    "",
    response_model=list[PredictionHistoryResponse],
    summary="Lấy danh sách lịch sử dự đoán",
)
def read_prediction_history(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Số bản ghi lịch sử cần lấy",
    ),
    db: Session = Depends(get_db),
) -> list[PredictionHistoryResponse]:
    """
    Lấy danh sách các lần dự đoán G3 gần nhất.
    """

    try:
        rows = get_prediction_history(
            db,
            limit=limit,
        )

        return [
            format_history_row(row)
            for row in rows
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Không thể lấy lịch sử dự đoán: "
                f"{exc}"
            ),
        ) from exc


@router.get(
    "/{prediction_id}",
    response_model=PredictionHistoryResponse,
    summary="Xem chi tiết một lần dự đoán",
)
def read_prediction_detail(
    prediction_id: int,
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """
    Lấy chi tiết một lần dự đoán theo prediction_id.
    """

    try:
        row = get_prediction_by_id(
            db,
            prediction_id=prediction_id,
        )

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Không tìm thấy lịch sử dự đoán.",
            )

        return format_history_row(row)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Không thể đọc lịch sử dự đoán: "
                f"{exc}"
            ),
        ) from exc