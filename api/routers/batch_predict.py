"""Endpoint dự đoán điểm G3 hàng loạt từ file CSV."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pandas.errors import EmptyDataError, ParserError
from pydantic import TypeAdapter, ValidationError

from api.schemas import PredictionFeatures
from src.inference.predict_batch import predict_batch as predict_batch_dataframe
from src.inference.predictor import PredictionMode

router = APIRouter(tags=["Batch Prediction"])

MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024
CSV_MEDIA_TYPE = "text/csv; charset=utf-8"


def _build_output_filename(original_filename: str | None) -> str:
    """Tạo tên file tải xuống an toàn và dễ đọc."""

    original_stem = Path(original_filename or "students").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", original_stem).strip("_")
    return f"{safe_stem or 'students'}_predictions.csv"


def _read_uploaded_csv(content: bytes) -> pd.DataFrame:
    """Đọc nội dung CSV UTF-8 và chuyển lỗi pandas thành lỗi đầu vào rõ ràng."""

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV đang rỗng.",
        )

    if len(content) > MAX_CSV_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File CSV vượt quá giới hạn 10 MB.",
        )

    try:
        dataframe = pd.read_csv(BytesIO(content))
    except EmptyDataError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV không chứa dữ liệu.",
        ) from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV phải sử dụng mã hóa UTF-8.",
        ) from exc
    except ParserError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không thể đọc cấu trúc CSV. Hãy kiểm tra dấu phân cách và số cột.",
        ) from exc

    if dataframe.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File CSV không có dòng học sinh nào.",
        )

    return dataframe


def _validate_and_normalize_features(
    dataframe: pd.DataFrame,
    prediction_mode: PredictionMode,
) -> pd.DataFrame:
    """Kiểm tra từng dòng theo cùng schema với endpoint dự đoán đơn."""

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicate_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "File CSV có tên cột bị trùng: "
                + ", ".join(map(str, duplicate_columns))
            ),
        )

    base_columns = [
        field_name
        for field_name in PredictionFeatures.model_fields
        if field_name not in {"G1", "G2"}
    ]
    required_columns = list(base_columns)
    if prediction_mode == "with_g1_g2":
        required_columns.extend(["G1", "G2"])

    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "File CSV thiếu các cột bắt buộc: "
                + ", ".join(missing_columns)
            ),
        )

    columns_with_missing_values = [
        column
        for column in required_columns
        if dataframe[column].isna().any()
    ]
    if columns_with_missing_values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Các cột bắt buộc có dữ liệu trống: "
                + ", ".join(columns_with_missing_values)
            ),
        )

    records = dataframe[required_columns].to_dict(orient="records")

    try:
        validated_rows = TypeAdapter(list[PredictionFeatures]).validate_python(
            records
        )
    except ValidationError as exc:
        error_messages: list[str] = []
        validation_errors = exc.errors()

        for error in validation_errors[:10]:
            location = error.get("loc", ())
            row_index = location[0] if location and isinstance(location[0], int) else None
            field_name = str(location[-1]) if location else "không xác định"
            csv_row = row_index + 2 if row_index is not None else "?"
            error_messages.append(
                f"Dòng {csv_row}, cột {field_name}: {error.get('msg', 'không hợp lệ')}"
            )

        if len(validation_errors) > 10:
            error_messages.append(
                f"Còn {len(validation_errors) - 10} lỗi khác chưa hiển thị."
            )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Dữ liệu trong file CSV không hợp lệ.",
                "errors": error_messages,
            },
        ) from exc

    normalized_dataframe = dataframe.copy()
    for column in required_columns:
        normalized_dataframe[column] = [
            getattr(validated_row, column) for validated_row in validated_rows
        ]

    return normalized_dataframe


@router.post(
    "/predict/batch",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Dự đoán G3 hàng loạt từ CSV",
)
async def predict_batch_students(
    file: Annotated[
        UploadFile,
        File(description="File CSV, mỗi dòng tương ứng với một học sinh."),
    ],
    prediction_mode: Annotated[
        PredictionMode,
        Form(description="with_g1_g2 hoặc early_prediction"),
    ],
    include_input: Annotated[
        bool,
        Form(description="Giữ các cột đầu vào trong file kết quả."),
    ] = True,
) -> StreamingResponse:
    """Nhận CSV, chạy model hồi quy và trả lại CSV chứa điểm G3 dự đoán."""

    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận file có phần mở rộng .csv.",
        )

    try:
        content = await file.read()
        input_dataframe = _read_uploaded_csv(content)
        normalized_dataframe = _validate_and_normalize_features(
            input_dataframe,
            prediction_mode,
        )
        output_dataframe = predict_batch_dataframe(
            normalized_dataframe,
            prediction_mode,
            include_input=include_input,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể thực hiện dự đoán hàng loạt.",
        ) from exc
    finally:
        await file.close()

    # utf-8-sig giúp Excel trên Windows hiển thị tiếng Việt đúng hơn.
    csv_bytes = output_dataframe.to_csv(index=False).encode("utf-8-sig")
    output_filename = _build_output_filename(file.filename)

    headers = {
        "Content-Disposition": f'attachment; filename="{output_filename}"',
        "X-Total-Rows": str(len(output_dataframe)),
    }

    return StreamingResponse(
        BytesIO(csv_bytes),
        media_type=CSV_MEDIA_TYPE,
        headers=headers,
    )
