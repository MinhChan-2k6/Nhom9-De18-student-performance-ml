from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentInformation(BaseModel):

    full_name: str = Field(
        min_length=2,
        max_length=100,
    )


class PredictionFeatures(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
    )

    # ==================================================
    # THÔNG TIN CƠ BẢN - BẮT BUỘC
    # ==================================================

    subject: Literal[
        "math",
        "portuguese",
    ]

    sex: Literal[
        "f",
        "m",
    ]

    age: int = Field(
        ge=15,
        le=22,
    )

    studytime: int = Field(
        ge=1,
        le=4,
    )

    failures: int = Field(
        ge=0,
        le=3,
    )

    absences: int = Field(
        ge=0,
        le=75,
    )

    # ==================================================
    # THÔNG TIN CHI TIẾT - KHÔNG BẮT BUỘC
    # ==================================================

    school: Literal[
        "gp",
        "ms",
    ] | None = None

    address: Literal[
        "u",
        "r",
    ] | None = None

    famsize: Literal[
        "le3",
        "gt3",
    ] | None = None

    Pstatus: Literal[
        "t",
        "a",
    ] | None = None

    Medu: int | None = Field(
        default=None,
        ge=0,
        le=4,
    )

    Fedu: int | None = Field(
        default=None,
        ge=0,
        le=4,
    )

    Mjob: Literal[
        "teacher",
        "health",
        "services",
        "at_home",
        "other",
    ] | None = None

    Fjob: Literal[
        "teacher",
        "health",
        "services",
        "at_home",
        "other",
    ] | None = None

    reason: Literal[
        "course",
        "home",
        "reputation",
        "other",
    ] | None = None

    guardian: Literal[
        "mother",
        "father",
        "other",
    ] | None = None

    traveltime: int | None = Field(
        default=None,
        ge=1,
        le=4,
    )

    schoolsup: Literal[
        "yes",
        "no",
    ] | None = None

    famsup: Literal[
        "yes",
        "no",
    ] | None = None

    paid: Literal[
        "yes",
        "no",
    ] | None = None

    activities: Literal[
        "yes",
        "no",
    ] | None = None

    nursery: Literal[
        "yes",
        "no",
    ] | None = None

    higher: Literal[
        "yes",
        "no",
    ] | None = None

    internet: Literal[
        "yes",
        "no",
    ] | None = None

    romantic: Literal[
        "yes",
        "no",
    ] | None = None

    famrel: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    freetime: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    goout: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    Dalc: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    Walc: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    health: int | None = Field(
        default=None,
        ge=1,
        le=5,
    )

    # ==================================================
    # ĐIỂM G1, G2
    # Chỉ bắt buộc khi prediction_mode = with_g1_g2
    # Việc kiểm tra được xử lý trong router /predict
    # ==================================================

    G1: int | None = Field(
        default=None,
        ge=0,
        le=20,
    )

    G2: int | None = Field(
        default=None,
        ge=0,
        le=20,
    )


class PredictRequest(BaseModel):

    student: StudentInformation

    prediction_mode: Literal[
        "with_g1_g2",
        "early_prediction",
    ]

    features: PredictionFeatures


class PredictResponse(BaseModel):
    """
    Kết quả dự đoán hồi quy.

    Hệ thống trả về:
    - Điểm G3 dự đoán.
    - MAE tham khảo của mô hình.
    - RMSE tham khảo của mô hình.
    - Khoảng dự đoán tham khảo dựa trên RMSE.

    Không phân loại Đạt hoặc Trượt.
    """

    full_name: str

    predicted_g3: float = Field(
        ge=0,
        le=20,
    )

    mae: float = Field(
        ge=0,
    )

    rmse: float = Field(
        ge=0,
    )

    prediction_lower: float = Field(
        ge=0,
        le=20,
    )

    prediction_upper: float = Field(
        ge=0,
        le=20,
    )

    model: str

    model_version: str

    scenario: str

    history_saved: bool

    history_id: int | None = None


class PredictionHistoryResponse(BaseModel):

    prediction_id: int

    student_code: str | None = None

    full_name: str | None = None

    subject: str | None = None

    prediction_mode: str

    scenario: str | None = None

    predicted_g3: float

    model_name: str

    model_version: str | None = None

    input_features: dict[str, Any] | None = None

    created_at: datetime


class HealthResponse(BaseModel):
    """
    Trạng thái hoạt động của FastAPI và kết nối SQL Server.
    """

    status: Literal["ok"]

    api: Literal["online"]

    database: Literal[
        "connected",
        "disconnected",
    ]