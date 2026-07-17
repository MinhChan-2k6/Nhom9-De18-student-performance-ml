from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentInformation(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)


class PredictionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Thông tin cơ bản bắt buộc
    subject: Literal["math", "portuguese"]
    sex: Literal["f", "m"]
    age: int = Field(ge=15, le=22)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    absences: int = Field(ge=0, le=75)

    # Thông tin chi tiết không bắt buộc
    school: Literal["gp", "ms"] | None = None
    address: Literal["u", "r"] | None = None
    famsize: Literal["le3", "gt3"] | None = None
    Pstatus: Literal["t", "a"] | None = None

    Medu: int | None = Field(default=None, ge=0, le=4)
    Fedu: int | None = Field(default=None, ge=0, le=4)

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

    traveltime: int | None = Field(default=None, ge=1, le=4)

    schoolsup: Literal["yes", "no"] | None = None
    famsup: Literal["yes", "no"] | None = None
    paid: Literal["yes", "no"] | None = None
    activities: Literal["yes", "no"] | None = None
    nursery: Literal["yes", "no"] | None = None
    higher: Literal["yes", "no"] | None = None
    internet: Literal["yes", "no"] | None = None
    romantic: Literal["yes", "no"] | None = None

    famrel: int | None = Field(default=None, ge=1, le=5)
    freetime: int | None = Field(default=None, ge=1, le=5)
    goout: int | None = Field(default=None, ge=1, le=5)
    Dalc: int | None = Field(default=None, ge=1, le=5)
    Walc: int | None = Field(default=None, ge=1, le=5)
    health: int | None = Field(default=None, ge=1, le=5)

    G1: int | None = Field(default=None, ge=0, le=20)
    G2: int | None = Field(default=None, ge=0, le=20)


class PredictRequest(BaseModel):
    student: StudentInformation
    prediction_mode: Literal["with_g1_g2", "early_prediction"]
    features: PredictionFeatures


class PredictResponse(BaseModel):
    full_name: str
    predicted_g3: float
    converted_result: Literal["Đạt", "Trượt"]
    model: str
    scenario: str