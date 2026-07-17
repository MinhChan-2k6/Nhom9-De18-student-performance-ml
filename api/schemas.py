from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentInformation(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)


class PredictionFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: Literal["math", "portuguese"]
    school: Literal["gp", "ms"]
    sex: Literal["f", "m"]
    age: int = Field(ge=15, le=22)
    address: Literal["u", "r"]
    famsize: Literal["le3", "gt3"]
    Pstatus: Literal["t", "a"]
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]
    reason: Literal["course", "home", "reputation", "other"]
    guardian: Literal["mother", "father", "other"]
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=3)
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]
    romantic: Literal["yes", "no"]
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=75)
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
