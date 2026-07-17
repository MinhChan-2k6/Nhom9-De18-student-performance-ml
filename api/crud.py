from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def create_prediction_history(
    db: Session,
    *,
    full_name: str | None,
    subject: str | None,
    prediction_mode: str,
    scenario: str | None,
    predicted_g3: float,
    predicted_result: str,
    model_name: str,
    input_features: dict[str, Any],
    student_code: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Lưu một lần dự đoán vào bảng PredictionHistory."""

    query = text(
        """
        INSERT INTO dbo.PredictionHistory (
            student_code,
            full_name,
            subject,
            prediction_mode,
            scenario,
            predicted_g3,
            predicted_result,
            model_name,
            model_version,
            input_json
        )
        OUTPUT
            INSERTED.prediction_id,
            INSERTED.student_code,
            INSERTED.full_name,
            INSERTED.subject,
            INSERTED.prediction_mode,
            INSERTED.scenario,
            INSERTED.predicted_g3,
            INSERTED.predicted_result,
            INSERTED.model_name,
            INSERTED.model_version,
            INSERTED.created_at
        VALUES (
            :student_code,
            :full_name,
            :subject,
            :prediction_mode,
            :scenario,
            :predicted_g3,
            :predicted_result,
            :model_name,
            :model_version,
            :input_json
        )
        """
    )

    parameters = {
        "student_code": student_code,
        "full_name": full_name,
        "subject": subject,
        "prediction_mode": prediction_mode,
        "scenario": scenario,
        "predicted_g3": round(float(predicted_g3), 2),
        "predicted_result": predicted_result,
        "model_name": model_name,
        "model_version": model_version,
        "input_json": json.dumps(
            input_features,
            ensure_ascii=False,
        ),
    }

    try:
        result = db.execute(query, parameters)
        inserted_row = result.mappings().one()
        db.commit()

        return dict(inserted_row)

    except Exception:
        db.rollback()
        raise


def get_prediction_history(
    db: Session,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Lấy danh sách lịch sử dự đoán mới nhất."""

    safe_limit = max(1, min(limit, 500))

    query = text(
        f"""
        SELECT TOP ({safe_limit})
            prediction_id,
            student_code,
            full_name,
            subject,
            prediction_mode,
            scenario,
            predicted_g3,
            predicted_result,
            model_name,
            model_version,
            input_json,
            created_at
        FROM dbo.PredictionHistory
        ORDER BY prediction_id DESC
        """
    )

    result = db.execute(query)

    return [
        dict(row)
        for row in result.mappings().all()
    ]


def get_prediction_by_id(
    db: Session,
    prediction_id: int,
) -> dict[str, Any] | None:
    """Tìm một lịch sử dự đoán theo ID."""

    query = text(
        """
        SELECT
            prediction_id,
            student_code,
            full_name,
            subject,
            prediction_mode,
            scenario,
            predicted_g3,
            predicted_result,
            model_name,
            model_version,
            input_json,
            created_at
        FROM dbo.PredictionHistory
        WHERE prediction_id = :prediction_id
        """
    )

    result = db.execute(
        query,
        {"prediction_id": prediction_id},
    )

    row = result.mappings().first()

    return dict(row) if row else None