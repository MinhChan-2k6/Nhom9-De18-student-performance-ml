"""Hàm tiện ích dự đoán điểm G3 cho một học sinh."""

from __future__ import annotations

from typing import Any, Mapping

from src.inference.predictor import (
    PredictionMode,
    get_prediction_service,
)


def predict_single(
    features: Mapping[str, Any],
    prediction_mode: PredictionMode,
) -> dict[str, Any]:
    """Dự đoán một học sinh bằng model tương ứng với ``prediction_mode``.

    Parameters
    ----------
    features:
        Mapping chứa các đặc trưng đầu vào. Model có G1/G2 cần đủ 33 đặc
        trưng; model dự báo sớm cần 31 đặc trưng và không bắt buộc G1/G2.
    prediction_mode:
        ``"with_g1_g2"`` hoặc ``"early_prediction"``.

    Returns
    -------
    dict[str, Any]
        Kết quả gồm ``predicted_g3``, ``converted_result``, ``model`` và
        ``scenario``.
    """

    service = get_prediction_service(prediction_mode)
    return service.predict(features)
