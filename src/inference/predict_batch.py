"""Dự đoán điểm G3 hàng loạt từ DataFrame hoặc file CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.predictor import (
    PREDICTION_OUTPUT_COLUMNS,
    PredictionMode,
    get_prediction_service,
)
from src.utils.io import read_csv, save_csv


def predict_batch(
    dataframe: pd.DataFrame,
    prediction_mode: PredictionMode,
    *,
    include_input: bool = True,
) -> pd.DataFrame:
    """Dự đoán toàn bộ dòng trong một DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe phải là pandas.DataFrame.")

    conflicting_columns = [
        column
        for column in PREDICTION_OUTPUT_COLUMNS
        if column in dataframe.columns
    ]
    if include_input and conflicting_columns:
        raise ValueError(
            "Dữ liệu đầu vào đã có cột kết quả dự đoán: "
            + ", ".join(conflicting_columns)
        )

    service = get_prediction_service(prediction_mode)
    prediction_result = service.predict_frame(dataframe)

    if not include_input:
        return prediction_result.reset_index(drop=True)

    input_copy = dataframe.copy().reset_index(drop=True)
    result_copy = prediction_result.reset_index(drop=True)
    return pd.concat([input_copy, result_copy], axis=1)


def predict_batch_csv(
    input_path: str | Path,
    output_path: str | Path,
    prediction_mode: PredictionMode,
    *,
    include_input: bool = True,
    read_csv_kwargs: dict[str, Any] | None = None,
    save_csv_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Đọc CSV, dự đoán hàng loạt và lưu file kết quả."""

    input_dataframe = read_csv(input_path, **(read_csv_kwargs or {}))
    output_dataframe = predict_batch(
        input_dataframe,
        prediction_mode,
        include_input=include_input,
    )

    return save_csv(
        output_dataframe,
        output_path,
        **(save_csv_kwargs or {}),
    )
