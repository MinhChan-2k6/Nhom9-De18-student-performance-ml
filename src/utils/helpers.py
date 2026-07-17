"""Các hàm hỗ trợ dùng chung cho bài toán hồi quy dự đoán điểm G3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import GRADE_MAX, GRADE_MIN, PREDICTION_DECIMALS


def validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str = "Dữ liệu",
) -> pd.DataFrame:
    """Kiểm tra DataFrame và trả về bản sao theo đúng thứ tự cột yêu cầu.

    Hàm kiểm tra kiểu dữ liệu, dữ liệu rỗng, tên cột trùng, cột bắt buộc bị
    thiếu và giá trị khuyết trong các cột model sử dụng.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{context} phải là pandas.DataFrame.")

    if dataframe.empty:
        raise ValueError(f"{context} không có dòng dữ liệu nào.")

    normalized_columns = [str(column) for column in required_columns]
    if not normalized_columns:
        raise ValueError("Danh sách cột bắt buộc không được rỗng.")

    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("Danh sách cột bắt buộc đang chứa tên cột trùng.")

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            f"{context} có tên cột bị trùng: "
            + ", ".join(map(str, duplicate_columns))
        )

    missing_columns = [
        column for column in normalized_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{context} thiếu cột bắt buộc: " + ", ".join(missing_columns)
        )

    selected = dataframe.loc[:, normalized_columns].copy()
    missing_counts = selected.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    if not columns_with_missing.empty:
        details = ", ".join(
            f"{column} ({int(count)} ô)"
            for column, count in columns_with_missing.items()
        )
        raise ValueError(
            f"{context} có giá trị thiếu trong các cột bắt buộc: {details}"
        )

    return selected


def as_finite_1d_array(values: Any, *, name: str = "Dữ liệu") -> np.ndarray:
    """Chuyển dữ liệu số thành mảng float một chiều hữu hạn."""

    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} không thể chuyển thành dữ liệu số.") from exc

    if array.size == 0:
        raise ValueError(f"{name} không được rỗng.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} chứa NaN hoặc giá trị vô hạn.")

    return array


def normalize_grade_scores(
    values: Any,
    *,
    minimum: float = GRADE_MIN,
    maximum: float = GRADE_MAX,
    decimals: int = PREDICTION_DECIMALS,
) -> np.ndarray:
    """Giới hạn điểm dự đoán vào thang G3 và làm tròn kết quả.

    Đây chỉ là hậu xử lý đầu ra hồi quy, không quy đổi điểm thành nhãn lớp.
    """

    if minimum >= maximum:
        raise ValueError("Điểm tối thiểu phải nhỏ hơn điểm tối đa.")

    if decimals < 0:
        raise ValueError("Số chữ số làm tròn không được âm.")

    scores = as_finite_1d_array(values, name="Điểm dự đoán")
    clipped_scores = np.clip(scores, minimum, maximum)

    # Điểm sau khi clip luôn không âm; cách này làm tròn .5 theo hướng lên,
    # tránh quy tắc làm tròn về số chẵn của ``numpy.round``.
    factor = 10**decimals
    return np.floor(clipped_scores * factor + 0.5) / factor
