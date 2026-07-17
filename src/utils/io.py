"""Các hàm đọc và ghi file dùng chung trong toàn bộ project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


def _as_path(file_path: str | Path) -> Path:
    """Chuẩn hóa đường dẫn đầu vào thành ``Path``."""

    return file_path if isinstance(file_path, Path) else Path(file_path)


def _validate_input_file(file_path: str | Path) -> Path:
    """Kiểm tra file đầu vào có tồn tại, là file và không rỗng."""

    path = _as_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    if not path.is_file():
        raise ValueError(f"Đường dẫn không phải là file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File đang rỗng: {path}")

    return path


def ensure_parent_directory(file_path: str | Path) -> Path:
    """Tạo thư mục cha trước khi ghi file và trả về đường dẫn đã chuẩn hóa."""

    path = _as_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Đọc CSV với thông báo lỗi rõ ràng."""

    path = _validate_input_file(file_path)

    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError as exc:
        raise ValueError(f"File CSV không có dữ liệu: {path}") from exc
    except (ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Không thể đọc file CSV hợp lệ: {path}") from exc


def save_csv(
    dataframe: pd.DataFrame,
    file_path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Ghi DataFrame ra CSV và tự tạo thư mục cha."""

    path = ensure_parent_directory(file_path)
    dataframe.to_csv(path, index=index, **kwargs)
    return path


def load_json(file_path: str | Path) -> Any:
    """Đọc và giải mã file JSON."""

    path = _validate_input_file(file_path)

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"File JSON không đúng định dạng: {path}") from exc


def save_json(
    data: Any,
    file_path: str | Path,
    *,
    indent: int = 4,
) -> Path:
    """Ghi dữ liệu ra JSON UTF-8 và tự tạo thư mục cha."""

    path = ensure_parent_directory(file_path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

    return path


def load_joblib(file_path: str | Path) -> Any:
    """Load model hoặc object đã lưu bằng joblib."""

    path = _validate_input_file(file_path)

    try:
        return joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Không thể load file joblib: {path}") from exc


def save_joblib(data: Any, file_path: str | Path) -> Path:
    """Lưu model hoặc object bằng joblib và tự tạo thư mục cha."""

    path = ensure_parent_directory(file_path)
    joblib.dump(data, path)
    return path


def save_text(
    content: str,
    file_path: str | Path,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Ghi nội dung văn bản và tự tạo thư mục cha."""

    path = ensure_parent_directory(file_path)
    path.write_text(content, encoding=encoding)
    return path
