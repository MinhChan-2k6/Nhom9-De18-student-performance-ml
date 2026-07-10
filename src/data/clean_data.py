"""Làm sạch bộ dữ liệu điểm học sinh sau khi đã gộp hai môn.

Luồng xử lý:

    data/interim/merged.csv -> data/interim/cleaned.csv

File dữ liệu gốc trong data/raw sẽ không bị thay đổi. Có thể chạy lại file này
mỗi khi chạy load_data.py để tạo lại dữ liệu đã làm sạch.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MERGED_FILE = INTERIM_DIR / "merged.csv"
CLEANED_FILE = INTERIM_DIR / "cleaned.csv"


EXPECTED_COLUMNS = [
    "subject",
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
]

NUMERIC_COLUMNS = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
]

STRING_COLUMNS = [column for column in EXPECTED_COLUMNS if column not in NUMERIC_COLUMNS]

# Khoảng giá trị hợp lệ của các cột số trong bộ dữ liệu.
VALID_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}

YES_NO_COLUMNS = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]


def load_merged_data(input_path: Path = MERGED_FILE) -> pd.DataFrame:
    """Đọc file merged.csv và kiểm tra tên, thứ tự các cột."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Merged data file not found: {input_path}. "
            "Hãy chạy `python src\\data\\load_data.py` trước."
        )

    df = pd.read_csv(input_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns or unexpected_columns or list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Cấu trúc merged.csv không đúng. "
            f"Các cột bị thiếu: {missing_columns}; "
            f"các cột không mong đợi: {unexpected_columns}; "
            f"thứ tự cột cần có: {EXPECTED_COLUMNS}"
        )

    return df


def _normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa khoảng trắng thừa và đưa dữ liệu dạng chữ về chữ thường."""
    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string").str.strip().str.lower()
    return df


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi các cột cần tính toán sang kiểu dữ liệu số."""
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _validate_no_missing_values(df: pd.DataFrame) -> None:
    """Kiểm tra dữ liệu thiếu hoặc giá trị không đổi được sang số."""
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_counts.items()
        )
        raise ValueError(
            "Phát hiện dữ liệu thiếu hoặc không hợp lệ sau khi chuyển kiểu: "
            f"{details}. Chương trình không tự động điền hoặc xóa các dòng này."
        )


def _validate_values(df: pd.DataFrame) -> None:
    """Kiểm tra các nhóm giá trị và khoảng điểm trước khi lưu file."""
    valid_subjects = {"math", "portuguese"}
    subjects = set(df["subject"].dropna().unique())
    invalid_subjects = subjects - valid_subjects
    if invalid_subjects:
        raise ValueError(
            f"Giá trị không hợp lệ trong cột subject: {sorted(invalid_subjects)}"
        )

    for column in YES_NO_COLUMNS:
        values = set(df[column].dropna().unique())
        invalid_values = values - {"yes", "no"}
        if invalid_values:
            raise ValueError(
                f"Giá trị không hợp lệ trong cột {column}: {sorted(invalid_values)}. "
                "Chỉ chấp nhận hai giá trị 'yes' hoặc 'no'."
            )

    for column, (minimum, maximum) in VALID_RANGES.items():
        invalid_mask = ~df[column].between(minimum, maximum)
        if invalid_mask.any():
            invalid_values = sorted(df.loc[invalid_mask, column].unique().tolist())
            raise ValueError(
                f"Cột {column} phải có giá trị từ {minimum} đến {maximum}. "
                f"Các giá trị sai: {invalid_values}"
            )


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Làm sạch dữ liệu và trả về dữ liệu mới cùng số dòng trùng đã xóa."""
    cleaned_df = df.copy()
    cleaned_df = _normalize_string_columns(cleaned_df)
    cleaned_df = _convert_numeric_columns(cleaned_df)

    before_deduplication = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(ignore_index=True)
    duplicate_count = before_deduplication - len(cleaned_df)

    _validate_no_missing_values(cleaned_df)
    _validate_values(cleaned_df)
    return cleaned_df, duplicate_count


def save_cleaned_data(
    input_path: Path = MERGED_FILE,
    output_path: Path = CLEANED_FILE,
) -> tuple[pd.DataFrame, int]:
    """Làm sạch merged.csv, lưu thành cleaned.csv và trả về kết quả."""
    merged_df = load_merged_data(input_path)
    cleaned_df, duplicate_count = clean_dataframe(merged_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df, duplicate_count


def main() -> None:
    # Để terminal Windows in được thông báo tiếng Việt có dấu.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cleaned_df, duplicate_count = save_cleaned_data()
    print(f"Đã lưu dữ liệu sạch tại: {CLEANED_FILE}")
    print(f"Số dòng: {cleaned_df.shape[0]}, số cột: {cleaned_df.shape[1]}")
    print(f"Số dòng trùng đã xóa: {duplicate_count}")
    print("Số giá trị bị thiếu còn lại: 0")


if __name__ == "__main__":
    main()
