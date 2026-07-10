"""Đọc và gộp dữ liệu điểm học sinh từ hai file CSV gốc.

Luồng xử lý:

    data/raw/student-mat.csv  \
        + data/raw/student-por.csv -> data/interim/merged.csv

Hai file trong data/raw được giữ nguyên. File này chỉ đọc dữ liệu, thêm cột
subject để phân biệt môn học rồi lưu dữ liệu đã gộp vào thư mục data/interim.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

MATH_FILE = RAW_DIR / "student-mat.csv"
PORTUGUESE_FILE = RAW_DIR / "student-por.csv"
MERGED_FILE = INTERIM_DIR / "merged.csv"


def load_subject_data(path: Path, subject: str) -> pd.DataFrame:
    """Đọc dữ liệu của một môn và thêm cột subject."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc: {path}")

    df = pd.read_csv(path, sep=";")
    df.insert(0, "subject", subject)
    return df


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc hai bộ dữ liệu Toán và Tiếng Bồ Đào Nha."""
    math_df = load_subject_data(MATH_FILE, "math")
    portuguese_df = load_subject_data(PORTUGUESE_FILE, "portuguese")
    return math_df, portuguese_df


def merge_raw_data() -> pd.DataFrame:
    """Kiểm tra schema rồi gộp hai bộ dữ liệu thành một bảng."""
    math_df, portuguese_df = load_raw_data()

    if list(math_df.columns) != list(portuguese_df.columns):
        missing_in_math = sorted(set(portuguese_df.columns) - set(math_df.columns))
        missing_in_portuguese = sorted(set(math_df.columns) - set(portuguese_df.columns))
        raise ValueError(
            "Hai file dữ liệu gốc không có cùng cấu trúc cột. "
            f"Cột bị thiếu trong file Toán: {missing_in_math}; "
            f"cột bị thiếu trong file Tiếng Bồ Đào Nha: {missing_in_portuguese}"
        )

    return pd.concat([math_df, portuguese_df], ignore_index=True)


def save_merged_data(output_path: Path = MERGED_FILE) -> pd.DataFrame:
    """Gộp dữ liệu và lưu kết quả vào file merged.csv."""
    merged_df = merge_raw_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return merged_df


def main() -> None:
    # Để terminal Windows in được thông báo tiếng Việt có dấu.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    merged_df = save_merged_data()
    print(f"Đã lưu dữ liệu đã gộp tại: {MERGED_FILE}")
    print(f"Số dòng: {merged_df.shape[0]}, số cột: {merged_df.shape[1]}")
    print("Số dòng theo từng môn:")
    print("- " + merged_df["subject"].value_counts().to_string())


if __name__ == "__main__":
    main()
