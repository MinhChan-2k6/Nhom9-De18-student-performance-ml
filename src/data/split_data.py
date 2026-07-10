"""
Chia dữ liệu sạch thành các tập train, validation và test.
Luồng xử lý:

    data/interim/cleaned.csv -> data/processed/train.csv
                             -> data/processed/valid.csv
                             -> data/processed/test.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLEANED_FILE = INTERIM_DIR / "cleaned.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALID_FILE = PROCESSED_DIR / "valid.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


def load_cleaned_data(input_path: Path = CLEANED_FILE) -> pd.DataFrame:
    """Đọc file cleaned.csv và kiểm tra các cột cần để chia dữ liệu."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu sạch: {input_path}. "
            "Hãy chạy `python src\\data\\clean_data.py` trước."
        )

    df = pd.read_csv(input_path)
    required_columns = {"subject", "G3"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"File cleaned.csv bị thiếu các cột cần thiết: {missing_columns}"
        )

    if df.empty:
        raise ValueError("File cleaned.csv đang không có dữ liệu để chia.")

    if df[["subject", "G3"]].isna().any().any():
        raise ValueError("Cột subject hoặc G3 đang có giá trị bị thiếu.")

    return df


def _make_stratify_key(df: pd.DataFrame) -> pd.Series:
    """Tạo nhãn tạm để giữ tỷ lệ môn học và đạt/trượt khi chia."""
    passed = (pd.to_numeric(df["G3"], errors="raise") >= 10).map(
        {True: "passed", False: "failed"}
    )
    return df["subject"].astype(str) + "_" + passed


def split_dataframe(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia DataFrame thành train, validation và test."""
    total_ratio = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("Tổng tỷ lệ train, validation và test phải bằng 1.")

    stratify_key = _make_stratify_key(df)

    train_df, temporary_df = train_test_split(
        df,
        test_size=VALID_RATIO + TEST_RATIO,
        random_state=random_state,
        stratify=stratify_key,
    )

    temporary_key = _make_stratify_key(temporary_df)
    valid_df, test_df = train_test_split(
        temporary_df,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        random_state=random_state,
        stratify=temporary_key,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    input_path: Path = CLEANED_FILE,
    train_path: Path = TRAIN_FILE,
    valid_path: Path = VALID_FILE,
    test_path: Path = TEST_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu sạch và lưu ba file vào data/processed."""
    df = load_cleaned_data(input_path)
    train_df, valid_df, test_df = split_dataframe(df)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, valid_df, test_df


def main() -> None:
    # Để terminal Windows in được thông báo tiếng Việt có dấu.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    train_df, valid_df, test_df = save_splits()
    print("Đã chia và lưu dữ liệu:")
    print(f"- train.csv: {len(train_df)} dòng")
    print(f"- valid.csv: {len(valid_df)} dòng")
    print(f"- test.csv: {len(test_df)} dòng")
    print(f"Tổng số dòng: {len(train_df) + len(valid_df) + len(test_df)}")
    print(f"Random state: {RANDOM_STATE}")


if __name__ == "__main__":
    main()
