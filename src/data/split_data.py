"""
Mục đích:
- Các bản ghi thuộc cùng một học sinh phải nằm trong cùng một tập.
- Tránh việc một học sinh xuất hiện đồng thời ở train, validation hoặc test.

Luồng xử lý:
    data/interim/cleaned.csv -> data/processed/train.csv
                             -> data/processed/valid.csv
                             -> data/processed/test.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


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

STUDENT_ID_COLUMNS = [
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
    "nursery",
    "internet",
]


def load_cleaned_data(input_path: Path = CLEANED_FILE) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu sạch: {input_path}. "
            "Hãy chạy `python src\\data\\clean_data.py` trước."
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "subject",
        "G3",
        *STUDENT_ID_COLUMNS,
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            "File cleaned.csv bị thiếu các cột cần thiết: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "File cleaned.csv đang không có dữ liệu để chia."
        )

    columns_to_check = [
        "subject",
        "G3",
        *STUDENT_ID_COLUMNS,
    ]

    if df[columns_to_check].isna().any().any():
        raise ValueError(
            "Các cột subject, G3 hoặc cột nhận dạng học sinh "
            "đang có giá trị bị thiếu."
        )

    return df


def create_student_group_id(df: pd.DataFrame) -> pd.Series:
    missing_columns = sorted(
        set(STUDENT_ID_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Không thể tạo mã học sinh vì thiếu các cột: "
            f"{missing_columns}"
        )

    student_information = (
        df[STUDENT_ID_COLUMNS]
        .astype(str)
        .fillna("")
    )

    # Tạo mã ổn định từ các thông tin nhận dạng.
    student_group_id = pd.util.hash_pandas_object(
        student_information,
        index=False,
    ).astype(str)

    student_group_id.name = "student_group_id"

    return student_group_id


def check_no_group_overlap(
    train_groups: pd.Series,
    valid_groups: pd.Series,
    test_groups: pd.Series,
) -> None:
    train_group_set = set(train_groups)
    valid_group_set = set(valid_groups)
    test_group_set = set(test_groups)

    train_valid_overlap = train_group_set & valid_group_set
    train_test_overlap = train_group_set & test_group_set
    valid_test_overlap = valid_group_set & test_group_set

    if train_valid_overlap:
        raise ValueError(
            "Phát hiện học sinh xuất hiện đồng thời "
            "trong train và validation."
        )

    if train_test_overlap:
        raise ValueError(
            "Phát hiện học sinh xuất hiện đồng thời "
            "trong train và test."
        )

    if valid_test_overlap:
        raise ValueError(
            "Phát hiện học sinh xuất hiện đồng thời "
            "trong validation và test."
        )


def split_dataframe(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_ratio = TRAIN_RATIO + VALID_RATIO + TEST_RATIO

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "Tổng tỷ lệ train, validation và test phải bằng 1."
        )

    working_df = df.copy()

    # Tạo mã nhóm học sinh.
    working_df["student_group_id"] = create_student_group_id(
        working_df
    )

    first_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_RATIO,
        random_state=random_state,
    )

    train_indices, temporary_indices = next(
        first_splitter.split(
            working_df,
            groups=working_df["student_group_id"],
        )
    )

    train_df = working_df.iloc[train_indices].copy()
    temporary_df = working_df.iloc[temporary_indices].copy()

    #chia tập tạm thời thành validation và test.
    temporary_test_ratio = TEST_RATIO / (
        VALID_RATIO + TEST_RATIO
    )

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=temporary_test_ratio,
        random_state=random_state,
    )

    valid_indices, test_indices = next(
        second_splitter.split(
            temporary_df,
            groups=temporary_df["student_group_id"],
        )
    )

    valid_df = temporary_df.iloc[valid_indices].copy()
    test_df = temporary_df.iloc[test_indices].copy()

    # Kiểm tra học sinh không bị trùng giữa các tập.
    check_no_group_overlap(
        train_df["student_group_id"],
        valid_df["student_group_id"],
        test_df["student_group_id"],
    )

    # Không lưu mã nhóm tạm vào file CSV.
    train_df = train_df.drop(
        columns=["student_group_id"]
    ).reset_index(drop=True)

    valid_df = valid_df.drop(
        columns=["student_group_id"]
    ).reset_index(drop=True)

    test_df = test_df.drop(
        columns=["student_group_id"]
    ).reset_index(drop=True)

    return train_df, valid_df, test_df


def save_splits(
    input_path: Path = CLEANED_FILE,
    train_path: Path = TRAIN_FILE,
    valid_path: Path = VALID_FILE,
    test_path: Path = TEST_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_cleaned_data(input_path)

    train_df, valid_df, test_df = split_dataframe(df)

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    valid_df.to_csv(
        valid_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    return train_df, valid_df, test_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    original_df = load_cleaned_data()
    original_groups = create_student_group_id(original_df)

    train_df, valid_df, test_df = save_splits()

    train_groups = create_student_group_id(train_df)
    valid_groups = create_student_group_id(valid_df)
    test_groups = create_student_group_id(test_df)

    # Kiểm tra lại sau khi đã chia và loại cột nhóm tạm.
    check_no_group_overlap(
        train_groups,
        valid_groups,
        test_groups,
    )

    total_rows = (
        len(train_df)
        + len(valid_df)
        + len(test_df)
    )

    print("Đã chia và lưu dữ liệu theo nhóm học sinh:")
    print(
        f"- train.csv: {len(train_df)} dòng, "
        f"{train_groups.nunique()} học sinh"
    )
    print(
        f"- valid.csv: {len(valid_df)} dòng, "
        f"{valid_groups.nunique()} học sinh"
    )
    print(
        f"- test.csv: {len(test_df)} dòng, "
        f"{test_groups.nunique()} học sinh"
    )

    print(f"Tổng số dòng: {total_rows}")
    print(
        "Tổng số học sinh: "
        f"{original_groups.nunique()}"
    )
    print(f"Random state: {RANDOM_STATE}")
    print(
        "Kiểm tra trùng học sinh: "
        "không có học sinh xuất hiện ở nhiều tập."
    )


if __name__ == "__main__":
    main()