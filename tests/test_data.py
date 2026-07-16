"""Kiểm thử dữ liệu cho bài toán hồi quy dự đoán điểm G3."""

from pathlib import Path
import sys

import pandas as pd
import pytest


# Thêm thư mục gốc vào sys.path để có thể import các file trong src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.clean_data import (  # noqa: E402
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
    STRING_COLUMNS,
    VALID_RANGES,
    YES_NO_COLUMNS,
    clean_dataframe,
    load_merged_data,
    save_cleaned_data,
)
from src.data.load_data import (  # noqa: E402
    MATH_FILE,
    PORTUGUESE_FILE,
    load_subject_data,
    merge_raw_data,
    save_merged_data,
)
from src.data.split_data import (  # noqa: E402
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    load_cleaned_data,
    save_splits,
    split_dataframe,
)


# Theo mô tả của bộ Student Performance, các cột dưới đây được dùng để nhận
# diện cùng một học sinh xuất hiện trong dữ liệu Toán và Tiếng Bồ Đào Nha.
STUDENT_KEY_COLUMNS = [
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


@pytest.fixture(scope="module")
def merged_df() -> pd.DataFrame:
    #Dữ liệu sau khi gộp hai môn, dùng chung cho nhiều bài test.
    return merge_raw_data()


@pytest.fixture(scope="module")
def cleaned_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    #Dữ liệu sau khi làm sạch, chưa chia train/valid/test.
    cleaned, _ = clean_dataframe(merged_df)
    return cleaned


@pytest.fixture(scope="module")
def split_dfs(
    cleaned_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Ba tập dữ liệu được chia với random_state mặc định. 
    return split_dataframe(cleaned_df)


def _row_hashes(df: pd.DataFrame) -> set[int]:
    #Tạo tập hash để kiểm tra dòng giống hệt nhau giữa các tập.
    return set(pd.util.hash_pandas_object(df, index=False).tolist())


def _student_ids(df: pd.DataFrame) -> set[str]:
    #Tạo mã học sinh từ các cột nhận dạng chung của hai môn.
    missing_columns = set(STUDENT_KEY_COLUMNS) - set(df.columns)
    assert not missing_columns, f"Thiếu cột nhận dạng học sinh: {missing_columns}"

    student_id = df[STUDENT_KEY_COLUMNS].astype(str).agg("|".join, axis=1)
    return set(student_id)


# ---------------------------------------------------------------------------
# 1. Test đọc và gộp dữ liệu
# ---------------------------------------------------------------------------


def test_raw_files_exist() -> None:
    #Hai file dữ liệu gốc phải tồn tại và không được rỗng.
    assert MATH_FILE.exists(), f"Không tìm thấy file: {MATH_FILE}"
    assert PORTUGUESE_FILE.exists(), f"Không tìm thấy file: {PORTUGUESE_FILE}"
    assert MATH_FILE.stat().st_size > 0
    assert PORTUGUESE_FILE.stat().st_size > 0


def test_load_math_data() -> None:
    #File Toán phải đọc được và có đúng số dòng của bộ dữ liệu gốc.
    math_df = load_subject_data(MATH_FILE, "math")

    assert math_df.shape == (395, 34)
    assert math_df.columns[0] == "subject"
    assert set(math_df["subject"].unique()) == {"math"}


def test_load_portuguese_data() -> None:
    #File Tiếng Bồ Đào Nha phải đọc được và có đúng số dòng.
    portuguese_df = load_subject_data(PORTUGUESE_FILE, "portuguese")

    assert portuguese_df.shape == (649, 34)
    assert portuguese_df.columns[0] == "subject"
    assert set(portuguese_df["subject"].unique()) == {"portuguese"}


def test_load_subject_data_raises_when_file_missing(tmp_path: Path) -> None:
    #Đường dẫn sai phải báo FileNotFoundError rõ ràng.
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_subject_data(missing_file, "math")


def test_merge_raw_data_shape_and_subject_counts(merged_df: pd.DataFrame) -> None:
    #Dữ liệu gộp phải giữ đủ dữ liệu của cả hai môn. 
    assert merged_df.shape == (1044, 34)
    assert list(merged_df.columns) == EXPECTED_COLUMNS
    assert merged_df["subject"].value_counts().to_dict() == {
        "portuguese": 649,
        "math": 395,
    }


def test_save_merged_data_uses_temporary_file(tmp_path: Path) -> None:
    #Hàm lưu merged.csv phải tạo file đúng mà không đụng dữ liệu thật 
    output_path = tmp_path / "merged.csv"
    result = save_merged_data(output_path)

    assert output_path.exists()
    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved_df, result, check_dtype=False)


def test_load_merged_data_rejects_wrong_schema(
    tmp_path: Path,
    merged_df: pd.DataFrame,
) -> None:
    #File merged.csv thiếu cột phải bị từ chối
    wrong_file = tmp_path / "wrong_merged.csv"
    merged_df.drop(columns=["G3"]).to_csv(wrong_file, index=False)

    with pytest.raises(ValueError, match="Cấu trúc merged.csv không đúng"):
        load_merged_data(wrong_file)


# ---------------------------------------------------------------------------
# 2. Test làm sạch dữ liệu
# ---------------------------------------------------------------------------


def test_clean_data_has_correct_schema(cleaned_df: pd.DataFrame) -> None:
    #Dữ liệu sạch phải giữ đúng tên và thứ tự các cột.
    assert list(cleaned_df.columns) == EXPECTED_COLUMNS
    assert cleaned_df.shape == (1044, 34)


def test_g3_is_valid_regression_target(cleaned_df: pd.DataFrame) -> None:
    #G3 phải là biến mục tiêu dạng số trong khoảng điểm từ 0 đến 20.
    assert "G3" in cleaned_df.columns
    assert pd.api.types.is_numeric_dtype(cleaned_df["G3"])
    assert cleaned_df["G3"].notna().all()
    assert cleaned_df["G3"].between(0, 20).all()
    assert cleaned_df["G3"].nunique() > 1


def test_clean_data_has_no_missing_or_duplicate_rows(
    cleaned_df: pd.DataFrame,
) -> None:
    #Sau làm sạch không còn ô trống hoặc dòng trùng hoàn toàn. 
    assert int(cleaned_df.isna().sum().sum()) == 0
    assert int(cleaned_df.duplicated().sum()) == 0


def test_clean_data_has_correct_column_types(cleaned_df: pd.DataFrame) -> None:
    #Cột số phải là kiểu số và cột chữ phải được chuẩn hóa.
    for column in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(cleaned_df[column]), (
            f"Cột {column} chưa phải kiểu số"
        )

    for column in STRING_COLUMNS:
        values = cleaned_df[column].dropna().astype(str)
        assert values.str.strip().equals(values)
        assert values.str.lower().equals(values)


def test_clean_data_values_are_in_valid_ranges(cleaned_df: pd.DataFrame) -> None:
    #Các cột số phải nằm trong khoảng giá trị đã quy định. 
    for column, (minimum, maximum) in VALID_RANGES.items():
        assert cleaned_df[column].between(minimum, maximum).all(), (
            f"Cột {column} có giá trị nằm ngoài khoảng {minimum}-{maximum}"
        )


def test_clean_data_yes_no_columns_are_valid(cleaned_df: pd.DataFrame) -> None:
    #Các cột dạng Có/Không chỉ được chứa yes hoặc no 
    for column in YES_NO_COLUMNS:
        assert set(cleaned_df[column].unique()).issubset({"yes", "no"})


def test_clean_dataframe_normalizes_and_removes_duplicates(
    merged_df: pd.DataFrame,
) -> None:
    #Hàm làm sạch phải sửa chữ, kiểu số và xóa dòng trùng 
    sample = merged_df.iloc[:3].copy()
    sample.loc[0, "school"] = "  GP  "
    sample["age"] = sample["age"].astype(object)
    sample.loc[0, "age"] = "18"
    sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)

    cleaned, duplicate_count = clean_dataframe(sample)

    assert duplicate_count == 1
    assert len(cleaned) == 3
    assert cleaned.loc[0, "school"] == "gp"
    assert pd.api.types.is_numeric_dtype(cleaned["age"])


def test_clean_dataframe_does_not_change_original_data(
    merged_df: pd.DataFrame,
) -> None:
    #Hàm clean_dataframe phải làm việc trên bản sao của DataFrame. 
    sample = merged_df.iloc[:5].copy()
    original = sample.copy(deep=True)

    clean_dataframe(sample)

    pd.testing.assert_frame_equal(sample, original)


def test_clean_dataframe_rejects_missing_value(merged_df: pd.DataFrame) -> None:
    #Dữ liệu thiếu phải được phát hiện thay vì âm thầm bỏ qua.
    sample = merged_df.iloc[:3].copy()
    sample.loc[sample.index[0], "age"] = None

    with pytest.raises(ValueError, match="dữ liệu thiếu|không hợp lệ"):
        clean_dataframe(sample)


def test_clean_dataframe_rejects_invalid_yes_no_value(
    merged_df: pd.DataFrame,
) -> None:
    #Giá trị khác yes/no trong cột nhị phân phải bị từ chối. 
    sample = merged_df.iloc[:3].copy()
    sample.loc[sample.index[0], "internet"] = "maybe"

    with pytest.raises(ValueError, match="internet"):
        clean_dataframe(sample)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("age", 14),
        ("studytime", 5),
        ("G1", 21),
        ("G2", -1),
        ("G3", 25),
    ],
)
def test_clean_dataframe_rejects_out_of_range_values(
    merged_df: pd.DataFrame,
    column: str,
    invalid_value: int,
) -> None:
    #Giá trị số ngoài khoảng hợp lệ phải báo lỗi. 
    sample = merged_df.iloc[:3].copy()
    sample.loc[sample.index[0], column] = invalid_value

    with pytest.raises(ValueError, match=column):
        clean_dataframe(sample)


def test_save_cleaned_data_uses_temporary_files(
    tmp_path: Path,
    merged_df: pd.DataFrame,
) -> None:
    #Hàm lưu cleaned.csv phải hoạt động với đường dẫn tùy chọn.
    input_path = tmp_path / "merged.csv"
    output_path = tmp_path / "cleaned.csv"
    merged_df.to_csv(input_path, index=False)

    result, duplicate_count = save_cleaned_data(input_path, output_path)

    assert output_path.exists()
    assert duplicate_count == 0
    saved_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved_df, result, check_dtype=False)


# ---------------------------------------------------------------------------
# 3. Test chia train, validation và test
# ---------------------------------------------------------------------------


def test_load_cleaned_data_rejects_missing_file(tmp_path: Path) -> None:
    """Không có cleaned.csv thì phải báo FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_cleaned_data(tmp_path / "missing_cleaned.csv")


def test_split_ratios_are_reasonable(
    cleaned_df: pd.DataFrame,
    split_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    #Kích thước các tập phải gần với tỷ lệ 70/15/15.
    train_df, valid_df, test_df = split_dfs
    total_rows = len(cleaned_df)

    assert len(train_df) + len(valid_df) + len(test_df) == total_rows
    assert abs(len(train_df) / total_rows - TRAIN_RATIO) <= 0.05
    assert abs(len(valid_df) / total_rows - VALID_RATIO) <= 0.05
    assert abs(len(test_df) / total_rows - TEST_RATIO) <= 0.05


def test_split_keeps_all_columns(
    cleaned_df: pd.DataFrame,
    split_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    #Sau khi chia, ba tập phải giữ nguyên schema.
    for frame in split_dfs:
        assert list(frame.columns) == list(cleaned_df.columns)


def test_split_has_no_exact_duplicate_rows_between_sets(
    split_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    #Không được có dòng giống hệt nhau xuất hiện ở hai tập
    train_df, valid_df, test_df = split_dfs
    train_hashes = _row_hashes(train_df)
    valid_hashes = _row_hashes(valid_df)
    test_hashes = _row_hashes(test_df)

    assert train_hashes.isdisjoint(valid_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert valid_hashes.isdisjoint(test_hashes)


def test_same_student_does_not_appear_in_different_sets(
    split_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    #Hai môn của cùng một học sinh phải nằm trong cùng một tập dữ liệu.
    train_df, valid_df, test_df = split_dfs
    train_ids = _student_ids(train_df)
    valid_ids = _student_ids(valid_df)
    test_ids = _student_ids(test_df)

    overlap_counts = {
        "train-valid": len(train_ids & valid_ids),
        "train-test": len(train_ids & test_ids),
        "valid-test": len(valid_ids & test_ids),
    }

    assert all(count == 0 for count in overlap_counts.values()), (
        f"Phát hiện học sinh bị chia chéo giữa các tập: {overlap_counts}"
    )


def test_split_is_reproducible(cleaned_df: pd.DataFrame) -> None:
    """Cùng random_state phải tạo ra kết quả giống nhau."""
    first_result = split_dataframe(cleaned_df, random_state=42)
    second_result = split_dataframe(cleaned_df, random_state=42)

    for first_df, second_df in zip(first_result, second_result):
        pd.testing.assert_frame_equal(first_df, second_df)


def test_subject_and_g3_distribution_are_not_too_different(
    cleaned_df: pd.DataFrame,
    split_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    #Phân phối môn học và điểm G3 không nên lệch quá nhiều sau khi chia
    original_subject_rates = cleaned_df["subject"].value_counts(normalize=True)
    original_g3_mean = float(cleaned_df["G3"].mean())
    original_g3_std = float(cleaned_df["G3"].std())
    original_g3_quartiles = cleaned_df["G3"].quantile([0.25, 0.50, 0.75])

    for frame in split_dfs:
        split_subject_rates = frame["subject"].value_counts(normalize=True)
        split_g3_mean = float(frame["G3"].mean())
        split_g3_std = float(frame["G3"].std())
        split_g3_quartiles = frame["G3"].quantile([0.25, 0.50, 0.75])

        for subject, original_rate in original_subject_rates.items():
            split_rate = float(split_subject_rates.get(subject, 0.0))
            assert abs(split_rate - original_rate) <= 0.08

        assert abs(split_g3_mean - original_g3_mean) <= 1.0
        assert abs(split_g3_std - original_g3_std) <= 1.0

        for quantile in original_g3_quartiles.index:
            assert (
                abs(
                    float(split_g3_quartiles.loc[quantile])
                    - float(original_g3_quartiles.loc[quantile])
                )
                <= 1.5
            )


def test_save_splits_uses_temporary_files(
    tmp_path: Path,
    cleaned_df: pd.DataFrame,
) -> None:
    #save_splits phải tạo đủ ba file ở đường dẫn được truyền vào
    input_path = tmp_path / "cleaned.csv"
    train_path = tmp_path / "processed" / "train.csv"
    valid_path = tmp_path / "processed" / "valid.csv"
    test_path = tmp_path / "processed" / "test.csv"
    cleaned_df.to_csv(input_path, index=False)

    train_df, valid_df, test_df = save_splits(
        input_path=input_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
    )

    assert train_path.exists()
    assert valid_path.exists()
    assert test_path.exists()
    assert len(pd.read_csv(train_path)) == len(train_df)
    assert len(pd.read_csv(valid_path)) == len(valid_df)
    assert len(pd.read_csv(test_path)) == len(test_df)
