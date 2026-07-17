"""Kiểm thử các helper dùng chung cho hồi quy G3."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import (  # noqa: E402
    as_finite_1d_array,
    normalize_grade_scores,
    validate_dataframe,
)


def test_validate_dataframe_selects_required_columns_in_order() -> None:
    dataframe = pd.DataFrame({"extra": [9], "G2": [12], "G1": [11]})

    result = validate_dataframe(dataframe, ["G1", "G2"])

    assert list(result.columns) == ["G1", "G2"]
    assert result.iloc[0].to_dict() == {"G1": 11, "G2": 12}


def test_validate_dataframe_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="G2"):
        validate_dataframe(pd.DataFrame({"G1": [10]}), ["G1", "G2"])


def test_validate_dataframe_rejects_missing_values() -> None:
    dataframe = pd.DataFrame({"G1": [10], "G2": [np.nan]})

    with pytest.raises(ValueError, match="G2"):
        validate_dataframe(dataframe, ["G1", "G2"])


def test_as_finite_1d_array_rejects_infinite_values() -> None:
    with pytest.raises(ValueError, match="vô hạn"):
        as_finite_1d_array([1.0, np.inf])


def test_normalize_grade_scores_clips_and_rounds() -> None:
    result = normalize_grade_scores([-1.5, 12.345, 21.0])

    np.testing.assert_array_equal(result, np.array([0.0, 12.35, 20.0]))
