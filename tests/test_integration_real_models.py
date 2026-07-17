"""Integration test dùng model artifact và dữ liệu test thật của project."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import TEST_PATH  # noqa: E402
from src.config.settings import (  # noqa: E402
    GRADE_MAX,
    GRADE_MIN,
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.inference.predictor import (  # noqa: E402
    PredictionMode,
    PredictionService,
)


@pytest.mark.parametrize(
    ("prediction_mode", "expected_scenario", "expected_feature_count"),
    [
        ("with_g1_g2", SCENARIO_WITH_PREVIOUS_GRADES, 33),
        ("early_prediction", SCENARIO_WITHOUT_PREVIOUS_GRADES, 31),
    ],
)
def test_real_model_predicts_one_test_row(
    prediction_mode: PredictionMode,
    expected_scenario: str,
    expected_feature_count: int,
) -> None:
    """Load model thật, dự đoán một dòng thật và kiểm tra đầu ra G3."""

    assert TEST_PATH.is_file(), f"Không tìm thấy dữ liệu test: {TEST_PATH}"

    test_dataframe = pd.read_csv(TEST_PATH)
    assert not test_dataframe.empty, "test.csv đang rỗng."

    # Coi cảnh báo lệch phiên bản scikit-learn là lỗi thật. Điều này giúp test
    # phát hiện model được train bằng phiên bản khác với môi trường đang chạy.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Setting the shape on a NumPy array has been deprecated.*"
            ),
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        warnings.simplefilter("error", InconsistentVersionWarning)
        service = PredictionService(prediction_mode)

    assert service.paths.model.is_file()
    assert service.best_scenario == expected_scenario
    assert len(service.input_features) == expected_feature_count
    assert "G3" not in service.input_features

    uses_previous_grades = prediction_mode == "with_g1_g2"
    assert ("G1" in service.input_features) is uses_previous_grades
    assert ("G2" in service.input_features) is uses_previous_grades

    sample_features = (
        test_dataframe.loc[:, service.input_features].iloc[0].to_dict()
    )
    result = service.predict(sample_features)

    assert set(result) == {"predicted_g3", "model", "scenario"}
    assert result["scenario"] == expected_scenario
    assert isinstance(result["model"], str) and result["model"]

    predicted_g3 = float(result["predicted_g3"])
    assert np.isfinite(predicted_g3)
    assert GRADE_MIN <= predicted_g3 <= GRADE_MAX
