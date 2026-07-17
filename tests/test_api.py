"""Kiểm thử schema và endpoint dự đoán điểm G3."""

from pathlib import Path
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402


BASE_FEATURES = {
    "subject": "math",
    "school": "gp",
    "sex": "f",
    "age": 17,
    "address": "u",
    "famsize": "gt3",
    "Pstatus": "t",
    "Medu": 3,
    "Fedu": 3,
    "Mjob": "services",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 2,
    "health": 4,
    "absences": 3,
}


class DummyPredictionService:
    def predict(self, features):
        return {
            "predicted_g3": 13.75,
            "model": "dummy_regressor",
            "scenario": "A_with_G1_G2",
        }


def test_predict_endpoint_returns_regression_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "api.routers.predict.get_prediction_service",
        lambda _mode: DummyPredictionService(),
    )
    client = TestClient(app)
    payload = {
        "student": {"full_name": "Nguyen Van An"},
        "prediction_mode": "with_g1_g2",
        "features": {**BASE_FEATURES, "G1": 12, "G2": 14},
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "full_name": "Nguyen Van An",
        "predicted_g3": 13.75,
        "model": "dummy_regressor",
        "scenario": "A_with_G1_G2",
    }


def test_with_previous_grades_mode_requires_g1_and_g2() -> None:
    client = TestClient(app)
    payload = {
        "student": {"full_name": "Nguyen Van An"},
        "prediction_mode": "with_g1_g2",
        "features": BASE_FEATURES,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_batch_predict_endpoint_returns_csv(monkeypatch) -> None:
    import pandas as pd

    def fake_predict_batch(dataframe, prediction_mode, *, include_input=True):
        assert prediction_mode == "with_g1_g2"
        assert include_input is True

        output = dataframe.copy() if include_input else pd.DataFrame(index=dataframe.index)
        output["predicted_g3"] = [13.5] * len(dataframe)
        output["model"] = "dummy_regressor"
        output["scenario"] = "A_with_G1_G2"
        return output

    monkeypatch.setattr(
        "api.routers.batch_predict.predict_batch_dataframe",
        fake_predict_batch,
    )

    client = TestClient(app)
    csv_dataframe = pd.DataFrame(
        [{**BASE_FEATURES, "G1": 12, "G2": 14}]
    )

    response = client.post(
        "/predict/batch",
        files={
            "file": (
                "students.csv",
                csv_dataframe.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        },
        data={
            "prediction_mode": "with_g1_g2",
            "include_input": "true",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "students_predictions.csv" in response.headers["content-disposition"]

    result = pd.read_csv(__import__("io").BytesIO(response.content))
    assert len(result) == 1
    assert result.loc[0, "predicted_g3"] == 13.5
    assert result.loc[0, "model"] == "dummy_regressor"
    assert result.loc[0, "scenario"] == "A_with_G1_G2"


def test_batch_predict_endpoint_rejects_non_csv_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        files={"file": ("students.txt", b"not a csv", "text/plain")},
        data={"prediction_mode": "early_prediction"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Chỉ chấp nhận file có phần mở rộng .csv."
