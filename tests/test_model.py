"""Kiểm thử model, pipeline và luồng tinh chỉnh siêu tham số."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.settings import (  # noqa: E402
    FEATURE_SET_ENGINEERED,
    FEATURE_SET_RAW,
    GRID_SEARCH_CV,
    GRID_SEARCH_SCORING,
    RANDOM_FOREST_GRID,
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)
from src.evaluation.metrics import (  # noqa: E402
    EvaluationResult,
    RegressionMetrics,
)
from src.features.build_features import (  # noqa: E402
    add_engineered_features,
    get_engineered_feature_names,
)
from src.features.transform import build_preprocessor  # noqa: E402
from src.models.model_registry import (  # noqa: E402
    build_candidate_models,
    build_pipeline,
)
from src.models.random_forest import (  # noqa: E402
    build_random_forest_grid_search,
)
from src.training import train as train_module  # noqa: E402


@pytest.fixture()
def regression_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Tạo train/validation nhỏ có cả biến số và biến phân loại."""

    rng = np.random.default_rng(42)
    rows = 48
    student_ids = np.arange(rows) // 2
    dataframe = pd.DataFrame(
        {
            "subject": np.where(np.arange(rows) % 2 == 0, "math", "portuguese"),
            "school": np.where(student_ids % 2 == 0, "GP", "MS"),
            "sex": np.where(student_ids % 2 == 0, "F", "M"),
            "age": 15 + (student_ids % 5),
            "address": np.where(student_ids % 3 == 0, "R", "U"),
            "famsize": np.where(student_ids % 2 == 0, "GT3", "LE3"),
            "Pstatus": np.where(student_ids % 4 == 0, "A", "T"),
            "Medu": 1 + (student_ids % 4),
            "Fedu": 1 + ((student_ids + 1) % 4),
            "Mjob": np.where(student_ids % 2 == 0, "teacher", "other"),
            "Fjob": np.where(student_ids % 3 == 0, "services", "other"),
            "reason": np.where(student_ids % 2 == 0, "course", "home"),
            "nursery": np.where(student_ids % 2 == 0, "yes", "no"),
            "internet": np.where(student_ids % 3 == 0, "no", "yes"),
            "Dalc": 1 + (student_ids % 3),
            "Walc": 1 + (student_ids % 5),
            "freetime": 1 + ((student_ids + 1) % 5),
            "goout": 1 + ((student_ids + 2) % 5),
            "schoolsup": np.where(student_ids % 3 == 0, "yes", "no"),
            "famsup": np.where(student_ids % 2 == 0, "yes", "no"),
            "absences": student_ids % 9,
            "studytime": rng.integers(1, 5, size=rows),
            "failures": rng.integers(0, 3, size=rows),
            "G1": rng.integers(4, 20, size=rows),
            "G2": rng.integers(4, 20, size=rows),
        }
    )
    noise = rng.normal(0, 0.4, size=rows)
    dataframe["G3"] = (
        0.2 * dataframe["G1"]
        + 0.65 * dataframe["G2"]
        + 0.35 * dataframe["studytime"]
        - 0.6 * dataframe["failures"]
        + noise
    ).clip(0, 20)

    feature_columns = ["subject", "studytime", "failures", "G1", "G2"]
    return dataframe.iloc[:36].copy(), dataframe.iloc[36:].copy(), feature_columns


def test_candidate_registry_contains_three_baselines() -> None:
    models = build_candidate_models()

    assert set(models) == {
        "linear_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert isinstance(models["linear_regression"], LinearRegression)
    assert isinstance(models["random_forest"], RandomForestRegressor)
    assert isinstance(models["gradient_boosting"], GradientBoostingRegressor)


def test_pipeline_can_fit_and_predict_mixed_features(
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, valid_df, feature_columns = regression_frames
    preprocessor = build_preprocessor(train_df, feature_columns)
    pipeline = build_pipeline(preprocessor, LinearRegression())

    pipeline.fit(train_df[feature_columns], train_df["G3"])
    predictions = pipeline.predict(valid_df[feature_columns])

    assert isinstance(pipeline, Pipeline)
    assert predictions.shape == (len(valid_df),)
    assert np.isfinite(predictions).all()


def test_grid_search_factory_has_expected_configuration(
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, _, feature_columns = regression_frames
    preprocessor = build_preprocessor(train_df, feature_columns)
    search = build_random_forest_grid_search(preprocessor)

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert search.scoring == GRID_SEARCH_SCORING
    assert isinstance(search.cv, GroupKFold)
    assert search.cv.get_n_splits() == GRID_SEARCH_CV
    assert search.param_grid == RANDOM_FOREST_GRID
    assert set(search.estimator.named_steps) == {"preprocessor", "model"}


def test_grid_search_fits_and_exposes_best_parameters(
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, valid_df, feature_columns = regression_frames
    preprocessor = build_preprocessor(train_df, feature_columns)
    search = build_random_forest_grid_search(preprocessor)
    search.set_params(
        cv=GroupKFold(n_splits=2),
        n_jobs=1,
        param_grid={
            "model__n_estimators": [5],
            "model__max_depth": [2, None],
            "model__min_samples_split": [2],
            "model__min_samples_leaf": [1],
            "model__max_features": ["sqrt"],
        },
    )

    groups = np.arange(len(train_df)) // 2
    search.fit(
        train_df[feature_columns],
        train_df["G3"],
        groups=groups,
    )
    predictions = search.best_estimator_.predict(valid_df[feature_columns])

    assert search.best_params_["model__n_estimators"] == 5
    assert search.best_params_["model__max_depth"] in {2, None}
    assert np.isfinite(search.best_score_)
    assert np.isfinite(predictions).all()


def test_train_tuned_random_forest_records_search_metadata(
    monkeypatch: pytest.MonkeyPatch,
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, valid_df, feature_columns = regression_frames

    def build_small_search(preprocessor, feature_engineer=None):
        search = build_random_forest_grid_search(
            preprocessor,
            feature_engineer,
        )
        return search.set_params(
            cv=GroupKFold(n_splits=2),
            n_jobs=1,
            param_grid={
                "model__n_estimators": [5],
                "model__max_depth": [2],
                "model__min_samples_split": [2],
                "model__min_samples_leaf": [1],
                "model__max_features": ["sqrt"],
            },
        )

    monkeypatch.setattr(
        train_module,
        "build_random_forest_grid_search",
        build_small_search,
    )
    result = train_module.train_tuned_random_forest(
        train_df=train_df,
        valid_df=valid_df,
        scenario_name=SCENARIO_WITH_PREVIOUS_GRADES,
        feature_columns=feature_columns,
        feature_set=FEATURE_SET_RAW,
    )

    assert result.model == "random_forest_grid_search"
    assert result.tuning is not None
    assert result.tuning["method"] == "GridSearchCV"
    assert result.tuning["cv_strategy"] == "GroupKFold"
    assert result.tuning["candidate_count"] == 1
    assert result.tuning["total_fits"] == 2
    assert result.tuning["best_params"]["model__n_estimators"] == 5
    assert np.isfinite(result.rmse)


def test_train_scenario_appends_grid_search_result(
    monkeypatch: pytest.MonkeyPatch,
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, valid_df, _ = regression_frames
    monkeypatch.setattr(train_module, "build_candidate_models", lambda: {})

    def build_tuned_result(**kwargs):
        return EvaluationResult(
            scenario=SCENARIO_WITH_PREVIOUS_GRADES,
            model="random_forest_grid_search",
            metrics=RegressionMetrics(mae=1.0, rmse=1.2, r2=0.8),
            estimator=Pipeline([]),
            feature_set=kwargs["feature_set"],
            tuning={"method": "GridSearchCV"},
        )

    monkeypatch.setattr(
        train_module,
        "train_tuned_random_forest",
        build_tuned_result,
    )

    results = train_module.train_scenario(
        train_df=train_df,
        valid_df=valid_df,
        scenario_name=SCENARIO_WITH_PREVIOUS_GRADES,
    )

    assert len(results) == 2
    assert {result.feature_set for result in results} == {
        FEATURE_SET_RAW,
        FEATURE_SET_ENGINEERED,
    }
    assert results[0].to_dict()["tuning"]["method"] == "GridSearchCV"


def test_feature_engineering_creates_mean_trend_and_support_features() -> None:
    dataframe = pd.DataFrame(
        {
            "Medu": [4],
            "Fedu": [2],
            "Dalc": [1],
            "Walc": [3],
            "freetime": [2],
            "goout": [4],
            "schoolsup": ["yes"],
            "famsup": ["no"],
            "absences": [9],
            "G1": [10],
            "G2": [13],
        }
    )

    result = add_engineered_features(
        dataframe,
        SCENARIO_WITH_PREVIOUS_GRADES,
    )

    assert result.loc[0, "parent_education_mean"] == 3.0
    assert result.loc[0, "alcohol_consumption_mean"] == 2.0
    assert result.loc[0, "social_activity_score"] == 6.0
    assert result.loc[0, "support_count"] == 1.0
    assert result.loc[0, "previous_grade_mean"] == 11.5
    assert result.loc[0, "grade_trend"] == 3.0
    assert result.loc[0, "absence_log1p"] == pytest.approx(np.log1p(9))


def test_early_prediction_engineering_does_not_create_grade_features() -> None:
    dataframe = pd.DataFrame(
        {
            "Medu": [3],
            "Fedu": [2],
            "Dalc": [1],
            "Walc": [2],
            "freetime": [3],
            "goout": [3],
            "schoolsup": ["no"],
            "famsup": ["yes"],
            "absences": [2],
        }
    )

    result = add_engineered_features(
        dataframe,
        SCENARIO_WITHOUT_PREVIOUS_GRADES,
    )

    expected_names = get_engineered_feature_names(
        SCENARIO_WITHOUT_PREVIOUS_GRADES
    )
    assert all(name in result.columns for name in expected_names)
    assert "previous_grade_mean" not in result.columns
    assert "grade_trend" not in result.columns


def test_metrics_payload_uses_tuning_only_from_selected_model() -> None:
    selected = EvaluationResult(
        scenario=SCENARIO_WITH_PREVIOUS_GRADES,
        model="linear_regression",
        metrics=RegressionMetrics(mae=0.8, rmse=1.0, r2=0.9),
        estimator=Pipeline([]),
        feature_set=FEATURE_SET_ENGINEERED,
        tuning=None,
    )
    tuned_but_not_selected = EvaluationResult(
        scenario=SCENARIO_WITH_PREVIOUS_GRADES,
        model="random_forest_grid_search",
        metrics=RegressionMetrics(mae=1.2, rmse=1.5, r2=0.7),
        estimator=Pipeline([]),
        feature_set=FEATURE_SET_RAW,
        tuning={"method": "GridSearchCV"},
    )

    payload = train_module.build_metrics_payload(
        best_result=selected,
        scenario_results=[selected, tuned_but_not_selected],
        test_metrics=RegressionMetrics(mae=0.9, rmse=1.1, r2=0.85),
        input_features=["G1", "G2"],
        model_feature_columns=["G1", "G2", "previous_grade_mean"],
        feature_names=["G1", "G2", "previous_grade_mean"],
        evaluation_artifacts={},
    )

    assert payload["hyperparameter_tuning"] is None
    assert payload["best_feature_set"] == FEATURE_SET_ENGINEERED
    assert payload["feature_engineering"]["enabled"] is True
