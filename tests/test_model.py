"""Kiểm thử model, pipeline và luồng tinh chỉnh siêu tham số."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.settings import (  # noqa: E402
    GRID_SEARCH_CV,
    GRID_SEARCH_SCORING,
    RANDOM_FOREST_GRID,
    SCENARIO_WITH_PREVIOUS_GRADES,
)
from src.evaluation.metrics import (  # noqa: E402
    EvaluationResult,
    RegressionMetrics,
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
    dataframe = pd.DataFrame(
        {
            "subject": np.where(np.arange(rows) % 2 == 0, "math", "portuguese"),
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
    assert search.cv == GRID_SEARCH_CV
    assert search.param_grid == RANDOM_FOREST_GRID
    assert set(search.estimator.named_steps) == {"preprocessor", "model"}


def test_grid_search_fits_and_exposes_best_parameters(
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, valid_df, feature_columns = regression_frames
    preprocessor = build_preprocessor(train_df, feature_columns)
    search = build_random_forest_grid_search(preprocessor)
    search.set_params(
        cv=2,
        n_jobs=1,
        param_grid={
            "model__n_estimators": [5],
            "model__max_depth": [2, None],
            "model__min_samples_split": [2],
            "model__min_samples_leaf": [1],
            "model__max_features": ["sqrt"],
        },
    )

    search.fit(train_df[feature_columns], train_df["G3"])
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

    def build_small_search(preprocessor):
        search = build_random_forest_grid_search(preprocessor)
        return search.set_params(
            cv=2,
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
    )

    assert result.model == "random_forest_grid_search"
    assert result.tuning is not None
    assert result.tuning["method"] == "GridSearchCV"
    assert result.tuning["candidate_count"] == 1
    assert result.tuning["total_fits"] == 2
    assert result.tuning["best_params"]["model__n_estimators"] == 5
    assert np.isfinite(result.rmse)


def test_train_scenario_appends_grid_search_result(
    monkeypatch: pytest.MonkeyPatch,
    regression_frames: tuple[pd.DataFrame, pd.DataFrame, list[str]],
) -> None:
    train_df, valid_df, _ = regression_frames
    tuned_result = EvaluationResult(
        scenario=SCENARIO_WITH_PREVIOUS_GRADES,
        model="random_forest_grid_search",
        metrics=RegressionMetrics(mae=1.0, rmse=1.2, r2=0.8),
        estimator=Pipeline([]),
        tuning={"method": "GridSearchCV"},
    )

    monkeypatch.setattr(train_module, "build_candidate_models", lambda: {})
    monkeypatch.setattr(
        train_module,
        "train_tuned_random_forest",
        lambda **_: tuned_result,
    )

    results = train_module.train_scenario(
        train_df=train_df,
        valid_df=valid_df,
        scenario_name=SCENARIO_WITH_PREVIOUS_GRADES,
    )

    assert results == [tuned_result]
    assert results[0].to_dict()["tuning"]["method"] == "GridSearchCV"
