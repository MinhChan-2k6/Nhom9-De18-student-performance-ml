"""Khởi tạo và cấu hình Random Forest Regression."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config.settings import (
    GRID_SEARCH_CV,
    GRID_SEARCH_N_JOBS,
    GRID_SEARCH_SCORING,
    GRID_SEARCH_VERBOSE,
    RANDOM_FOREST_BASE_PARAMS,
    RANDOM_FOREST_GRID,
    RANDOM_FOREST_TUNING_PARAMS,
)


def build_random_forest() -> RandomForestRegressor:
    """Tạo Random Forest cơ sở dùng để so sánh model."""

    return RandomForestRegressor(
        **RANDOM_FOREST_BASE_PARAMS
    )


def build_random_forest_for_tuning() -> RandomForestRegressor:
    """Tạo Random Forest dùng bên trong Grid Search."""

    return RandomForestRegressor(
        **RANDOM_FOREST_TUNING_PARAMS
    )


def build_random_forest_grid_search(
    preprocessor: ColumnTransformer
) -> GridSearchCV:
    """Tạo pipeline và Grid Search để tối ưu Random Forest."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_random_forest_for_tuning()),
        ]
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=RANDOM_FOREST_GRID,
        scoring=GRID_SEARCH_SCORING,
        cv=GRID_SEARCH_CV,
        n_jobs=GRID_SEARCH_N_JOBS,
        verbose=GRID_SEARCH_VERBOSE,
    )
