"""Khởi tạo và cấu hình Random Forest Regression."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
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
    preprocessor: ColumnTransformer,
    feature_engineer: Any | None = None,
) -> GridSearchCV:
    """Tạo Grid Search dùng GroupKFold để tránh trùng học sinh giữa fold."""

    steps: list[tuple[str, Any]] = []

    if feature_engineer is not None:
        steps.append(("feature_engineering", feature_engineer))

    steps.extend(
        [
            ("preprocessor", preprocessor),
            ("model", build_random_forest_for_tuning()),
        ]
    )

    pipeline = Pipeline(steps=steps)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=RANDOM_FOREST_GRID,
        scoring=GRID_SEARCH_SCORING,
        cv=GroupKFold(n_splits=GRID_SEARCH_CV),
        n_jobs=GRID_SEARCH_N_JOBS,
        verbose=GRID_SEARCH_VERBOSE,
    )
