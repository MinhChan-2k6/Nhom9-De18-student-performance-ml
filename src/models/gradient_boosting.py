"""Khởi tạo mô hình Gradient Boosting Regression."""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor

from src.config.settings import RANDOM_STATE


def build_gradient_boosting() -> GradientBoostingRegressor:
    """Tạo Gradient Boosting với random state cố định."""

    return GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )
