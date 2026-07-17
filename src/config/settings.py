from __future__ import annotations

from typing import Final

# ==================================================
# Cấu hình bài toán
# ==================================================
RANDOM_STATE: Final[int] = 42
TARGET_COLUMN: Final[str] = "G3"

SCENARIO_WITH_PREVIOUS_GRADES: Final[str] = "A_with_G1_G2"
SCENARIO_WITHOUT_PREVIOUS_GRADES: Final[str] = "B_without_G1_G2"

SCENARIOS: Final[tuple[str, ...]] = (
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)

EARLY_PREDICTION_EXCLUDED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"G1", "G2"}
)

# ==================================================
# Cấu hình model
# ==================================================
RANDOM_FOREST_BASE_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOM_FOREST_TUNING_PARAMS: Final[dict[str, object]] = {
    "random_state": RANDOM_STATE,
    # GridSearchCV chạy ở tầng ngoài; để 1 nhằm tránh nested parallelism.
    "n_jobs": 1,
}

RANDOM_FOREST_GRID: Final[dict[str, list[object]]] = {
    # Grid vừa đủ để so sánh, tránh hàng trăm lượt fit gây treo máy.
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1],
    # Bao gồm 1.0 (mặc định của RandomForestRegressor) để GridSearch không
    # loại bỏ cấu hình baseline mạnh, đồng thời vẫn so sánh với "sqrt".
    "model__max_features": [1.0, "sqrt"],
}

GRID_SEARCH_SCORING: Final[str] = "neg_root_mean_squared_error"
GRID_SEARCH_CV: Final[int] = 3
GRID_SEARCH_N_JOBS: Final[int] = 1
GRID_SEARCH_VERBOSE: Final[int] = 0

# ==================================================
# Cấu hình logging
# ==================================================
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(message)s"
