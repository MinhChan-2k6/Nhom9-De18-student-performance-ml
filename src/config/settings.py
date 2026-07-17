<<<<<<< HEAD
=======
<<<<<<< Updated upstream
=======
>>>>>>> origin/MinhChan
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
<<<<<<< HEAD
    "n_jobs": -1,
}

RANDOM_FOREST_GRID: Final[dict[str, list[object]]] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", 0.8],
}

GRID_SEARCH_SCORING: Final[str] = "neg_root_mean_squared_error"
GRID_SEARCH_CV: Final[int] = 5
GRID_SEARCH_N_JOBS: Final[int] = -1
GRID_SEARCH_VERBOSE: Final[int] = 1
=======
    # GridSearchCV đã chạy song song ở tầng ngoài; để 1 tránh nested parallelism.
    "n_jobs": 1,
}

RANDOM_FOREST_GRID: Final[dict[str, list[object]]] = {
    # Grid vừa đủ để so sánh, tránh hàng trăm lượt fit gây treo máy.
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1],
    "model__max_features": ["sqrt"],
}

GRID_SEARCH_SCORING: Final[str] = "neg_root_mean_squared_error"
GRID_SEARCH_CV: Final[int] = 3
GRID_SEARCH_N_JOBS: Final[int] = 1
GRID_SEARCH_VERBOSE: Final[int] = 0
>>>>>>> origin/MinhChan

# ==================================================
# Cấu hình logging
# ==================================================
LOG_LEVEL: Final[str] = "INFO"
<<<<<<< HEAD
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(message)s"
=======
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(message)s"
>>>>>>> Stashed changes
>>>>>>> origin/MinhChan
