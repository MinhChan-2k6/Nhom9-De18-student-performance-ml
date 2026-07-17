<<<<<<< HEAD
from __future__ import annotations

from pathlib import Path


# ==================================================
# CẤU HÌNH CHUNG
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

=======
<<<<<<< Updated upstream
=======
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.settings import (
    SCENARIO_WITH_PREVIOUS_GRADES,
    SCENARIO_WITHOUT_PREVIOUS_GRADES,
)


# ==================================================
# Thư mục gốc
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==================================================
# Dữ liệu đã chia
# ==================================================
>>>>>>> origin/MinhChan
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train.csv"
VALID_PATH = DATA_DIR / "valid.csv"
TEST_PATH = DATA_DIR / "test.csv"

<<<<<<< HEAD
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_regression_model.pkl"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"
=======

# ==================================================
# Artifact dùng chung
# ==================================================
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TRAINING_SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"

# Hai thư mục model theo đúng hai chế độ dự đoán.
MODEL_HAVE_A_B_DIR = ARTIFACT_DIR / "Model_Have_A_B"
MODEL_DONT_HAVE_A_B_DIR = ARTIFACT_DIR / "Model_Dont_Have_A_B"


@dataclass(frozen=True)
class ScenarioArtifactPaths:
    """Tập hợp toàn bộ đường dẫn output của một scenario."""

    directory: Path
    model: Path
    preprocessor: Path
    metrics: Path
    feature_names: Path
    model_card: Path
    test_predictions: Path
    plots: Path


def _build_scenario_paths(directory: Path) -> ScenarioArtifactPaths:
    return ScenarioArtifactPaths(
        directory=directory,
        model=directory / "best_regression_model.pkl",
        preprocessor=directory / "preprocessor.pkl",
        metrics=directory / "metrics.json",
        feature_names=directory / "feature_names.json",
        model_card=directory / "model_card.md",
        test_predictions=directory / "test_predictions.csv",
        plots=directory / "plots",
    )


SCENARIO_ARTIFACT_PATHS: dict[str, ScenarioArtifactPaths] = {
    SCENARIO_WITH_PREVIOUS_GRADES: _build_scenario_paths(
        MODEL_HAVE_A_B_DIR
    ),
    SCENARIO_WITHOUT_PREVIOUS_GRADES: _build_scenario_paths(
        MODEL_DONT_HAVE_A_B_DIR
    ),
}


def get_scenario_artifact_paths(
    scenario_name: str,
) -> ScenarioArtifactPaths:
    """Lấy bộ đường dẫn artifact tương ứng với scenario."""

    try:
        return SCENARIO_ARTIFACT_PATHS[scenario_name]
    except KeyError as exc:
        raise ValueError(f"Scenario không hợp lệ: {scenario_name}") from exc
>>>>>>> Stashed changes
>>>>>>> origin/MinhChan
