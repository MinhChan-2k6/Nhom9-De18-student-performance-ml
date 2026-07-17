from __future__ import annotations

from pathlib import Path


# ==================================================
# CẤU HÌNH CHUNG
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train.csv"
VALID_PATH = DATA_DIR / "valid.csv"
TEST_PATH = DATA_DIR / "test.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "best_regression_model.pkl"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"