"""Public API cho chức năng suy luận điểm G3."""

from src.inference.predict_batch import predict_batch, predict_batch_csv
from src.inference.predict_single import predict_single
from src.inference.predictor import (
    MODE_TO_SCENARIO,
    PREDICTION_OUTPUT_COLUMNS,
    PredictionMode,
    PredictionService,
    get_prediction_service,
)

__all__ = [
    "MODE_TO_SCENARIO",
    "PREDICTION_OUTPUT_COLUMNS",
    "PredictionMode",
    "PredictionService",
    "get_prediction_service",
    "predict_single",
    "predict_batch",
    "predict_batch_csv",
]
