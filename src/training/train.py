<<<<<<< Updated upstream
=======
<<<<<<< HEAD
=======
"""Điều phối quy trình train và lưu riêng hai model dự đoán G3."""

>>>>>>> origin/MinhChan
from __future__ import annotations

import json
import logging
<<<<<<< HEAD
from dataclasses import dataclass
=======
>>>>>>> origin/MinhChan
from pathlib import Path
from typing import Any

import joblib
<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
=======
import pandas as pd
from sklearn.base import clone
>>>>>>> origin/MinhChan
from sklearn.pipeline import Pipeline

from src.config.paths import (
    ARTIFACT_DIR,
<<<<<<< HEAD
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODEL_CARD_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VALID_PATH,
)
from src.config.settings import (
    LOG_FORMAT,
    LOG_LEVEL,
    SCENARIOS,
    TARGET_COLUMN,
)

from src.features.build_features import (
    get_scenario_features,
    split_features_target,
)
from src.features.transform import (
    build_preprocessor,
    get_feature_names,
)
from src.models.model_registry import (
    build_candidate_models,
    build_pipeline,
)
from src.models.random_forest import build_random_forest_grid_search


# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FORMAT
)

LOGGER = logging.getLogger(__name__)


# ==================================================
# CẤU TRÚC DỮ LIỆU KẾT QUẢ
# ==================================================

@dataclass
class EvaluationResult:
    scenario: str
    model: str
    mae: float
    rmse: float
    r2: float
    estimator: Pipeline

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "model": self.model,
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
            "r2": round(self.r2, 6)
        }


# ==================================================
# ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==================================================

=======
    TEST_PATH,
    TRAIN_PATH,
    TRAINING_SUMMARY_PATH,
    VALID_PATH,
    ScenarioArtifactPaths,
    get_scenario_artifact_paths,
)
from src.config.settings import LOG_FORMAT, LOG_LEVEL, SCENARIOS, TARGET_COLUMN
from src.evaluation.compare import select_best_result
from src.evaluation.metrics import (
    EvaluationResult,
    calculate_pass_fail_metrics,
    evaluate_pipeline,
    predict_and_evaluate,
)
from src.evaluation.plots import generate_evaluation_artifacts
from src.features.build_features import get_scenario_features, split_features_target
from src.features.transform import build_preprocessor, get_feature_names
from src.models.model_registry import build_candidate_models, build_pipeline


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FORMAT,
)
LOGGER = logging.getLogger(__name__)


>>>>>>> origin/MinhChan
def load_dataset(file_path: Path) -> pd.DataFrame:
    """Đọc CSV và kiểm tra dữ liệu cơ bản."""

    if not file_path.exists():
<<<<<<< HEAD
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {file_path}"
        )
=======
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")
>>>>>>> origin/MinhChan

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
<<<<<<< HEAD
        raise ValueError(
            f"File dữ liệu không có dòng nào: {file_path}"
        )
=======
        raise ValueError(f"File dữ liệu không có dòng nào: {file_path}")
>>>>>>> origin/MinhChan

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} "
<<<<<<< HEAD
            f"trong file {file_path.name}"
=======
            f"trong file {file_path.name}."
>>>>>>> origin/MinhChan
        )

    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()
<<<<<<< HEAD

    if duplicate_columns:
        raise ValueError(
            "Dữ liệu có tên cột bị trùng: "
            + ", ".join(duplicate_columns)
=======
    if duplicate_columns:
        raise ValueError(
            "Dữ liệu có tên cột bị trùng: " + ", ".join(duplicate_columns)
>>>>>>> origin/MinhChan
        )

    return dataframe


def validate_dataset_compatibility(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
<<<<<<< HEAD
    test_df: pd.DataFrame
) -> None:
    """Đảm bảo ba tập dữ liệu có cùng cấu trúc cột."""

    train_columns = list(train_df.columns)
    valid_columns = list(valid_df.columns)
    test_columns = list(test_df.columns)

    if train_columns != valid_columns:
        raise ValueError(
            "train.csv và valid.csv không có cùng cấu trúc cột."
        )

    if train_columns != test_columns:
        raise ValueError(
            "train.csv và test.csv không có cùng cấu trúc cột."
        )



# ==================================================
# ĐÁNH GIÁ
# ==================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Tính MAE, RMSE và R²."""

    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def evaluate_pipeline(
    estimator: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    scenario_name: str,
    model_name: str
) -> EvaluationResult:
    """Đánh giá pipeline trên tập validation."""

    predictions = estimator.predict(X_valid)

    mae, rmse, r2 = calculate_metrics(
        y_valid,
        predictions
    )

    return EvaluationResult(
        scenario=scenario_name,
        model=model_name,
        mae=mae,
        rmse=rmse,
        r2=r2,
        estimator=estimator
    )


def is_better_result(
    candidate: EvaluationResult,
    current_best: EvaluationResult | None
) -> bool:
    """
    Ưu tiên RMSE thấp nhất.
    Nếu RMSE bằng nhau, ưu tiên MAE thấp hơn.
    """

    if current_best is None:
        return True

    if candidate.rmse < current_best.rmse:
        return True

    if np.isclose(
        candidate.rmse,
        current_best.rmse
    ):
        return candidate.mae < current_best.mae

    return False


# ==================================================
# GRID SEARCH CHO RANDOM FOREST
# ==================================================

def tune_random_forest(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    scenario_name: str
) -> EvaluationResult:
    """Tối ưu Random Forest trên tập train, rồi đánh giá trên validation."""

    X_train, y_train = split_features_target(
        train_df,
        feature_columns
    )

    X_valid, y_valid = split_features_target(
        valid_df,
        feature_columns
    )

    preprocessor = build_preprocessor(
        train_df,
        feature_columns
    )

    grid_search = build_random_forest_grid_search(
        preprocessor
    )

    LOGGER.info(
        "Bắt đầu Grid Search cho %s",
        scenario_name
    )

    grid_search.fit(
        X_train,
        y_train
    )

    LOGGER.info(
        "Best params (%s): %s",
        scenario_name,
        grid_search.best_params_
    )

    result = evaluate_pipeline(
        estimator=grid_search.best_estimator_,
        X_valid=X_valid,
        y_valid=y_valid,
        scenario_name=scenario_name,
        model_name="random_forest_grid_search"
    )

    return result


# ==================================================
# HUẤN LUYỆN MỘT SCENARIO
# ==================================================

def train_scenario(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    scenario_name: str
) -> list[EvaluationResult]:
    """Train và đánh giá các model trong một scenario."""

    feature_columns = get_scenario_features(
        train_df,
        scenario_name
    )

    X_train, y_train = split_features_target(
        train_df,
        feature_columns
    )

    X_valid, y_valid = split_features_target(
        valid_df,
        feature_columns
    )
=======
    test_df: pd.DataFrame,
) -> None:
    """Đảm bảo train, valid và test có cùng cấu trúc cột."""

    train_columns = list(train_df.columns)

    if train_columns != list(valid_df.columns):
        raise ValueError("train.csv và valid.csv không có cùng cấu trúc cột.")

    if train_columns != list(test_df.columns):
        raise ValueError("train.csv và test.csv không có cùng cấu trúc cột.")



def train_scenario(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    scenario_name: str,
) -> list[EvaluationResult]:
    """Huấn luyện và đánh giá các model trong một scenario."""

    feature_columns = get_scenario_features(train_df, scenario_name)
    X_train, y_train = split_features_target(train_df, feature_columns)
    X_valid, y_valid = split_features_target(valid_df, feature_columns)
>>>>>>> origin/MinhChan

    results: list[EvaluationResult] = []

    for model_name, model in build_candidate_models().items():
<<<<<<< HEAD
        LOGGER.info(
            "Đang train %s | %s",
            scenario_name,
            model_name
        )

        preprocessor = build_preprocessor(
            train_df,
            feature_columns
        )

        pipeline = build_pipeline(
            preprocessor,
            model
        )

        pipeline.fit(
            X_train,
            y_train
        )
=======
        LOGGER.info("Đang train %s | %s", scenario_name, model_name)

        preprocessor = build_preprocessor(train_df, feature_columns)
        pipeline = build_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
>>>>>>> origin/MinhChan

        result = evaluate_pipeline(
            estimator=pipeline,
            X_valid=X_valid,
            y_valid=y_valid,
            scenario_name=scenario_name,
<<<<<<< HEAD
            model_name=model_name
        )

=======
            model_name=model_name,
        )
>>>>>>> origin/MinhChan
        results.append(result)

        LOGGER.info(
            "%s | %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            scenario_name,
            model_name,
            result.mae,
            result.rmse,
<<<<<<< HEAD
            result.r2
        )

    tuned_rf_result = tune_random_forest(
        train_df=train_df,
        valid_df=valid_df,
        feature_columns=feature_columns,
        scenario_name=scenario_name
    )

    results.append(tuned_rf_result)

    LOGGER.info(
        "%s | tuned RF | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        scenario_name,
        tuned_rf_result.mae,
        tuned_rf_result.rmse,
        tuned_rf_result.r2
    )

    return results


# ==================================================
# REFIT VÀ TEST
# ==================================================

def refit_best_pipeline(
    best_result: EvaluationResult,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame
) -> tuple[Pipeline, list[str]]:
    """
    Sau khi chọn model bằng validation,
    refit model trên train + validation.
    """

    combined_df = pd.concat(
        [train_df, valid_df],
        ignore_index=True
    )

    feature_columns = get_scenario_features(
        combined_df,
        best_result.scenario
    )

    X_combined, y_combined = split_features_target(
        combined_df,
        feature_columns
    )

    # Tạo bản sao sạch của model đã chọn trước khi refit.
    chosen_model = clone(
        best_result.estimator.named_steps["model"]
    )

    preprocessor = build_preprocessor(
        combined_df,
        feature_columns
    )

    final_pipeline = build_pipeline(
        preprocessor,
        chosen_model
    )

    final_pipeline.fit(
        X_combined,
        y_combined
    )
=======
            result.r2,
        )

    return results


def refit_best_pipeline(
    best_result: EvaluationResult,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[Pipeline, list[str]]:
    """Refit model tốt nhất của scenario trên train + validation."""

    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    feature_columns = get_scenario_features(combined_df, best_result.scenario)
    X_combined, y_combined = split_features_target(combined_df, feature_columns)

    chosen_model = clone(best_result.estimator.named_steps["model"])
    preprocessor = build_preprocessor(combined_df, feature_columns)
    final_pipeline = build_pipeline(preprocessor, chosen_model)
    final_pipeline.fit(X_combined, y_combined)
>>>>>>> origin/MinhChan

    return final_pipeline, feature_columns


<<<<<<< HEAD
def evaluate_on_test(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_columns: list[str]
) -> dict[str, float]:
    """Đánh giá model cuối cùng trên test set."""

    X_test, y_test = split_features_target(
        test_df,
        feature_columns
    )

    predictions = pipeline.predict(X_test)

    mae, rmse, r2 = calculate_metrics(
        y_test,
        predictions
    )

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6)
    }


# ==================================================
# LƯU ARTIFACTS
# ==================================================

def save_artifacts(
    final_pipeline: Pipeline,
    feature_names: list[str],
    metrics_payload: dict[str, Any]
) -> None:
    """Lưu model, preprocessor, metrics và feature names."""

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        final_pipeline,
        MODEL_PATH
    )

    joblib.dump(
        final_pipeline.named_steps["preprocessor"],
        PREPROCESSOR_PATH
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics_payload,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    FEATURE_NAMES_PATH.write_text(
        json.dumps(
            feature_names,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
=======
def save_artifacts(
    final_pipeline: Pipeline,
    feature_names: list[str],
    metrics_payload: dict[str, Any],
    paths: ScenarioArtifactPaths,
) -> None:
    """Lưu model và metadata vào đúng thư mục scenario."""

    paths.directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_pipeline, paths.model)
    joblib.dump(final_pipeline.named_steps["preprocessor"], paths.preprocessor)

    paths.metrics.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    paths.feature_names.write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=4),
        encoding="utf-8",
>>>>>>> origin/MinhChan
    )


def write_model_card(
    metrics_payload: dict[str, Any],
<<<<<<< HEAD
    input_features: list[str]
) -> None:
    """Tạo model card Markdown."""

    test_metrics = metrics_payload["test_metrics"]
=======
    input_features: list[str],
    paths: ScenarioArtifactPaths,
) -> None:
    """Tạo model card riêng cho từng scenario."""

    test_metrics = metrics_payload["test_metrics"]
    pass_fail_metrics = metrics_payload.get("test_pass_fail_metrics", {})
    plot_count = len(
        metrics_payload.get("evaluation_artifacts", {}).get(
            "generated_plots", []
        )
    )

    excluded_note = (
        "Có sử dụng `G1`, `G2`."
        if "G1" in input_features and "G2" in input_features
        else "Không sử dụng `G1`, `G2`; phù hợp cho dự báo sớm."
    )
>>>>>>> origin/MinhChan

    model_card = f"""# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `{TARGET_COLUMN}`
- Thang điểm mục tiêu: 0–20
<<<<<<< HEAD
- Model được chọn: `{metrics_payload["best_model"]}`
- Scenario được chọn: `{metrics_payload["best_scenario"]}`

## 2. Scenario

- `A_with_G1_G2`: sử dụng G1 và G2.
- `B_without_G1_G2`: không sử dụng G1, G2; phục vụ dự báo sớm.

## 3. Chỉ số trên tập test

- MAE: `{test_metrics["mae"]}`
- RMSE: `{test_metrics["rmse"]}`
- R²: `{test_metrics["r2"]}`

## 4. Số lượng đặc trưng đầu vào

- Số cột đầu vào trước tiền xử lý: `{len(input_features)}`

## 5. Lưu ý

- MAE và RMSE càng thấp càng tốt.
- R² càng gần 1 càng tốt.
- Kết quả Đạt/Trượt chỉ là quy đổi từ điểm G3 dự đoán.
- Model chính vẫn là mô hình hồi quy.
- Scenario có G1, G2 thường chính xác hơn nhưng chỉ dùng được khi đã có điểm trước đó.
- Scenario không có G1, G2 phù hợp cho cảnh báo sớm nhưng có thể kém chính xác hơn.

## 6. Giới hạn

- Dữ liệu gốc sử dụng thang điểm 0–20.
- Model có thể không khái quát tốt cho mọi trường học hoặc hệ thống giáo dục.
- Không nên sử dụng dự đoán như quyết định duy nhất về kết quả học tập.
"""

    MODEL_CARD_PATH.write_text(
        model_card,
        encoding="utf-8"
    )


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    LOGGER.info("Đang đọc dữ liệu...")

    train_df = load_dataset(TRAIN_PATH)
    valid_df = load_dataset(VALID_PATH)
    test_df = load_dataset(TEST_PATH)

    validate_dataset_compatibility(
        train_df,
        valid_df,
        test_df
    )

    LOGGER.info(
        "Train: %s | Valid: %s | Test: %s",
        train_df.shape,
        valid_df.shape,
        test_df.shape
    )

    all_results: list[EvaluationResult] = []

    for scenario_name in SCENARIOS:
        scenario_results = train_scenario(
            train_df=train_df,
            valid_df=valid_df,
            scenario_name=scenario_name
        )

        all_results.extend(
            scenario_results
        )

    best_result: EvaluationResult | None = None

    for result in all_results:
        if is_better_result(
            candidate=result,
            current_best=best_result
        ):
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "Không chọn được model tốt nhất."
        )

    LOGGER.info(
        "Model tốt nhất trên validation: %s | %s",
        best_result.scenario,
        best_result.model
=======
- Model được chọn: `{metrics_payload['best_model']}`
- Scenario: `{metrics_payload['best_scenario']}`
- Cách dùng đặc trưng: {excluded_note}

## 2. Chỉ số trên tập test

- MAE: `{test_metrics['mae']}`
- RMSE: `{test_metrics['rmse']}`
- R²: `{test_metrics['r2']}`
- Accuracy Đạt/Trượt: `{pass_fail_metrics.get('accuracy', 'N/A')}`
- F1 Đạt/Trượt: `{pass_fail_metrics.get('f1', 'N/A')}`
- Ngưỡng Đạt: `G3 >= {pass_fail_metrics.get('threshold', 10)}`

## 3. Đặc trưng và output

- Số cột đầu vào trước tiền xử lý: `{len(input_features)}`
- Danh sách đầu vào: `{', '.join(input_features)}`
- Số biểu đồ đánh giá: `{plot_count}`
- Model: `{paths.model.relative_to(ARTIFACT_DIR.parent).as_posix()}`
- Metrics: `{paths.metrics.relative_to(ARTIFACT_DIR.parent).as_posix()}`
- Biểu đồ: `{paths.plots.relative_to(ARTIFACT_DIR.parent).as_posix()}`

## 4. Lưu ý

- MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt.
- Đạt/Trượt chỉ là kết quả quy đổi từ điểm G3 dự đoán.
- Không nên dùng dự đoán như quyết định duy nhất về kết quả học tập.
"""

    paths.model_card.write_text(model_card, encoding="utf-8")


def train_and_save_scenario(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scenario_name: str,
) -> dict[str, Any]:
    """Train, chọn model tốt nhất và lưu artifact cho một scenario."""

    scenario_results = train_scenario(
        train_df=train_df,
        valid_df=valid_df,
        scenario_name=scenario_name,
    )
    best_result = select_best_result(scenario_results)

    LOGGER.info(
        "Model tốt nhất của %s: %s",
        scenario_name,
        best_result.model,
>>>>>>> origin/MinhChan
    )

    final_pipeline, input_features = refit_best_pipeline(
        best_result=best_result,
        train_df=train_df,
<<<<<<< HEAD
        valid_df=valid_df
    )

    test_metrics = evaluate_on_test(
        pipeline=final_pipeline,
        test_df=test_df,
        feature_columns=input_features
    )

    feature_names = get_feature_names(
        final_pipeline
=======
        valid_df=valid_df,
    )

    X_test, y_test = split_features_target(test_df, input_features)
    test_predictions, test_metrics = predict_and_evaluate(
        estimator=final_pipeline,
        X=X_test,
        y=y_test,
    )
    pass_fail_metrics = calculate_pass_fail_metrics(
        y_true=y_test,
        y_pred=test_predictions,
    )
    feature_names = get_feature_names(final_pipeline)
    paths = get_scenario_artifact_paths(scenario_name)

    LOGGER.info("Đang tạo biểu đồ cho %s", scenario_name)
    evaluation_artifacts = generate_evaluation_artifacts(
        results=scenario_results,
        final_pipeline=final_pipeline,
        feature_names=feature_names,
        y_test=y_test,
        y_pred=test_predictions,
        test_metrics=test_metrics,
        output_dir=paths.plots,
        predictions_path=paths.test_predictions,
>>>>>>> origin/MinhChan
    )

    metrics_payload = {
        "task": "regression",
        "target": TARGET_COLUMN,
        "best_model": best_result.model,
        "best_scenario": best_result.scenario,
        "selection_metric": "validation_rmse",
<<<<<<< HEAD
        "validation_results": [
            result.to_dict()
            for result in all_results
        ],
        "test_metrics": test_metrics,
        "input_features": input_features,
        "total_input_features": len(input_features),
        "total_features_after_preprocessing": len(feature_names)
=======
        "validation_results": [result.to_dict() for result in scenario_results],
        "test_metrics": test_metrics.to_dict(),
        "test_pass_fail_metrics": pass_fail_metrics.to_dict(),
        "input_features": input_features,
        "total_input_features": len(input_features),
        "total_features_after_preprocessing": len(feature_names),
        "evaluation_artifacts": evaluation_artifacts,
>>>>>>> origin/MinhChan
    }

    save_artifacts(
        final_pipeline=final_pipeline,
        feature_names=feature_names,
<<<<<<< HEAD
        metrics_payload=metrics_payload
    )

    write_model_card(
        metrics_payload=metrics_payload,
        input_features=input_features
    )

    LOGGER.info("Đã lưu model tại: %s", MODEL_PATH)
    LOGGER.info("Test metrics: %s", test_metrics)
=======
        metrics_payload=metrics_payload,
        paths=paths,
    )
    write_model_card(
        metrics_payload=metrics_payload,
        input_features=input_features,
        paths=paths,
    )

    LOGGER.info("Đã lưu model tại: %s", paths.model)
    LOGGER.info("Test metrics (%s): %s", scenario_name, test_metrics.to_dict())

    return {
        "scenario": scenario_name,
        "best_model": best_result.model,
        "artifact_directory": paths.directory.relative_to(
            ARTIFACT_DIR.parent
        ).as_posix(),
        "model_path": paths.model.relative_to(ARTIFACT_DIR.parent).as_posix(),
        "total_input_features": len(input_features),
        "input_features": input_features,
        "validation_metrics": {
            "mae": best_result.mae,
            "rmse": best_result.rmse,
            "r2": best_result.r2,
        },
        "test_metrics": test_metrics.to_dict(),
    }


def main() -> None:
    LOGGER.info("Đang đọc dữ liệu...")

    train_df = load_dataset(TRAIN_PATH)
    valid_df = load_dataset(VALID_PATH)
    test_df = load_dataset(TEST_PATH)
    validate_dataset_compatibility(train_df, valid_df, test_df)

    LOGGER.info(
        "Train: %s | Valid: %s | Test: %s",
        train_df.shape,
        valid_df.shape,
        test_df.shape,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    scenario_summaries = []

    for scenario_name in SCENARIOS:
        scenario_summaries.append(
            train_and_save_scenario(
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                scenario_name=scenario_name,
            )
        )

    training_summary = {
        "task": "regression",
        "target": TARGET_COLUMN,
        "models_created": len(scenario_summaries),
        "scenarios": scenario_summaries,
    }
    TRAINING_SUMMARY_PATH.write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    LOGGER.info("Đã tạo đủ %d model.", len(scenario_summaries))
    LOGGER.info("Tóm tắt huấn luyện: %s", TRAINING_SUMMARY_PATH)
>>>>>>> origin/MinhChan
    LOGGER.info("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
