<<<<<<< Updated upstream
=======
"""Điều phối quy trình train và lưu riêng hai model dự đoán G3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.config.paths import (
    ARTIFACT_DIR,
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


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Đọc CSV và kiểm tra dữ liệu cơ bản."""

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {file_path}")

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(f"File dữ liệu không có dòng nào: {file_path}")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} "
            f"trong file {file_path.name}."
        )

    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()
    if duplicate_columns:
        raise ValueError(
            "Dữ liệu có tên cột bị trùng: " + ", ".join(duplicate_columns)
        )

    return dataframe


def validate_dataset_compatibility(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
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

    results: list[EvaluationResult] = []

    for model_name, model in build_candidate_models().items():
        LOGGER.info("Đang train %s | %s", scenario_name, model_name)

        preprocessor = build_preprocessor(train_df, feature_columns)
        pipeline = build_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)

        result = evaluate_pipeline(
            estimator=pipeline,
            X_valid=X_valid,
            y_valid=y_valid,
            scenario_name=scenario_name,
            model_name=model_name,
        )
        results.append(result)

        LOGGER.info(
            "%s | %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            scenario_name,
            model_name,
            result.mae,
            result.rmse,
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

    return final_pipeline, feature_columns


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
    )


def write_model_card(
    metrics_payload: dict[str, Any],
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

    model_card = f"""# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `{TARGET_COLUMN}`
- Thang điểm mục tiêu: 0–20
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
    )

    final_pipeline, input_features = refit_best_pipeline(
        best_result=best_result,
        train_df=train_df,
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
    )

    metrics_payload = {
        "task": "regression",
        "target": TARGET_COLUMN,
        "best_model": best_result.model,
        "best_scenario": best_result.scenario,
        "selection_metric": "validation_rmse",
        "validation_results": [result.to_dict() for result in scenario_results],
        "test_metrics": test_metrics.to_dict(),
        "test_pass_fail_metrics": pass_fail_metrics.to_dict(),
        "input_features": input_features,
        "total_input_features": len(input_features),
        "total_features_after_preprocessing": len(feature_names),
        "evaluation_artifacts": evaluation_artifacts,
    }

    save_artifacts(
        final_pipeline=final_pipeline,
        feature_names=feature_names,
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
    LOGGER.info("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
