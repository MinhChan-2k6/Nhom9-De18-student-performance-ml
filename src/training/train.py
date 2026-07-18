"""Điều phối quy trình train và lưu riêng hai model dự đoán G3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
from src.config.settings import (
    FEATURE_SET_ENGINEERED,
    FEATURE_SETS,
    LOG_FORMAT,
    LOG_LEVEL,
    SCENARIOS,
    TARGET_COLUMN,
)
from src.data.split_data import create_student_group_id
from src.evaluation.compare import (
    select_best_result,
    summarize_feature_set_results,
)
from src.evaluation.metrics import (
    EvaluationResult,
    RegressionMetrics,
    evaluate_pipeline,
    predict_and_evaluate,
)
from src.evaluation.plots import generate_evaluation_artifacts
from src.features.build_features import (
    build_feature_engineer,
    get_engineered_feature_names,
    get_scenario_features,
    split_features_target,
)
from src.features.transform import build_preprocessor, get_feature_names
from src.models.model_registry import build_candidate_models, build_pipeline
from src.models.random_forest import build_random_forest_grid_search
from src.utils.io import read_csv, save_joblib, save_json, save_text


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format=LOG_FORMAT,
)
LOGGER = logging.getLogger(__name__)


def load_dataset(file_path: Path) -> pd.DataFrame:

    dataframe = read_csv(file_path)

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

    train_columns = list(train_df.columns)

    if train_columns != list(valid_df.columns):
        raise ValueError("train.csv và valid.csv không có cùng cấu trúc cột.")

    if train_columns != list(test_df.columns):
        raise ValueError("train.csv và test.csv không có cùng cấu trúc cột.")


def build_training_pipeline(
    dataframe: pd.DataFrame,
    scenario_name: str,
    feature_set: str,
    model: Any,
) -> tuple[Pipeline, list[str], list[str]]:

    input_features = get_scenario_features(dataframe, scenario_name)
    raw_feature_frame = dataframe.loc[:, input_features]
    feature_engineer = build_feature_engineer(scenario_name, feature_set)
    model_feature_frame = feature_engineer.fit_transform(raw_feature_frame)
    model_feature_columns = model_feature_frame.columns.tolist()
    preprocessor = build_preprocessor(
        model_feature_frame,
        model_feature_columns,
    )
    pipeline = build_pipeline(
        preprocessor=preprocessor,
        model=model,
        feature_engineer=feature_engineer,
    )
    return pipeline, input_features, model_feature_columns


def train_scenario(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    scenario_name: str,
) -> list[EvaluationResult]:

    input_features = get_scenario_features(train_df, scenario_name)
    X_train, y_train = split_features_target(train_df, input_features)
    X_valid, y_valid = split_features_target(valid_df, input_features)
    results: list[EvaluationResult] = []

    for feature_set in FEATURE_SETS:
        for model_name, model in build_candidate_models().items():
            LOGGER.info(
                "Đang train %s | %s | feature_set=%s",
                scenario_name,
                model_name,
                feature_set,
            )

            pipeline, _, _ = build_training_pipeline(
                dataframe=train_df,
                scenario_name=scenario_name,
                feature_set=feature_set,
                model=model,
            )
            pipeline.fit(X_train, y_train)

            result = evaluate_pipeline(
                estimator=pipeline,
                X_valid=X_valid,
                y_valid=y_valid,
                scenario_name=scenario_name,
                model_name=model_name,
                feature_set=feature_set,
            )
            results.append(result)

            LOGGER.info(
                "%s | %s | %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
                scenario_name,
                feature_set,
                model_name,
                result.mae,
                result.rmse,
                result.r2,
            )

        tuned_result = train_tuned_random_forest(
            train_df=train_df,
            valid_df=valid_df,
            scenario_name=scenario_name,
            feature_columns=input_features,
            feature_set=feature_set,
        )
        results.append(tuned_result)

    return results


def _get_cv_folds(grid_search: Any) -> int:
    cv = grid_search.cv
    if isinstance(cv, int):
        return cv
    if hasattr(cv, "get_n_splits"):
        return int(cv.get_n_splits())
    raise TypeError("Không xác định được số fold của GridSearchCV.")


def train_tuned_random_forest(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    scenario_name: str,
    feature_columns: list[str],
    feature_set: str,
) -> EvaluationResult:

    X_train, y_train = split_features_target(train_df, feature_columns)
    X_valid, y_valid = split_features_target(valid_df, feature_columns)

    feature_engineer = build_feature_engineer(scenario_name, feature_set)
    model_feature_frame = feature_engineer.fit_transform(X_train)
    preprocessor = build_preprocessor(
        model_feature_frame,
        model_feature_frame.columns.tolist(),
    )
    grid_search = build_random_forest_grid_search(
        preprocessor=preprocessor,
        feature_engineer=feature_engineer,
    )

    candidate_count = 1
    for values in grid_search.param_grid.values():
        candidate_count *= len(values)
    cv_folds = _get_cv_folds(grid_search)

    training_groups = create_student_group_id(train_df)
    unique_groups = int(training_groups.nunique())
    if unique_groups < cv_folds:
        raise ValueError(
            "Không đủ nhóm học sinh cho GroupKFold: "
            f"{unique_groups} nhóm < {cv_folds} folds."
        )

    LOGGER.info(
        "Đang GridSearchCV %s | feature_set=%s | "
        "%d cấu hình x %d GroupKFold",
        scenario_name,
        feature_set,
        candidate_count,
        cv_folds,
    )
    grid_search.fit(X_train, y_train, groups=training_groups)

    tuning = {
        "method": "GridSearchCV",
        "cv_strategy": "GroupKFold",
        "grouping": "student_identity",
        "unique_training_groups": unique_groups,
        "scoring": str(grid_search.scoring),
        "cv_folds": cv_folds,
        "candidate_count": candidate_count,
        "total_fits": candidate_count * cv_folds,
        "best_cv_rmse": round(float(-grid_search.best_score_), 6),
        "best_params": dict(grid_search.best_params_),
    }

    result = evaluate_pipeline(
        estimator=grid_search.best_estimator_,
        X_valid=X_valid,
        y_valid=y_valid,
        scenario_name=scenario_name,
        model_name="random_forest_grid_search",
        feature_set=feature_set,
        tuning=tuning,
    )

    LOGGER.info(
        "%s | %s | random_forest_grid_search | CV RMSE=%.4f | "
        "validation RMSE=%.4f | best_params=%s",
        scenario_name,
        feature_set,
        tuning["best_cv_rmse"],
        result.rmse,
        tuning["best_params"],
    )
    return result


def refit_best_pipeline(
    best_result: EvaluationResult,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[Pipeline, list[str], list[str]]:

    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    chosen_model = clone(best_result.estimator.named_steps["model"])
    final_pipeline, input_features, model_feature_columns = (
        build_training_pipeline(
            dataframe=combined_df,
            scenario_name=best_result.scenario,
            feature_set=best_result.feature_set,
            model=chosen_model,
        )
    )
    X_combined, y_combined = split_features_target(
        combined_df,
        input_features,
    )
    final_pipeline.fit(X_combined, y_combined)

    return final_pipeline, input_features, model_feature_columns


def save_artifacts(
    final_pipeline: Pipeline,
    feature_names: list[str],
    metrics_payload: dict[str, Any],
    paths: ScenarioArtifactPaths,
) -> None:

    save_joblib(final_pipeline, paths.model)
    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                final_pipeline.named_steps["feature_engineering"],
            ),
            (
                "preprocessor",
                final_pipeline.named_steps["preprocessor"],
            ),
        ]
    )
    save_joblib(preprocessing_pipeline, paths.preprocessor)
    save_json(metrics_payload, paths.metrics)
    save_json(feature_names, paths.feature_names)


def build_metrics_payload(
    *,
    best_result: EvaluationResult,
    scenario_results: list[EvaluationResult],
    test_metrics: RegressionMetrics,
    input_features: list[str],
    model_feature_columns: list[str],
    feature_names: list[str],
    evaluation_artifacts: dict[str, Any],
) -> dict[str, Any]:

    engineered_features = [
        feature
        for feature in model_feature_columns
        if feature not in input_features
    ]

    return {
        "task": "regression",
        "target": TARGET_COLUMN,
        "best_model": best_result.model,
        "best_scenario": best_result.scenario,
        "best_feature_set": best_result.feature_set,
        "selection_metric": "validation_rmse",
        "validation_results": [result.to_dict() for result in scenario_results],
        # Không lấy tuning từ một model thua cuộc.
        "hyperparameter_tuning": best_result.tuning,
        "feature_engineering": {
            "selected_feature_set": best_result.feature_set,
            "enabled": best_result.feature_set == FEATURE_SET_ENGINEERED,
            "engineered_features": engineered_features,
            "validation_comparison": summarize_feature_set_results(
                scenario_results
            ),
        },
        "test_metrics": test_metrics.to_dict(),
        "input_features": input_features,
        "total_input_features": len(input_features),
        "model_features_before_preprocessing": model_feature_columns,
        "total_model_features_before_preprocessing": len(
            model_feature_columns
        ),
        "total_features_after_preprocessing": len(feature_names),
        "evaluation_artifacts": evaluation_artifacts,
    }


def write_model_card(
    metrics_payload: dict[str, Any],
    input_features: list[str],
    paths: ScenarioArtifactPaths,
) -> None:

    test_metrics = metrics_payload["test_metrics"]
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

    feature_metadata = metrics_payload.get("feature_engineering", {})
    selected_feature_set = feature_metadata.get(
        "selected_feature_set",
        "N/A",
    )
    engineered_features = feature_metadata.get("engineered_features", [])
    comparison = feature_metadata.get("validation_comparison", {})
    comparison_lines = []
    for feature_set, values in comparison.items():
        comparison_lines.append(
            f"- `{feature_set}`: model tốt nhất "
            f"`{values.get('best_model', 'N/A')}`, "
            f"RMSE=`{values.get('rmse', 'N/A')}`, "
            f"MAE=`{values.get('mae', 'N/A')}`, "
            f"R²=`{values.get('r2', 'N/A')}`"
        )
    comparison_text = "\n".join(comparison_lines) or "- Chưa có dữ liệu so sánh."

    tuning = metrics_payload.get("hyperparameter_tuning")
    tuning_section = ""
    if tuning:
        best_params = ", ".join(
            f"`{name}`={value}"
            for name, value in tuning.get("best_params", {}).items()
        )
        tuning_section = f"""
## 4. Tinh chỉnh siêu tham số của model được chọn

- Phương pháp: `{tuning.get('method', 'N/A')}`
- Chiến lược CV: `{tuning.get('cv_strategy', 'N/A')}`
- Cách nhóm: `{tuning.get('grouping', 'N/A')}`
- Số nhóm học sinh trong train: `{tuning.get('unique_training_groups', 'N/A')}`
- Số fold: `{tuning.get('cv_folds', 'N/A')}`
- Số cấu hình: `{tuning.get('candidate_count', 'N/A')}`
- Tổng lượt fit: `{tuning.get('total_fits', 'N/A')}`
- CV RMSE tốt nhất: `{tuning.get('best_cv_rmse', 'N/A')}`
- Tham số tốt nhất: {best_params or 'N/A'}
"""

    model_card = f"""# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `{TARGET_COLUMN}`
- Thang điểm mục tiêu: 0–20
- Model được chọn: `{metrics_payload['best_model']}`
- Scenario: `{metrics_payload['best_scenario']}`
- Feature set được chọn: `{selected_feature_set}`
- Cách dùng đặc trưng: {excluded_note}

## 2. Chỉ số trên tập test

- MAE: `{test_metrics['mae']}`
- RMSE: `{test_metrics['rmse']}`
- R²: `{test_metrics['r2']}`

## 3. So sánh feature engineering trên validation

{comparison_text}

- Đặc trưng dẫn xuất được model cuối sử dụng: `{', '.join(engineered_features) or 'Không có'}`

{tuning_section}

## 5. Đặc trưng và output

- Số cột đầu vào thô: `{len(input_features)}`
- Số cột trước tiền xử lý sau feature engineering: `{metrics_payload.get('total_model_features_before_preprocessing', len(input_features))}`
- Danh sách đầu vào thô: `{', '.join(input_features)}`
- Đầu ra của model: Điểm `G3` dạng số thực
- Số biểu đồ đánh giá: `{plot_count}`
- Model: `{paths.model.relative_to(ARTIFACT_DIR.parent).as_posix()}`
- Metrics: `{paths.metrics.relative_to(ARTIFACT_DIR.parent).as_posix()}`
- Biểu đồ: `{paths.plots.relative_to(ARTIFACT_DIR.parent).as_posix()}`

## 6. Lưu ý

- GroupKFold giữ các bản ghi của cùng học sinh trong cùng một fold khi tuning.
- MAE và RMSE càng thấp càng tốt; R² càng gần 1 càng tốt.
- Đầu ra duy nhất của hệ thống là điểm G3 dự đoán trên thang 0–20.
- Không nên dùng dự đoán như quyết định duy nhất về kết quả học tập.
"""

    save_text(model_card, paths.model_card)


def train_and_save_scenario(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scenario_name: str,
) -> dict[str, Any]:
    """Train, chọn model/feature set tốt nhất và lưu artifact."""

    scenario_results = train_scenario(
        train_df=train_df,
        valid_df=valid_df,
        scenario_name=scenario_name,
    )
    best_result = select_best_result(scenario_results)

    LOGGER.info(
        "Model tốt nhất của %s: %s | feature_set=%s",
        scenario_name,
        best_result.model,
        best_result.feature_set,
    )

    final_pipeline, input_features, model_feature_columns = (
        refit_best_pipeline(
            best_result=best_result,
            train_df=train_df,
            valid_df=valid_df,
        )
    )

    X_test, y_test = split_features_target(test_df, input_features)
    test_predictions, test_metrics = predict_and_evaluate(
        estimator=final_pipeline,
        X=X_test,
        y=y_test,
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

    metrics_payload = build_metrics_payload(
        best_result=best_result,
        scenario_results=scenario_results,
        test_metrics=test_metrics,
        input_features=input_features,
        model_feature_columns=model_feature_columns,
        feature_names=feature_names,
        evaluation_artifacts=evaluation_artifacts,
    )

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
        "best_feature_set": best_result.feature_set,
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
        "feature_engineering": metrics_payload["feature_engineering"],
        "hyperparameter_tuning": metrics_payload["hyperparameter_tuning"],
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
    save_json(training_summary, TRAINING_SUMMARY_PATH)

    LOGGER.info("Đã tạo đủ %d model.", len(scenario_summaries))
    LOGGER.info("Tóm tắt huấn luyện: %s", TRAINING_SUMMARY_PATH)
    LOGGER.info("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
