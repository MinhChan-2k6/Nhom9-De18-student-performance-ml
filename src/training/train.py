from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# CẤU HÌNH CHUNG
# ==================================================

RANDOM_STATE = 42
TARGET_COLUMN = "G3"

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


# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
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

def load_dataset(file_path: Path) -> pd.DataFrame:
    """Đọc CSV và kiểm tra dữ liệu cơ bản."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"File dữ liệu không có dòng nào: {file_path}"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột mục tiêu {TARGET_COLUMN} "
            f"trong file {file_path.name}"
        )

    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            "Dữ liệu có tên cột bị trùng: "
            + ", ".join(duplicate_columns)
        )

    return dataframe


def validate_dataset_compatibility(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
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
# SCENARIO
# ==================================================

def get_scenario_features(
    dataframe: pd.DataFrame,
    scenario_name: str
) -> list[str]:
    """
    Scenario A: sử dụng G1 và G2.
    Scenario B: loại bỏ G1 và G2 để dự báo sớm.
    """

    feature_columns = [
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
    ]

    if scenario_name == "A_with_G1_G2":
        return feature_columns

    if scenario_name == "B_without_G1_G2":
        return [
            column
            for column in feature_columns
            if column not in {"G1", "G2"}
        ]

    raise ValueError(
        f"Scenario không hợp lệ: {scenario_name}"
    )


# ==================================================
# TIỀN XỬ LÝ
# ==================================================

def build_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str]
) -> ColumnTransformer:
    """Tạo bộ tiền xử lý cho cột số và cột phân loại."""

    feature_df = dataframe[feature_columns]

    numeric_columns = feature_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = feature_df.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


# ==================================================
# MODEL
# ==================================================

def build_candidate_models() -> dict[str, Any]:
    """Danh sách model cơ sở để so sánh."""

    return {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    }


def build_pipeline(
    preprocessor: ColumnTransformer,
    model: Any
) -> Pipeline:
    """Gộp tiền xử lý và model thành một pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
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

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]

    preprocessor = build_preprocessor(
        train_df,
        feature_columns
    )

    pipeline = build_pipeline(
        preprocessor,
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    parameter_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.8]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
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

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    X_valid = valid_df[feature_columns]
    y_valid = valid_df[TARGET_COLUMN]

    results: list[EvaluationResult] = []

    for model_name, model in build_candidate_models().items():
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

        result = evaluate_pipeline(
            estimator=pipeline,
            X_valid=X_valid,
            y_valid=y_valid,
            scenario_name=scenario_name,
            model_name=model_name
        )

        results.append(result)

        LOGGER.info(
            "%s | %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            scenario_name,
            model_name,
            result.mae,
            result.rmse,
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

    X_combined = combined_df[feature_columns]
    y_combined = combined_df[TARGET_COLUMN]

    # Clone bằng cách tái sử dụng tham số model đã chọn.
    chosen_model = best_result.estimator.named_steps["model"]

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

    return final_pipeline, feature_columns


def evaluate_on_test(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_columns: list[str]
) -> dict[str, float]:
    """Đánh giá model cuối cùng trên test set."""

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

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

def get_feature_names(
    fitted_pipeline: Pipeline
) -> list[str]:
    """Lấy tên feature sau tiền xử lý."""

    preprocessor = fitted_pipeline.named_steps[
        "preprocessor"
    ]

    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except AttributeError:
        return []


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
    )


def write_model_card(
    metrics_payload: dict[str, Any],
    input_features: list[str]
) -> None:
    """Tạo model card Markdown."""

    test_metrics = metrics_payload["test_metrics"]

    model_card = f"""# Model Card — Student Performance Regression

## 1. Bài toán

- Loại bài toán: Hồi quy
- Biến mục tiêu: `{TARGET_COLUMN}`
- Thang điểm mục tiêu: 0–20
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

    for scenario_name in [
        "A_with_G1_G2",
        "B_without_G1_G2"
    ]:
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
    )

    final_pipeline, input_features = refit_best_pipeline(
        best_result=best_result,
        train_df=train_df,
        valid_df=valid_df
    )

    test_metrics = evaluate_on_test(
        pipeline=final_pipeline,
        test_df=test_df,
        feature_columns=input_features
    )

    feature_names = get_feature_names(
        final_pipeline
    )

    metrics_payload = {
        "task": "regression",
        "target": TARGET_COLUMN,
        "best_model": best_result.model,
        "best_scenario": best_result.scenario,
        "selection_metric": "validation_rmse",
        "validation_results": [
            result.to_dict()
            for result in all_results
        ],
        "test_metrics": test_metrics,
        "input_features": input_features,
        "total_input_features": len(input_features),
        "total_features_after_preprocessing": len(feature_names)
    }

    save_artifacts(
        final_pipeline=final_pipeline,
        feature_names=feature_names,
        metrics_payload=metrics_payload
    )

    write_model_card(
        metrics_payload=metrics_payload,
        input_features=input_features
    )

    LOGGER.info("Đã lưu model tại: %s", MODEL_PATH)
    LOGGER.info("Test metrics: %s", test_metrics)
    LOGGER.info("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
