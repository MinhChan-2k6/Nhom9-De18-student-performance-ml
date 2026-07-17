from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """Phân chia đặc trưng thành cột số và cột phân loại."""

    missing_features = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Dữ liệu thiếu các cột đặc trưng: "
            + ", ".join(missing_features)
        )

    feature_df = dataframe.loc[:, feature_columns]

    numeric_columns = feature_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = feature_df.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def build_numeric_transformer() -> Pipeline:
    """Tạo pipeline điền khuyết và chuẩn hóa cho đặc trưng số."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_transformer() -> Pipeline:
    """Tạo pipeline điền khuyết và one-hot cho đặc trưng phân loại."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def build_preprocessor(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> ColumnTransformer:
    """Tạo bộ tiền xử lý chung cho các cột số và phân loại."""

    numeric_columns, categorical_columns = get_feature_types(
        dataframe,
        feature_columns,
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                build_numeric_transformer(),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                build_categorical_transformer(),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("Không có cột đặc trưng để tiền xử lý.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Lấy tên đặc trưng sau khi pipeline tiền xử lý đã được fit."""

    if "preprocessor" not in fitted_pipeline.named_steps:
        raise ValueError(
            "Pipeline không có bước 'preprocessor'."
        )

    preprocessor = fitted_pipeline.named_steps["preprocessor"]

    if not hasattr(preprocessor, "get_feature_names_out"):
        return []

    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]
