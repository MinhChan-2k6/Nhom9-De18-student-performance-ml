"""Tạo bộ biểu đồ đánh giá mô hình và lưu dưới dạng PNG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.compare import build_result_label, results_to_dataframe
from src.evaluation.metrics import (
    EvaluationResult,
    RegressionMetrics,
    calculate_pass_fail_metrics,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_DPI = 160
DEFAULT_FIGSIZE = (10, 6)


def _prepare_output_directory(output_dir: Path) -> None:
    """Tạo thư mục chứa biểu đồ nếu chưa tồn tại."""

    output_dir.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Lưu figure với cấu hình đồng nhất và giải phóng bộ nhớ."""

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    return output_path


def plot_model_comparison(
    results: list[EvaluationResult],
    output_dir: Path,
) -> list[Path]:
    """Vẽ riêng MAE, RMSE và R² để tránh trộn các thang đo."""

    _prepare_output_directory(output_dir)
    dataframe = results_to_dataframe(results)
    dataframe["label"] = [
        build_result_label(scenario, model)
        for scenario, model in zip(
            dataframe["scenario"],
            dataframe["model"],
            strict=True,
        )
    ]

    output_paths: list[Path] = []
    metric_specs = (
        ("mae", "MAE trên validation", "MAE (thấp hơn tốt hơn)"),
        ("rmse", "RMSE trên validation", "RMSE (thấp hơn tốt hơn)"),
        ("r2", "R² trên validation", "R² (cao hơn tốt hơn)"),
    )

    for metric_name, title, x_label in metric_specs:
        metric_df = dataframe.sort_values(
            metric_name,
            ascending=metric_name != "r2",
        )

        height = max(5.5, 0.62 * len(metric_df))
        fig, ax = plt.subplots(figsize=(11, height))
        bars = ax.barh(metric_df["label"], metric_df[metric_name])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.grid(axis="x", alpha=0.25)

        if metric_name != "r2":
            ax.invert_yaxis()

        for bar, value in zip(bars, metric_df[metric_name], strict=True):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}",
                va="center",
                fontsize=9,
            )

        output_path = output_dir / f"model_comparison_{metric_name}.png"
        output_paths.append(_save_figure(fig, output_path))

    return output_paths


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    metrics: RegressionMetrics,
    output_path: Path,
) -> Path:
    """Vẽ điểm dự đoán so với điểm thực tế cùng đường lý tưởng y=x."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)

    lower_bound = float(min(true_values.min(), predicted_values.min()))
    upper_bound = float(max(true_values.max(), predicted_values.max()))
    padding = max(0.5, (upper_bound - lower_bound) * 0.05)
    limits = (lower_bound - padding, upper_bound + padding)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        true_values,
        predicted_values,
        alpha=0.68,
        s=38,
        edgecolors="none",
    )
    ax.plot(limits, limits, linestyle="--", linewidth=1.8, label="Dự đoán lý tưởng")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Điểm thực tế và điểm dự đoán", fontweight="bold")
    ax.set_xlabel("G3 thực tế")
    ax.set_ylabel("G3 dự đoán")
    ax.grid(alpha=0.25)
    ax.legend()

    metric_text = (
        f"MAE = {metrics.mae:.3f}\n"
        f"RMSE = {metrics.rmse:.3f}\n"
        f"R² = {metrics.r2:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    return _save_figure(fig, output_path)


def plot_score_distribution(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """So sánh phân phối điểm thực tế và điểm dự đoán."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    bins = np.arange(-0.5, 21.5, 1.0)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(
        true_values,
        bins=bins,
        alpha=0.58,
        label="G3 thực tế",
    )
    ax.hist(
        predicted_values,
        bins=bins,
        alpha=0.58,
        label="G3 dự đoán",
    )
    ax.set_title("Phân phối điểm thực tế và dự đoán", fontweight="bold")
    ax.set_xlabel("Điểm G3")
    ax.set_ylabel("Số học sinh")
    ax.set_xticks(range(0, 21, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    return _save_figure(fig, output_path)


def plot_residuals_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """Kiểm tra residual có phân bố ngẫu nhiên quanh đường 0 hay không."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    residuals = true_values - predicted_values

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.scatter(
        predicted_values,
        residuals,
        alpha=0.68,
        s=36,
        edgecolors="none",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.8)
    ax.set_title("Residual theo điểm dự đoán", fontweight="bold")
    ax.set_xlabel("G3 dự đoán")
    ax.set_ylabel("Residual = thực tế - dự đoán")
    ax.grid(alpha=0.25)

    return _save_figure(fig, output_path)


def plot_residual_distribution(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """Vẽ phân phối residual và đánh dấu sai số trung bình."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    residuals = true_values - predicted_values
    residual_mean = float(residuals.mean())

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.hist(residuals, bins="auto", alpha=0.78, edgecolor="white")
    ax.axvline(0.0, linestyle="--", linewidth=1.8, label="Residual = 0")
    ax.axvline(
        residual_mean,
        linestyle=":",
        linewidth=1.8,
        label=f"Trung bình = {residual_mean:.3f}",
    )
    ax.set_title("Phân phối residual", fontweight="bold")
    ax.set_xlabel("Residual = thực tế - dự đoán")
    ax.set_ylabel("Tần suất")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    return _save_figure(fig, output_path)


def plot_absolute_error_by_score(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """Cho biết model thường sai nhiều ở vùng điểm thực tế nào."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    absolute_errors = np.abs(true_values - predicted_values)

    error_df = pd.DataFrame(
        {
            "actual_score": true_values,
            "absolute_error": absolute_errors,
        }
    )
    grouped = (
        error_df.groupby("actual_score", as_index=False)["absolute_error"]
        .agg(["mean", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.plot(
        grouped["actual_score"],
        grouped["mean"],
        marker="o",
        linewidth=1.8,
    )
    ax.set_title("Sai số tuyệt đối trung bình theo điểm G3", fontweight="bold")
    ax.set_xlabel("G3 thực tế")
    ax.set_ylabel("Sai số tuyệt đối trung bình")
    ax.set_xticks(sorted(grouped["actual_score"].astype(int).unique()))
    ax.grid(alpha=0.25)

    for _, row in grouped.iterrows():
        if int(row["count"]) >= 3:
            ax.annotate(
                f"n={int(row['count'])}",
                (row["actual_score"], row["mean"]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    return _save_figure(fig, output_path)


def plot_pass_fail_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    pass_threshold: float = 10.0,
) -> Path:
    """Đánh giá quy đổi Đạt/Trượt từ điểm G3 dự đoán."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)
    pass_fail_metrics = calculate_pass_fail_metrics(
        y_true=true_values,
        y_pred=predicted_values,
        threshold=pass_threshold,
    )
    matrix = np.asarray(pass_fail_metrics.confusion_matrix, dtype=int)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Ma trận nhầm lẫn Đạt/Trượt", fontweight="bold")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_xticks([0, 1], labels=["Trượt", "Đạt"])
    ax.set_yticks([0, 1], labels=["Trượt", "Đạt"])

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=13,
                fontweight="bold",
            )

    ax.text(
        0.5,
        -0.16,
        f"Ngưỡng đạt: G3 ≥ {pass_threshold:g} | "
        f"Accuracy = {pass_fail_metrics.accuracy:.3f} | "
        f"F1 = {pass_fail_metrics.f1:.3f}",
        transform=ax.transAxes,
        ha="center",
    )

    return _save_figure(fig, output_path)


def save_model_comparison(
    results: list[EvaluationResult],
    output_path: Path,
) -> Path:
    """Lưu bảng so sánh model dưới dạng CSV để dùng trong báo cáo."""

    dataframe = results_to_dataframe(results)
    dataframe["display_name"] = [
        build_result_label(scenario, model)
        for scenario, model in zip(
            dataframe["scenario"],
            dataframe["model"],
            strict=True,
        )
    ]
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _extract_feature_importance(
    fitted_pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """Lấy feature importance từ model cây hoặc hệ số tuyệt đối từ model tuyến tính."""

    if "model" not in fitted_pipeline.named_steps:
        raise ValueError("Pipeline không có bước 'model'.")

    model = fitted_pipeline.named_steps["model"]
    importance_values: np.ndarray | None = None
    importance_type = ""

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_, dtype=float)
        importance_type = "Feature importance"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        importance_values = np.abs(coefficients).reshape(-1)
        importance_type = "|Hệ số|"

    if importance_values is None:
        return None

    if len(feature_names) != len(importance_values):
        LOGGER.warning(
            "Không vẽ feature importance vì số tên feature (%d) khác số importance (%d).",
            len(feature_names),
            len(importance_values),
        )
        return None

    dataframe = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
            "importance_type": importance_type,
        }
    )
    return dataframe.sort_values("importance", ascending=False, ignore_index=True)


def plot_feature_importance(
    fitted_pipeline: Pipeline,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> Path | None:
    """Vẽ top đặc trưng ảnh hưởng mạnh nhất đến dự đoán."""

    importance_df = _extract_feature_importance(
        fitted_pipeline=fitted_pipeline,
        feature_names=feature_names,
    )

    if importance_df is None or importance_df.empty:
        LOGGER.warning("Model không hỗ trợ biểu đồ feature importance.")
        return None

    top_features = importance_df.head(top_n).sort_values("importance")
    fig_height = max(6.0, 0.4 * len(top_features))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title(f"Top {len(top_features)} đặc trưng quan trọng", fontweight="bold")
    ax.set_xlabel(str(top_features["importance_type"].iloc[0]))
    ax.set_ylabel("Đặc trưng sau tiền xử lý")
    ax.grid(axis="x", alpha=0.25)

    max_importance = float(top_features["importance"].max())
    for bar, value in zip(bars, top_features["importance"], strict=True):
        ax.text(
            bar.get_width() + max_importance * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )

    return _save_figure(fig, output_path)


def save_test_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """Lưu dữ liệu dùng trong biểu đồ để dễ kiểm chứng và làm báo cáo."""

    true_values = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_values = np.asarray(y_pred, dtype=float).reshape(-1)

    prediction_df = pd.DataFrame(
        {
            "actual_g3": true_values,
            "predicted_g3": predicted_values,
            "residual": true_values - predicted_values,
            "absolute_error": np.abs(true_values - predicted_values),
        }
    )
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def generate_evaluation_artifacts(
    results: list[EvaluationResult],
    final_pipeline: Pipeline,
    feature_names: list[str],
    y_test: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    test_metrics: RegressionMetrics,
    output_dir: Path,
    predictions_path: Path,
) -> dict[str, Any]:
    """Tạo toàn bộ biểu đồ và file dự đoán sau khi huấn luyện."""

    _prepare_output_directory(output_dir)
    generated_paths: list[Path] = []

    generated_paths.extend(
        plot_model_comparison(
            results=results,
            output_dir=output_dir,
        )
    )
    generated_paths.append(
        plot_actual_vs_predicted(
            y_true=y_test,
            y_pred=y_pred,
            metrics=test_metrics,
            output_path=output_dir / "actual_vs_predicted.png",
        )
    )
    generated_paths.append(
        plot_score_distribution(
            y_true=y_test,
            y_pred=y_pred,
            output_path=output_dir / "score_distribution.png",
        )
    )
    generated_paths.append(
        plot_residuals_vs_predicted(
            y_true=y_test,
            y_pred=y_pred,
            output_path=output_dir / "residuals_vs_predicted.png",
        )
    )
    generated_paths.append(
        plot_residual_distribution(
            y_true=y_test,
            y_pred=y_pred,
            output_path=output_dir / "residual_distribution.png",
        )
    )
    generated_paths.append(
        plot_absolute_error_by_score(
            y_true=y_test,
            y_pred=y_pred,
            output_path=output_dir / "absolute_error_by_score.png",
        )
    )
    generated_paths.append(
        plot_pass_fail_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            output_path=output_dir / "pass_fail_confusion_matrix.png",
        )
    )

    feature_importance_path = plot_feature_importance(
        fitted_pipeline=final_pipeline,
        feature_names=feature_names,
        output_path=output_dir / "feature_importance.png",
    )
    if feature_importance_path is not None:
        generated_paths.append(feature_importance_path)

    predictions_file = save_test_predictions(
        y_true=y_test,
        y_pred=y_pred,
        output_path=predictions_path,
    )
    comparison_file = save_model_comparison(
        results=results,
        output_path=output_dir / "model_comparison.csv",
    )

    project_root = output_dir.parent.parent

    def portable_path(path: Path) -> str:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()

    return {
        "plots_directory": portable_path(output_dir),
        "test_predictions": portable_path(predictions_file),
        "model_comparison_table": portable_path(comparison_file),
        "generated_plots": [portable_path(path) for path in generated_paths],
    }
