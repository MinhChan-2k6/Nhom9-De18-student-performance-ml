
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DPI = 150
REQUIRED_COLUMNS = {"subject", "G1", "G2", "G3", "absences"}


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by looking for the data and src folders."""

    current = (start or Path(__file__)).resolve()
    for folder in [current, *current.parents]:
        if (folder / "data").exists() and (folder / "src").exists():
            return folder

    raise FileNotFoundError(
        "Could not find the project root. Run this script inside the repo."
    )


PROJECT_ROOT = find_project_root()
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACT_DIR / "figures"
EDA_SUMMARY_PATH = ARTIFACT_DIR / "eda_summary.json"
MERGED_FILE = PROJECT_ROOT / "data" / "interim" / "merged.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MATH_FILE = RAW_DIR / "student-mat.csv"
PORTUGUESE_FILE = RAW_DIR / "student-por.csv"


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
        }
    )


def load_subject_data(path: Path, subject: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path, sep=";")
    df.insert(0, "subject", subject)
    return df


def load_eda_data(input_path: Path | None = None) -> pd.DataFrame:
    """Load merged/interim data, or rebuild the EDA frame from raw CSV files."""

    if input_path is not None:
        df = pd.read_csv(input_path)
    elif MERGED_FILE.exists():
        df = pd.read_csv(MERGED_FILE)
    else:
        math_df = load_subject_data(MATH_FILE, "math")
        portuguese_df = load_subject_data(PORTUGUESE_FILE, "portuguese")
        df = pd.concat([math_df, portuguese_df], ignore_index=True)

    validate_eda_data(df)
    return df


def validate_eda_data(df: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError("EDA data is missing columns: " + ", ".join(missing_columns))

    for column in ["G1", "G2", "G3", "absences"]:
        df[column] = pd.to_numeric(df[column], errors="raise")


def save_json(output_path: Path, payload: dict) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_eda_summary(df: pd.DataFrame, output_path: Path = EDA_SUMMARY_PATH) -> Path:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "subjects": {str(k): int(v) for k, v in df["subject"].value_counts().items()},
        "g3_mean": float(df["G3"].mean()),
        "g3_median": float(df["G3"].median()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return save_json(output_path, summary)


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_g3_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=df,
        x="G3",
        hue="subject",
        discrete=True,
        multiple="dodge",
        shrink=0.8,
        ax=ax,
    )
    ax.set_title("Final grade G3 distribution by subject")
    ax.set_xlabel("G3 final grade")
    ax.set_ylabel("Number of students")
    return save_figure(fig, output_dir / "eda_g3_distribution.png")


def plot_g3_by_subject(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="subject", y="G3", ax=ax)
    ax.set_title("G3 distribution by subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("G3 final grade")
    return save_figure(fig, output_dir / "eda_g3_by_subject.png")


def plot_previous_grades_vs_g3(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=df, x="G1", y="G3", hue="subject", alpha=0.65, ax=axes[0])
    axes[0].set_title("Relationship between G1 and G3")
    sns.scatterplot(data=df, x="G2", y="G3", hue="subject", alpha=0.65, ax=axes[1])
    axes[1].set_title("Relationship between G2 and G3")

    for ax in axes:
        ax.set_xlabel("Previous period grade")
        ax.set_ylabel("G3 final grade")

    return save_figure(fig, output_dir / "eda_previous_grades_vs_g3.png")


def plot_numeric_correlation(df: pd.DataFrame, output_dir: Path) -> Path:
    corr = df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix for numeric variables")
    return save_figure(fig, output_dir / "eda_numeric_correlation.png")


def plot_absences_vs_g3(df: pd.DataFrame, output_dir: Path) -> Path:
    absence_bins = [-1, 5, 10, 20, 100]
    absence_labels = ["0-5", "6-10", "11-20", "21+"]
    plot_df = df.assign(
        absence_group=pd.cut(df["absences"], bins=absence_bins, labels=absence_labels)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="absence_group", y="G3", ax=ax)
    ax.set_title("G3 by absence group")
    ax.set_xlabel("Number of absences")
    ax.set_ylabel("G3 final grade")
    return save_figure(fig, output_dir / "eda_absences_vs_g3.png")


def generate_eda_figures(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    return [
        plot_g3_distribution(df, output_dir),
        plot_g3_by_subject(df, output_dir),
        plot_previous_grades_vs_g3(df, output_dir),
        plot_numeric_correlation(df, output_dir),
        plot_absences_vs_g3(df, output_dir),
    ]


def print_generated(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"- {display_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate EDA figures from student performance data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional CSV input. Defaults to data/interim/merged.csv, then raw CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for generated PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    configure_output()
    configure_plots()
    args = parse_args()

    df = load_eda_data(args.input)
    summary_path = write_eda_summary(df)
    figure_paths = generate_eda_figures(df, args.output_dir)

    print("Generated EDA artifacts:")
    print_generated([summary_path, *figure_paths])


if __name__ == "__main__":
    main()
