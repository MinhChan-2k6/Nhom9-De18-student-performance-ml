from __future__ import annotations
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

MATH_FILE = RAW_DIR / "student-mat.csv"
PORTUGUESE_FILE = RAW_DIR / "student-por.csv"
MERGED_FILE = INTERIM_DIR / "merged.csv"


def load_subject_data(path: Path, subject: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path, sep=";")
    df.insert(0, "subject", subject)
    return df


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    math_df = load_subject_data(MATH_FILE, "math")
    portuguese_df = load_subject_data(PORTUGUESE_FILE, "portuguese")
    return math_df, portuguese_df


def merge_raw_data() -> pd.DataFrame:
    math_df, portuguese_df = load_raw_data()

    if list(math_df.columns) != list(portuguese_df.columns):
        missing_in_math = sorted(set(portuguese_df.columns) - set(math_df.columns))
        missing_in_portuguese = sorted(set(math_df.columns) - set(portuguese_df.columns))
        raise ValueError(
            "Raw files do not have the same schema. "
            f"Missing in math: {missing_in_math}; "
            f"missing in portuguese: {missing_in_portuguese}"
        )

    return pd.concat([math_df, portuguese_df], ignore_index=True)


def save_merged_data(output_path: Path = MERGED_FILE) -> pd.DataFrame:
    merged_df = merge_raw_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return merged_df


def main() -> None:
    merged_df = save_merged_data()
    print(f"Saved merged data to: {MERGED_FILE}")
    print(f"Rows: {merged_df.shape[0]}, columns: {merged_df.shape[1]}")
    print("Subject counts:")
    print(merged_df["subject"].value_counts().to_string())


if __name__ == "__main__":
    main()
