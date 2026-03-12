"""Preprocessing pipeline for CICIoMT2024 merged training data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

INPUT_RELATIVE_PATH = Path("data/interim/train_dataset.csv")

PROCESSED_RELATIVE_PATH = Path("data/processed/train_processed.csv")
LABEL_MAPPING_RELATIVE_PATH = Path("models/label_mapping.json")
METADATA_COLUMNS = ["label", "source_file"]


def load_merged_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the merged training dataset from CSV.

    Uses a development sample of up to 500,000 rows to reduce memory pressure.
    """
    df = pd.read_csv(dataset_path, low_memory=False)

    # Sample for development to avoid memory crashes on full dataset loads.
    sample_size = min(500_000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and normalize string columns."""
    cleaned = df.drop_duplicates().copy()

    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in string_columns:
        cleaned[col] = cleaned[col].astype("string").str.strip()

    return cleaned


def preprocess_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, int]]:
    """Build processed features and encoded target label.

    Steps:
    - Detect numeric and categorical columns automatically.
    - Fill missing numeric values with median.
    - Fill missing categorical values with "unknown".
    - Scale numeric features using MinMaxScaler.
    - One-hot encode categorical features.
    - Encode labels with LabelEncoder.
    """
    if "label" not in df.columns:
        raise ValueError("Input dataset must contain a 'label' column.")

    y_raw = df["label"].astype("string").fillna("unknown")
    feature_df = df.drop(columns=[col for col in METADATA_COLUMNS if col in df.columns]).copy()

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

    # Numeric preprocessing: median imputation followed by MinMax scaling.
    if numeric_cols:
        feature_df[numeric_cols] = feature_df[numeric_cols].apply(
            lambda col: col.fillna(col.median()),
            axis=0,
        )
        scaler = MinMaxScaler()
        feature_df[numeric_cols] = scaler.fit_transform(feature_df[numeric_cols])

    # Categorical preprocessing: fill missing values and one-hot encode.
    if categorical_cols:
        for col in categorical_cols:
            feature_df[col] = feature_df[col].astype("string").fillna("unknown")

        feature_df = pd.get_dummies(
            feature_df,
            columns=categorical_cols,
            prefix=categorical_cols,
            prefix_sep="__",
            dtype=int,
        )

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(
        label_encoder.fit_transform(y_raw),
        name="label_encoded",
    )
    label_mapping = {
        str(label): int(index)
        for index, label in enumerate(label_encoder.classes_)
    }

    feature_names = feature_df.columns.tolist()
    return feature_df, y_encoded, feature_names, label_mapping


def preprocess_training_dataset(base_path: str) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, int]]:
    """Preprocess CICIoMT2024 merged training data and save outputs."""
    project_root = Path(base_path).resolve()
    input_path = project_root / INPUT_RELATIVE_PATH
    processed_output_path = project_root / PROCESSED_RELATIVE_PATH
    mapping_output_path = project_root / LABEL_MAPPING_RELATIVE_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = load_merged_dataset(input_path)
    cleaned_df = clean_dataset(df)
    x_processed, y_encoded, feature_names, label_mapping = preprocess_features_and_target(cleaned_df)

    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_output = x_processed.copy()
    if "source_file" in cleaned_df.columns:
        processed_output["source_file"] = cleaned_df["source_file"].astype("string").values
    processed_output["label_encoded"] = y_encoded.values
    processed_output.to_csv(processed_output_path, index=False)
    mapping_output_path.write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    return x_processed, y_encoded, feature_names, label_mapping


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source_path = root / INPUT_RELATIVE_PATH

    original_df = load_merged_dataset(source_path)
    cleaned_df = clean_dataset(original_df)
    x_processed, y_encoded, feature_names, label_mapping = preprocess_training_dataset(str(root))

    print(f"Original shape: {original_df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("Label mapping:")
    print(label_mapping)
    print(f"Encoded labels shape: {y_encoded.shape}")
