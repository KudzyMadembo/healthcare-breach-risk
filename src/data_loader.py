"""Utilities for loading CICIoMT2024 training CSV data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Relative dataset paths from the project root.
TRAIN_RELATIVE_DIR = Path("data/raw/CICIoMT2024/WiFi and MQTT/train")
OUTPUT_RELATIVE_PATH = Path("data/interim/train_dataset.csv")


def _standardize_column_name(column_name: str) -> str:
    """Convert a column name to lowercase snake_case."""
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", column_name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _extract_label_from_filename(filename: str) -> str:
    """Extract the attack label from CICIoMT2024 file naming format.

    Examples:
        TCP_IP-DDoS-TCP3_train.pcap.csv -> TCP_IP-DDoS-TCP
        TCP_IP-DoS-SYN2_train.pcap.csv -> TCP_IP-DoS-SYN
    """
    match = re.match(r"^(?P<label>.+?)(?:\d+)?_train\.pcap\.csv$", filename)
    if match:
        return match.group("label")

    # Fallback for unexpected names: remove known suffix and trailing instance digits.
    raw_label = re.sub(r"_train\.pcap\.csv$", "", filename)
    return re.sub(r"\d+$", "", raw_label)


def load_training_dataset(base_path: str) -> pd.DataFrame:
    """Load and merge CICIoMT2024 training data into one DataFrame.

    Args:
        base_path: Project root path containing the `data/` directory.

    Returns:
        A merged pandas DataFrame across all `*.pcap.csv` files in the train folder.

    Raises:
        FileNotFoundError: If the expected train directory does not exist.
        ValueError: If no matching training CSV files are found.
    """
    base_dir = Path(base_path).resolve()
    train_dir = base_dir / TRAIN_RELATIVE_DIR

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    csv_files = sorted(train_dir.rglob("*.pcap.csv"))
    if not csv_files:
        raise ValueError(f"No '*.pcap.csv' files found under: {train_dir}")

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, low_memory=False)
        frame.columns = [_standardize_column_name(col) for col in frame.columns]
        frame["label"] = _extract_label_from_filename(csv_file.name)
        frame["source_file"] = str(csv_file.relative_to(base_dir)).replace("\\", "/")
        frames.append(frame)

    merged_df = pd.concat(frames, ignore_index=True, sort=False)

    print(f"Total files loaded: {len(csv_files)}")
    print(f"DataFrame shape: {merged_df.shape}")
    print("Label distribution:")
    print(merged_df["label"].value_counts())

    return merged_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / OUTPUT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(str(project_root))
    dataset.to_csv(output_path, index=False)

    print(f"Merged training dataset saved to: {output_path}")
