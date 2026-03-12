"""Build temporally valid sliding-window sequences for CICIoMT2024."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_RELATIVE_PATH = Path("data/processed/train_processed.csv")
OUTPUT_X_RELATIVE_PATH = Path("data/processed/X_seq.npy")
OUTPUT_Y_RELATIVE_PATH = Path("data/processed/y_seq.npy")
TARGET_COLUMN = "label_encoded"
SOURCE_COLUMN = "source_file"
ORDER_COLUMN = "iat"
SUPPORTED_LABEL_STRATEGIES = {"last_label", "majority_label"}


def load_processed_tabular_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load processed tabular dataset from CSV."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path, low_memory=False)


def _derive_sequence_label(window_labels: np.ndarray, strategy: str) -> int:
    """Derive one sequence label from labels inside a window."""
    if strategy == "last_label":
        return int(window_labels[-1])

    if strategy == "majority_label":
        # Labels are encoded integers, so bincount is efficient.
        return int(np.bincount(window_labels).argmax())

    raise ValueError(f"Unsupported label strategy: {strategy}")


def _num_sequences_for_rows(num_rows: int, sequence_length: int, stride: int) -> int:
    """Compute number of sliding windows for a group length."""
    if num_rows < sequence_length:
        return 0
    return ((num_rows - sequence_length) // stride) + 1


def _validate_inputs(
    df: pd.DataFrame,
    sequence_length: int,
    stride: int,
    label_strategy: str,
) -> None:
    """Validate required columns and sequence parameters."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if label_strategy not in SUPPORTED_LABEL_STRATEGIES:
        raise ValueError(
            f"label_strategy must be one of {sorted(SUPPORTED_LABEL_STRATEGIES)}"
        )

    required_columns = {TARGET_COLUMN, SOURCE_COLUMN, ORDER_COLUMN}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            "Processed dataset is missing required columns for temporal sequencing: "
            f"{missing}. Ensure preprocessing preserves '{SOURCE_COLUMN}' and '{ORDER_COLUMN}'."
        )


def _preallocate_sequence_arrays(
    df: pd.DataFrame,
    sequence_length: int,
    stride: int,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Preallocate output arrays and compute sequence counts per source file."""
    sequences_per_file: dict[str, int] = {}
    total_sequences = 0

    grouped_sizes = df.groupby(SOURCE_COLUMN, sort=False).size()
    for source_file, count in grouped_sizes.items():
        seq_count = _num_sequences_for_rows(int(count), sequence_length, stride)
        sequences_per_file[str(source_file)] = seq_count
        total_sequences += seq_count

    num_features = len(feature_columns)
    x_seq = np.empty((total_sequences, sequence_length, num_features), dtype=np.float32)
    y_seq = np.empty((total_sequences,), dtype=np.int64)
    return x_seq, y_seq, sequences_per_file


def build_temporal_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    stride: int,
    label_strategy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build group-aware temporal sequences.

    Sequences are created strictly within each `source_file` after sorting by `iat`.
    """
    feature_columns = [
        col for col in df.columns if col not in {TARGET_COLUMN, SOURCE_COLUMN}
    ]
    if not feature_columns:
        raise ValueError("No feature columns found after removing metadata columns.")

    x_seq, y_seq, sequences_per_file = _preallocate_sequence_arrays(
        df=df,
        sequence_length=sequence_length,
        stride=stride,
        feature_columns=feature_columns,
    )

    write_idx = 0
    for source_file, group_df in df.groupby(SOURCE_COLUMN, sort=False):
        group_df = group_df.sort_values(ORDER_COLUMN, ascending=True, kind="mergesort")
        x_group = group_df[feature_columns].to_numpy(dtype=np.float32, copy=False)
        y_group = group_df[TARGET_COLUMN].to_numpy(dtype=np.int64, copy=False)

        seq_count = sequences_per_file[str(source_file)]
        if seq_count == 0:
            continue

        for start in range(0, x_group.shape[0] - sequence_length + 1, stride):
            end = start + sequence_length
            x_seq[write_idx] = x_group[start:end]
            y_seq[write_idx] = _derive_sequence_label(y_group[start:end], label_strategy)
            write_idx += 1

    return x_seq, y_seq, sequences_per_file


def create_and_save_sequences(
    base_path: str,
    sequence_length: int = 20,
    stride: int = 5,
    label_strategy: str = "last_label",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, int]]:
    """Load processed data, build temporal sequences, and save `.npy` artifacts.

    Returns:
        x_seq: Sequence features array.
        y_seq: Sequence label array.
        original_shape: Original tabular shape as (rows, cols).
        sequences_per_file: Number of sequences created per source file.
    """
    project_root = Path(base_path).resolve()
    input_path = project_root / INPUT_RELATIVE_PATH
    output_x_path = project_root / OUTPUT_X_RELATIVE_PATH
    output_y_path = project_root / OUTPUT_Y_RELATIVE_PATH

    df = load_processed_tabular_dataset(input_path)
    original_shape = df.shape
    _validate_inputs(
        df=df,
        sequence_length=sequence_length,
        stride=stride,
        label_strategy=label_strategy,
    )
    x_seq, y_seq, sequences_per_file = build_temporal_sequences(
        df=df,
        sequence_length=sequence_length,
        stride=stride,
        label_strategy=label_strategy,
    )

    output_x_path.parent.mkdir(parents=True, exist_ok=True)
    output_y_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_x_path, x_seq)
    np.save(output_y_path, y_seq)

    return x_seq, y_seq, original_shape, sequences_per_file


def _print_label_distribution(y_seq: np.ndarray) -> None:
    """Print sequence-level label distribution."""
    labels, counts = np.unique(y_seq, return_counts=True)
    print("Label distribution across sequences:")
    for label, count in zip(labels, counts):
        print(f"  label {int(label)}: {int(count)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build sliding-window sequences.")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--label-strategy",
        type=str,
        default="last_label",
        choices=sorted(SUPPORTED_LABEL_STRATEGIES),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    x_seq_out, y_seq_out, tabular_shape, sequences_per_file_out = create_and_save_sequences(
        base_path=str(root),
        sequence_length=args.sequence_length,
        stride=args.stride,
        label_strategy=args.label_strategy,
    )

    print(f"Original tabular shape: {tabular_shape}")
    print(f"Number of files processed: {len(sequences_per_file_out)}")
    print("Sequences per file:")
    for source_file, seq_count in sequences_per_file_out.items():
        print(f"  - {source_file}: {seq_count}")
    print(f"Sequence shape: {x_seq_out.shape}")
    _print_label_distribution(y_seq_out)
    print(f"Saved X sequences to: {root / OUTPUT_X_RELATIVE_PATH}")
    print(f"Saved y sequences to: {root / OUTPUT_Y_RELATIVE_PATH}")
