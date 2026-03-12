"""Train and evaluate baseline multiclass models on processed CICIoMT2024 data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_models import (  # noqa: E402
    load_label_mapping,
    load_processed_dataset,
    split_train_validation,
    train_and_evaluate_baselines,
)
from sklearn.model_selection import train_test_split  # noqa: E402

DATASET_RELATIVE_PATH = Path("data/processed/train_processed.csv")
LABEL_MAPPING_RELATIVE_PATH = Path("models/label_mapping.json")
METRICS_RELATIVE_PATH = Path("reports/baseline_metrics.json")
MODELS_DIR_RELATIVE_PATH = Path("models")
REPORTS_DIR_RELATIVE_PATH = Path("reports")
TARGET_COLUMN = "label_encoded"
MAX_ROWS_FOR_BASELINE = 200_000


def train_baseline_models(base_path: str) -> dict[str, Any]:
    """Run full baseline training pipeline and save all artifacts."""
    root = Path(base_path).resolve()
    dataset_path = root / DATASET_RELATIVE_PATH
    label_mapping_path = root / LABEL_MAPPING_RELATIVE_PATH
    metrics_path = root / METRICS_RELATIVE_PATH
    models_dir = root / MODELS_DIR_RELATIVE_PATH
    reports_dir = root / REPORTS_DIR_RELATIVE_PATH

    x, y = load_processed_dataset(dataset_path, target_column=TARGET_COLUMN)
    id_to_label = load_label_mapping(label_mapping_path)

    # Cap row count for safer baseline experiments on very large datasets.
    if len(x) > MAX_ROWS_FOR_BASELINE:
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=MAX_ROWS_FOR_BASELINE,
            random_state=42,
            stratify=y,
        )

    x_train, x_valid, y_train, y_valid = split_train_validation(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    per_model_summary = train_and_evaluate_baselines(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        id_to_label=id_to_label,
        models_dir=models_dir,
        reports_dir=reports_dir,
        random_state=42,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "dataset_path": str(dataset_path).replace("\\", "/"),
        "target_column": TARGET_COLUMN,
        "max_rows_for_baseline": MAX_ROWS_FOR_BASELINE,
        "rows_used": int(len(x)),
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "validation_shape": [int(x_valid.shape[0]), int(x_valid.shape[1])],
        "models": per_model_summary,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return metrics_payload


if __name__ == "__main__":
    results = train_baseline_models(str(PROJECT_ROOT))

    print("Baseline training complete.")
    print(f"Train shape: {tuple(results['train_shape'])}")
    print(f"Validation shape: {tuple(results['validation_shape'])}")
    print("Model metrics (accuracy / macro_f1):")
    for model_name, details in results["models"].items():
        metrics = details["metrics"]
        print(
            f"  - {model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    print(f"Saved metrics: {PROJECT_ROOT / METRICS_RELATIVE_PATH}")
