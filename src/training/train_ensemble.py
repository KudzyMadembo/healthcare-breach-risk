"""Evaluate probability-level ensemble of trained LSTM and XGBoost models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Ensure project root is importable for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble_model import (  # noqa: E402
    compute_metrics,
    get_validation_indices,
    load_label_mapping,
    load_lstm_model,
    load_sequence_arrays,
    load_xgboost_model,
    predict_lstm_probabilities,
    predict_xgb_probabilities,
    save_confusion_matrix_plot,
    weighted_soft_voting,
)

LSTM_MODEL_RELATIVE_PATH = Path("models/lstm_model.pt")
XGB_MODEL_RELATIVE_PATH = Path("models/xgboost_model.joblib")
X_SEQ_RELATIVE_PATH = Path("data/processed/X_seq.npy")
Y_SEQ_RELATIVE_PATH = Path("data/processed/y_seq.npy")
PROCESSED_CSV_RELATIVE_PATH = Path("data/processed/train_processed.csv")
LABEL_MAPPING_RELATIVE_PATH = Path("models/label_mapping.json")
METRICS_OUTPUT_RELATIVE_PATH = Path("reports/ensemble_metrics.json")
CM_OUTPUT_RELATIVE_PATH = Path("reports/ensemble_confusion_matrix.png")


def _load_processed_csv_for_validation(path: Path) -> pd.DataFrame:
    """Load processed CSV for shape/schema validation."""
    if not path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {path}")
    return pd.read_csv(path, nrows=5, low_memory=False)


def run_ensemble_evaluation(
    base_path: str,
    w_lstm: float = 0.4,
    w_xgb: float = 0.6,
    batch_size: int = 256,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run ensemble evaluation and save report artifacts."""
    root = Path(base_path).resolve()
    lstm_model_path = root / LSTM_MODEL_RELATIVE_PATH
    xgb_model_path = root / XGB_MODEL_RELATIVE_PATH
    x_seq_path = root / X_SEQ_RELATIVE_PATH
    y_seq_path = root / Y_SEQ_RELATIVE_PATH
    processed_csv_path = root / PROCESSED_CSV_RELATIVE_PATH
    mapping_path = root / LABEL_MAPPING_RELATIVE_PATH
    metrics_output_path = root / METRICS_OUTPUT_RELATIVE_PATH
    cm_output_path = root / CM_OUTPUT_RELATIVE_PATH

    # Load processed CSV (required input) for schema validation.
    processed_preview = _load_processed_csv_for_validation(processed_csv_path)
    if "label_encoded" not in processed_preview.columns:
        raise ValueError("Processed CSV must contain 'label_encoded'.")

    x_seq, y_seq = load_sequence_arrays(x_seq_path, y_seq_path)
    id_to_label = load_label_mapping(mapping_path)
    val_idx = get_validation_indices(y_seq, random_state=random_state)

    x_val_seq = x_seq[val_idx]
    y_val = y_seq[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = load_lstm_model(
        model_path=lstm_model_path,
        input_size=int(x_seq.shape[2]),
        num_classes=len(id_to_label),
        device=device,
    )
    xgb_model = load_xgboost_model(xgb_model_path)

    lstm_probs = predict_lstm_probabilities(
        model=lstm_model,
        x_val_seq=x_val_seq,
        batch_size=batch_size,
        device=device,
    )
    xgb_probs = predict_xgb_probabilities(xgb_model=xgb_model, x_val_seq=x_val_seq)

    final_probs = weighted_soft_voting(
        lstm_probs=lstm_probs,
        xgb_probs=xgb_probs,
        w_lstm=w_lstm,
        w_xgb=w_xgb,
    )
    y_pred = np.argmax(final_probs, axis=1)
    metrics = compute_metrics(y_true=y_val, y_pred=y_pred)

    class_names = [id_to_label[idx] for idx in sorted(id_to_label)]
    save_confusion_matrix_plot(
        confusion=metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_output_path,
    )

    payload = {
        "weights": {"w_lstm": float(w_lstm), "w_xgb": float(w_xgb)},
        "validation_samples": int(len(y_val)),
        "sequence_shape_used": [int(x_val_seq.shape[0]), int(x_val_seq.shape[1]), int(x_val_seq.shape[2])],
        "metrics": metrics,
        "artifacts": {
            "lstm_model": str(lstm_model_path).replace("\\", "/"),
            "xgboost_model": str(xgb_model_path).replace("\\", "/"),
            "metrics_json": str(metrics_output_path).replace("\\", "/"),
            "confusion_matrix_plot": str(cm_output_path).replace("\\", "/"),
        },
    }

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM + XGBoost soft-voting ensemble.")
    parser.add_argument("--w-lstm", type=float, default=0.4)
    parser.add_argument("--w-xgb", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    results = run_ensemble_evaluation(
        base_path=str(PROJECT_ROOT),
        w_lstm=args.w_lstm,
        w_xgb=args.w_xgb,
        batch_size=args.batch_size,
        random_state=42,
    )
    m = results["metrics"]
    print("Ensemble evaluation complete.")
    print(
        f"accuracy={m['accuracy']:.4f}, "
        f"macro_precision={m['macro_precision']:.4f}, "
        f"macro_recall={m['macro_recall']:.4f}, "
        f"macro_f1={m['macro_f1']:.4f}"
    )
    print(f"Saved metrics: {PROJECT_ROOT / METRICS_OUTPUT_RELATIVE_PATH}")
    print(f"Saved confusion matrix: {PROJECT_ROOT / CM_OUTPUT_RELATIVE_PATH}")
