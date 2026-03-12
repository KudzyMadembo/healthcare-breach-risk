"""Probability-level ensemble utilities for LSTM + XGBoost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_model import LSTMClassifier


def load_label_mapping(mapping_path: Path) -> dict[int, str]:
    """Load label mapping JSON as {encoded_id: class_name}."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(encoded): label for label, encoded in raw.items()}


def load_sequence_arrays(x_seq_path: Path, y_seq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load sequence feature and label arrays."""
    if not x_seq_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {x_seq_path}")
    if not y_seq_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {y_seq_path}")
    return np.load(x_seq_path), np.load(y_seq_path)


def load_xgboost_model(model_path: Path) -> Any:
    """Load trained XGBoost model from joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    return joblib.load(model_path)


def load_lstm_model(
    model_path: Path,
    input_size: int,
    num_classes: int,
    device: torch.device,
) -> LSTMClassifier:
    """Load trained LSTM weights into model architecture."""
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found: {model_path}")

    model = LSTMClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_validation_indices(y_seq: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Recreate validation split indices with stratification."""
    indices = np.arange(len(y_seq))
    _, val_indices, _, _ = train_test_split(
        indices,
        y_seq,
        test_size=0.2,
        random_state=random_state,
        stratify=y_seq,
    )
    return np.sort(val_indices)


def predict_lstm_probabilities(
    model: LSTMClassifier,
    x_val_seq: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict class probabilities from LSTM model."""
    dataset = TensorDataset(torch.from_numpy(x_val_seq).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(batch_probs)
    return np.vstack(probs)


def predict_xgb_probabilities(xgb_model: Any, x_val_seq: np.ndarray) -> np.ndarray:
    """Predict class probabilities from XGBoost model.

    Uses the last timestep feature vector for each sequence to align with sequence labels.
    """
    x_val_last_step = x_val_seq[:, -1, :].astype(np.float32, copy=False)
    probs = xgb_model.predict_proba(x_val_last_step)
    return np.asarray(probs)


def weighted_soft_voting(
    lstm_probs: np.ndarray,
    xgb_probs: np.ndarray,
    w_lstm: float = 0.4,
    w_xgb: float = 0.6,
) -> np.ndarray:
    """Combine probabilities using weighted soft voting."""
    if lstm_probs.shape != xgb_probs.shape:
        raise ValueError(
            "Probability arrays must have the same shape. "
            f"Got LSTM={lstm_probs.shape}, XGB={xgb_probs.shape}"
        )
    return (w_lstm * lstm_probs) + (w_xgb * xgb_probs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute multiclass metrics for ensemble predictions."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_confusion_matrix_plot(
    confusion: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save confusion matrix heatmap plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("LSTM + XGBoost Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
