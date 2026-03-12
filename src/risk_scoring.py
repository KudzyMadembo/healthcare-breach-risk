"""Utilities for converting model probabilities to breach risk outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _validate_class_alignment(probabilities: np.ndarray, class_names: list[str]) -> None:
    """Validate probability/class-name alignment for one sample."""
    if probabilities.ndim != 1:
        raise ValueError("Expected a 1D probability array for a single sample.")
    if len(class_names) != probabilities.shape[0]:
        raise ValueError(
            "class_names length must match probability vector length. "
            f"Got class_names={len(class_names)}, probs={probabilities.shape[0]}."
        )


def _risk_level_from_probability(predicted_probability: float) -> str:
    """Map predicted class probability to qualitative risk level."""
    if predicted_probability < 0.40:
        return "Low"
    if predicted_probability < 0.70:
        return "Medium"
    if predicted_probability < 0.90:
        return "High"
    return "Critical"


def score_risk(probabilities: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Score breach risk for one sample from class probability outputs.

    Args:
        probabilities: 1D probability array for one sample across all classes.
        class_names: Class names aligned with probability indices.

    Returns:
        Dictionary containing:
        - predicted_class
        - predicted_probability
        - risk_score (0-100 scale)
        - risk_level
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    _validate_class_alignment(probs, class_names)

    pred_idx = int(np.argmax(probs))
    pred_prob = float(probs[pred_idx])
    risk_score = float(pred_prob * 100.0)

    return {
        "predicted_class": class_names[pred_idx],
        "predicted_probability": pred_prob,
        "risk_score": risk_score,
        "risk_level": _risk_level_from_probability(pred_prob),
    }


def score_risk_batch(prob_matrix: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    """Score breach risk for a batch of samples.

    Args:
        prob_matrix: 2D array with shape [num_samples, num_classes].
        class_names: Class names aligned with probability column indices.

    Returns:
        DataFrame with one row per sample containing risk outputs.
    """
    probs = np.asarray(prob_matrix, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("Expected a 2D probability matrix for batch scoring.")
    if probs.shape[1] != len(class_names):
        raise ValueError(
            "class_names length must match probability matrix width. "
            f"Got class_names={len(class_names)}, num_classes={probs.shape[1]}."
        )

    records = [score_risk(row, class_names) for row in probs]
    return pd.DataFrame(records)


if __name__ == "__main__":
    example_class_names = ["Benign", "Recon", "DoS", "DDoS"]
    example_prob_matrix = np.array(
        [
            [0.15, 0.20, 0.50, 0.15],
            [0.03, 0.04, 0.08, 0.85],
            [0.01, 0.02, 0.03, 0.94],
        ],
        dtype=np.float64,
    )

    print("Single-sample risk scoring:")
    single_result = score_risk(example_prob_matrix[0], example_class_names)
    print(single_result)

    print("\nBatch risk scoring:")
    batch_results = score_risk_batch(example_prob_matrix, example_class_names)
    print(batch_results.to_string(index=False))
