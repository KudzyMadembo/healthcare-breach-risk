"""End-to-end inference utilities using ensemble probability outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mitigation import recommend_action  # noqa: E402
from src.risk_scoring import score_risk  # noqa: E402

LABEL_MAPPING_RELATIVE_PATH = Path("models/label_mapping.json")
ENSEMBLE_METRICS_RELATIVE_PATH = Path("reports/ensemble_metrics.json")


def load_class_names_from_mapping(mapping_path: Path) -> list[str]:
    """Load class names aligned by encoded class index."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")

    mapping_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    # mapping_raw shape: {class_name: encoded_id}
    id_to_name = {int(idx): name for name, idx in mapping_raw.items()}
    return [id_to_name[i] for i in sorted(id_to_name)]


def load_ensemble_reference(metrics_path: Path) -> dict[str, Any]:
    """Load ensemble metrics metadata for optional reference/use."""
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def infer_with_risk_and_mitigation(
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Produce final inference output from class probabilities.

    Args:
        probabilities: Class probabilities for one sample.
        class_names: Class names aligned with probability indices.

    Returns:
        Dictionary containing:
        - predicted_class
        - predicted_probability
        - risk_score
        - risk_level
        - recommended_action
        - priority
    """
    risk_output = score_risk(probabilities=probabilities, class_names=class_names)
    mitigation_output = recommend_action(
        predicted_class=risk_output["predicted_class"],
        risk_level=risk_output["risk_level"],
    )

    return {
        "predicted_class": risk_output["predicted_class"],
        "predicted_probability": risk_output["predicted_probability"],
        "risk_score": risk_output["risk_score"],
        "risk_level": risk_output["risk_level"],
        "recommended_action": mitigation_output["recommended_action"],
        "priority": mitigation_output["priority"],
    }


if __name__ == "__main__":
    mapping_path = PROJECT_ROOT / LABEL_MAPPING_RELATIVE_PATH
    metrics_path = PROJECT_ROOT / ENSEMBLE_METRICS_RELATIVE_PATH

    class_names_out = load_class_names_from_mapping(mapping_path)
    ensemble_meta = load_ensemble_reference(metrics_path)

    # Mocked example probabilities for testing the end-to-end output.
    mock_probabilities = np.full(len(class_names_out), 0.01, dtype=np.float64)
    target_class = "TCP_IP-DDoS-UDP"
    if target_class in class_names_out:
        target_index = class_names_out.index(target_class)
    else:
        target_index = int(np.argmax(mock_probabilities))
    mock_probabilities[target_index] = 0.92
    mock_probabilities /= mock_probabilities.sum()

    result = infer_with_risk_and_mitigation(
        probabilities=mock_probabilities,
        class_names=class_names_out,
    )

    print("Inference output:")
    print(result)
    if ensemble_meta:
        print("Loaded ensemble reference metadata from reports/ensemble_metrics.json")
