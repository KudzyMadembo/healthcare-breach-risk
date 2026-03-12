"""Create a thesis-ready comparison summary across trained models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASELINE_METRICS_PATH = Path("reports/baseline_metrics.json")
LSTM_METRICS_PATH = Path("reports/lstm_metrics.json")
ENSEMBLE_METRICS_PATH = Path("reports/ensemble_metrics.json")

COMPARISON_CSV_PATH = Path("reports/model_comparison.csv")
COMPARISON_JSON_PATH = Path("reports/model_comparison.json")
F1_CHART_PATH = Path("reports/model_comparison_f1.png")

METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def load_json(path: Path) -> dict:
    """Load a JSON file into a Python dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_baseline_metrics(payload: dict) -> list[dict]:
    """Parse baseline metrics JSON into comparable row records."""
    rows: list[dict] = []
    models_payload = payload.get("models", {})

    for model_name, details in models_payload.items():
        metrics = details.get("metrics", {})
        row = {"model_name": model_name}
        for metric in METRIC_COLUMNS:
            row[metric] = float(metrics.get(metric, 0.0))
        rows.append(row)

    return rows


def parse_single_model_metrics(payload: dict, model_name: str) -> dict:
    """Parse single-model payloads (LSTM, ensemble) into one row."""
    metrics = payload.get("metrics", {})
    row = {"model_name": model_name}
    for metric in METRIC_COLUMNS:
        row[metric] = float(metrics.get(metric, 0.0))
    return row


def build_comparison_table(base_path: str) -> pd.DataFrame:
    """Build a model comparison DataFrame from all reports."""
    root = Path(base_path).resolve()
    baseline_payload = load_json(root / BASELINE_METRICS_PATH)
    lstm_payload = load_json(root / LSTM_METRICS_PATH)
    ensemble_payload = load_json(root / ENSEMBLE_METRICS_PATH)

    rows = []
    rows.extend(parse_baseline_metrics(baseline_payload))
    rows.append(parse_single_model_metrics(lstm_payload, "lstm"))
    rows.append(parse_single_model_metrics(ensemble_payload, "ensemble_lstm_xgboost"))

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
    return comparison_df


def save_comparison_outputs(comparison_df: pd.DataFrame, base_path: str) -> None:
    """Save comparison outputs to CSV and JSON files."""
    root = Path(base_path).resolve()
    csv_path = root / COMPARISON_CSV_PATH
    json_path = root / COMPARISON_JSON_PATH

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(comparison_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def plot_macro_f1_chart(comparison_df: pd.DataFrame, base_path: str) -> None:
    """Create and save a bar chart comparing macro-F1 scores."""
    root = Path(base_path).resolve()
    output_path = root / F1_CHART_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(comparison_df["model_name"], comparison_df["macro_f1"], color="steelblue")
    plt.title("Macro-F1 Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_best_model_summary(comparison_df: pd.DataFrame) -> None:
    """Print short summary identifying best-performing model."""
    best_row = comparison_df.iloc[0]
    print(
        "Best-performing model: "
        f"{best_row['model_name']} "
        f"(macro_f1={best_row['macro_f1']:.4f}, "
        f"accuracy={best_row['accuracy']:.4f})"
    )


def run_summary(base_path: str) -> pd.DataFrame:
    """Run full comparison summary workflow."""
    comparison_df = build_comparison_table(base_path)
    save_comparison_outputs(comparison_df, base_path)
    plot_macro_f1_chart(comparison_df, base_path)
    return comparison_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    comparison = run_summary(str(project_root))
    print(comparison.to_string(index=False))
    print_best_model_summary(comparison)
    print(f"Saved CSV: {project_root / COMPARISON_CSV_PATH}")
    print(f"Saved JSON: {project_root / COMPARISON_JSON_PATH}")
    print(f"Saved chart: {project_root / F1_CHART_PATH}")
