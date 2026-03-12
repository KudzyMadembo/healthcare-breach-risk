"""Baseline model training utilities for CICIoMT2024 multiclass classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_processed_dataset(dataset_path: Path, target_column: str = "label_encoded") -> tuple[pd.DataFrame, pd.Series]:
    """Load processed dataset and split into features and encoded target."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in {dataset_path}. "
            "Ensure preprocessing saves the encoded label column."
        )

    y = df[target_column].astype(int)
    x = df.drop(columns=[target_column]).copy()

    # Keep memory footprint lower for large tabular datasets.
    x = x.astype("float32")
    return x, y


def load_label_mapping(mapping_path: Path) -> dict[int, str]:
    """Load label mapping JSON and return {encoded_id: class_name}."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")

    mapping_raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(encoded): label for label, encoded in mapping_raw.items()}


def split_train_validation(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data with stratification for multiclass balance."""
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_baseline_models(num_classes: int, random_state: int = 42) -> dict[str, Any]:
    """Create baseline multiclass models with practical defaults."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=400,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softmax",
            num_class=num_classes,
            n_estimators=250,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    return models


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Compute required evaluation metrics for multiclass predictions."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_model(model: Any, output_path: Path) -> None:
    """Persist a trained model to disk using joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def save_confusion_matrix_plot(
    confusion: list[list[int]],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Save confusion matrix heatmap for a model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_and_evaluate_baselines(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    id_to_label: dict[int, str],
    models_dir: Path,
    reports_dir: Path,
    random_state: int = 42,
) -> dict[str, dict[str, Any]]:
    """Train all baseline models, save artifacts, and return metrics summary."""
    models = build_baseline_models(num_classes=len(id_to_label), random_state=random_state)
    class_names = [id_to_label[idx] for idx in sorted(id_to_label)]

    summary: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = pd.Series(model.predict(x_valid), index=y_valid.index)

        metrics = evaluate_predictions(y_valid, y_pred)

        model_path = models_dir / f"{model_name}_model.joblib"
        cm_plot_path = reports_dir / f"{model_name}_confusion_matrix.png"
        save_model(model, model_path)
        save_confusion_matrix_plot(
            confusion=metrics["confusion_matrix"],
            class_names=class_names,
            output_path=cm_plot_path,
            title=f"{model_name.replace('_', ' ').title()} Confusion Matrix",
        )

        summary[model_name] = {
            "metrics": metrics,
            "model_path": str(model_path).replace("\\", "/"),
            "confusion_matrix_plot": str(cm_plot_path).replace("\\", "/"),
        }

    return summary
