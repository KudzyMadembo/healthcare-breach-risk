"""Train and evaluate an LSTM model on CICIoMT2024 sequence data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

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
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is importable for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_model import LSTMClassifier  # noqa: E402

X_SEQ_RELATIVE_PATH = Path("data/processed/X_seq.npy")
Y_SEQ_RELATIVE_PATH = Path("data/processed/y_seq.npy")
LABEL_MAPPING_RELATIVE_PATH = Path("models/label_mapping.json")
MODEL_OUTPUT_PATH = Path("models/lstm_model.pt")
METRICS_OUTPUT_PATH = Path("reports/lstm_metrics.json")
CM_PLOT_OUTPUT_PATH = Path("reports/lstm_confusion_matrix.png")


def load_sequence_data(x_path: Path, y_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-built sequence arrays."""
    if not x_path.exists():
        raise FileNotFoundError(f"Sequence features file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Sequence labels file not found: {y_path}")

    x_seq = np.load(x_path)
    y_seq = np.load(y_path)
    return x_seq, y_seq


def load_label_mapping(mapping_path: Path) -> dict[int, str]:
    """Load label mapping from JSON as {encoded_id: label_name}."""
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")

    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    return {int(encoded): label for label, encoded in raw.items()}


def create_data_loaders(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    batch_size: int,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Create stratified train/validation PyTorch data loaders."""
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_seq,
        y_seq,
        test_size=0.2,
        random_state=random_state,
        stratify=y_seq,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long(),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid).float(),
        torch.from_numpy(y_valid).long(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, valid_loader, y_train, y_valid


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train model for one epoch and return average training loss."""
    model.train()
    running_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * features.size(0)

    return running_loss / len(loader.dataset)


def evaluate_loss_and_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model loss and return true/pred arrays for metrics."""
    model.eval()
    running_loss = 0.0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += float(loss.item()) * features.size(0)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.concatenate(y_true), np.concatenate(y_pred)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute multiclass metrics payload for reporting."""
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
    """Save confusion matrix as a heatmap image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("LSTM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_lstm_pipeline(
    base_path: str,
    batch_size: int = 256,
    epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full LSTM training, evaluation, and artifact export pipeline."""
    root = Path(base_path).resolve()
    x_path = root / X_SEQ_RELATIVE_PATH
    y_path = root / Y_SEQ_RELATIVE_PATH
    mapping_path = root / LABEL_MAPPING_RELATIVE_PATH
    model_output = root / MODEL_OUTPUT_PATH
    metrics_output = root / METRICS_OUTPUT_PATH
    cm_output = root / CM_PLOT_OUTPUT_PATH

    x_seq, y_seq = load_sequence_data(x_path, y_path)
    id_to_label = load_label_mapping(mapping_path)

    train_loader, valid_loader, y_train, y_valid = create_data_loaders(
        x_seq=x_seq,
        y_seq=y_seq,
        batch_size=batch_size,
        random_state=random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = int(x_seq.shape[2])
    num_classes = len(np.unique(y_seq))

    model = LSTMClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, _, _ = evaluate_loss_and_predictions(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        history.append(
            {"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)}
        )
        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state_dict is None:
        raise RuntimeError("Training ended without capturing a best model state.")

    model.load_state_dict(best_state_dict)
    val_loss, y_true, y_pred = evaluate_loss_and_predictions(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        device=device,
    )

    class_names = [id_to_label[idx] for idx in sorted(id_to_label)]
    metrics = compute_classification_metrics(y_true, y_pred)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output)
    save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, cm_output)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "input_shape": [int(x_seq.shape[0]), int(x_seq.shape[1]), int(x_seq.shape[2])],
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_valid)),
        "batch_size": int(batch_size),
        "epochs_requested": int(epochs),
        "early_stopping_patience": int(patience),
        "learning_rate": float(learning_rate),
        "best_validation_loss": float(best_val_loss),
        "final_validation_loss": float(val_loss),
        "training_history": history,
        "metrics": metrics,
        "model_path": str(model_output).replace("\\", "/"),
        "confusion_matrix_plot": str(cm_output).replace("\\", "/"),
    }
    metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM on CICIoMT2024 sequences.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    results = train_lstm_pipeline(
        base_path=str(PROJECT_ROOT),
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        random_state=42,
    )

    m = results["metrics"]
    print("LSTM training complete.")
    print(
        f"Validation metrics - "
        f"accuracy: {m['accuracy']:.4f}, "
        f"macro_precision: {m['macro_precision']:.4f}, "
        f"macro_recall: {m['macro_recall']:.4f}, "
        f"macro_f1: {m['macro_f1']:.4f}"
    )
    print(f"Saved model: {PROJECT_ROOT / MODEL_OUTPUT_PATH}")
    print(f"Saved metrics: {PROJECT_ROOT / METRICS_OUTPUT_PATH}")
    print(f"Saved confusion matrix: {PROJECT_ROOT / CM_PLOT_OUTPUT_PATH}")
