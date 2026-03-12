"""PyTorch LSTM architecture for multiclass sequence classification."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Multiclass LSTM classifier.

    Architecture:
    - LSTM encoder
    - Last time-step representation
    - Linear classification head
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return class logits."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits
