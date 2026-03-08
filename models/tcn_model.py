"""
Temporal Convolutional Network (TCN) for Remaining Useful Life prediction.

Architecture:
    Input  →  Stack of TCN blocks (dilated causal convolutions + residuals)
           →  Global average pooling  →  FC regression head  →  RUL scalar

TCN block:
    x  →  Conv1d (dilated, causal)  →  WeightNorm  →  ReLU  →  Dropout
       →  Conv1d (dilated, causal)  →  WeightNorm  →  ReLU  →  Dropout
       →  (+residual connection via 1×1 conv if channels differ)  →  ReLU

The dilation doubles every block: 1, 2, 4, 8, …
Receptive field = 2 × kernel_size × (2^n_blocks - 1) + 1.

References:
    Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation
    of generic convolutional and recurrent networks for sequence modeling.
    arXiv:1803.01271.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from tqdm import tqdm

from .base_model import BaseModel


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    """Causal (left-padded) dilated 1-D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, T)  →  (B, C_out, T)"""
        out = self.conv(x)
        return out[:, :, : x.size(2)]   # trim right-side padding


class _TCNBlock(nn.Module):
    """One residual TCN block with two dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.dropout(self.relu(self.conv1(x)))
        out = self.dropout(self.relu(self.conv2(out)))
        return self.relu(out + residual)


class _TCNNetwork(nn.Module):
    """Full TCN: stacked blocks + global average pooling + FC head."""

    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                _TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
            in_channels = out_channels
        self.tcn_blocks = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, F)  — batch-first input

        Returns
        -------
        out : Tensor, shape (B, 1)
        """
        x = x.permute(0, 2, 1)        # → (B, F, T)
        x = self.tcn_blocks(x)         # → (B, C_last, T)
        x = x.mean(dim=-1)             # global avg pooling → (B, C_last)
        return self.fc(x)


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "input_size": 14,
    "num_channels": [64, 64, 128, 128],
    "kernel_size": 3,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "epochs": 50,
    "patience": 10,
    "device": "auto",
}


class TCNModel(BaseModel):
    """
    Temporal Convolutional Network for RUL regression.

    Parameters
    ----------
    config : dict
        Model and training hyperparameters.

    Config keys
    -----------
    input_size   : int         — number of sensor features
    num_channels : list[int]   — output channels per TCN block  (default: [64,64,128,128])
    kernel_size  : int         — convolution kernel size        (default: 3)
    dropout      : float                                        (default: 0.2)
    lr           : float                                        (default: 1e-3)
    batch_size   : int                                          (default: 256)
    epochs       : int                                          (default: 50)
    patience     : int                                          (default: 10)

    Receptive field (default config):
        kernel_size=3, 4 blocks → RF = 2 × 3 × (16 - 1) + 1 = 91 steps.

    Examples
    --------
    >>> model = TCNModel.from_config("configs/tcn_config.yaml")
    >>> history = model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    def __init__(self, config: dict) -> None:
        merged = {**_DEFAULT_CONFIG, **config}
        if merged["device"] == "auto":
            merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(merged)

    def _build_network(self) -> None:
        self.network = _TCNNetwork(
            input_size=self.config["input_size"],
            num_channels=self.config["num_channels"],
            kernel_size=self.config["kernel_size"],
            dropout=self.config["dropout"],
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        epochs = epochs or self.config["epochs"]
        batch_size = batch_size or self.config["batch_size"]
        lr = lr or self.config["lr"]

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=self.config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        train_loader = self._make_dataloader(X_train, y_train, batch_size)
        val_loader = (
            self._make_dataloader(X_val, y_val, batch_size, shuffle=False)
            if X_val is not None
            else None
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(1, epochs + 1), desc="Training TCN"):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience_counter = 0
                    self._best_state = {
                        k: v.clone() for k, v in self.network.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["patience"]:
                        print(f"Early stopping at epoch {epoch}")
                        break

        if val_loader is not None and hasattr(self, "_best_state"):
            self.network.load_state_dict(self._best_state)

        return history

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(X)),
            batch_size=self.config["batch_size"],
        )
        preds = []
        for (x_batch,) in loader:
            out = self.network(x_batch)
            preds.append(out.squeeze(-1).cpu().numpy())
        return np.concatenate(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        from evaluation.rul_metrics import rmse, mae, nasa_score
        preds = self.predict(X)
        return {
            "rmse": float(rmse(y, preds)),
            "mae": float(mae(y, preds)),
            "nasa_score": float(nasa_score(y, preds)),
        }

    @property
    def receptive_field(self) -> int:
        """Compute the model's temporal receptive field in samples."""
        k = self.config["kernel_size"]
        n = len(self.config["num_channels"])
        return 2 * k * (2 ** n - 1) + 1
