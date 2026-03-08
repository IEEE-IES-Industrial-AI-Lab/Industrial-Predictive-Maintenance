"""
Encoder-only Transformer for Remaining Useful Life (RUL) prediction.

Architecture:
    Input  →  Linear projection  →  Sinusoidal positional encoding
           →  N × TransformerEncoderLayer  →  Global average pooling
           →  FC regression head  →  RUL scalar

References:
    Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.

    Mo, Y. et al. (2021). Multi-head attention-based long short-term memory
    for depression detection from speech. Frontiers in Neurorobotics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_model import BaseModel


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding as in Vaswani et al. (2017).

    Parameters
    ----------
    d_model : int
        Model embedding dimension.
    max_len : int
        Maximum sequence length to pre-compute encodings for.
    dropout : float
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)              # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _TransformerNetwork(nn.Module):
    """Inner PyTorch module for the encoder-only Transformer."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads + 1) * n_heads

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, F)

        Returns
        -------
        out : Tensor, shape (B, 1)
        """
        x = self.input_projection(x)          # (B, T, d_model)
        x = self.pos_encoding(x)
        x = self.encoder(x)                   # (B, T, d_model)
        x = x.mean(dim=1)                     # global average pooling → (B, d_model)
        return self.fc(x)


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "input_size": 14,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "max_seq_len": 512,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "epochs": 50,
    "patience": 10,
    "device": "auto",
}


class TransformerRULModel(BaseModel):
    """
    Encoder-only Transformer for RUL regression.

    Parameters
    ----------
    config : dict
        Model and training hyperparameters. Missing keys fall back to defaults.

    Config keys
    -----------
    input_size     : int   — number of sensor features per time step
    d_model        : int   — model embedding dimension  (default: 128)
    n_heads        : int   — attention heads            (default: 8)
    n_layers       : int   — encoder layers             (default: 3)
    dim_feedforward: int   — FFN inner dimension        (default: 256)
    dropout        : float                              (default: 0.1)
    lr             : float — Adam learning rate         (default: 5e-4)
    batch_size     : int                                (default: 256)
    epochs         : int                                (default: 50)
    patience       : int   — early stopping patience   (default: 10)

    Examples
    --------
    >>> model = TransformerRULModel.from_config("configs/transformer_config.yaml")
    >>> history = model.fit(X_train, y_train, X_val, y_val)
    >>> preds = model.predict(X_test)
    """

    def __init__(self, config: dict) -> None:
        merged = {**_DEFAULT_CONFIG, **config}
        if merged["device"] == "auto":
            merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(merged)

    def _build_network(self) -> None:
        self.network = _TransformerNetwork(
            input_size=self.config["input_size"],
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            n_layers=self.config["n_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            max_seq_len=self.config["max_seq_len"],
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

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=self.config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
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

        for epoch in tqdm(range(1, epochs + 1), desc="Training Transformer"):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)
            scheduler.step()

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader, criterion)
                history["val_loss"].append(val_loss)

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
