"""
LSTM with Bahdanau attention for Remaining Useful Life (RUL) prediction.

Architecture:
    Input  →  2-layer stacked LSTM  →  Bahdanau attention  →  FC head  →  RUL scalar

References:
    Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation
    by jointly learning to align and translate. ICLR 2015.

    Li, X. et al. (2018). Remaining useful life estimation in prognostics
    using deep convolution neural networks. Reliability Engineering &
    System Safety, 172, 1–11.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_model import BaseModel


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention over LSTM hidden states.

    Parameters
    ----------
    hidden_size : int
        Size of the LSTM hidden dimension.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_keys = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : Tensor, shape (B, H)   — last hidden state
        keys  : Tensor, shape (B, T, H) — all hidden states

        Returns
        -------
        context : Tensor, shape (B, H)
        weights : Tensor, shape (B, T)  — attention distribution
        """
        q = self.W_query(query).unsqueeze(1)        # (B, 1, H)
        k = self.W_keys(keys)                        # (B, T, H)
        energy = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, T)
        weights = torch.softmax(energy, dim=-1)      # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, H)
        return context, weights


class _LSTMNetwork(nn.Module):
    """Inner PyTorch module — 2-layer LSTM + Bahdanau attention + FC."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, F)

        Returns
        -------
        out    : Tensor, shape (B, 1)
        weights: Tensor, shape (B, T)
        """
        hidden_states, (h_n, _) = self.lstm(x)          # (B, T, H)
        query = h_n[-1]                                   # last layer's final hidden
        context, weights = self.attention(query, hidden_states)
        out = self.fc(context)
        return out, weights


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "input_size": 14,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "epochs": 50,
    "patience": 10,
    "device": "auto",
}


class LSTMPredictiveModel(BaseModel):
    """
    Stacked LSTM with Bahdanau attention for RUL regression.

    Parameters
    ----------
    config : dict
        Model and training hyperparameters. Missing keys fall back to defaults.

    Config keys
    -----------
    input_size   : int   — number of sensor features per time step
    hidden_size  : int   — LSTM hidden dimension          (default: 128)
    num_layers   : int   — stacked LSTM layers            (default: 2)
    dropout      : float — dropout probability            (default: 0.2)
    lr           : float — Adam learning rate             (default: 1e-3)
    weight_decay : float — L2 regularisation              (default: 1e-5)
    batch_size   : int   — mini-batch size                (default: 256)
    epochs       : int   — maximum training epochs        (default: 50)
    patience     : int   — early stopping patience        (default: 10)

    Examples
    --------
    >>> model = LSTMPredictiveModel.from_config("configs/lstm_config.yaml")
    >>> history = model.fit(X_train, y_train, X_val, y_val)
    >>> preds = model.predict(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(self, config: dict) -> None:
        merged = {**_DEFAULT_CONFIG, **config}
        if merged["device"] == "auto":
            merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(merged)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def _build_network(self) -> None:
        self.network = _LSTMNetwork(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
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
        """
        Train the model with optional early stopping.

        Parameters
        ----------
        X_train : ndarray, shape (N, T, F)
        y_train : ndarray, shape (N,)
        X_val, y_val : validation arrays (enables early stopping)
        epochs, batch_size, lr : override config values

        Returns
        -------
        history : {"train_loss": [...], "val_loss": [...]}
        """
        epochs = epochs or self.config["epochs"]
        batch_size = batch_size or self.config["batch_size"]
        lr = lr or self.config["lr"]

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=self.config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
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

        for epoch in tqdm(range(1, epochs + 1), desc="Training LSTM"):
            train_loss = self._train_epoch_attention(train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._eval_epoch_attention(val_loader, criterion)
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
        """Predict RUL values. Returns ndarray of shape (N,)."""
        self.network.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(X)),
            batch_size=self.config["batch_size"],
        )
        preds = []
        for (x_batch,) in loader:
            out, _ = self.network(x_batch)
            preds.append(out.squeeze(-1).cpu().numpy())
        return np.concatenate(preds)

    @torch.no_grad()
    def predict_with_attention(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict RUL and return attention weights for interpretability."""
        self.network.eval()
        x_t = self._to_tensor(X)
        out, weights = self.network(x_t)
        return out.squeeze(-1).cpu().numpy(), weights.cpu().numpy()

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute RMSE, MAE, and NASA asymmetric score.

        Returns
        -------
        dict with keys: ``"rmse"``, ``"mae"``, ``"nasa_score"``
        """
        from evaluation.rul_metrics import rmse, mae, nasa_score
        preds = self.predict(X)
        return {
            "rmse": float(rmse(y, preds)),
            "mae": float(mae(y, preds)),
            "nasa_score": float(nasa_score(y, preds)),
        }

    # ------------------------------------------------------------------
    # Internal training loops (overridden to handle attention output)
    # ------------------------------------------------------------------

    def _train_epoch_attention(self, loader, optimizer, criterion) -> float:
        self.network.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out, _ = self.network(X_batch)
            loss = criterion(out.squeeze(-1), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch_attention(self, loader, criterion) -> float:
        self.network.eval()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            out, _ = self.network(X_batch)
            loss = criterion(out.squeeze(-1), y_batch)
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)
