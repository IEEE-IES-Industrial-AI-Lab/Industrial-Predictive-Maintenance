"""
LSTM Autoencoder for unsupervised anomaly detection.

Architecture:
    Encoder: LSTM(input_size → latent_size)
    Decoder: LSTM(latent_size → input_size)  — sequence reconstructed step by step

Training:
    Fit on healthy (normal) data only.
    Reconstruction error (MSE per sample) is the anomaly score.
    Threshold calibrated at a chosen percentile of training reconstruction errors.

References:
    Malhotra, P. et al. (2016). LSTM-based encoder-decoder for multi-sensor
    anomaly detection. ICML 2016 Anomaly Detection Workshop.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_model import BaseModel


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class _LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x : (B, T, F)  →  last hidden/cell state"""
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class _LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(
        self,
        h_n: torch.Tensor,
        c_n: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent state back to sequence.

        Returns
        -------
        reconstruction : Tensor, shape (B, T, F)
        """
        B = h_n.size(1)
        # Use last hidden state as repeated input token
        dec_input = h_n[-1].unsqueeze(1).expand(B, self.seq_len, -1)  # (B, T, H)
        out, _ = self.lstm(dec_input, (h_n, c_n))
        return self.output_proj(out)                                    # (B, T, F)


class _AENetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.encoder = _LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = _LSTMDecoder(input_size, hidden_size, num_layers, dropout, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_n, c_n = self.encoder(x)
        return self.decoder(h_n, c_n)


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "input_size": 1,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "seq_len": 50,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 128,
    "epochs": 50,
    "patience": 10,
    "threshold_percentile": 95,
    "device": "auto",
}


class LSTMAutoencoder(BaseModel):
    """
    LSTM Autoencoder trained on normal data for anomaly detection.

    Parameters
    ----------
    config : dict
        Model and training hyperparameters.

    Config keys
    -----------
    input_size            : int   — number of features (1 for univariate)
    hidden_size           : int   — LSTM hidden dimension     (default: 64)
    num_layers            : int                               (default: 2)
    seq_len               : int   — input sequence length     (default: 50)
    dropout               : float                             (default: 0.1)
    lr                    : float                             (default: 1e-3)
    batch_size            : int                               (default: 128)
    epochs                : int                               (default: 50)
    patience              : int                               (default: 10)
    threshold_percentile  : int   — percentile for threshold  (default: 95)

    Examples
    --------
    >>> model = LSTMAutoencoder.from_config("configs/autoencoder_config.yaml")
    >>> model.fit(X_normal)                          # train on healthy data
    >>> scores = model.predict_anomaly_score(X_test) # reconstruction error
    >>> labels = model.classify(X_test)              # 0 = normal, 1 = anomaly
    """

    def __init__(self, config: dict) -> None:
        merged = {**_DEFAULT_CONFIG, **config}
        if merged["device"] == "auto":
            merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold: Optional[float] = None
        super().__init__(merged)

    def _build_network(self) -> None:
        self.network = _AENetwork(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            seq_len=self.config["seq_len"],
        ).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,   # unused — only for interface compat
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        calibrate_threshold: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train autoencoder on normal (healthy) data.

        Parameters
        ----------
        X_train : ndarray, shape (N, T, F) or (N, T) for univariate
        calibrate_threshold : bool
            After training, calibrate the anomaly threshold from training
            reconstruction errors at ``threshold_percentile``.

        Returns
        -------
        history : dict
        """
        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]
        if X_val is not None and X_val.ndim == 2:
            X_val = X_val[:, :, np.newaxis]

        epochs = epochs or self.config["epochs"]
        batch_size = batch_size or self.config["batch_size"]
        lr = lr or self.config["lr"]

        # Dummy y for dataloader (reconstruction target = input)
        y_dummy = np.zeros(len(X_train), dtype=np.float32)
        train_loader = self._make_dataloader(X_train, y_dummy, batch_size)
        val_loader = None
        if X_val is not None:
            y_val_dummy = np.zeros(len(X_val), dtype=np.float32)
            val_loader = self._make_dataloader(X_val, y_val_dummy, batch_size, shuffle=False)

        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, weight_decay=self.config["weight_decay"]
        )
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(1, epochs + 1), desc="Training Autoencoder"):
            train_loss = self._ae_train_epoch(train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._ae_eval_epoch(val_loader, criterion)
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

        if calibrate_threshold:
            scores = self.predict_anomaly_score(X_train)
            self.threshold = float(
                np.percentile(scores, self.config["threshold_percentile"])
            )
            print(f"Anomaly threshold set to {self.threshold:.6f} "
                  f"(p{self.config['threshold_percentile']} of train scores)")

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed sequences. Shape same as input."""
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        self.network.eval()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._to_tensor(X)),
            batch_size=self.config["batch_size"],
        )
        recons = []
        for (x_batch,) in loader:
            recon = self.network(x_batch)
            recons.append(recon.cpu().numpy())
        return np.concatenate(recons, axis=0)

    @torch.no_grad()
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample mean-squared reconstruction error.

        Parameters
        ----------
        X : ndarray, shape (N, T) or (N, T, F)

        Returns
        -------
        ndarray, shape (N,)  — anomaly scores (higher = more anomalous)
        """
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        recon = self.predict(X)
        return np.mean((X - recon) ** 2, axis=(1, 2)).astype(np.float32)

    def classify(self, X: np.ndarray) -> np.ndarray:
        """
        Classify samples as normal (0) or anomaly (1) using calibrated threshold.

        Raises
        ------
        RuntimeError
            If threshold has not been calibrated (call ``fit()`` first).
        """
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not calibrated. Call fit() with calibrate_threshold=True."
            )
        scores = self.predict_anomaly_score(X)
        return (scores > self.threshold).astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance.

        Parameters
        ----------
        X : ndarray, shape (N, T) or (N, T, F)
        y : ndarray, shape (N,) — binary labels (0 = normal, 1 = fault)

        Returns
        -------
        dict with keys: ``"roc_auc"``, ``"f1"``, ``"precision"``, ``"recall"``
        """
        from evaluation.fault_metrics import classification_metrics
        labels = self.classify(X)
        scores = self.predict_anomaly_score(X)
        return classification_metrics(y, labels, scores)

    def set_threshold(self, threshold: float) -> None:
        """Manually override the anomaly detection threshold."""
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _ae_train_epoch(self, loader, optimizer, criterion) -> float:
        self.network.train()
        total_loss = 0.0
        for X_batch, _ in loader:
            optimizer.zero_grad()
            recon = self.network(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _ae_eval_epoch(self, loader, criterion) -> float:
        self.network.eval()
        total_loss = 0.0
        for X_batch, _ in loader:
            recon = self.network(X_batch)
            loss = criterion(recon, X_batch)
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)
