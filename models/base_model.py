"""
Abstract base class for all predictive maintenance models.

Defines a consistent interface:
    model.fit(X_train, y_train, ...)
    model.predict(X)
    model.evaluate(X, y)
    model.save(path)
    model.load(path)
    model.from_config(config_path)
"""

from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import yaml


class BaseModel(abc.ABC):
    """
    Abstract base class for all RUL / anomaly models.

    Subclasses must implement ``_build_network``, ``fit``, ``predict``,
    and ``evaluate``.  ``save`` / ``load`` / ``from_config`` are provided
    as concrete methods that work with the ``network`` attribute.

    Attributes
    ----------
    config : dict
        Hyperparameters (populated from YAML or kwargs).
    device : torch.device
        Inferred from CUDA availability or overridden via config.
    network : nn.Module or None
        The underlying PyTorch module; populated by ``_build_network``.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.network: Optional[nn.Module] = None
        self._build_network()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_network(self) -> None:
        """Instantiate ``self.network`` from ``self.config``."""

    @abc.abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, list]:
        """
        Train the model.

        Parameters
        ----------
        X_train : ndarray, shape (N, T, F) or (N, F)
        y_train : ndarray, shape (N,)
        X_val, y_val : optional validation arrays
        **kwargs : extra training arguments (epochs, batch_size, lr, …)

        Returns
        -------
        history : dict with keys ``"train_loss"`` and optionally ``"val_loss"``.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference.

        Parameters
        ----------
        X : ndarray

        Returns
        -------
        ndarray of predictions.
        """

    @abc.abstractmethod
    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute task-specific evaluation metrics.

        Returns
        -------
        dict of metric name → value.
        """

    # ------------------------------------------------------------------
    # Concrete save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model weights and config to ``path``.

        The file is a ``torch.save`` dict with keys ``"state_dict"``
        and ``"config"``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.network is None:
            raise RuntimeError("No network to save. Call fit() first.")
        torch.save(
            {"state_dict": self.network.state_dict(), "config": self.config},
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """
        Load a saved model.

        Parameters
        ----------
        path : str or Path
            Path to ``.pt`` file written by ``save()``.

        Returns
        -------
        Instantiated model with weights restored.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(checkpoint["config"])
        model.network.load_state_dict(checkpoint["state_dict"])
        model.network.to(model.device)
        model.network.eval()
        print(f"Model loaded from {path}")
        return model

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "BaseModel":
        """
        Instantiate a model from a YAML config file.

        Parameters
        ----------
        config_path : str or Path

        Returns
        -------
        Instantiated (untrained) model.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ------------------------------------------------------------------
    # Shared training utilities
    # ------------------------------------------------------------------

    def _to_tensor(
        self, array: np.ndarray, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.tensor(array, dtype=dtype, device=self.device)

    def _make_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(
            self._to_tensor(X), self._to_tensor(y)
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        self.network.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = self.network(X_batch)
            loss = criterion(preds.squeeze(-1), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> float:
        self.network.eval()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            preds = self.network(X_batch)
            loss = criterion(preds.squeeze(-1), y_batch)
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)
