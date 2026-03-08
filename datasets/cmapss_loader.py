"""
NASA CMAPSS Turbofan Engine Degradation Dataset Loader.

Dataset source:
    https://ti.arc.nasa.gov/c/6/

Four subsets:
    FD001 — single fault mode, single operating condition   (train: 100 units)
    FD002 — single fault mode, 6 operating conditions       (train: 260 units)
    FD003 — two fault modes,   single operating condition   (train: 100 units)
    FD004 — two fault modes,   6 operating conditions       (train: 249 units)

Columns (26 total):
    unit_nr, time, op_setting_1, op_setting_2, op_setting_3,
    sensor_1 … sensor_21

Reference:
    Saxena, A., & Goebel, K. (2008). Turbofan engine degradation simulation
    data set. NASA Ames Prognostics Data Repository.
"""

from __future__ import annotations

import os
import io
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# URL of the publicly hosted CMAPSS zip on NASA's server
_CMAPSS_URL = (
    "https://ti.arc.nasa.gov/c/6/"
    # Falls back to Kaggle mirror if the NASA link requires auth
)
_FALLBACK_URL = (
    "https://raw.githubusercontent.com/IEEE-IES-Industrial-AI-Lab/"
    "Industrial-Predictive-Maintenance/main/datasets/data/CMAPSSData.zip"
)

_COLUMN_NAMES = (
    ["unit_nr", "time", "op_1", "op_2", "op_3"]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

# Sensors with near-zero variance across the dataset (excluded by convention)
_DROPPED_SENSORS = ["sensor_01", "sensor_05", "sensor_06", "sensor_10",
                    "sensor_16", "sensor_18", "sensor_19"]

_FEATURE_COLS = [c for c in _COLUMN_NAMES
                 if c not in ["unit_nr", "time"] + _DROPPED_SENSORS]


class CMAPSSLoader:
    """
    Load and preprocess the NASA CMAPSS turbofan engine dataset.

    Parameters
    ----------
    subset : str
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.
    data_dir : str or Path
        Directory where raw data files are stored (or will be downloaded to).
    max_rul : int
        Piece-wise linear RUL cap (cycles). Labels are clipped at this value.
        Typical value: 125.
    window_size : int
        Number of time steps per input sequence.
    stride : int
        Sliding-window stride when building training sequences.
    normalize : bool
        Apply min-max normalization per feature (fit on training split).

    Examples
    --------
    >>> loader = CMAPSSLoader(subset="FD001", max_rul=125, window_size=30)
    >>> X_train, y_train, X_test, y_test = loader.load()
    >>> X_train.shape
    (17631, 30, 14)
    """

    def __init__(
        self,
        subset: str = "FD001",
        data_dir: str | Path = "datasets/data",
        max_rul: int = 125,
        window_size: int = 30,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        valid_subsets = {"FD001", "FD002", "FD003", "FD004"}
        if subset not in valid_subsets:
            raise ValueError(f"subset must be one of {valid_subsets}")
        self.subset = subset
        self.data_dir = Path(data_dir)
        self.max_rul = max_rul
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.scaler: MinMaxScaler | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(X_train, y_train, X_test, y_test)`` as numpy arrays.

        Shapes
        ------
        X_train : (N_train, window_size, n_features)
        y_train : (N_train,)  — RUL labels (capped at max_rul)
        X_test  : (N_units, window_size, n_features)  — last window per unit
        y_test  : (N_units,)  — ground-truth RUL from RUL_FDxxx.txt
        """
        self._ensure_data()
        train_df = self._read_split("train")
        test_df = self._read_split("test")
        rul_df = self._read_rul()

        train_df = self._add_rul_labels(train_df)

        if self.normalize:
            self.scaler = MinMaxScaler()
            train_df[_FEATURE_COLS] = self.scaler.fit_transform(
                train_df[_FEATURE_COLS]
            )
            test_df[_FEATURE_COLS] = self.scaler.transform(
                test_df[_FEATURE_COLS]
            )

        X_train, y_train = self._build_train_sequences(train_df)
        X_test, y_test = self._build_test_sequences(test_df, rul_df)

        return X_train, y_train, X_test, y_test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_data(self) -> None:
        """Download CMAPSSData.zip if files are not already present."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        train_file = self.data_dir / f"train_{self.subset}.txt"
        if train_file.exists():
            return
        zip_path = self.data_dir / "CMAPSSData.zip"
        if not zip_path.exists():
            print(f"Downloading CMAPSS data to {self.data_dir} …")
            try:
                urllib.request.urlretrieve(_CMAPSS_URL, zip_path)
            except Exception:
                print(
                    "Automatic download failed.\n"
                    "Please download CMAPSSData.zip manually from:\n"
                    "  https://ti.arc.nasa.gov/c/6/\n"
                    f"and place it in:  {self.data_dir}"
                )
                return
        print(f"Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

    def _read_split(self, split: str) -> pd.DataFrame:
        path = self.data_dir / f"{split}_{self.subset}.txt"
        df = pd.read_csv(path, sep=r"\s+", header=None, names=_COLUMN_NAMES)
        df.drop(columns=_DROPPED_SENSORS, inplace=True)
        return df

    def _read_rul(self) -> pd.Series:
        path = self.data_dir / f"RUL_{self.subset}.txt"
        return pd.read_csv(path, header=None, names=["rul"])["rul"]

    def _add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute piecewise-linear (capped) RUL labels for each cycle."""
        max_cycles = df.groupby("unit_nr")["time"].max()
        df = df.join(max_cycles.rename("max_time"), on="unit_nr")
        df["rul"] = df["max_time"] - df["time"]
        df["rul"] = df["rul"].clip(upper=self.max_rul).astype(np.float32)
        df.drop(columns=["max_time"], inplace=True)
        return df

    def _build_train_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for _, unit_df in df.groupby("unit_nr"):
            values = unit_df[_FEATURE_COLS].values.astype(np.float32)
            rul = unit_df["rul"].values.astype(np.float32)
            n = len(values)
            for start in range(0, n - self.window_size + 1, self.stride):
                end = start + self.window_size
                X_list.append(values[start:end])
                y_list.append(rul[end - 1])
        return np.stack(X_list), np.array(y_list, dtype=np.float32)

    def _build_test_sequences(
        self, df: pd.DataFrame, rul_series: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Take the last `window_size` timesteps for each test unit."""
        X_list = []
        for _, unit_df in df.groupby("unit_nr"):
            values = unit_df[_FEATURE_COLS].values.astype(np.float32)
            if len(values) < self.window_size:
                pad = np.zeros(
                    (self.window_size - len(values), values.shape[1]),
                    dtype=np.float32,
                )
                values = np.vstack([pad, values])
            X_list.append(values[-self.window_size:])
        X_test = np.stack(X_list)
        y_test = rul_series.values.clip(max=self.max_rul).astype(np.float32)
        return X_test, y_test

    @property
    def feature_names(self) -> list[str]:
        return _FEATURE_COLS

    @property
    def n_features(self) -> int:
        return len(_FEATURE_COLS)
