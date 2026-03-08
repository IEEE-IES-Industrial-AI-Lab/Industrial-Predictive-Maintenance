"""
IMS (University of Cincinnati) Bearing Dataset Loader.

Dataset source:
    https://ti.arc.nasa.gov/c/3/

Description:
    Run-to-failure experiments on rolling element bearings.
    Three experiments (runs), each containing 4 bearing channels.
    Data sampled at 20 kHz; each file contains one second of data
    (20 480 data points).

    Run 1: Bearings 3 and 4 failed (outer race faults).
    Run 2: Bearing 1 failed (outer race fault).
    Run 3: Bearing 3 failed (outer race fault).

File naming convention:
    <timestamp>     (e.g. 2003.10.22.12.06.24)

Reference:
    J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services,
    "Bearing Data Set", IMS, University of Cincinnati.
    NASA Ames Prognostics Data Repository, 2007.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class IMSLoader:
    """
    Load and preprocess the IMS bearing run-to-failure dataset.

    The loader operates on pre-extracted data directory containing
    space-delimited text files from one experimental run.

    Parameters
    ----------
    data_dir : str or Path
        Path to the experiment directory (e.g. ``"datasets/data/IMS/1st_test"``).
    run : int
        Experiment run number (1, 2, or 3).
    bearing_channels : list of int
        Bearing channel indices to use (1–4). Defaults to all four.
    window_size : int
        Number of data points per analysis window (subset of 20 480-point file).
    normalize : bool
        Apply z-score normalization per channel.

    Examples
    --------
    >>> loader = IMSLoader(data_dir="datasets/data/IMS/1st_test", run=1)
    >>> X, health_index, timestamps = loader.load()
    >>> X.shape
    (984, 1024, 4)
    """

    _RUN_DIRS = {1: "1st_test", 2: "2nd_test", 3: "3rd_test"}
    _N_CHANNELS = {1: 4, 2: 4, 3: 4}
    _FAULT_BEARINGS = {1: [3, 4], 2: [1], 3: [3]}

    def __init__(
        self,
        data_dir: str | Path = "datasets/data/IMS",
        run: int = 1,
        bearing_channels: Optional[List[int]] = None,
        window_size: int = 1024,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.run = run
        self.bearing_channels = bearing_channels or [1, 2, 3, 4]
        self.window_size = window_size
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all files in the run directory.

        Returns
        -------
        X : ndarray, shape (n_files, window_size, n_channels)
            Raw vibration windows (first `window_size` samples per file).
        health_index : ndarray, shape (n_files,)
            Normalised RMS-based health index (0 = healthy, 1 = degraded).
        timestamps : list of str
            File name timestamps, ordered chronologically.
        """
        run_dir = self.data_dir / self._RUN_DIRS[self.run]
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Run directory not found: {run_dir}\n"
                "Download the IMS dataset from https://ti.arc.nasa.gov/c/3/ "
                f"and extract to {self.data_dir}"
            )

        file_paths = sorted(run_dir.glob("*"))
        file_paths = [p for p in file_paths if p.is_file() and not p.name.startswith(".")]

        X_list, rms_list, timestamps = [], [], []
        for fp in file_paths:
            try:
                df = pd.read_csv(fp, sep=r"\s+", header=None)
            except Exception:
                continue

            n_cols = df.shape[1]
            channel_indices = [c - 1 for c in self.bearing_channels if c - 1 < n_cols]
            data = df.iloc[:, channel_indices].values.astype(np.float32)

            window = data[: self.window_size]
            if len(window) < self.window_size:
                pad = np.zeros(
                    (self.window_size - len(window), window.shape[1]),
                    dtype=np.float32,
                )
                window = np.vstack([window, pad])

            rms = np.sqrt(np.mean(window ** 2, axis=0))
            rms_list.append(rms.mean())
            X_list.append(window)
            timestamps.append(fp.name)

        X = np.stack(X_list)

        if self.normalize:
            scaler = MinMaxScaler()
            orig_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_flat = scaler.fit_transform(X_flat)
            X = X_flat.reshape(orig_shape)

        rms_array = np.array(rms_list, dtype=np.float32)
        rms_min, rms_max = rms_array.min(), rms_array.max()
        health_index = (rms_array - rms_min) / (rms_max - rms_min + 1e-8)

        return X, health_index, timestamps

    def load_split(
        self,
        healthy_fraction: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into healthy and degraded segments by health index threshold.

        Parameters
        ----------
        healthy_fraction : float
            Fraction of earliest files to treat as "healthy" baseline.

        Returns
        -------
        X_healthy : ndarray
        X_degraded : ndarray
        """
        X, health_index, _ = self.load()
        n = len(X)
        split = int(n * healthy_fraction)
        return X[:split], X[split:]

    @property
    def fault_bearings(self) -> List[int]:
        return self._FAULT_BEARINGS[self.run]
