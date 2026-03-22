"""
NASA CMAPSS Turbofan Engine Dataset Loader
==========================================
Source: Saxena, A., et al. "Damage propagation modeling for aircraft engine
        run-to-failure simulation." PHM 2008.
Data:   https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data

RUL computation: piece-wise linear, capped at max_rul=125.
Normalization: per-sensor min-max, computed on TRAIN set only (no leakage).
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# Official column names from the CMAPSS documentation
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]   # 21 sensors
SETTING_COLS = ["setting1", "setting2", "setting3"]
ALL_COLS = ["engine_id", "cycle"] + SETTING_COLS + SENSOR_COLS

# Sensors dropped in most published baselines (near-zero variance on FD001)
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
FEATURE_COLS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]


class CMAPSSLoader:
    """
    Load and preprocess the NASA CMAPSS dataset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    max_rul : int
        RUL cap for piece-wise linear target (default 125, per PHM08 protocol).
    window_size : int
        Sliding window length for sequence models.
    data_dir : str
        Directory to store downloaded data.
    """

    BASE_URL = "https://data.nasa.gov/api/views/ff5v-kuh6/rows.csv?accessType=DOWNLOAD"

    def __init__(
        self,
        subset: str = "FD001",
        max_rul: int = 125,
        window_size: int = 30,
        data_dir: str = "data/cmapss",
    ):
        assert subset in ("FD001", "FD002", "FD003", "FD004"), \
            f"subset must be one of FD001–FD004, got {subset}"
        self.subset = subset
        self.max_rul = max_rul
        self.window_size = window_size
        self.data_dir = Path(data_dir)
        self._min = None  # fit on train, applied to test
        self._max = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X_train, y_train, X_test, y_test
            X shape: (N, window_size, n_features)
            y shape: (N,) — RUL values
        """
        self._ensure_data()
        train_df = self._read_csv("train")
        test_df  = self._read_csv("test")
        rul_df   = self._read_rul()

        # Compute RUL labels
        train_df = self._add_rul_train(train_df)
        test_df  = self._add_rul_test(test_df, rul_df)

        # Normalize (fit on train only — no data leakage)
        train_df, test_df = self._normalize(train_df, test_df)

        # Build sliding windows
        X_train, y_train = self._make_windows(train_df)
        X_test,  y_test  = self._make_windows(test_df)

        return X_train, y_train, X_test, y_test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_data(self):
        """Download CMAPSS zip if not already present."""
        zip_path = self.data_dir / "CMAPSSData.zip"
        train_path = self.data_dir / f"train_{self.subset}.txt"

        if train_path.exists():
            return  # already downloaded

        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CMAPSS data to {self.data_dir} ...")

        # The official NASA zip contains all 4 subsets
        url = (
            "https://data.nasa.gov/api/views/ff5v-kuh6/rows.csv?"
            "accessType=DOWNLOAD"
        )
        # Fallback: many mirrors exist; use the PHM Society mirror
        try:
            urllib.request.urlretrieve(
                "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
                zip_path,
            )
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.data_dir)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Auto-download failed: {e}\n"
                "Manual download: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data\n"
                f"Extract to: {self.data_dir}"
            )

    def _read_csv(self, split: str) -> pd.DataFrame:
        path = self.data_dir / f"{split}_{self.subset}.txt"
        df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
        return df

    def _read_rul(self) -> pd.DataFrame:
        path = self.data_dir / f"RUL_{self.subset}.txt"
        rul = pd.read_csv(path, header=None, names=["rul"])
        rul["engine_id"] = np.arange(1, len(rul) + 1)
        return rul

    def _add_rul_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Piece-wise linear RUL: capped at max_rul, computed per engine.
        This is the standard PHM08 protocol.
        """
        max_cycles = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.merge(max_cycles, on="engine_id")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df["rul"] = df["rul"].clip(upper=self.max_rul)
        df.drop(columns="max_cycle", inplace=True)
        return df

    def _add_rul_test(self, df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        """
        For test set: use the last cycle of each engine + the provided RUL.
        """
        # Keep only the last window of each engine
        last_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
        last_cycles.columns = ["engine_id", "max_cycle"]
        df = df.merge(last_cycles, on="engine_id")
        df = df[df["cycle"] == df["max_cycle"]].copy()
        df = df.merge(rul_df, on="engine_id")
        df["rul"] = df["rul"].clip(upper=self.max_rul)
        return df

    def _normalize(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Min-max normalization. Fit on train only."""
        self._min = train_df[FEATURE_COLS].min()
        self._max = train_df[FEATURE_COLS].max()
        rng = (self._max - self._min).replace(0, 1)  # avoid div-by-zero

        train_df[FEATURE_COLS] = (train_df[FEATURE_COLS] - self._min) / rng
        test_df[FEATURE_COLS]  = (test_df[FEATURE_COLS]  - self._min) / rng
        return train_df, test_df

    def _make_windows(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sliding windows per engine.
        Windows shorter than window_size (early cycles) are zero-padded on the left.
        """
        X_list, y_list = [], []
        w = self.window_size

        for _, engine_df in df.groupby("engine_id"):
            feats = engine_df[FEATURE_COLS].values.astype(np.float32)
            ruls  = engine_df["rul"].values.astype(np.float32)

            for i in range(len(feats)):
                start = max(0, i - w + 1)
                window = feats[start : i + 1]
                if len(window) < w:
                    pad = np.zeros((w - len(window), feats.shape[1]), dtype=np.float32)
                    window = np.vstack([pad, window])
                X_list.append(window)
                y_list.append(ruls[i])

        return np.array(X_list), np.array(y_list)
