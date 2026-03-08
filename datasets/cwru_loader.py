"""
Case Western Reserve University (CWRU) Bearing Dataset Loader.

Dataset source:
    https://engineering.case.edu/bearingdatacenter

Description:
    Vibration data collected from drive end (DE) and fan end (FE) accelerometers
    on test stand with SKF and Rexnord bearings. Four fault categories:
        - Normal (baseline)
        - Inner race fault  (IR)
        - Ball fault        (B)
        - Outer race fault  (OR) — centred at 3, 6, or 12 o'clock

    Fault diameters: 0.007", 0.014", 0.021" (and 0.028" for some conditions).
    Motor loads: 0, 1, 2, 3 HP (1797, 1772, 1750, 1730 RPM).

File naming convention (example):
    IR007_DE_time  → inner race, 0.007" fault, drive end
    Normal_0_DE    → normal baseline, 0 HP load, drive end

Reference:
    Smith, W. A., & Randall, R. B. (2015). Rolling element bearing diagnostics
    using the Case Western Reserve University data: A benchmark study.
    Mechanical Systems and Signal Processing, 64–65, 100–131.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.io as sio
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


_FAULT_LABELS: Dict[str, int] = {
    "normal": 0,
    "inner_race": 1,
    "ball": 2,
    "outer_race": 3,
}

_FILE_LABEL_MAP: Dict[str, str] = {
    "Normal": "normal",
    "IR": "inner_race",
    "B": "ball",
    "OR": "outer_race",
}

# CWRU files available at the official site
_CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"


class CWRULoader:
    """
    Load the CWRU bearing dataset from .mat files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing downloaded .mat files.
    fault_sizes : list of str, optional
        Fault diameters to include: ``["normal", "0.007", "0.014", "0.021"]``.
        Defaults to all four.
    channel : str
        Accelerometer channel: ``"DE"`` (drive end) or ``"FE"`` (fan end).
    motor_load : int, optional
        Motor load in HP (0–3). None means all loads are included.
    window_size : int
        Number of samples per input window.
    stride : int
        Sliding window stride.
    normalize : bool
        Z-score normalise each window.

    Examples
    --------
    >>> loader = CWRULoader(data_dir="datasets/data/CWRU", fault_sizes=["normal", "0.007"])
    >>> X, y, label_names = loader.load()
    >>> X.shape  # (N, window_size)

    >>> X_normal, X_fault = loader.load_split()
    """

    def __init__(
        self,
        data_dir: str | Path = "datasets/data/CWRU",
        fault_sizes: Optional[List[str]] = None,
        channel: str = "DE",
        motor_load: Optional[int] = None,
        window_size: int = 1024,
        stride: int = 512,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.fault_sizes = fault_sizes or ["normal", "0.007", "0.014", "0.021"]
        self.channel = channel.upper()
        self.motor_load = motor_load
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all matching .mat files and return windowed arrays.

        Returns
        -------
        X : ndarray, shape (N, window_size)
        y : ndarray, shape (N,)  — integer labels
        label_names : list of str
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Please download CWRU .mat files from:\n"
                "  https://engineering.case.edu/bearingdatacenter/download-data-file\n"
                f"and place them in:  {self.data_dir}"
            )

        mat_files = list(self.data_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found in {self.data_dir}"
            )

        X_list, y_list = [], []
        for mat_file in sorted(mat_files):
            fault_type, fault_size = self._parse_filename(mat_file.name)
            if fault_type is None:
                continue
            if fault_size not in self.fault_sizes and fault_size != "normal":
                continue
            if fault_type == "normal" and "normal" not in self.fault_sizes:
                continue

            signal = self._load_signal(mat_file)
            if signal is None:
                continue

            label = _FAULT_LABELS.get(fault_type, -1)
            windows = self._sliding_windows(signal)
            X_list.append(windows)
            y_list.append(np.full(len(windows), label, dtype=np.int64))

        if not X_list:
            raise RuntimeError("No data loaded — check fault_sizes and data_dir.")

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        if self.normalize:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True) + 1e-8
            X = (X - mean) / std

        return X.astype(np.float32), y, list(_FAULT_LABELS.keys())

    def load_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return separate arrays for normal and fault data.

        Useful for training anomaly detection models on healthy data only.
        """
        X, y, _ = self.load()
        X_normal = X[y == _FAULT_LABELS["normal"]]
        X_fault = X[y != _FAULT_LABELS["normal"]]
        return X_normal, X_fault

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_filename(self, filename: str) -> Tuple[Optional[str], str]:
        """
        Extract fault type and fault size from CWRU filename convention.
        Returns (fault_type, fault_size_str) or (None, None) if unrecognised.
        """
        name = filename.replace(".mat", "")

        # Normal files: "Normal_0", "Normal_1", etc.
        if name.lower().startswith("normal"):
            parts = name.split("_")
            if self.motor_load is not None:
                load_part = next((p for p in parts if p.isdigit()), None)
                if load_part is not None and int(load_part) != self.motor_load:
                    return None, ""
            return "normal", "normal"

        # Fault files: "IR007_DE_time", "B021_FE_time", "OR014@6_DE_time"
        for prefix, fault_type in _FILE_LABEL_MAP.items():
            if name.startswith(prefix) and prefix != "Normal":
                rest = name[len(prefix):]
                # Extract numeric fault size (first 3 digits)
                digits = "".join(c for c in rest[:3] if c.isdigit())
                if len(digits) == 3:
                    size_str = f"0.{digits}"
                    # Check motor load
                    if self.motor_load is not None:
                        parts = name.split("_")
                        load_part = next(
                            (p for p in parts if p.isdigit() and len(p) == 1),
                            None,
                        )
                        if load_part is not None and int(load_part) != self.motor_load:
                            return None, ""
                    # Check channel
                    if f"_{self.channel}_" not in name and not name.endswith(
                        f"_{self.channel}"
                    ):
                        return None, ""
                    return fault_type, size_str

        return None, ""

    def _load_signal(self, path: Path) -> Optional[np.ndarray]:
        if not _HAS_SCIPY:
            raise ImportError("scipy is required to load CWRU .mat files.")
        try:
            mat = sio.loadmat(str(path))
        except Exception:
            return None

        # Find the key matching the desired channel
        channel_key = next(
            (k for k in mat if self.channel in k.upper() and "time" in k.lower()),
            None,
        )
        if channel_key is None:
            # Fallback: first array key that is not metadata
            channel_key = next(
                (k for k in mat if not k.startswith("__")), None
            )
        if channel_key is None:
            return None

        data = mat[channel_key].ravel().astype(np.float32)
        return data

    def _sliding_windows(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        indices = range(0, n - self.window_size + 1, self.stride)
        return np.stack([signal[i: i + self.window_size] for i in indices])
