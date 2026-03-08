"""
Paderborn University Bearing Dataset Loader.

Dataset source:
    https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/

Description:
    Bearings with artificial (EDM, drilling, engraving) and real accelerated
    lifetime damage. 32 operating conditions × multiple damage states.

    Data files are MATLAB v7.3 (.mat) files, each containing:
        - bearing.vibration.y_values   — vibration signal (64 000 samples)
        - bearing.motor_current.y_values — motor current (64 000 samples)

Damage categories:
    K001–K005 : undamaged (healthy) bearings
    KA04–KA30 : outer ring damage (various methods)
    KI04–KI21 : inner ring damage
    KB23–KB27 : rolling element damage

Reference:
    Lessmeier, C. et al. (2016). Condition Monitoring of Bearing Damage in
    Electromechanical Drive Systems by Using Motor Current Signals.
    PHM Society European Conference, Bilbao, Spain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mat73
    _HAS_MAT73 = True
except ImportError:
    _HAS_MAT73 = False

try:
    import scipy.io as sio
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


_LABEL_MAP: Dict[str, int] = {
    "healthy": 0,
    "outer_ring": 1,
    "inner_ring": 2,
    "rolling_element": 3,
}

_BEARING_PREFIXES: Dict[str, str] = {
    "K": "healthy",
    "KA": "outer_ring",
    "KI": "inner_ring",
    "KB": "rolling_element",
}


def _infer_label(bearing_id: str) -> Tuple[str, int]:
    """Map bearing ID (e.g. 'KA04') to (fault_type, label_int)."""
    for prefix in sorted(_BEARING_PREFIXES, key=len, reverse=True):
        if bearing_id.startswith(prefix):
            fault_type = _BEARING_PREFIXES[prefix]
            return fault_type, _LABEL_MAP[fault_type]
    return "unknown", -1


class PaderbornLoader:
    """
    Load and preprocess the Paderborn bearing dataset.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing one sub-folder per bearing ID.
    bearing_ids : list of str, optional
        Specific bearing IDs to load (e.g. ``["K001", "KA04", "KI04"]``).
        If None, all bearing folders found in ``data_dir`` are loaded.
    signal : str
        Which signal to use: ``"vibration"`` or ``"motor_current"``.
    window_size : int
        Length of each analysis window (samples).
    stride : int
        Stride when creating sliding windows from each recording.
    normalize : bool
        Z-score normalise each window independently.

    Examples
    --------
    >>> loader = PaderbornLoader(
    ...     data_dir="datasets/data/Paderborn",
    ...     bearing_ids=["K001", "KA04", "KI04"],
    ... )
    >>> X, y, label_names = loader.load()
    >>> X.shape   # (N_windows, window_size)
    """

    def __init__(
        self,
        data_dir: str | Path = "datasets/data/Paderborn",
        bearing_ids: Optional[List[str]] = None,
        signal: str = "vibration",
        window_size: int = 4096,
        stride: int = 2048,
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.bearing_ids = bearing_ids
        self.signal = signal
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load bearings and return windowed data with integer labels.

        Returns
        -------
        X : ndarray, shape (N, window_size)
        y : ndarray, shape (N,)  — integer class labels
        label_names : list of str  — class name for each label integer
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                "Download the Paderborn dataset and extract to this path."
            )

        bearing_dirs = self._discover_bearing_dirs()

        X_list, y_list = [], []
        for bearing_id, bearing_dir in bearing_dirs:
            _, label_int = _infer_label(bearing_id)
            signal_data = self._load_bearing(bearing_dir)
            if signal_data is None:
                continue
            windows = self._sliding_windows(signal_data)
            X_list.append(windows)
            y_list.append(np.full(len(windows), label_int, dtype=np.int64))

        if not X_list:
            raise RuntimeError(
                "No data loaded. Check that .mat files exist in the data directory."
            )

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        if self.normalize:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True) + 1e-8
            X = (X - mean) / std

        label_names = list(_LABEL_MAP.keys())
        return X, y, label_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_bearing_dirs(self) -> List[Tuple[str, Path]]:
        if self.bearing_ids is not None:
            return [
                (bid, self.data_dir / bid)
                for bid in self.bearing_ids
                if (self.data_dir / bid).exists()
            ]
        return [
            (p.name, p)
            for p in sorted(self.data_dir.iterdir())
            if p.is_dir()
        ]

    def _load_bearing(self, bearing_dir: Path) -> Optional[np.ndarray]:
        mat_files = list(bearing_dir.glob("*.mat"))
        if not mat_files:
            return None

        all_segments = []
        for mat_file in mat_files:
            data = self._read_mat(mat_file)
            if data is not None:
                all_segments.append(data)

        if not all_segments:
            return None
        return np.concatenate(all_segments)

    def _read_mat(self, path: Path) -> Optional[np.ndarray]:
        """Try to extract vibration/motor_current signal from a .mat file."""
        if _HAS_MAT73:
            try:
                mat = mat73.loadmat(str(path))
                return self._extract_signal_mat73(mat)
            except Exception:
                pass

        if _HAS_SCIPY:
            try:
                mat = sio.loadmat(str(path))
                return self._extract_signal_scipy(mat)
            except Exception:
                pass

        return None

    def _extract_signal_mat73(self, mat: dict) -> Optional[np.ndarray]:
        bearing_key = next(
            (k for k in mat if k.startswith("bearing")), None
        )
        if bearing_key is None:
            return None
        bearing = mat[bearing_key]
        if self.signal == "vibration" and "vibration" in bearing:
            return np.array(bearing["vibration"]["y_values"]).ravel().astype(np.float32)
        if self.signal == "motor_current" and "motor_current" in bearing:
            return np.array(bearing["motor_current"]["y_values"]).ravel().astype(np.float32)
        return None

    def _extract_signal_scipy(self, mat: dict) -> Optional[np.ndarray]:
        # Scipy loads .mat v5 — key is typically like 'bearing'
        for key in mat:
            if not key.startswith("__"):
                try:
                    obj = mat[key]
                    # Try nested struct access
                    signal_data = obj["vibration"][0, 0]["y_values"][0, 0].ravel()
                    return signal_data.astype(np.float32)
                except Exception:
                    pass
        return None

    def _sliding_windows(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        indices = range(0, n - self.window_size + 1, self.stride)
        return np.stack([signal[i: i + self.window_size] for i in indices])
