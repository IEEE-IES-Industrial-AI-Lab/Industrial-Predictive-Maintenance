"""
Time-domain feature extraction for vibration / accelerometer signals.

All features are well-established in the condition monitoring literature.

References:
    Caesarendra, W., & Tjahjowidodo, T. (2017). A review of feature
    extraction methods in vibration-based condition monitoring.
    Applied Sciences, 7(11), 1152.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Union

import numpy as np


@dataclass
class TimeFeatures:
    """Container for all computed time-domain features."""

    rms: float
    mean: float
    std: float
    variance: float
    peak: float
    peak_to_peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    shape_factor: float
    impulse_factor: float
    margin_factor: float
    entropy: float
    zero_crossing_rate: float
    mean_absolute_value: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return list(TimeFeatures.__dataclass_fields__.keys())


def extract_time_features(signal: np.ndarray) -> TimeFeatures:
    """
    Compute 15 time-domain statistical features from a 1-D vibration signal.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Raw or pre-processed vibration signal (zero-mean recommended).

    Returns
    -------
    TimeFeatures
        Dataclass containing all computed features.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.random.randn(1024)
    >>> feats = extract_time_features(sig)
    >>> feats.rms
    0.9983...
    >>> arr = feats.to_array()   # shape (15,)
    """
    x = signal.astype(np.float64)
    n = len(x)

    rms = float(np.sqrt(np.mean(x ** 2)))
    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    variance = float(np.var(x))
    peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.max(x) - np.min(x))
    crest_factor = peak / (rms + 1e-10)

    # Kurtosis: 4th-order standardised central moment
    kurtosis = float(_central_moment(x, 4) / (std_val ** 4 + 1e-10))

    # Skewness: 3rd-order standardised central moment
    skewness = float(_central_moment(x, 3) / (std_val ** 3 + 1e-10))

    # Mean absolute value
    mav = float(np.mean(np.abs(x)))

    # Shape factor = RMS / MAV
    shape_factor = rms / (mav + 1e-10)

    # Impulse factor = peak / MAV
    impulse_factor = peak / (mav + 1e-10)

    # Margin factor = peak / (sqrt(mean(|x|)))^2
    smr = float(np.mean(np.sqrt(np.abs(x)))) ** 2  # square mean root
    margin_factor = peak / (smr + 1e-10)

    # Shannon entropy (discretised on 64 bins)
    entropy = _shannon_entropy(x, bins=64)

    # Zero-crossing rate
    zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / (n - 1)

    return TimeFeatures(
        rms=float(rms),
        mean=float(mean_val),
        std=float(std_val),
        variance=float(variance),
        peak=float(peak),
        peak_to_peak=float(peak_to_peak),
        crest_factor=float(crest_factor),
        kurtosis=float(kurtosis),
        skewness=float(skewness),
        shape_factor=float(shape_factor),
        impulse_factor=float(impulse_factor),
        margin_factor=float(margin_factor),
        entropy=float(entropy),
        zero_crossing_rate=float(zcr),
        mean_absolute_value=float(mav),
    )


def extract_time_features_batch(signals: np.ndarray) -> np.ndarray:
    """
    Extract time-domain features from a batch of signals.

    Parameters
    ----------
    signals : ndarray, shape (N, L)
        Batch of N signals, each of length L.

    Returns
    -------
    ndarray, shape (N, 15)
        Feature matrix.
    """
    return np.stack(
        [extract_time_features(s).to_array() for s in signals]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _central_moment(x: np.ndarray, order: int) -> float:
    """Compute the ``order``-th central moment of array ``x``."""
    return float(np.mean((x - x.mean()) ** order))


def _shannon_entropy(x: np.ndarray, bins: int = 64) -> float:
    """Estimate Shannon entropy via histogram binning."""
    counts, _ = np.histogram(x, bins=bins)
    probabilities = counts / (counts.sum() + 1e-10)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))
