"""
Signal cleaning and conditioning utilities for industrial sensor data.

Functions cover:
    - Normalisation  (z-score, min-max)
    - Filtering      (Butterworth bandpass / lowpass)
    - Denoising      (wavelet soft-thresholding)
    - Baseline       (DC offset removal)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_zscore(
    data: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalise along ``axis``.

    Parameters
    ----------
    data : ndarray
        Input array of any shape.
    axis : int
        Axis along which mean and std are computed.
    eps : float
        Small constant added to std for numerical stability.

    Returns
    -------
    ndarray
        Normalised array, same shape as ``data``.
    """
    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True) + eps
    return (data - mean) / std


def normalize_minmax(
    data: np.ndarray,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Min-max normalise along ``axis`` to ``feature_range``.

    Parameters
    ----------
    data : ndarray
    feature_range : (min_val, max_val)
    axis : int
    eps : float

    Returns
    -------
    ndarray
    """
    lo, hi = feature_range
    xmin = data.min(axis=axis, keepdims=True)
    xmax = data.max(axis=axis, keepdims=True)
    scale = (xmax - xmin) + eps
    return lo + (data - xmin) / scale * (hi - lo)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def butterworth_bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to a 1-D signal.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    ndarray, shape (N,)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = np.clip(low, 1e-6, 1.0 - 1e-6)
    high = np.clip(high, 1e-6, 1.0 - 1e-6)
    b, a = scipy_signal.butter(order, [low, high], btype="band")
    return scipy_signal.filtfilt(b, a, signal).astype(signal.dtype)


def butterworth_lowpass(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter to a 1-D signal.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int

    Returns
    -------
    ndarray, shape (N,)
    """
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 1.0 - 1e-6)
    b, a = scipy_signal.butter(order, normal_cutoff, btype="low")
    return scipy_signal.filtfilt(b, a, signal).astype(signal.dtype)


def apply_filter_batch(
    signals: np.ndarray,
    filter_fn,
    **kwargs,
) -> np.ndarray:
    """
    Apply ``filter_fn`` to each row of a 2-D array of signals.

    Parameters
    ----------
    signals : ndarray, shape (N, L)
    filter_fn : callable
        One of ``butterworth_bandpass`` or ``butterworth_lowpass``.
    **kwargs
        Extra keyword arguments forwarded to ``filter_fn``.

    Returns
    -------
    ndarray, shape (N, L)
    """
    return np.stack([filter_fn(s, **kwargs) for s in signals])


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """
    Denoise a 1-D signal using wavelet soft/hard thresholding (VisuShrink).

    Parameters
    ----------
    signal : ndarray, shape (N,)
    wavelet : str
        PyWavelets wavelet name (e.g. ``"db4"``, ``"sym8"``).
    level : int, optional
        Decomposition level. If None, uses the maximum possible level.
    threshold_mode : str
        ``"soft"`` or ``"hard"``.

    Returns
    -------
    ndarray, shape (N,)

    Raises
    ------
    ImportError
        If PyWavelets is not installed.
    """
    if not _HAS_PYWT:
        raise ImportError(
            "PyWavelets is required for wavelet denoising. "
            "Install with: pip install PyWavelets"
        )
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet)[: len(signal)].astype(signal.dtype)


# ---------------------------------------------------------------------------
# Baseline removal
# ---------------------------------------------------------------------------

def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """
    Remove the mean (DC component) from a signal.

    Parameters
    ----------
    signal : ndarray, shape (N,) or (N, C)

    Returns
    -------
    ndarray
        Zero-mean signal of the same shape.
    """
    return signal - signal.mean(axis=0, keepdims=True)


def rolling_mean_subtract(
    signal: np.ndarray,
    window: int = 256,
) -> np.ndarray:
    """
    Subtract a rolling mean baseline from a 1-D signal.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    window : int
        Rolling window length in samples.

    Returns
    -------
    ndarray, shape (N,)
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(signal)
    baseline = np.empty(n, dtype=signal.dtype)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        baseline[i] = signal[lo:hi].mean()
    return signal - baseline
