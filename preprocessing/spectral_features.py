"""
Frequency-domain (spectral) feature extraction for vibration signals.

Features are derived from the FFT magnitude spectrum and Welch PSD.

References:
    Randall, R. B., & Antoni, J. (2011). Rolling element bearing diagnostics —
    A tutorial. Mechanical Systems and Signal Processing, 25(2), 485–520.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal


@dataclass
class SpectralFeatures:
    """Container for computed spectral features."""

    spectral_centroid: float
    spectral_variance: float
    spectral_skewness: float
    spectral_kurtosis: float
    spectral_entropy: float
    spectral_flatness: float
    spectral_rolloff: float
    dominant_frequency: float
    dominant_amplitude: float
    total_power: float
    # Band energy ratios (low / mid / high thirds)
    band_energy_low: float
    band_energy_mid: float
    band_energy_high: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return list(SpectralFeatures.__dataclass_fields__.keys())


def extract_spectral_features(
    signal: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> SpectralFeatures:
    """
    Compute spectral features from a 1-D signal using Welch's PSD estimate.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    fs : float
        Sampling frequency in Hz. Use 1.0 for normalised frequency.
    nperseg : int, optional
        Segment length for Welch's method. Defaults to ``min(len(signal), 256)``.

    Returns
    -------
    SpectralFeatures

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.sin(2 * np.pi * 50 * np.arange(1024) / 1000)
    >>> feats = extract_spectral_features(sig, fs=1000.0)
    >>> round(feats.dominant_frequency, 0)
    50.0
    """
    x = signal.astype(np.float64)
    n = len(x)
    seg = nperseg or min(n, 256)

    # Welch PSD
    freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=seg)
    total_power = float(np.sum(psd))

    # Normalised PSD (probability-like)
    psd_norm = psd / (total_power + 1e-10)

    # Spectral centroid (mean frequency)
    centroid = float(np.sum(freqs * psd_norm))

    # Spectral variance
    spec_var = float(np.sum((freqs - centroid) ** 2 * psd_norm))

    # Spectral skewness
    spec_std = np.sqrt(spec_var) + 1e-10
    spec_skew = float(np.sum(((freqs - centroid) / spec_std) ** 3 * psd_norm))

    # Spectral kurtosis
    spec_kurt = float(np.sum(((freqs - centroid) / spec_std) ** 4 * psd_norm))

    # Spectral entropy
    spec_entropy = float(-np.sum(psd_norm[psd_norm > 0] * np.log2(psd_norm[psd_norm > 0])))

    # Spectral flatness (geometric mean / arithmetic mean of PSD)
    log_psd = np.log(psd + 1e-10)
    geo_mean = np.exp(log_psd.mean())
    arith_mean = psd.mean() + 1e-10
    spec_flatness = float(geo_mean / arith_mean)

    # Spectral rolloff (frequency below which 85% of energy lies)
    cumulative = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    rolloff_idx = min(rolloff_idx, len(freqs) - 1)
    spec_rolloff = float(freqs[rolloff_idx])

    # Dominant frequency and amplitude
    dom_idx = int(np.argmax(psd))
    dominant_freq = float(freqs[dom_idx])
    dominant_amp = float(psd[dom_idx])

    # Band energy ratios (divide spectrum into thirds)
    n_freq = len(freqs)
    t1, t2 = n_freq // 3, 2 * n_freq // 3
    low_e = float(psd[:t1].sum() / (total_power + 1e-10))
    mid_e = float(psd[t1:t2].sum() / (total_power + 1e-10))
    high_e = float(psd[t2:].sum() / (total_power + 1e-10))

    return SpectralFeatures(
        spectral_centroid=centroid,
        spectral_variance=spec_var,
        spectral_skewness=spec_skew,
        spectral_kurtosis=spec_kurt,
        spectral_entropy=spec_entropy,
        spectral_flatness=spec_flatness,
        spectral_rolloff=spec_rolloff,
        dominant_frequency=dominant_freq,
        dominant_amplitude=dominant_amp,
        total_power=total_power,
        band_energy_low=low_e,
        band_energy_mid=mid_e,
        band_energy_high=high_e,
    )


def extract_spectral_features_batch(
    signals: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
) -> np.ndarray:
    """
    Extract spectral features from a batch of signals.

    Parameters
    ----------
    signals : ndarray, shape (N, L)
    fs : float
    nperseg : int, optional

    Returns
    -------
    ndarray, shape (N, 13)
    """
    return np.stack(
        [extract_spectral_features(s, fs=fs, nperseg=nperseg).to_array()
         for s in signals]
    )


def compute_fft(
    signal: np.ndarray,
    fs: float = 1.0,
    one_sided: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of a 1-D signal.

    Parameters
    ----------
    signal : ndarray, shape (N,)
    fs : float
        Sampling frequency in Hz.
    one_sided : bool
        Return only the positive-frequency half.

    Returns
    -------
    freqs : ndarray
        Frequency bins in Hz.
    magnitude : ndarray
        FFT magnitude spectrum.
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)

    if one_sided:
        half = n // 2
        freqs = freqs[:half]
        magnitude = (2.0 / n) * np.abs(fft_vals[:half])
    else:
        magnitude = np.abs(fft_vals)

    return freqs, magnitude.astype(np.float32)


def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    bandpass_low: float,
    bandpass_high: float,
    filter_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the envelope spectrum (Hilbert-transform based).

    Useful for detecting bearing fault frequencies (BPFI, BPFO, BSF).

    Parameters
    ----------
    signal : ndarray, shape (N,)
    fs : float
    bandpass_low, bandpass_high : float
        Bandpass filter cutoffs in Hz to isolate resonance band.
    filter_order : int

    Returns
    -------
    freqs : ndarray
    envelope_magnitude : ndarray
    """
    from scipy.signal import hilbert

    from .signal_cleaning import butterworth_bandpass
    filtered = butterworth_bandpass(signal, bandpass_low, bandpass_high, fs, filter_order)

    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    envelope -= envelope.mean()

    return compute_fft(envelope, fs=fs, one_sided=True)
