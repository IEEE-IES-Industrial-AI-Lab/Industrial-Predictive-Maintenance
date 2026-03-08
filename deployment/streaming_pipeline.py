"""
Real-time sliding window inference pipeline for industrial sensor streams.

Designed for deployment scenarios where sensor data arrives sample-by-sample
or in small bursts (e.g. from PLCs, IoT gateways, OPC-UA servers).

Components:
    SlidingWindowBuffer  — maintains a fixed-length window of recent samples
    StreamingPredictor   — wraps a trained model for stream-compatible inference
    MockSensorStream     — deterministic sensor stream for testing

Usage
-----
    from deployment.streaming_pipeline import StreamingPredictor
    from models.lstm_predictive import LSTMPredictiveModel

    model = LSTMPredictiveModel.load("checkpoints/lstm_best.pt")
    predictor = StreamingPredictor(model, window_size=30, n_features=14)

    for new_sample in sensor_stream:          # shape: (n_features,)
        predictor.push(new_sample)
        if predictor.ready:
            rul = predictor.predict_latest()
            print(f"RUL estimate: {rul:.1f} cycles")
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Sliding window buffer
# ---------------------------------------------------------------------------

class SlidingWindowBuffer:
    """
    Thread-safe fixed-length sliding window over 1-D or multi-channel data.

    Once the buffer is full, each ``push`` evicts the oldest sample.

    Parameters
    ----------
    window_size : int
        Number of time steps to maintain.
    n_features : int
        Number of sensor channels per time step.

    Examples
    --------
    >>> buf = SlidingWindowBuffer(window_size=5, n_features=3)
    >>> buf.push(np.ones(3))
    >>> buf.is_full
    False
    >>> for _ in range(4): buf.push(np.ones(3))
    >>> buf.is_full
    True
    >>> buf.get_window().shape
    (5, 3)
    """

    def __init__(self, window_size: int, n_features: int) -> None:
        self.window_size = window_size
        self.n_features = n_features
        self._buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def push(self, sample: np.ndarray) -> None:
        """
        Add a new sample to the buffer.

        Parameters
        ----------
        sample : ndarray, shape (n_features,)
        """
        sample = np.asarray(sample, dtype=np.float32).ravel()
        if len(sample) != self.n_features:
            raise ValueError(
                f"Expected sample of size {self.n_features}, got {len(sample)}"
            )
        with self._lock:
            self._buffer.append(sample)

    def push_batch(self, batch: np.ndarray) -> None:
        """
        Push multiple samples at once.

        Parameters
        ----------
        batch : ndarray, shape (N, n_features)
        """
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        for sample in batch:
            self.push(sample)

    def get_window(self) -> np.ndarray:
        """
        Return the current window as an array.

        Returns
        -------
        ndarray, shape (window_size, n_features)
            Zero-padded on the left if buffer is not yet full.
        """
        with self._lock:
            n = len(self._buffer)
            window = np.zeros((self.window_size, self.n_features), dtype=np.float32)
            if n > 0:
                window[self.window_size - n:] = np.stack(list(self._buffer))
        return window

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) == self.window_size

    @property
    def n_samples(self) -> int:
        with self._lock:
            return len(self._buffer)


# ---------------------------------------------------------------------------
# Streaming predictor
# ---------------------------------------------------------------------------

class StreamingPredictor:
    """
    Real-time RUL predictor that wraps a trained model with a sliding buffer.

    Parameters
    ----------
    model
        Trained model with a ``predict(X)`` method.
        X should have shape ``(1, window_size, n_features)``.
    window_size : int
        Number of time steps per inference window.
    n_features : int
        Number of sensor channels.
    predict_every : int
        Run inference every N samples (reduces CPU load). Default 1.
    callback : callable, optional
        Function called on every prediction: ``callback(rul_estimate)``.

    Examples
    --------
    >>> predictor = StreamingPredictor(model, window_size=30, n_features=14)
    >>> for sample in stream:
    ...     predictor.push(sample)
    ...     if predictor.ready:
    ...         print(predictor.predict_latest())
    """

    def __init__(
        self,
        model,
        window_size: int,
        n_features: int,
        predict_every: int = 1,
        callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.predict_every = predict_every
        self.callback = callback

        self._buffer = SlidingWindowBuffer(window_size, n_features)
        self._sample_count = 0
        self._last_prediction: Optional[float] = None
        self._prediction_history: List[float] = []
        self._lock = threading.Lock()

    def push(self, sample: np.ndarray) -> Optional[float]:
        """
        Push a single sensor sample and optionally run inference.

        Parameters
        ----------
        sample : ndarray, shape (n_features,)

        Returns
        -------
        float or None
            RUL estimate if an inference was triggered; None otherwise.
        """
        self._buffer.push(sample)
        with self._lock:
            self._sample_count += 1

        if self._buffer.is_full and (self._sample_count % self.predict_every == 0):
            prediction = self.predict_latest()
            if self.callback is not None:
                self.callback(prediction)
            return prediction
        return None

    def push_batch(self, batch: np.ndarray) -> List[Optional[float]]:
        """
        Push a batch of samples and collect any triggered predictions.

        Parameters
        ----------
        batch : ndarray, shape (N, n_features)

        Returns
        -------
        list of float or None
        """
        results = []
        for sample in batch:
            results.append(self.push(sample))
        return results

    def predict_latest(self) -> float:
        """
        Run a single inference on the current window.

        Returns
        -------
        float
            RUL estimate in the model's output unit (cycles for CMAPSS).

        Raises
        ------
        RuntimeError
            If the buffer is not yet full.
        """
        if not self._buffer.is_full:
            raise RuntimeError(
                f"Buffer not full yet ({self._buffer.n_samples}/{self.window_size} samples). "
                "Push more samples before calling predict_latest()."
            )
        window = self._buffer.get_window()                       # (T, F)
        X = window[np.newaxis, ...]                              # (1, T, F)
        prediction = float(self.model.predict(X)[0])
        with self._lock:
            self._last_prediction = prediction
            self._prediction_history.append(prediction)
        return prediction

    def reset(self) -> None:
        """Clear buffer and reset state."""
        self._buffer.reset()
        with self._lock:
            self._sample_count = 0
            self._last_prediction = None
            self._prediction_history.clear()

    @property
    def ready(self) -> bool:
        """True when the buffer contains a full window."""
        return self._buffer.is_full

    @property
    def last_prediction(self) -> Optional[float]:
        return self._last_prediction

    @property
    def prediction_history(self) -> List[float]:
        with self._lock:
            return list(self._prediction_history)


# ---------------------------------------------------------------------------
# Mock sensor stream (for testing and demos)
# ---------------------------------------------------------------------------

class MockSensorStream:
    """
    Deterministic mock sensor stream that replays a data array sample by sample.

    Useful for integration testing the streaming pipeline without hardware.

    Parameters
    ----------
    data : ndarray, shape (N, n_features)
        Pre-recorded sensor data to replay.
    sampling_rate_hz : float
        Simulated sampling rate. Controls ``time.sleep`` between samples.
        Set to ``None`` (default) to replay without artificial delay.

    Examples
    --------
    >>> stream = MockSensorStream(X_test_unit, sampling_rate_hz=None)
    >>> for sample in stream:
    ...     predictor.push(sample)
    """

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate_hz: Optional[float] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self._delay = (1.0 / sampling_rate_hz) if sampling_rate_hz else None

    def __iter__(self):
        for sample in self.data:
            if self._delay is not None:
                time.sleep(self._delay)
            yield sample

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_features(self) -> int:
        return self.data.shape[1]
