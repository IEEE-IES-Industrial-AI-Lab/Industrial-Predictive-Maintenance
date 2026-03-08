"""
Edge inference utilities: ONNX export, INT8 quantization, and latency benchmarking.

Workflow:
    1. Train a model using models/ classes.
    2. Export to ONNX with export_to_onnx().
    3. Optionally quantize with quantize_onnx_int8().
    4. Run latency benchmarks with benchmark_latency().
    5. Use ONNXPredictor for production inference.

Usage
-----
    from models.lstm_predictive import LSTMPredictiveModel
    from deployment.edge_inference import export_to_onnx, ONNXPredictor

    model = LSTMPredictiveModel.load("checkpoints/lstm_best.pt")
    export_to_onnx(model, "checkpoints/lstm.onnx", input_shape=(1, 30, 14))

    predictor = ONNXPredictor("checkpoints/lstm.onnx")
    rul = predictor.predict(X_test)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import onnx
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model,
    output_path: str | Path,
    input_shape: Tuple[int, ...],
    opset_version: int = 17,
    dynamic_batch: bool = True,
    verify: bool = True,
) -> Path:
    """
    Export a trained model to ONNX format.

    Parameters
    ----------
    model : BaseModel subclass
        A trained model with a ``network`` attribute (``nn.Module``).
    output_path : str or Path
        Destination ``.onnx`` file path.
    input_shape : tuple
        Shape of a single dummy input, e.g. ``(1, 30, 14)``
        for (batch=1, seq_len=30, n_features=14).
    opset_version : int
        ONNX opset version.
    dynamic_batch : bool
        Register the batch dimension as dynamic for variable batch sizes.
    verify : bool
        Run onnx.checker after export.

    Returns
    -------
    Path to the exported ONNX file.

    Raises
    ------
    ImportError
        If ``onnx`` is not installed.
    """
    if not _HAS_ONNX:
        raise ImportError("onnx is required: pip install onnx")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.network.eval()
    dummy_input = torch.zeros(input_shape, device=model.device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    with torch.no_grad():
        torch.onnx.export(
            model.network,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    if verify:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model verified successfully.")

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Exported ONNX model to: {output_path}  ({size_mb:.2f} MB)")
    return output_path


def quantize_onnx_int8(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Apply static INT8 quantization to an ONNX model (reduces size ~4×).

    Parameters
    ----------
    input_path : str or Path
        Path to the input ONNX file.
    output_path : str or Path, optional
        Destination path. Defaults to ``<input_stem>_int8.onnx``.

    Returns
    -------
    Path to the quantized ONNX file.

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed or lacks quantization support.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime quantization tools are required. "
            "Install with: pip install onnxruntime"
        )

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_int8.onnx"
    else:
        output_path = Path(output_path)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    original_mb = input_path.stat().st_size / (1024 ** 2)
    quantized_mb = output_path.stat().st_size / (1024 ** 2)
    ratio = original_mb / quantized_mb
    print(
        f"INT8 quantization complete.\n"
        f"  Original:  {original_mb:.2f} MB\n"
        f"  Quantized: {quantized_mb:.2f} MB  ({ratio:.1f}× smaller)"
    )
    return output_path


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_latency(
    onnx_path: str | Path,
    input_shape: Tuple[int, ...],
    n_warmup: int = 10,
    n_runs: int = 100,
    providers: Optional[list] = None,
) -> Dict[str, float]:
    """
    Benchmark inference latency of an ONNX model.

    Parameters
    ----------
    onnx_path : str or Path
    input_shape : tuple  — shape of a single inference call
    n_warmup : int       — warm-up runs (discarded)
    n_runs : int         — timed runs
    providers : list, optional  — ONNX Runtime execution providers

    Returns
    -------
    dict with keys: ``"mean_ms"``, ``"std_ms"``, ``"min_ms"``, ``"max_ms"``,
    ``"throughput_qps"``
    """
    if not _HAS_ORT:
        raise ImportError("onnxruntime is required: pip install onnxruntime")

    providers = providers or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name

    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    results = {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "throughput_qps": float(1000.0 / latencies.mean()),
    }

    print(
        f"\n── ONNX Latency Benchmark ──────────────────\n"
        f"  Input shape : {input_shape}\n"
        f"  Mean        : {results['mean_ms']:.3f} ms\n"
        f"  Std         : {results['std_ms']:.3f} ms\n"
        f"  Min / Max   : {results['min_ms']:.3f} / {results['max_ms']:.3f} ms\n"
        f"  Throughput  : {results['throughput_qps']:.1f} QPS\n"
        f"────────────────────────────────────────────\n"
    )
    return results


# ---------------------------------------------------------------------------
# Production predictor
# ---------------------------------------------------------------------------

class ONNXPredictor:
    """
    Lightweight ONNX Runtime wrapper for production inference.

    Parameters
    ----------
    onnx_path : str or Path
        Path to a ``.onnx`` model file.
    providers : list, optional
        ONNX Runtime execution providers.
        Defaults to ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
        (falls back to CPU if CUDA is unavailable).

    Examples
    --------
    >>> predictor = ONNXPredictor("checkpoints/lstm.onnx")
    >>> rul = predictor.predict(X_test)  # shape: (N,)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: Optional[list] = None,
    ) -> None:
        if not _HAS_ORT:
            raise ImportError("onnxruntime is required: pip install onnxruntime")

        self.onnx_path = Path(onnx_path)
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Run batched inference.

        Parameters
        ----------
        X : ndarray, shape (N, T, F)
        batch_size : int

        Returns
        -------
        ndarray, shape (N,)
        """
        X = X.astype(np.float32)
        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i: i + batch_size]
            out = self._session.run(None, {self._input_name: batch})[0]
            results.append(out.ravel())
        return np.concatenate(results)

    @property
    def input_shape(self) -> list:
        return self._session.get_inputs()[0].shape

    @property
    def model_size_mb(self) -> float:
        return self.onnx_path.stat().st_size / (1024 ** 2)
