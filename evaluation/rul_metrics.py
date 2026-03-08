"""
Evaluation metrics for Remaining Useful Life (RUL) prediction.

Standard metrics used in the PHM and prognostics community:
    - RMSE  — root mean squared error
    - MAE   — mean absolute error
    - NASA Asymmetric Score  — penalises late predictions more heavily
    - PHM Score  — variant used in the 2008 PHM Data Challenge

References:
    Saxena, A. et al. (2008). Metrics for evaluating performance of
    prognostic techniques. 2008 International Conference on Prognostics
    and Health Management (PHM).

    Heimes, F. (2008). Recurrent neural networks for remaining useful
    life estimation. 2008 International Conference on PHM.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (N,)

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA asymmetric scoring function for RUL evaluation.

    The score is:

        S = Σ f(d_i)

    where d_i = predicted_i - true_i and

        f(d) = exp(-d / 13) - 1   if d < 0  (early prediction)
        f(d) = exp( d / 10) - 1   if d >= 0 (late prediction — penalised more)

    A lower score indicates better performance.

    Parameters
    ----------
    y_true : array-like
        Ground-truth RUL values.
    y_pred : array-like
        Predicted RUL values.

    Returns
    -------
    float
        Sum of asymmetric penalties.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(score))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PHM 2008 Data Challenge scoring function.

    Identical to ``nasa_score`` but normalised by the number of samples,
    giving a per-unit average score.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float
    """
    n = len(np.asarray(y_true))
    return nasa_score(y_true, y_pred) / (n + 1e-10)


def evaluate_rul(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute all RUL metrics at once.

    Parameters
    ----------
    y_true, y_pred : array-like
    verbose : bool
        Print a summary table.

    Returns
    -------
    dict with keys: ``"rmse"``, ``"mae"``, ``"nasa_score"``, ``"phm_score"``
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
    }
    if verbose:
        print("\n── RUL Evaluation Metrics ──────────────────")
        for k, v in metrics.items():
            print(f"  {k:<20s} {v:.4f}")
        print("────────────────────────────────────────────\n")
    return metrics


def plot_rul_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_ids: np.ndarray | None = None,
    title: str = "RUL Prediction vs Ground Truth",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of predicted vs true RUL.

    Parameters
    ----------
    y_true, y_pred : array-like
    unit_ids : optional array of unit identifiers for colouring
    title : str
    save_path : str, optional  — save figure to this path

    Returns
    -------
    matplotlib Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: predicted vs true
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10, color="steelblue")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{title}\nRMSE={rmse(y_true, y_pred):.2f}  MAE={mae(y_true, y_pred):.2f}")
    ax.legend()

    # Error distribution
    ax2 = axes[1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax2.axvline(0, color="r", linewidth=1.5, linestyle="--", label="Zero error")
    ax2.set_xlabel("Prediction error (cycles)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")
    ax2.legend()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
