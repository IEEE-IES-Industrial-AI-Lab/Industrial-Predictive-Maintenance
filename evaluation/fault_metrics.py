"""
Classification metrics for fault detection and anomaly detection evaluation.

Covers:
    - Accuracy, Precision, Recall, F1 (macro and per-class)
    - AUC-ROC
    - Confusion matrix (heatmap)
    - ROC curve plot
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        roc_curve,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    average: str = "macro",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        Integer class labels.
    y_pred : array-like, shape (N,)
        Predicted integer class labels.
    y_scores : array-like, shape (N,) or (N, C), optional
        Decision scores or probabilities for AUC-ROC calculation.
        Required for binary problems if AUC-ROC is desired.
    average : str
        Averaging strategy for multi-class: ``"macro"``, ``"micro"``,
        ``"weighted"``, or ``"binary"``.
    verbose : bool
        Print summary.

    Returns
    -------
    dict with keys: ``"accuracy"``, ``"precision"``, ``"recall"``,
    ``"f1"``, and optionally ``"roc_auc"``.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_scores is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass

    if verbose:
        print("\n── Fault Detection Metrics ─────────────────")
        for k, v in metrics.items():
            print(f"  {k:<20s} {v:.4f}")
        print("────────────────────────────────────────────\n")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
    class_names : list of str, optional
    title : str
    save_path : str, optional
    normalize : bool
        Normalise each row to [0, 1].

    Returns
    -------
    matplotlib Figure
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn required for confusion matrix.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = ".2f"
    else:
        cm_display = cm.astype(float)
        fmt = "d"

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(5, n_classes * 1.2), max(4, n_classes * 1.0)))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm_display.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = f"{cm_display[i, j]:{fmt}}"
            ax.text(
                j, i, val,
                ha="center", va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the binary ROC curve.

    Parameters
    ----------
    y_true : array-like — binary labels (0 / 1)
    y_scores : array-like — anomaly / decision scores
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn required for ROC curve.")

    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_scores))
    auc = roc_auc_score(np.asarray(y_true), np.asarray(y_scores))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Timeline",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Time-series plot of anomaly scores with optional fault labels overlay.

    Parameters
    ----------
    scores : ndarray, shape (N,)
    labels : ndarray, shape (N,), optional — binary ground truth
    threshold : float, optional — detection threshold line
    title : str
    save_path : str, optional

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(len(scores))
    ax.plot(x, scores, color="steelblue", linewidth=0.8, label="Anomaly score")

    if threshold is not None:
        ax.axhline(threshold, color="red", linewidth=1.5, linestyle="--",
                   label=f"Threshold ({threshold:.4f})")

    if labels is not None:
        fault_idx = np.where(np.asarray(labels) == 1)[0]
        if len(fault_idx):
            ax.axvspan(
                fault_idx[0], fault_idx[-1],
                alpha=0.15, color="red", label="Fault region"
            )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
