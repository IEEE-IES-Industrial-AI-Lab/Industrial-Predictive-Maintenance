"""
Unified benchmark runner for all predictive maintenance models.

Trains and evaluates all models (LSTM, Transformer, TCN) on the selected
CMAPSS dataset subset and saves results as JSON and a Markdown table.

Usage
-----
    # Benchmark all models on FD001
    python benchmarks/run_benchmarks.py --dataset FD001 --all-models

    # Benchmark a single model
    python benchmarks/run_benchmarks.py --dataset FD001 --model lstm

    # Benchmark all models on all subsets
    python benchmarks/run_benchmarks.py --all-datasets --all-models

Output
------
    benchmarks/results/results_FD001.json
    benchmarks/results/README.md  (updated with new rows)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.cmapss_loader import CMAPSSLoader
from models.lstm_predictive import LSTMPredictiveModel
from models.transformer_rul import TransformerRULModel
from models.tcn_model import TCNModel
from evaluation.rul_metrics import evaluate_rul


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_REGISTRY = {
    "lstm": (LSTMPredictiveModel, "configs/lstm_config.yaml"),
    "transformer": (TransformerRULModel, "configs/transformer_config.yaml"),
    "tcn": (TCNModel, "configs/tcn_config.yaml"),
}

CMAPSS_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_single_benchmark(
    model_name: str,
    dataset_subset: str,
    data_dir: str = "datasets/data",
    max_rul: int = 125,
    window_size: int = 30,
    val_fraction: float = 0.1,
    epochs: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Train and evaluate one model on one dataset subset.

    Returns
    -------
    dict with keys: model, dataset, rmse, mae, nasa_score, phm_score,
    n_params, train_time_s, timestamp
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name.upper()}   Dataset: {dataset_subset}")
        print(f"{'=' * 60}")

    # Load data
    loader = CMAPSSLoader(
        subset=dataset_subset,
        data_dir=data_dir,
        max_rul=max_rul,
        window_size=window_size,
        stride=1,
        normalize=True,
    )
    X_train, y_train, X_test, y_test = loader.load()

    # Train / val split
    n_val = max(1, int(len(X_train) * val_fraction))
    idx = np.random.permutation(len(X_train))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    if verbose:
        print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Build model
    model_cls, config_path = MODEL_REGISTRY[model_name]
    config = {"input_size": X_tr.shape[-1]}

    config_file = Path(config_path)
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            yaml_cfg = yaml.safe_load(f)
        config = {**yaml_cfg, **config}  # override input_size

    model = model_cls(config)

    n_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    if verbose:
        print(f"  Parameters: {n_params:,}")

    # Train
    t0 = time.time()
    model.fit(X_tr, y_tr, X_val, y_val, epochs=epochs)
    train_time = time.time() - t0

    # Evaluate
    metrics = evaluate_rul(y_test, model.predict(X_test), verbose=verbose)

    result = {
        "model": model_name,
        "dataset": dataset_subset,
        "rmse": round(metrics["rmse"], 4),
        "mae": round(metrics["mae"], 4),
        "nasa_score": round(metrics["nasa_score"], 2),
        "phm_score": round(metrics["phm_score"], 4),
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # Save checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    model.save(ckpt_dir / f"{model_name}_{dataset_subset}.pt")

    return result


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: List[Dict], subset: str) -> Path:
    path = RESULTS_DIR / f"results_{subset}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


def results_to_markdown_table(results: List[Dict]) -> str:
    header = (
        "| Model | Dataset | RMSE ↓ | MAE ↓ | NASA Score ↓ | Params | Train Time |\n"
        "|-------|---------|--------|-------|-------------|--------|------------|\n"
    )
    rows = []
    for r in results:
        params_k = f"{r['n_params'] / 1000:.0f}K"
        rows.append(
            f"| {r['model'].upper()} | {r['dataset']} "
            f"| {r['rmse']:.2f} | {r['mae']:.2f} | {r['nasa_score']:.1f} "
            f"| {params_k} | {r['train_time_s']:.0f}s |"
        )
    return header + "\n".join(rows)


def update_results_readme(all_results: List[Dict]) -> None:
    readme_path = RESULTS_DIR / "README.md"
    table = results_to_markdown_table(all_results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    content = f"""# Benchmark Results

Last updated: {timestamp}

## CMAPSS Turbofan Dataset — RUL Prediction

Evaluation on test split. Lower RMSE, MAE, and NASA Score = better.
NASA Score is asymmetric: late predictions penalised more than early ones.

{table}

## Methodology

- All models trained from scratch with early stopping (patience = 15).
- Input: sliding window of 30 time steps, 14 sensor features.
- Max RUL clipped at 125 cycles (piecewise-linear labelling).
- 90/10 train/val split; test set uses NASA-provided ground-truth RUL.
- Reported metrics are from the best checkpoint (lowest val loss).

## Reproduce

```bash
python benchmarks/run_benchmarks.py --dataset FD001 --all-models
```
"""
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"Results README updated: {readme_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark predictive maintenance models on CMAPSS."
    )
    parser.add_argument(
        "--dataset",
        choices=CMAPSS_SUBSETS,
        default="FD001",
        help="CMAPSS subset to use.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run on all four CMAPSS subsets.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="lstm",
        help="Model to benchmark.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all registered models.",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/data",
        help="Directory containing CMAPSS data files.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    datasets = CMAPSS_SUBSETS if args.all_datasets else [args.dataset]
    models = list(MODEL_REGISTRY.keys()) if args.all_models else [args.model]

    all_results: List[Dict] = []

    for subset in datasets:
        subset_results = []
        for model_name in models:
            try:
                result = run_single_benchmark(
                    model_name=model_name,
                    dataset_subset=subset,
                    data_dir=args.data_dir,
                    epochs=args.epochs,
                )
                subset_results.append(result)
                all_results.append(result)
            except Exception as exc:
                print(f"  ERROR: {model_name} on {subset}: {exc}")

        if subset_results:
            save_results(subset_results, subset)

    if all_results:
        update_results_readme(all_results)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
