"""Test-set metrics, model size, and CPU latency for the comparison table."""

import io
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import count_parameters

DR_LABEL = 1
WARMUP_RUNS = 3
TIMED_RUNS = 10


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true labels for every item in a loader (eval mode, no gradients)."""
    model.eval()
    preds, labels = [], []
    for images, batch_labels in loader:
        preds.append(model(images).argmax(dim=1))
        labels.append(batch_labels)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Accuracy, and sensitivity: the share of true DR images predicted as DR."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    is_dr = labels == DR_LABEL
    if not is_dr.any():
        raise ValueError("no DR labels, so sensitivity is undefined")
    return {
        "accuracy": float((preds == labels).mean()),
        "sensitivity": float((preds[is_dr] == DR_LABEL).mean()),
    }


def model_size_mb(model: nn.Module) -> float:
    """Size of the saved state_dict (weights and buffers), in MB (1 MB = 1024**2 bytes)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1024**2


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    images: torch.Tensor,
    warmup: int = WARMUP_RUNS,
    runs: int = TIMED_RUNS,
) -> dict[str, float]:
    """Forward-pass time for one batch, in milliseconds: mean and sample std.

    Eval mode, no gradients. Warm-up runs aren't timed, because the first
    calls pay one-time costs such as memory allocation.
    """
    model.eval()
    for _ in range(warmup):
        model(images)

    times_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        model(images)
        times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "latency_mean_ms": float(np.mean(times_ms)),
        "latency_std_ms": float(np.std(times_ms, ddof=1)),
    }


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, latency_batch: torch.Tensor
) -> dict[str, float]:
    """One comparison-table row: test metrics, parameters, size, and latency."""
    preds, labels = predict(model, test_loader)
    return {
        **classification_metrics(labels, preds),
        "parameters": count_parameters(model)["total"],
        "size_mb": model_size_mb(model),
        **measure_latency(model, latency_batch),
    }


def majority_baseline(train_labels: np.ndarray, test_labels: np.ndarray) -> dict[str, float]:
    """Row for a model that always predicts the most common training label.

    The majority class comes from training labels only, never from test.
    """
    majority = np.bincount(np.asarray(train_labels)).argmax()
    preds = np.full(len(test_labels), majority)
    return {
        **classification_metrics(test_labels, preds),
        "parameters": 0,
        "size_mb": 0.0,
        "latency_mean_ms": np.nan,
        "latency_std_ms": np.nan,
    }


def cpu_description() -> str:
    """Processor name and PyTorch thread count, for recording latency settings."""
    return f"{platform.processor() or platform.machine()}, {torch.get_num_threads()} PyTorch threads"


def results_table(rows: dict[str, dict], batch_size: int) -> pd.DataFrame:
    """Format comparison rows for display, one row per model."""
    df = pd.DataFrame(rows).T
    latency = [
        "—" if pd.isna(mean) else f"{mean:.1f} ± {std:.1f}"
        for mean, std in zip(df["latency_mean_ms"], df["latency_std_ms"])
    ]
    return pd.DataFrame(
        {
            "Test accuracy": df["accuracy"].map("{:.3f}".format),
            "Sensitivity (DR recall)": df["sensitivity"].map("{:.3f}".format),
            "Parameters": df["parameters"].map("{:,.0f}".format),
            "Size (MB)": df["size_mb"].map("{:.2f}".format),
            f"CPU latency, batch of {batch_size} (ms)": latency,
        },
        index=df.index,
    )
