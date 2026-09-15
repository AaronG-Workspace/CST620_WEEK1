"""CPU training, evaluation, and frozen-feature extraction."""

import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
NUM_WORKERS = 0  # load in the main process; worker processes are fragile in Windows notebooks


def make_loader(dataset: Dataset, shuffle: bool, seed: int = 42) -> DataLoader:
    """DataLoader with the project's batch size and workers. Shuffling is seeded."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over a loader: trains if an optimizer is given, otherwise evaluates.

    Returns (mean loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)

    return total_loss / seen, correct / seen


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
) -> list[dict]:
    """Train with Adam, then restore the epoch with the best validation accuracy.

    Only parameters with requires_grad are updated. Ties go to the earlier
    epoch. The test set is never used here.
    Returns one dict of metrics per epoch.
    """
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    history = []
    best_val_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seconds": time.perf_counter() - start,
            }
        )
        print(
            f"epoch {epoch:>2}/{epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}  "
            f"{history[-1]['seconds']:.1f}s{'  <- best' if is_best else ''}"
        )

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader) -> TensorDataset:
    """Run a timm model's frozen backbone once and keep what its head receives.

    For DeiT this is the final class-token embedding (192 values per image).
    Returns a TensorDataset of (features, labels), in loader order.
    """
    model.to(DEVICE).eval()
    features, labels = [], []
    for images, batch_labels in loader:
        tokens = model.forward_features(images.to(DEVICE))
        features.append(model.forward_head(tokens, pre_logits=True))
        labels.append(batch_labels)
    return TensorDataset(torch.cat(features), torch.cat(labels))
