"""OpenCV preprocessing pipeline and PyTorch Dataset for retinal fundus images."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Every pipeline setting, in pipeline order.
PIPELINE_CONFIG = {
    "clahe": {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
    # sigma 0 lets OpenCV derive sigma from the kernel size (1.1 for 5x5).
    "blur": {"kernel_size": (5, 5), "sigma": 0},
    # Chosen by measuring gradient strength inside the fundus circle on 200
    # training images after CLAHE and blur (median 22, 75th percentile 38,
    # 95th 96, 98th 140):
    # - low 40 is just above the 75th percentile, so background texture
    #   doesn't turn into edges.
    # - high 120 is between the 95th and 98th percentiles, so edges only start
    #   at strong boundaries: vessel walls, the optic disc, lesions, the rim.
    # - The 1:3 ratio is in the 1:2 to 1:3 range Canny recommended.
    # Canny's 3x3 Sobel filter scales an intensity step by about 4, so these
    # match steps of roughly 10 and 30 gray levels.
    "canny": {"low_threshold": 40, "high_threshold": 120},
    "closing": {"kernel_size": (3, 3)},
    # (width, height): the order cv2.resize expects.
    "model_input": {"size": (224, 224)},
}


def load_bgr(path: str | Path) -> np.ndarray:
    """Read an image with cv2.imread. The result is uint8 in BGR channel order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cv2.imread could not read {path}")
    return image


def _grayscale_and_clahe(bgr: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected a 3-channel BGR image, got shape {bgr.shape}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    settings = config["clahe"]
    clahe = cv2.createCLAHE(
        clipLimit=settings["clip_limit"], tileGridSize=settings["tile_grid_size"]
    )
    return gray, clahe.apply(gray)


def _to_model_input(clahe_image: np.ndarray, config: dict) -> np.ndarray:
    rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, config["model_input"]["size"], interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def run_pipeline(bgr: np.ndarray, config: dict = PIPELINE_CONFIG) -> dict[str, np.ndarray]:
    """Run every stage on one BGR image and return all of them, in order.

    Keys: bgr, gray, clahe, blurred, edges, closed, model_input.
    `model_input` is the CLAHE image as 3 channels, 224x224, float32 in [0, 1].
    """
    gray, clahe = _grayscale_and_clahe(bgr, config)
    blurred = cv2.GaussianBlur(clahe, config["blur"]["kernel_size"], config["blur"]["sigma"])
    edges = cv2.Canny(
        blurred, config["canny"]["low_threshold"], config["canny"]["high_threshold"]
    )
    kernel = np.ones(config["closing"]["kernel_size"], np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return {
        "bgr": bgr,
        "gray": gray,
        "clahe": clahe,
        "blurred": blurred,
        "edges": edges,
        "closed": closed,
        "model_input": _to_model_input(clahe, config),
    }


def make_model_input(bgr: np.ndarray, config: dict = PIPELINE_CONFIG) -> np.ndarray:
    """Build only the model input, skipping the blur, Canny, and closing stages."""
    _, clahe = _grayscale_and_clahe(bgr, config)
    return _to_model_input(clahe, config)


def compute_mean_std(
    train_df: pd.DataFrame, config: dict = PIPELINE_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std of the model inputs, from training images only.

    Uses running sums, so the whole split never has to be in memory at once.
    """
    if "split" in train_df and not (train_df["split"] == "train").all():
        raise ValueError("compute_mean_std must only be given training rows")

    total = np.zeros(3)
    total_sq = np.zeros(3)
    n_pixels = 0
    for path in train_df["path"]:
        image = make_model_input(load_bgr(path), config).astype(np.float64)
        total += image.sum(axis=(0, 1))
        total_sq += (image**2).sum(axis=(0, 1))
        n_pixels += image.shape[0] * image.shape[1]

    mean = total / n_pixels
    std = np.sqrt(total_sq / n_pixels - mean**2)
    return mean.astype(np.float32), std.astype(np.float32)


class RetinaDataset(Dataset):
    """One split's images and labels, run through the OpenCV pipeline.

    Each item is (image, label): a float32 tensor of shape (3, 224, 224),
    normalized with the training split's mean and std, and an int label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        config: dict = PIPELINE_CONFIG,
        label_col: str = "label",
    ):
        self.paths = df["path"].tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.config = config

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = make_model_input(load_bgr(self.paths[index]), self.config)
        tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        return (tensor - self.mean) / self.std, self.labels[index]
