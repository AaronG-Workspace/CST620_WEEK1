"""Load, deduplicate, label, and split the diabetic retinopathy dataset."""

import hashlib
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

DATASET_HANDLE = "sovitrath/diabetic-retinopathy-224x224-2019-data/versions/4"

# Folder in `colored_images/` for each 0-4 diagnosis grade.
GRADE_FOLDERS = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR",
}

BINARY_NAMES = {0: "No DR", 1: "DR"}

SEED = 42
SPLITS = ("train", "val", "test")


def download_dataset(handle: str = DATASET_HANDLE) -> Path:
    """Download the dataset, or reuse the kagglehub cache, and return its folder."""
    return Path(kagglehub.dataset_download(handle))


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Read train.csv and attach each image's file path.

    Returns one row per image (columns: id_code, grade, path), sorted by
    id_code so later steps don't depend on file order.
    """
    df = pd.read_csv(data_dir / "train.csv").rename(columns={"diagnosis": "grade"})
    image_dir = data_dir / "colored_images"
    df["path"] = [
        image_dir / GRADE_FOLDERS[grade] / f"{id_code}.png"
        for id_code, grade in zip(df["id_code"], df["grade"])
    ]

    missing = [path for path in df["path"] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images listed in train.csv are missing, e.g. {missing[0]}"
        )
    return df.sort_values("id_code", ignore_index=True)


def file_md5(path: Path) -> str:
    """Return the MD5 hex digest of a file's raw bytes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove byte-identical images, found by the MD5 hash of each file.

    - A group whose copies have different 0-4 grades is removed entirely,
      because its true label is unknown.
    - A group whose copies agree keeps its first row.

    Run this before splitting, so copies can't end up in different splits.
    Returns the cleaned DataFrame (with an md5 column) and summary counts.
    """
    df = df.assign(md5=[file_md5(path) for path in df["path"]])

    by_hash = df.groupby("md5")
    in_group = by_hash["id_code"].transform("size") > 1
    conflicting = by_hash["grade"].transform("nunique") > 1
    extra_copy = df.duplicated("md5", keep="first")

    keep = ~conflicting & ~extra_copy
    clean = df[keep].reset_index(drop=True)

    stats = {
        "images before": len(df),
        "duplicate groups": int(df.loc[in_group, "md5"].nunique()),
        "files in duplicate groups": int(in_group.sum()),
        "conflicting groups": int(df.loc[conflicting, "md5"].nunique()),
        "files in conflicting groups": int(conflicting.sum()),
        "files removed": int((~keep).sum()),
        "images after": len(clean),
    }
    return clean, stats


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary labels: grade 0 -> 0 (No DR), grades 1-4 -> 1 (DR)."""
    label = (df["grade"] > 0).astype(int)
    return df.assign(label=label, label_name=label.map(BINARY_NAMES))


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> pd.DataFrame:
    """Assign every row to train, val, or test, keeping class proportions.

    Each class is shuffled with a seeded generator. Validation and test sizes
    are rounded per class, and train gets the rest, so no row is dropped.

    Returns a copy of df with a `split` column.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)

    for label in sorted(df[label_col].unique()):
        rows = rng.permutation(df.index[df[label_col] == label])
        n_val = round(len(rows) * val_frac)
        n_test = round(len(rows) * test_frac)
        split.loc[rows[:n_val]] = "val"
        split.loc[rows[n_val : n_val + n_test]] = "test"

    return df.assign(split=split)
