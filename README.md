# Classical Pipeline to CNN/ViT Selector

**CST 620-100 · Computer Vision — Week 01 Assignment**

A decision-support tool that compares a classical OpenCV preprocessing pipeline with CNN and Vision Transformer (ViT) models on a retinal image classification task. It then uses the measured results to recommend an architecture.

## Scenario

A medical imaging startup is starting a new retinal scan classification task. As the team's junior computer vision engineer, the engineering manager has asked for a lightweight tool to quickly compare classical OpenCV-based pipelines against CNN and Vision Transformer architectures. The final recommendation has to account for the startup's data volume and deployment constraints.

## Objectives

1. **Preprocessing pipeline.** A Python/OpenCV pipeline that performs:
   - grayscale conversion
   - histogram equalization (e.g., CLAHE)
   - edge detection (e.g., Canny)
   - morphological operations

   It must handle color spaces correctly. OpenCV loads images as BGR, while PIL and torchvision return RGB.

   **Operation order and parameters:** _TBD. Document the chosen order and why each step comes where it does._

2. **Comparison notebook.** A Jupyter notebook that:
   - loads a publicly available medical or natural image dataset from Kaggle or TensorFlow Datasets
   - trains and evaluates at least two architecture families in PyTorch (e.g., a small CNN and a ViT variant such as DeiT-tiny from `timm`)
   - outputs a comparison table of **accuracy**, **parameter count**, and **inference latency**

3. **Architecture recommendation.** A justified choice based on the measured metrics, the startup's data volume, and its deployment constraints.

## Project structure

```
cst620_week1/
├── config/      # configuration (pipeline and training parameters)
├── data/        # optional local data files (ignored by git)
├── src/         # OpenCV preprocessing pipeline and model code
├── tests/       # tests
└── README.md
```

The dataset is stored in the kagglehub cache, not in the project folder (see [Loading](#loading)). The comparison notebook will be added as the project is built.

## Setup

Requires Python 3.10 or newer.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Install packages and launch Jupyter** from the project root:

```bash
pip install opencv-python torch torchvision timm tensorflow-datasets numpy pandas matplotlib jupyter kaggle kagglehub
jupyter notebook
```

## Dataset

The dataset must be found and loaded independently. Course-provided datasets may not be used.

**Chosen dataset:** [Diabetic Retinopathy 224x224 (2019 Data)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data), published on Kaggle by `sovitrath` (version 4).

- **Contents:** 3,662 retinal fundus photographs saved as 224×224 RGB PNG files (239 MB), plus `train.csv` with `id_code` and `diagnosis` columns.
- **Origin:** The image count and CSV columns match the APTOS 2019 Blindness Detection training set. This dataset appears to be a resized copy of it.
- **Labels:** 5 diabetic retinopathy grades. Each image's folder name agrees with its CSV label.

| `diagnosis` | Folder | Images |
|---|---|---|
| 0 | `No_DR` | 1,805 |
| 1 | `Mild` | 370 |
| 2 | `Moderate` | 999 |
| 3 | `Severe` | 193 |
| 4 | `Proliferate_DR` | 295 |

**Label mapping for this project:** _TBD. Either 5-class, or binary (No DR vs. DR)._

**Why this dataset:** The assignment suggests `retinopathy_detection`, whose TFDS catalog name is `diabetic_retinopathy_detection`. That dataset requires a manual download of the Kaggle Diabetic Retinopathy Detection competition files, and its full-resolution version is about 89 GiB. This dataset is public, 239 MB, and downloads directly through the Kaggle API.

### Loading

Load the dataset with `kagglehub`, pinned to version 4 so results are reproducible:

```python
import kagglehub

data_dir = kagglehub.dataset_download(
    "sovitrath/diabetic-retinopathy-224x224-2019-data/versions/4"
)
```

The first call downloads the files to the kagglehub cache (`~/.cache/kagglehub/`). Later calls return the cached copy without downloading again.

Public datasets like this one don't require a Kaggle login. Other Kaggle API features do: sign in with `kaggle auth login`, or see the [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) for token-based options. **Never commit an API token.** `.gitignore` excludes `kaggle.json`, `access_token`, and `.env`.

### Data quality checks

These checks were run on the downloaded files (version 4):

- **Complete and readable.** Every image has a CSV row and every CSV row has an image. All 3,662 images open correctly and are 224×224 with 3 channels.
- **Exact duplicates.** 123 groups of byte-identical images (251 files). In 93 groups, every copy has the same label. In the other 30 groups (62 files), the copies have different labels.
- **No patient IDs.** Each `id_code` identifies an image, not a patient, so images can't be grouped by patient.
- **Class imbalance.** 49.3% of images are `No_DR`. A 5-class model that always predicts `No_DR` would score about 49% accuracy.

**Planned duplicate handling (not yet implemented):** Before splitting, keep one copy from each group whose copies share a label, and remove every group whose copies disagree. This removes 158 files and leaves 3,504 unique, consistently labeled images. These counts use the 5-class labels. If the project switches to binary labels, fewer groups will disagree, so the check needs to be re-run.

## Evaluation protocol

These steps, in this order, keep the evaluation leakage-free:

1. **Remove duplicates before splitting.** This has to come first. After a split, copies of the same image could already be in different splits. The step compares file hashes and labels; no statistics are fitted.
2. **Stratified train / validation / test split** (e.g., 70/15/15) with a fixed random seed, so each split keeps the same class proportions. The smallest class (`Severe`) has fewer than 200 images, so a purely random split could leave very few in the test set.
3. **Per-image OpenCV preprocessing.** Each OpenCV operation uses only the image it is processing, so no information passes between splits.
4. **Fit on training data only.** Any parameter learned from data (e.g., normalization mean and standard deviation) is computed on the training split and applied unchanged to validation and test. Fixed values, such as the ImageNet statistics used with a pretrained ViT, don't leak either.
5. **Test once.** Choose models with the validation split. The test split is used only for the final reported metrics.

Every model uses the same splits and the same preprocessing (matched data). Accuracy, parameter count, and latency all come from this project's own runs, not from published numbers.

## Results

| Model | Family | Test accuracy | Parameters | Mean inference latency (ms) |
|---|---|---|---|---|
| Majority-class baseline (always predicts the most common class) | — | — | — | — |
| _TBD_ (small CNN) | CNN | — | — | — |
| _TBD_ (e.g., DeiT-tiny) | ViT | — | — | — |

_Latency measurement settings (device, batch size, warm-up runs, number of timed runs): TBD._

## Architecture recommendation

_TBD. Write this once results are in. Base the choice on the measured numbers above, and name the data-volume and deployment constraints that drive it._

## Built with

[Claude Code](https://claude.com/claude-code) · OpenCV · PyTorch · timm · kagglehub · pandas · matplotlib · Jupyter
