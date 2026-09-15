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

   **Operation order and parameters.** All settings live in one dictionary, `PIPELINE_CONFIG`, in [`src/preprocess.py`](src/preprocess.py).

   | Step | Operation | Settings | Why it comes here |
   |---|---|---|---|
   | 1 | Grayscale | `cv2.COLOR_BGR2GRAY` | `cv2.imread` returns BGR, so the BGR conversion code is required. |
   | 2 | CLAHE | clip limit 2.0, 8×8 tiles | Boosts local contrast so faint vessels and lesions stand out. |
   | 3 | Gaussian blur | 5×5 kernel, sigma derived from kernel size | Smooths the noise CLAHE amplifies, which Canny would otherwise mark as edges. |
   | 4 | Canny edges | low 40, high 120 | Runs on the smoothed, contrast-enhanced image. |
   | 5 | Morphological closing | 3×3 kernel | Bridges 1-pixel gaps in the edge lines. |

   **Canny thresholds** were chosen by measuring gradient strength inside the fundus circle on 200 training images after CLAHE and blur. The low threshold (40) is just above the 75th percentile, so background texture is ignored. The high threshold (120) is between the 95th and 98th percentiles, so edges only start at strong boundaries: vessels, the optic disc, lesions, and the image rim. The 1:3 ratio is within Canny's recommended range of 1:2 to 1:3.

   **Model input.** The CLAHE image is copied to 3 channels, resized to 224×224, scaled to 0–1, and normalized with the training split's mean (0.312) and standard deviation (0.191). The blur, Canny, and closing stages are shown in the notebook for inspection; the models don't use them.

2. **Comparison notebook.** A Jupyter notebook that:
   - loads a publicly available medical or natural image dataset from Kaggle or TensorFlow Datasets
   - trains and evaluates at least two architecture families in PyTorch (e.g., a small CNN and a ViT variant such as DeiT-tiny from `timm`)
   - outputs a comparison table of **accuracy**, **parameter count**, and **inference latency**

3. **Architecture recommendation.** A justified choice based on the measured metrics, the startup's data volume, and its deployment constraints.

## Project structure

```
cst620_week1/
├── comparison.ipynb    # data prep, pipeline stages, training, comparison table
├── config/             # reserved for configuration files (empty for now)
├── data/               # optional local data files (ignored by git)
├── src/
│   ├── data.py         # loading, duplicate removal, binary labels, stratified split
│   ├── preprocess.py   # OpenCV pipeline and PyTorch Dataset
│   ├── models.py       # small CNN and DeiT-tiny
│   ├── training.py     # training loop and frozen-feature extraction
│   └── benchmark.py    # test metrics, model size, CPU latency
├── tests/              # tests (none yet)
└── README.md
```

The dataset is stored in the kagglehub cache, not in the project folder (see [Loading](#loading)). Run `comparison.ipynb` from top to bottom. A full run takes about 11 minutes on the development machine's CPU.

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

**Label mapping for this project:** binary. Grade 0 is No DR (label 0), and grades 1–4 are DR (label 1). After duplicate removal, there are 1,796 No DR images (51.3%) and 1,708 DR images (48.7%).

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

**Duplicate handling** ([`src/data.py`](src/data.py)): Before splitting, files are grouped by MD5 hash. Each group whose copies share a grade keeps one copy, and each group whose copies disagree is removed. This removes 158 files and leaves 3,504 unique, consistently labeled images. Disagreement is judged on the 0–4 grades, not the binary labels, so a group graded 1 and 2 is removed even though both copies are DR. This is the stricter choice.

## Evaluation protocol

These steps, in this order, keep the evaluation leakage-free:

1. **Remove duplicates before splitting.** This has to come first. After a split, copies of the same image could already be in different splits. The step compares file hashes and labels; no statistics are fitted.
2. **Stratified train / validation / test split** (70/15/15, seed 42), so each split keeps the same class proportions. The result is 2,454 / 525 / 525 images, with 48.7–48.8% DR in each split. The notebook checks that no image ID or file hash appears in more than one split.
3. **Per-image OpenCV preprocessing.** Each OpenCV operation uses only the image it is processing, so no information passes between splits.
4. **Fit on training data only.** Any parameter learned from data (e.g., normalization mean and standard deviation) is computed on the training split and applied unchanged to validation and test. Both models, including the pretrained DeiT-tiny, use the training split's statistics instead of ImageNet statistics, so they receive identical inputs.
5. **Test once.** Choose models with the validation split. The test split is used only for the final reported metrics.

Every model uses the same splits and the same preprocessing (matched data). Accuracy, parameter count, and latency all come from this project's own runs, not from published numbers.

## Results

Measured by [`comparison.ipynb`](comparison.ipynb) on the test split (525 images, 256 with DR), which was used once.

| Model | Family | Test accuracy | Sensitivity (DR recall) | Parameters | Size (MB) | CPU latency, batch of 32 (ms) |
|---|---|---|---|---|---|---|
| Majority-class baseline (always predicts the most common training label) | — | 0.512 | 0.000 | 0 | 0.00 | — |
| Small CNN (3 conv layers, trained from scratch) | CNN | 0.886 | 0.906 | 32,162 | 0.13 | 275.9 ± 35.9 |
| DeiT-tiny (`deit_tiny_patch16_224`, frozen pretrained backbone, trained head) | ViT | 0.966 | 0.945 | 5,524,802 | 21.13 | 1138.2 ± 97.6 |

**Training settings:** CPU, batch size 32, `num_workers=0`, Adam (learning rate 0.001), cross-entropy loss. The small CNN trained for 5 epochs. For DeiT-tiny, the frozen backbone's features were computed once, and only the head (386 parameters) trained for 20 epochs. Each model keeps the epoch with the best validation accuracy: epoch 5 for both (small CNN 0.890, DeiT-tiny 0.960).

**Latency settings:** the development machine's Intel CPU (Intel64 Family 6 Model 170) with 12 PyTorch threads; one batch of 32 test images; eval mode with no gradients; 3 warm-up runs, then the mean ± standard deviation of 10 timed runs. DeiT-tiny is timed as the full model on images, not on cached features. Times cover the model's forward pass only, not image loading or OpenCV preprocessing. Size is the saved `state_dict`, where 1 MB = 1024² bytes.

## Architecture recommendation

**Recommendation: DeiT-tiny with a frozen backbone**, for a startup with about 3,500 images and CPU-only clinic computers.

- **Best accuracy.** Test accuracy is 0.966, vs. 0.886 for the small CNN and 0.512 for the majority-class baseline.
- **Fewer missed DR cases**, which is the costliest error in screening. Sensitivity is 0.945 vs. 0.906 for the CNN: a 5.5% miss rate vs. 9.4%.
- **Workable CPU speed.** A batch of 32 takes 1138.2 ± 97.6 ms, about 36 ms per image, so even a full batch finishes in about a second.
- **Size isn't a barrier.** Its 5,524,802 parameters take 21.13 MB, which fits easily on a clinic computer. With about 3,500 images, the from-scratch CNN (32,162 parameters) still scored 8 points lower in accuracy.
- **Keep the small CNN as a backup for very slow machines.** It's about 4× faster (275.9 ± 35.9 ms per batch) and only 0.13 MB, but gives up accuracy (0.886) and sensitivity (0.906). Time both models on an actual clinic computer before deciding.

### Why this comparison could be misleading

1. **It's not a fair CNN-vs-ViT contest.** DeiT-tiny started from ImageNet pretraining, while the CNN learned from scratch. The CNN also got only 5 epochs and was still improving; its best epoch was the last one. So the 8-point gap mixes the architecture with pretraining and training time. A fair match would be a pretrained CNN (e.g., ResNet-18) with the same frozen-backbone setup.
2. **A single small test set makes some gaps look more certain than they are.** The test set has 525 images (256 with DR), from one split and one seed. Rough 95% confidence intervals overlap for sensitivity (DeiT-tiny 0.92–0.97, CNN 0.87–0.94), so DeiT-tiny's sensitivity advantage may be partly noise. A paired test on the same test images (McNemar's) would settle it. The accuracy intervals don't overlap (0.95–0.98 vs. 0.86–0.91), so that gap is more reliable.

## Built with

[Claude Code](https://claude.com/claude-code) · OpenCV · PyTorch · timm · kagglehub · pandas · matplotlib · Jupyter
