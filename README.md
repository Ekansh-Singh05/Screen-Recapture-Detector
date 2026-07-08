# screen-recapture-detector

Binary classifier that distinguishes **genuine camera photos** (class 0) from
**photos of screens / recaptures** (class 1).

```
python predict.py photo.jpg
0.93
```

**Technical report:** [report.md](report.md) — methodology, accuracy, benchmark results, and deployment recommendations.

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Solution overview](#solution-overview)
3. [Project structure](#project-structure)
4. [Setup](#setup)
5. [Quickstart](#quickstart)
6. [Feature engineering](#feature-engineering)
7. [Training pipeline](#training-pipeline)
8. [Evaluation](#evaluation)
9. [Benchmarking](#benchmarking)
10. [Streamlit demo](#streamlit-demo)
11. [Design decisions](#design-decisions)
12. [Extending the project](#extending-the-project)
13. [Results](#results)
14. [License](#license)

---

## Problem statement

Screen recaptures (photographing a monitor, phone, or printed display) are used
to bypass liveness checks, forge documents, and spoof identity systems.
A fast, lightweight detector is needed that runs on CPU with no GPU dependency.

**Input**: any JPEG / PNG image  
**Output**: probability in `[0.0, 1.0]` that the image is a screen photo

---

## Solution overview

### Solution A — Classical CV pipeline (primary)

```
Image
  └─ Preprocessing        letterbox resize → NL-means denoise → CLAHE
       └─ Feature extraction  193-dim float32 vector (19 feature groups)
            └─ StandardScaler + optional SelectKBest
                 └─ Classifier  RF / LR / SVM / XGBoost
                      └─ Probability [0, 1]
```

The 193-dimensional feature vector captures the physical artefacts that
screen photos introduce: Moiré patterns (FFT), compression ringing (Laplacian),
dot-matrix regularity (LBP), specular glare (HSV), perspective distortion
(Harris / Hough), and colour-gamut narrowing (LAB statistics).

### Solution B — MobileNetV3 (experimental)

Fine-tuned MobileNetV3-Small head for comparison against Solution A.
See [Extending the project](#extending-the-project).

---

## Project structure

```
screen-recapture-detector/
├── data/
│   ├── real/           # class 0 — genuine camera photos
│   └── screen/         # class 1 — photos of screens / recaptures
├── models/             # saved artefacts (git-ignored)
├── outputs/
│   ├── plots/          # PNG figures
│   ├── reports/        # JSON reports
│   └── logs/           # rotating log files
├── src/
│   ├── logger.py       # logging setup — every module uses get_logger()
│   ├── config.py       # single source of truth for all constants (CFG)
│   ├── utils.py        # shared plumbing (paths, JSON, timing)
│   ├── preprocessing.py        # load → validate → letterbox → denoise → CLAHE
│   ├── feature_extractor.py    # 193-dim feature vector
│   ├── trainer.py      # RandomizedSearchCV × 4 classifiers
│   ├── predictor.py    # inference — loads artefacts once, reuses across calls
│   ├── evaluation.py   # metrics + 5 plots
│   ├── benchmark.py    # latency / memory / CPU / cost analysis
│   └── visualization.py        # feature distributions, correlation, PCA
├── train.py            # entry point: extract → train → save → visualise
├── predict.py          # entry point: print one float to stdout
├── evaluate.py         # entry point: reconstruct test set → full eval suite
├── benchmark.py        # entry point: latency + cost report
├── demo.py             # Streamlit UI
├── requirements.txt
└── .gitignore
```

---

## Setup

### Requirements

- Python 3.9+
- No GPU required — all inference runs on CPU

### Install dependencies

```bash
pip install -r requirements.txt
```

### Prepare data

Place images in the two class directories:

```
data/real/      ← genuine camera photos   (JPEG / PNG / BMP / TIFF / WebP)
data/screen/    ← photos of screens       (same formats)
```

A balanced dataset of ≥ 200 images per class is recommended.  
The 80/20 stratified split is handled automatically.

---

## Quickstart

```bash
# 1. Train
python train.py

# 2. Predict a single image
python predict.py path/to/image.jpg
# → 0.87

# 3. Full evaluation (metrics + plots)
python evaluate.py

# 4. Latency & cost benchmark
python benchmark.py

# 5. Interactive demo
streamlit run demo.py
```

---

## Feature engineering

The 193-dim vector is built from 19 hand-crafted groups, each targeting a
specific physical artefact of screen photography.

| # | Group | Dims | Physical signal |
|---|-------|------|----------------|
| 1 | Sharpness | 3 | Laplacian variance, Brenner, Tenengrad — screen photos show ringing at edges |
| 2 | FFT Frequency | 5 | Magnitude mean/std, high/low ratio, peak distance — dot-matrix regularity |
| 3 | Moiré Detection | 2 | Peak count (> 3σ), spectral flatness — interference between pixel grids |
| 4 | Gradient Stats | 6 | Sobel magnitude + X/Y channel statistics |
| 5 | Gradient Orientation | 8 | Weighted orientation histogram — screens skew toward axis-aligned gradients |
| 6 | Edge Density | 2 | Canny fraction, Sobel fraction |
| 7 | LBP Texture | 26 | Uniform LBP P=24 R=3 — dot matrix and pixel grid regularity |
| 8 | Entropy | 1 | Shannon entropy — screen photos have lower entropy |
| 9 | Brightness | 3 | HSV-V mean/std/median |
| 10 | Contrast | 2 | Gray std, RMS contrast |
| 11 | Saturation | 3 | HSV-S mean/std/P90 — colour gamut narrowed by screen |
| 12 | HSV Statistics | 9 | H/S/V × mean/std/skew |
| 13 | LAB Statistics | 9 | L/A/B × mean/std/skew — LAB gamut compression artefacts |
| 14 | Color Histogram | 96 | B/G/R × 32 bins normalised |
| 15 | Glare/Reflection | 4 | Pixel fraction, blob count, max area, specular index |
| 16 | Screen Border | 3 | Edge ratio, LR symmetry, corner contrast — bezel detection |
| 17 | Perspective | 3 | Harris corner count, border ratio, Hough lines — keystoning |
| 18 | Noise Stats | 4 | Mean/std/ratio/SNR via Gaussian residual |
| 19 | Texture Stats | 4 | Energy, homogeneity, coarseness, local contrast |

### Preprocessing chain

```
load (BGR uint8)
  → validate (gray/BGRA → BGR, bad shape check)
    → letterbox resize to 256×256 (aspect-ratio preserving, black pad)
      → NL-means denoise (h=5, conservative — preserves LBP micro-texture)
        → CLAHE on LAB L-channel (clip=2.0, tile=8×8 — avoids colour shift)
```

**Why letterbox, not squash?**  
Squash distorts aspect ratios, corrupting the perspective and border
features that detect keystoning and bezel geometry.

**Why NL-means with h=5 (not default 10)?**  
Default h=10 smooths away the dot-matrix regularity that LBP detects.
h=5 removes sensor noise while preserving screen-grid texture.

**Why CLAHE on LAB L-channel only?**  
Global histogram equalisation shifts hue and saturation, corrupting the
colour-gamut features (HSV-S, LAB A/B) that detect screen colour narrowing.

---

## Training pipeline

```
load_features()         preprocess + extract all images, shape (N, 193)
train_test_split()      80/20 stratified, seed=42
fit_preprocessors()     StandardScaler fit on X_train only (no leakage)
                        + optional SelectKBest (mutual information)
train_all_models()      RandomizedSearchCV(n_iter=20) × 4 classifiers
                            Random Forest, Logistic Regression, SVM, XGBoost
select_best()           highest mean stratified-CV F1
save()                  model_a.joblib, scaler_a.joblib, feature_names.json
```

**Why RandomizedSearchCV over GridSearchCV?**  
Grid search is O(∏|param_values|) — 4 parameters with 4 values each is
256 fits per model. Randomized search samples 20 combinations uniformly,
covering the space at a fixed compute budget with comparable result quality.

**Why F1 as the selection criterion?**  
F1 is robust to class imbalance. Accuracy rewards the majority-class bias;
F1 penalises both false positives and false negatives equally.

**Why StratifiedKFold?**  
Preserves class ratios in every fold — critical when real/screen counts differ.

---

## Evaluation

```bash
python evaluate.py
```

Outputs:

| File | Description |
|------|-------------|
| `outputs/reports/metrics.json` | accuracy, precision, recall, F1, ROC-AUC |
| `outputs/reports/classification_report.txt` | per-class precision/recall/F1 |
| `outputs/plots/confusion_matrix.png` | TP/FP/TN/FN counts |
| `outputs/plots/roc_curve.png` | ROC curve with AUC |
| `outputs/plots/precision_recall_curve.png` | PR curve with AP |
| `outputs/plots/calibration_curve.png` | Reliability diagram |
| `outputs/plots/feature_importance.png` | MDI / \|coef\| / permutation importance |

---

## Benchmarking

```bash
python benchmark.py [--n-images 50]
```

Measures end-to-end latency (warm-up runs excluded), peak memory, CPU
utilisation, and artefact disk size.  Generates cost estimates for six
deployment platforms using the *measured* median latency — not assumed numbers.
Results are saved to `outputs/reports/benchmark_report.json`.

| Platform | Notes |
|----------|-------|
| On-device | Zero cloud cost; ~50 MB RAM |
| AWS EC2 t3.medium | Single-threaded baseline |
| AWS EC2 c5.xlarge | 4 parallel workers |
| AWS Lambda 512 MB | Scales to zero; cold-start penalty |
| Google Cloud Run | 1 vCPU, 256 MB; REST API |
| Azure Functions 512 MB | Consumption plan |

---

## Streamlit demo

```bash
streamlit run demo.py
```

- Drag-and-drop any image
- Displays: verdict badge, probability gauge, feature-group breakdown bar chart
- Expandable table of all 193 raw feature values
- Requires trained model — shows a clear error with fix instructions if missing

---

## Design decisions

### No magic numbers anywhere
All constants (target size, LBP radius, CLAHE clip limit, CV folds, …) live
in `src/config.py` as a typed dataclass.  One place to change, zero hunt.

### StandardScaler fitted on train set only
Fitting the scaler on the full dataset leaks test-set mean/variance into
the transformation, inflating held-out scores.  The scaler is persisted to
disk so `predict.py` applies identical normalisation at inference time.

### `predict.py` stdout purity
`src.logger.silence()` is called before any project import.  The process
prints exactly one float — safe for shell piping and scripting.

### `uint8` throughout preprocessing
Keeping images as `uint8` until feature extraction halves RAM compared to
an early `float32` cast with zero accuracy benefit.

### NaN/Inf sanitisation in feature extractor
`_sanitise()` replaces any NaN or Inf in the feature vector with 0.0
before it reaches the scaler.  A single corrupt feature would otherwise
propagate silently through the entire pipeline.

### Calibration curve with small datasets
`n_bins = min(n_bins, max(2, len(y_true) // 5))` prevents a crash when
the test set is too small to fill the default 10 bins.

---

## Extending the project

### Adding a new feature group
1. Add the extraction logic as a private method in
   `src/feature_extractor.py` following the naming convention
   `_<group>_features(img) -> np.ndarray`.
2. Call it in `extract()` and append the result to `parts`.
3. Add corresponding names in `get_feature_names()` with the same
   `<group>_<detail>` prefix convention.
4. Update the dimension constant and the table in this README.

### Solution B — MobileNetV3
Uncomment the `torch` / `torchvision` lines in `requirements.txt`.
A `FeatureExtractorB` class using a frozen MobileNetV3-Small backbone
can be dropped into `src/` and wired into `trainer.py` by swapping
the extractor reference.

### Swapping the classifier
Replace the estimator in `trainer._build_candidates()`.  The rest of the
pipeline (CV, selection, persistence, inference) requires no changes.

---

## Results

After training, run `python evaluate.py` to generate a full metrics report.

| Metric | Description |
|--------|-------------|
| Accuracy | Overall fraction of correct predictions |
| Precision | Fraction of screen flags that are genuine screen photos |
| Recall | Fraction of screen photos correctly flagged |
| F1 | Harmonic mean of precision and recall (primary metric) |
| ROC-AUC | Threshold-independent ranking quality |

Results are saved to `outputs/reports/metrics.json` and five diagnostic plots
are written to `outputs/plots/`.  Reported numbers depend on the training
dataset; the target is ≥ 95% accuracy on held-out images.

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions about the approach or implementation, open a GitHub issue.
