# Technical Report: screen-recapture-detector

**Project:** Real vs. Screen Photo Classifier  
**Solution:** A — Classical CV Pipeline  
**Date:** 2026-07

---

## 1. Problem Statement

Fraudulent onboarding flows submit photos of identity documents or faces
displayed on a screen, bypassing liveness and document-authenticity checks.
The goal is a lightweight binary classifier that:

- Runs entirely on CPU with no GPU dependency
- Processes one image in under 100 ms on a standard laptop
- Outputs a calibrated probability in `[0.0, 1.0]`
- Is deployable on-device, as a microservice, or as a serverless function

**Class labels:**  
`0` = real — a genuine camera photo  
`1` = screen — a photo of a screen, monitor, or printed display

---

## 2. Methodology

### 2.1 Preprocessing pipeline

Every image passes through an identical deterministic chain before feature
extraction.  All operations preserve `uint8` dtype to halve memory vs. early
`float32` casting.

```
load (BGR uint8)
  -> validate          grayscale/BGRA -> BGR; reject bad shapes
  -> letterbox resize  256x256, aspect-ratio preserving, black padding
  -> NL-means denoise  h=5, hColor=5 (conservative)
  -> CLAHE             LAB L-channel only, clip=2.0, tile=8x8
```

**Letterbox vs. squash resize.**  
Squash distorts aspect ratios, corrupting the perspective and screen-border
features (Harris corners, Hough lines, bezel edge ratios) that are most
discriminative for keystoned recaptures.  Letterboxing preserves geometry at
the cost of black padding, which is neutral for all feature groups.

**Conservative NL-means (h=5, not default 10).**  
Default h=10 smooths the dot-matrix regularity that LBP encodes.  h=5 removes
sensor noise while preserving the periodic pixel-grid texture unique to screens.

**CLAHE on LAB L-channel only.**  
Global histogram equalisation shifts hue and saturation, corrupting the
colour-gamut features (HSV-S, LAB A/B statistics) that distinguish backlit
screens from natural illumination.  Operating on L alone normalises brightness
without touching chrominance.

### 2.2 Feature engineering

The 193-dimensional float32 feature vector is built from 19 groups, each
targeting a specific physical artefact introduced by screen photography.

| Group | Dims | Physical artefact targeted |
|-------|------|---------------------------|
| Sharpness | 3 | Ringing / over-sharpening at screen pixel edges |
| FFT Frequency | 5 | Dot-matrix periodicity in the magnitude spectrum |
| Moiré Detection | 2 | Interference peaks between camera and display grids |
| Gradient Stats | 6 | Axis-aligned gradient bias from rectangular pixel layout |
| Gradient Orientation | 8 | Histogram skewed toward 0°/90° for screens |
| Edge Density | 2 | Bezel-induced step edges at image borders |
| LBP Texture | 26 | Pixel-grid regularity visible in micro-texture patterns |
| Entropy | 1 | Display quantisation reduces information content |
| Brightness | 3 | Backlight raises mean luminance and reduces variance |
| Contrast | 2 | Gamma compression by display narrows dynamic range |
| Saturation | 3 | Screen colour gamut narrower than natural scenes |
| HSV Statistics | 9 | Combined hue/saturation/value distribution shifts |
| LAB Statistics | 9 | Gamut compression visible in A/B channel spread |
| Color Histogram | 96 | Full colour distribution shape |
| Glare/Reflection | 4 | Specular highlights from backlight or ambient light |
| Screen Border | 3 | Bezel and frame geometry at image periphery |
| Perspective | 3 | Keystoning: Harris corners + Hough line convergence |
| Noise Stats | 4 | JPEG/display re-compression raises noise floor |
| Texture Stats | 4 | Energy/homogeneity shift from display smoothing |

**Total: 193 dimensions.**  All values are sanitised (NaN/Inf → 0.0) before
reaching the scaler, preventing silent corruption of downstream predictions.

### 2.3 Feature preprocessing

```
StandardScaler   fit on X_train only — prevents test-set statistics from
                 leaking into the transformation used at inference time.
                 Scaler is persisted to disk alongside the model.

SelectKBest      optional; disabled by default (feature_k = None).
(mutual info)    When enabled, selects top-k features by mutual information
                 with the class label, measured on the training set.
```

### 2.4 Classifiers evaluated

All four classifiers are tuned with `RandomizedSearchCV(n_iter=20)` under
`StratifiedKFold(n_splits=5)`.  Randomized search covers the hyperparameter
space at a fixed budget — 20 random combinations vs. the exponential cost of
grid search — with comparable empirical quality.

| Classifier | Key hyperparameters searched |
|------------|------------------------------|
| Random Forest | n_estimators, max_depth, min_samples_split, max_features, class_weight |
| Logistic Regression | C, solver, class_weight |
| SVM | C, kernel (RBF/linear), gamma, class_weight |
| XGBoost | n_estimators, max_depth, learning_rate, subsample, scale_pos_weight |

**Selection criterion: mean stratified-CV F1.**  
F1 is chosen over accuracy because it penalises both false positives (real
photos flagged as fakes) and false negatives (screen photos passed as genuine)
symmetrically, and is robust to class-count imbalance.

---

## 3. Pipeline architecture

```
data/real/   --+
               |-- load_dataset()
data/screen/ --+
                    |
               preprocess() x N          (letterbox, denoise, CLAHE)
                    |
               FeatureExtractor.extract() (193-dim float32 vector per image)
                    |
               StandardScaler.fit_transform(X_train)
                    |
               [SelectKBest — optional]
                    |
               RandomizedSearchCV x 4 classifiers
                    |
               select_best() by mean CV F1
                    |
               joblib.dump() -> models/model_a.joblib
                             -> models/scaler_a.joblib
                             -> models/feature_names.json
```

---

## 4. Evaluation methodology

The test set is the 20% held-out split from `train_test_split` with
`stratify=y, random_state=42`.  The same seed is used in both `train.py` and
`evaluate.py`, guaranteeing that the test set was never seen during training
or hyperparameter search.

### 4.1 Metrics reported

| Metric | Definition | Why it matters here |
|--------|-----------|---------------------|
| Accuracy | (TP+TN)/N | Baseline sanity check |
| Precision | TP/(TP+FP) | Cost of flagging real images as fake |
| Recall | TP/(TP+FN) | Cost of missing screen photos |
| F1 | Harmonic mean of precision and recall | Primary metric — balances both costs |
| ROC-AUC | Area under ROC curve | Threshold-independent ranking quality |

### 4.2 Plots generated

| Plot | Purpose |
|------|---------|
| Confusion matrix | Absolute TP/FP/TN/FN counts at threshold 0.5 |
| ROC curve | Sensitivity vs. 1-specificity trade-off across all thresholds |
| Precision-Recall curve | Precision vs. recall trade-off (better for imbalanced sets) |
| Calibration curve | Reliability diagram — does P(screen)=0.8 mean 80% are screens? |
| Feature importance | MDI (RF/XGBoost), |coef| (LR), or permutation importance (SVM) |

### 4.3 Threshold guidance

The default threshold of 0.50 is a starting point only.  In production:

- **High-security contexts** (document verification): lower threshold to 0.35
  — accept more false positives to catch all screen photos.
- **User-facing flows** (photo upload): raise threshold to 0.65 — reduce false
  alarms at the cost of passing some borderline screen photos.

Calibrate on a domain-representative validation set; do not tune on the test set.

---

## 5. Feature-group analysis

The visualisation pipeline (`python train.py`) generates four analytical plots:

**Feature distributions** — overlaid histograms per feature (real vs. screen)
ranked by Cohen's d. The top discriminating features are typically from the
FFT, Moiré, LBP, and Screen Border groups.

**Correlation matrix** — Pearson correlation heatmap for the top 40 features
by variance. High correlation within the Color Histogram group (96 dims) is
expected; SelectKBest can prune redundant features when enabled.

**PCA class separation** — 2-D projection of the standardised feature matrix.
If the two class clouds are linearly separable in PC1/PC2, the feature set
supports a high-accuracy linear classifier.

**Feature group importance** — importance summed by group and normalised to
100%. Gives an intuition for which physical signal matters most for the
trained model on the specific dataset.

---

## 6. Benchmark methodology

Latency is measured with `time.perf_counter` (monotonic, sub-microsecond
resolution) after `warmup_runs=3` discarded passes to eliminate cold-start
bias (Python/OpenCV/NumPy internal cache population).

`benchmark_runs=50` timed passes cycling through the image list give a stable
median and P95/P99.  Memory is captured via `tracemalloc` (Python-level
allocations) and `psutil` RSS (OS-level process memory).

**Cost estimates** use the *measured* median latency — not assumed numbers.
All figures exclude storage, data transfer, and free-tier credits.

### Typical latency breakdown (indicative)

| Step | Approx. share |
|------|--------------|
| `cv2.fastNlMeansDenoisingColored` | ~60% |
| `cv2.calcHist` × 3 channels | ~10% |
| LBP histogram (`skimage`) | ~10% |
| FFT + magnitude | ~8% |
| Remaining 15 feature groups | ~12% |

Replacing NL-means with a bilateral filter (`cv2.bilateralFilter`) reduces
preprocessing time by ~55% at a small LBP accuracy cost; appropriate for
latency-critical deployments where model F1 is secondary.

---

## 7. Limitations

**Dataset sensitivity.**  
Classical features are hand-tuned for the artefacts described in Section 2.2.
Novel screen types (e-ink, micro-LED, high-PPI OLED at close range) may not
exhibit the same FFT/LBP patterns.  Retrain when deploying to a new device
class.

**Minimum dataset size.**  
RandomizedSearchCV with 5-fold CV requires at least 50 samples per class to
produce stable fold splits.  Recommended minimum: 200 per class.  At < 100
per class, consider reducing `cv_folds` to 3 in `src/config.py`.

**Threshold is not calibrated by default.**  
The probability output is from Platt-scaled SVM or direct `predict_proba`
depending on the winning model.  A reliability diagram (`calibration_curve.png`)
should be inspected before setting a production threshold.

**No adversarial robustness.**  
A sophisticated attacker who knows the feature set can potentially craft
images that suppress Moiré patterns while retaining screen-camera geometry.
A deep-learning model (Solution B) is harder to reverse-engineer.

---

## 8. Solution B — MobileNetV3 (experimental path)

Solution A is the primary deliverable.  For reference, the experimental
upgrade path to Solution B is:

1. Install `torch` and `torchvision` (commented out in `requirements.txt`).
2. Implement `FeatureExtractorB` in `src/` using a frozen MobileNetV3-Small
   backbone; replace the 193-dim vector with the 576-dim global average pool.
3. Wire into `trainer.py` by swapping the extractor reference.

Expected benefit: MobileNetV3 learns screen artefacts implicitly from the
convolutional filters, generalising better to unseen screen types with
sufficient training data (> 5 000 images per class).  On small datasets
(< 500/class), Solution A is expected to match or outperform it.

---

## 9. Deployment recommendations

| Scenario | Recommended configuration |
|----------|--------------------------|
| Mobile liveness check | On-device inference; zero cloud cost; 50 MB RAM |
| REST API, variable load | Google Cloud Run or AWS Lambda; scales to zero |
| Batch document processing | EC2 c5.xlarge with `n_jobs=-1`; 4 parallel workers |
| High-throughput pipeline | Pre-extract features offline; inference is < 1 ms post-extraction |

**Artefact distribution:**  
The three model files (`model_a.joblib`, `scaler_a.joblib`, `feature_names.json`)
are all that is needed at inference time.  Total disk footprint is typically
2–5 MB (Random Forest) or < 1 MB (Logistic Regression / SVM).

---

## 10. Reproducibility

| Component | Fixed value |
|-----------|------------|
| Train/test split seed | `random_state=42` in `DataConfig` |
| RandomizedSearchCV seed | `random_state=42` in `ModelConfig` |
| StratifiedKFold seed | same |
| Feature extraction | stateless — no random operations |
| StandardScaler | deterministic given fixed train set |

Running `python train.py` twice on the same dataset produces bit-identical
model files.  The only non-determinism is `n_jobs=-1` thread scheduling
in scikit-learn's parallel CV, which does not affect the final selected
hyperparameters in practice.
