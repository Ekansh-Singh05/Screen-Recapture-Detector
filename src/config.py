"""Central configuration for the screen-recapture-detector pipeline.

All paths, hyperparameters, and constants live here.  No other module
should contain magic numbers or hardcoded paths.

Usage::

    from src.config import CFG

    img = cv2.resize(img, CFG.preprocessing.target_size)

Design rationale
----------------
* ``dataclass`` with ``field(default_factory=...)`` avoids Python's
  mutable-default-argument trap when fields hold ``Path`` objects.
* Nested sub-configs by concern (data, preprocessing, features, model,
  output) follow the Single Responsibility Principle — changes to model
  paths never touch preprocessing constants.
* One module-level singleton ``CFG`` is imported by every other module.
  Re-instantiation is never needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Absolute anchor — resolves correctly regardless of CWD at runtime.
# ---------------------------------------------------------------------------
_ROOT: Path = Path(__file__).resolve().parent.parent


# ===========================================================================
# Sub-configurations
# ===========================================================================

@dataclass
class DataConfig:
    """Paths to raw data directories and dataset split parameters."""

    # Class directories — the loader scans these for images.
    real_dir: Path = field(default_factory=lambda: _ROOT / "data" / "real")
    screen_dir: Path = field(default_factory=lambda: _ROOT / "data" / "screen")

    # 80/20 split is the conventional default for datasets < 10 k images.
    test_size: float = 0.20

    # Fixed seed ensures reproducible splits across runs.
    random_state: int = 42

    # Supported image extensions (lower-case).
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


@dataclass
class PreprocessingConfig:
    """Image pre-processing hyperparameters."""

    # 256×256 is a deliberate trade-off: large enough to preserve Moiré
    # patterns and LBP textures; small enough for < 5 ms preprocessing.
    # (width, height) — matches cv2.resize convention.
    target_size: Tuple[int, int] = (256, 256)

    # Padding colour for letterboxing (black — neutral for feature stats).
    pad_value: int = 0

    # Non-local means denoising — conservative parameters to avoid
    # destroying the micro-texture LBP depends on.
    denoise_h: int = 5           # luminance filter strength
    denoise_h_color: int = 5     # colour filter strength
    denoise_template_win: int = 7
    denoise_search_win: int = 21

    # CLAHE — applied to L channel of LAB to avoid colour shift.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Toggle steps — useful during ablation studies.
    apply_denoise: bool = True
    apply_clahe: bool = True


@dataclass
class FeatureConfig:
    """Feature extraction hyperparameters."""

    # --- LBP ---
    # radius=3 captures 3-pixel neighbourhood; 8*radius = 24 sample points
    # (uniform method gives 26 stable, rotation-invariant patterns).
    lbp_radius: int = 3
    lbp_n_points: int = 24      # must equal 8 * lbp_radius
    lbp_method: str = "uniform"  # rotation-invariant, sparse histogram

    # --- Color histogram ---
    # 32 bins per channel balances resolution vs. dimensionality.
    hist_bins: int = 32

    # --- Glare / reflection ---
    # 240/255 ≈ 94% brightness — isolates true specular highlights.
    glare_threshold: int = 240

    # --- Screen border ---
    # 20-pixel margin ≈ 7.8% of 256 px — enough to capture bezel edges.
    border_margin: int = 20

    # --- Gradient orientation histogram ---
    # 8 bins = 45° per bin — captures axis-aligned (screen) vs. isotropic
    # (natural scene) gradient distributions.
    grad_orient_bins: int = 8

    # --- FFT high-frequency split ---
    # Fraction of the spectrum considered "high frequency".
    # 0.3 means the outer 70% of the magnitude spectrum is high-freq.
    fft_high_freq_frac: float = 0.3

    # --- Noise statistics ---
    # Estimated via difference between image and its Gaussian-blurred version.
    noise_blur_kernel: Tuple[int, int] = (5, 5)

    # --- Perspective distortion ---
    # Harris corner detector parameters.
    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04


@dataclass
class ModelConfig:
    """ML model paths and training hyperparameters."""

    models_dir: Path = field(default_factory=lambda: _ROOT / "models")

    # Solution A — classical CV pipeline
    model_a_path: Path = field(default_factory=lambda: _ROOT / "models" / "model_a.joblib")
    scaler_a_path: Path = field(default_factory=lambda: _ROOT / "models" / "scaler_a.joblib")
    selector_a_path: Path = field(default_factory=lambda: _ROOT / "models" / "selector_a.joblib")
    feature_names_path: Path = field(default_factory=lambda: _ROOT / "models" / "feature_names.json")

    # Cross-validation
    cv_folds: int = 5
    n_jobs: int = -1             # use all available CPU cores
    random_state: int = 42

    # RandomizedSearchCV — 20 iterations is enough to explore the space
    # without the O(n²) cost of full GridSearchCV.
    n_search_iter: int = 20

    # Feature selection — keep top-k features by mutual information.
    # None = disable selection (use all features).
    feature_k: int | None = None


@dataclass
class OutputConfig:
    """Paths for all generated artefacts."""

    outputs_dir: Path = field(default_factory=lambda: _ROOT / "outputs")
    plots_dir: Path = field(default_factory=lambda: _ROOT / "outputs" / "plots")
    reports_dir: Path = field(default_factory=lambda: _ROOT / "outputs" / "reports")
    logs_dir: Path = field(default_factory=lambda: _ROOT / "outputs" / "logs")

    # Plot resolution — 150 dpi is sharp enough for reports without
    # producing multi-MB PNG files.
    plot_dpi: int = 150
    plot_style: str = "seaborn-v0_8-whitegrid"


@dataclass
class BenchmarkConfig:
    """Benchmarking parameters."""

    # Number of warm-up passes before timing (to avoid cold-start bias).
    warmup_runs: int = 3

    # Number of timed passes — enough for stable median / P95.
    benchmark_runs: int = 50

    # Cost analysis — latency assumption used when real measurement
    # is unavailable (e.g. cost report generated without hardware).
    assumed_latency_ms: float = 10.0

    n_images_cost: int = 1_000_000  # cost per 1 M images


# ===========================================================================
# Root singleton
# ===========================================================================

@dataclass
class Config:
    """Aggregated application configuration.

    Import and use as::

        from src.config import CFG

        print(CFG.model.cv_folds)   # 5
        print(CFG.data.real_dir)    # .../data/real
    """

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


# Module-level singleton — the one true config object.
CFG: Config = Config()
