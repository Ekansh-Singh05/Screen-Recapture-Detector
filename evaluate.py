"""Evaluation entry point for screen-recapture-detector — Solution A.

Reconstructs the held-out test set using the same stratified split as
training (identical seed + test_size), then runs the full Evaluator suite.

Outputs (outputs/reports/ and outputs/plots/)
---------------------------------------------
metrics.json                  accuracy / precision / recall / F1 / ROC-AUC
classification_report.txt     scikit-learn text report
classification_report.json    machine-readable version
evaluation_summary.json       top-level summary with paths to all outputs
confusion_matrix.png
roc_curve.png
precision_recall_curve.png
calibration_curve.png
feature_importance.png

Usage::

    python evaluate.py [--log-level {DEBUG,INFO,WARNING}]
"""
from __future__ import annotations

import argparse
import logging
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import CFG
from src.evaluation import Evaluator
from src.feature_extractor import FeatureExtractor
from src.logger import setup as setup_logging, get_logger
from src.preprocessing import preprocess
from src.utils import ensure_dirs, load_dataset, load_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the trained screen-recapture-detector Solution-A model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return p.parse_args()


def _load_artefacts(log: logging.Logger):
    """Load model, scaler, selector, and feature names from disk.

    Returns:
        ``(model, scaler, selector, feature_names)``
        *selector* may be ``None`` if feature selection was not used.

    Exits with code 1 if required artefacts are missing.
    """
    cfg = CFG.model

    for path, name in [(cfg.model_a_path, "model"), (cfg.scaler_a_path, "scaler")]:
        if not path.exists():
            log.error("Required artefact not found: %s\nRun: python train.py", path)
            sys.exit(1)

    model  = joblib.load(cfg.model_a_path)
    scaler = joblib.load(cfg.scaler_a_path)

    selector = None
    if cfg.selector_a_path.exists():
        selector = joblib.load(cfg.selector_a_path)
        log.info("Loaded selector <- %s", cfg.selector_a_path)

    feature_names: list[str] = []
    if cfg.feature_names_path.exists():
        meta = load_json(cfg.feature_names_path)
        feature_names = meta.get("feature_names", [])

    log.info("Loaded model   <- %s", cfg.model_a_path)
    log.info("Loaded scaler  <- %s", cfg.scaler_a_path)
    return model, scaler, selector, feature_names


def _build_test_set(log: logging.Logger):
    """Extract features for every image and return the held-out test split.

    Uses the same ``test_size`` and ``random_state`` as ``train.py`` so
    the test set is identical to what was held out during training.

    Returns:
        ``(X_test_raw, y_test)`` — raw (unscaled) feature matrix and labels.

    Exits with code 1 if no valid images are found.
    """
    cfg = CFG.data
    try:
        paths, labels = load_dataset(cfg.real_dir, cfg.screen_dir)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    extractor = FeatureExtractor()
    vectors, valid_labels = [], []

    log.info("Extracting features from %d images ...", len(paths))
    for path, label in tqdm(zip(paths, labels), total=len(paths),
                            desc="Extracting", unit="img"):
        try:
            img = preprocess(path)
            vec = extractor.extract(img)
            vectors.append(vec)
            valid_labels.append(label)
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)

    if not vectors:
        log.error("No valid images found.  Check data/real/ and data/screen/.")
        sys.exit(1)

    X = np.vstack(vectors)
    y = np.array(valid_labels, dtype=np.int32)

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    log.info("Test set: %d samples  (%d real / %d screen)",
             len(y_test), (y_test == 0).sum(), (y_test == 1).sum())
    return X_test, y_test


def main() -> None:
    args = _parse_args()

    ensure_dirs(CFG.output.logs_dir, CFG.output.reports_dir, CFG.output.plots_dir)
    setup_logging(
        level=getattr(logging, args.log_level),
        log_dir=CFG.output.logs_dir,
        log_filename="evaluate.log",
    )
    log = get_logger(__name__)
    log.info("=" * 60)
    log.info("screen-recapture-detector  Solution A  Evaluation")
    log.info("=" * 60)

    model, scaler, selector, feature_names = _load_artefacts(log)
    X_test_raw, y_test = _build_test_set(log)

    evaluator = Evaluator()
    metrics = evaluator.run(model, scaler, selector, X_test_raw, y_test, feature_names)

    # ------------------------------------------------------------------
    # Human-readable summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("  Evaluation Results")
    print("=" * 40)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        val = metrics.get(key)
        if val is not None:
            print(f"  {key:<12s}  {val:.4f}")
    print("=" * 40)
    print(f"  Plots  -> {CFG.output.plots_dir}")
    print(f"  Report -> {CFG.output.reports_dir}")
    print("=" * 40 + "\n")

    log.info("Evaluation complete.  Run: python benchmark.py")


if __name__ == "__main__":
    main()
