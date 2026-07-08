"""Training entry point for screen-recapture-detector — Solution A.

Runs the complete pipeline:
  1. Extract 193-dim feature vectors from data/real/ and data/screen/
  2. Train four classifiers with RandomizedSearchCV (RF / LR / SVM / XGBoost)
  3. Select the best by mean stratified-CV F1
  4. Persist model, scaler, selector, feature_names.json to models/
  5. Generate feature-analysis visualisations (outputs/plots/)
  6. Save training summary JSON to outputs/reports/

Usage::

    python train.py [--no-viz] [--log-level {DEBUG,INFO,WARNING}]

Flags:
    --no-viz      Skip visualisation plots (faster, no matplotlib required).
    --log-level   Console log verbosity.  Default: INFO.
"""
from __future__ import annotations

import argparse
import logging

from src.logger import setup as setup_logging, get_logger
from src.config import CFG
from src.trainer import ModelTrainer
from src.utils import ensure_dirs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the screen-recapture-detector Solution-A classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip feature visualisation plots after training.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ensure_dirs(CFG.output.logs_dir)
    setup_logging(
        level=getattr(logging, args.log_level),
        log_dir=CFG.output.logs_dir,
        log_filename="train.log",
    )
    log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log.info("Starting Solution-A training pipeline.")
    trainer = ModelTrainer()
    summary = trainer.run()

    log.info("Training done.  Best model: %s", summary.get("best_model", "?"))

    # ------------------------------------------------------------------
    # Feature visualisation
    # ------------------------------------------------------------------
    if not args.no_viz:
        try:
            import numpy as np
            from src.visualization import FeatureAnalyzer
            from src.preprocessing import preprocess
            from src.utils import load_dataset

            log.info("Generating feature analysis plots ...")

            paths, labels = load_dataset(CFG.data.real_dir, CFG.data.screen_dir)
            vectors, valid_labels = [], []
            for path, label in zip(paths, labels):
                try:
                    img = preprocess(path)
                    vec = trainer.extractor.extract(img)
                    vectors.append(vec)
                    valid_labels.append(label)
                except Exception as exc:
                    log.debug("Skipping %s for viz: %s", path.name, exc)

            if vectors:
                X_raw = np.vstack(vectors)
                y_raw = np.array(valid_labels, dtype=np.int32)

                # Pull importances from the best model if available.
                importances = None
                if hasattr(trainer.best_model, "feature_importances_"):
                    imp = trainer.best_model.feature_importances_
                    if trainer.selector is not None:
                        # Expand back to full n_features-dim space (zeros for dropped features).
                        n_features = len(trainer.extractor.get_feature_names())
                        full = np.zeros(n_features, dtype=np.float64)
                        full[trainer.selector.get_support()] = imp
                        imp = full
                    importances = imp

                FeatureAnalyzer().run(
                    X_raw,
                    y_raw,
                    trainer.extractor.get_feature_names(),
                    importances,
                )
            else:
                log.warning("No valid images found for visualisation — skipping.")

        except Exception as exc:
            log.warning("Visualisation skipped due to error: %s", exc)
    else:
        log.info("--no-viz set: skipping feature visualisation.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log.info("All artefacts saved.  Run: python evaluate.py")


if __name__ == "__main__":
    main()
