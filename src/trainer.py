"""Training pipeline for Solution A (classical CV) classifier.

Responsibilities
----------------
1. Load image paths and labels from ``data/real/`` + ``data/screen/``.
2. Extract the 193-dim feature vector for every image.
3. Split into stratified train / test sets.
4. Fit a ``StandardScaler`` on the training features only.
5. Optionally apply mutual-information feature selection (SelectKBest).
6. Train four classifiers with ``RandomizedSearchCV``:
       Random Forest, Logistic Regression, SVM, XGBoost (optional).
7. Select the best model by mean stratified-CV F1 score.
8. Persist model, scaler, selector, and feature names via Joblib.
9. Save a JSON training report to ``outputs/reports/``.

Usage::

    python train.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

try:
    from xgboost import XGBClassifier  # type: ignore[import]
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.config import CFG
from src.feature_extractor import FeatureExtractor
from src.logger import get_logger
from src.preprocessing import preprocess
from src.utils import ensure_dirs, load_dataset, save_json

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hyper-parameter search spaces
# ---------------------------------------------------------------------------

_RF_PARAMS: Dict[str, List] = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3],
    "class_weight": [None, "balanced"],
}

_LR_PARAMS: Dict[str, List] = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"],
    "max_iter": [500],
}

_SVM_PARAMS: Dict[str, List] = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto", 0.001, 0.01],
    "class_weight": [None, "balanced"],
}

_XGB_PARAMS: Dict[str, List] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "scale_pos_weight": [1, 2, 5],
}


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Orchestrates the end-to-end Solution-A training pipeline.

    Attributes:
        extractor: Stateless :class:`~src.feature_extractor.FeatureExtractor`.
        scaler: ``StandardScaler`` fitted on training features.
        selector: ``SelectKBest`` fitted on training features, or ``None``
            if feature selection is disabled (``CFG.model.feature_k is None``).
        best_model: Fitted estimator with the highest mean CV F1.
        best_model_name: Human-readable name of the selected model.
        feature_names: Names of the features *after* selection (or all
            names if selection is disabled).
    """

    def __init__(self) -> None:
        self.extractor = FeatureExtractor()
        self.scaler: Optional[StandardScaler] = None
        self.selector: Optional[SelectKBest] = None
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Step 1 — Dataset loading
    # ------------------------------------------------------------------

    def load_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load images, run preprocessing + feature extraction for all images.

        Returns:
            ``(X, y)`` where *X* has shape ``(N, 193)`` (float32) and
            *y* has shape ``(N,)`` (int32, values 0 or 1).

        Raises:
            ValueError: If either class directory is empty.
        """
        cfg = CFG.data
        paths, labels = load_dataset(cfg.real_dir, cfg.screen_dir)

        vectors: List[np.ndarray] = []
        valid_labels: List[int] = []
        failed: List[str] = []

        log.info("Extracting features from %d images ...", len(paths))
        for path, label in tqdm(zip(paths, labels), total=len(paths),
                                desc="Extracting features", unit="img"):
            try:
                img = preprocess(path)
                vec = self.extractor.extract(img)
                vectors.append(vec)
                valid_labels.append(label)
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
                log.warning("Skipping %s — %s", path.name, exc)

        if failed:
            log.warning("%d image(s) skipped due to errors.", len(failed))

        if not vectors:
            raise ValueError(
                "Feature extraction produced no valid samples.  "
                "Check that data/real/ and data/screen/ contain valid images."
            )

        X = np.vstack(vectors)
        y = np.array(valid_labels, dtype=np.int32)
        log.info("Feature matrix: shape=%s  class balance: %d real / %d screen",
                 X.shape, (y == 0).sum(), (y == 1).sum())
        return X, y

    # ------------------------------------------------------------------
    # Step 2 — Preprocessing (scale + optional feature selection)
    # ------------------------------------------------------------------

    def fit_preprocessors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Fit StandardScaler and (optionally) SelectKBest on training data.

        Why fit only on training data?
            Fitting on the full dataset leaks test-set statistics into the
            scaler's mean/variance, inflating held-out evaluation scores.

        Args:
            X_train: Raw training features, shape ``(N_train, 193)``.
            y_train: Training labels.

        Returns:
            ``(X_train_processed, selected_feature_names)`` — the scaled
            (and optionally selected) training matrix plus name list.
        """
        all_names = self.extractor.get_feature_names()

        # Scale.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        log.info("StandardScaler fitted  mean~0, std~1 per feature")

        # Optional mutual-information feature selection.
        k = CFG.model.feature_k
        if k is not None:
            k = min(k, X_scaled.shape[1])
            log.info("Selecting top %d features by mutual information ...", k)
            self.selector = SelectKBest(mutual_info_classif, k=k)
            X_scaled = self.selector.fit_transform(X_scaled, y_train)
            mask = self.selector.get_support()
            selected_names = [n for n, m in zip(all_names, mask) if m]
            log.info("Feature selection: %d -> %d features", len(all_names), k)
        else:
            self.selector = None
            selected_names = all_names

        self.feature_names = selected_names
        return X_scaled, selected_names

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler (+ selector) to new data.

        Args:
            X: Raw feature matrix of shape ``(N, 193)``.

        Returns:
            Transformed matrix.

        Raises:
            RuntimeError: If called before :meth:`fit_preprocessors`.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted — call fit_preprocessors() first.")
        X_out = self.scaler.transform(X)
        if self.selector is not None:
            X_out = self.selector.transform(X_out)
        return X_out

    # ------------------------------------------------------------------
    # Step 3 — Model training
    # ------------------------------------------------------------------

    def _build_candidates(self) -> Dict[str, Tuple[Any, Dict]]:
        """Instantiate estimator objects and their search-space dicts.

        XGBoost is included only if the package is installed.

        Returns:
            ``{model_name: (estimator, param_dist)}`` mapping.
        """
        rs = CFG.model.random_state
        nj = CFG.model.n_jobs

        candidates: Dict[str, Tuple[Any, Dict]] = {
            "RandomForest": (
                RandomForestClassifier(random_state=rs, n_jobs=nj),
                _RF_PARAMS,
            ),
            "LogisticRegression": (
                LogisticRegression(random_state=rs, n_jobs=nj),
                _LR_PARAMS,
            ),
            "SVM": (
                SVC(probability=True, random_state=rs),
                _SVM_PARAMS,
            ),
        }

        if _XGBOOST_AVAILABLE:
            candidates["XGBoost"] = (
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=rs,
                    n_jobs=nj,
                    verbosity=0,
                ),
                _XGB_PARAMS,
            )
        else:
            log.warning(
                "XGBoost not installed — skipping.  "
                "Install with: pip install xgboost"
            )

        return candidates

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Dict]:
        """Run RandomizedSearchCV for every candidate model.

        Why RandomizedSearchCV instead of GridSearchCV?
            Grid search is O(prod(|param_values|)) — exponential.
            With 6 parameters each having 4 values that is 4^6 = 4096 fits.
            RandomizedSearch samples ``n_iter=20`` combinations, covering
            the space more efficiently at a fixed computation budget.

        Args:
            X_train: Scaled (and optionally selected) training features.
            y_train: Training labels.

        Returns:
            Dict mapping model name to a results dict containing:
            ``best_estimator``, ``mean_cv_f1``, ``std_cv_f1``,
            ``best_params``, ``fit_time_s``.
        """
        cfg = CFG.model
        cv = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )
        candidates = self._build_candidates()
        results: Dict[str, Dict] = {}

        for name, (estimator, param_dist) in candidates.items():
            log.info("--- Training %s ---", name)
            t0 = time.perf_counter()

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=cfg.n_search_iter,
                scoring="f1",
                cv=cv,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            # Full CV evaluation on the best hyper-params.
            best_est = search.best_estimator_
            cv_scores = cross_val_score(
                best_est, X_train, y_train,
                cv=cv, scoring="f1", n_jobs=cfg.n_jobs,
            )

            results[name] = {
                "best_estimator": best_est,
                "mean_cv_f1":     float(cv_scores.mean()),
                "std_cv_f1":      float(cv_scores.std()),
                "best_params":    search.best_params_,
                "cv_scores":      cv_scores.tolist(),
                "fit_time_s":     round(elapsed, 2),
            }

            log.info(
                "%s  mean_F1=%.4f (+/-%.4f)  best_params=%s  time=%.1fs",
                name, cv_scores.mean(), cv_scores.std(),
                search.best_params_, elapsed,
            )

        return results

    # ------------------------------------------------------------------
    # Step 4 — Model selection
    # ------------------------------------------------------------------

    def select_best(self, results: Dict[str, Dict]) -> None:
        """Set :attr:`best_model` and :attr:`best_model_name`.

        Selection criterion: highest mean cross-validation F1 score.
        F1 is preferred over accuracy because it is robust to class
        imbalance (a pure-accuracy metric would reward predicting the
        majority class for every sample).

        Args:
            results: Output of :meth:`train_all_models`.
        """
        best_name = max(results, key=lambda k: results[k]["mean_cv_f1"])
        self.best_model      = results[best_name]["best_estimator"]
        self.best_model_name = best_name
        log.info(
            "Best model: %s  (mean CV F1 = %.4f)",
            best_name, results[best_name]["mean_cv_f1"],
        )

    # ------------------------------------------------------------------
    # Step 5 — Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist model, scaler, selector, and feature names to disk.

        Raises:
            RuntimeError: If called before training is complete.
        """
        if self.best_model is None or self.scaler is None:
            raise RuntimeError("Nothing to save — run the full pipeline first.")

        cfg = CFG.model
        ensure_dirs(cfg.models_dir)

        joblib.dump(self.best_model, cfg.model_a_path)
        joblib.dump(self.scaler,     cfg.scaler_a_path)
        log.info("Saved model  -> %s", cfg.model_a_path)
        log.info("Saved scaler -> %s", cfg.scaler_a_path)

        if self.selector is not None:
            joblib.dump(self.selector, cfg.selector_a_path)
            log.info("Saved selector -> %s", cfg.selector_a_path)

        save_json(
            {
                "model_name":    self.best_model_name,
                "feature_names": self.feature_names,
                "n_features":    len(self.feature_names),
                "n_features_raw": len(self.extractor.get_feature_names()),
                "feature_selection_k": CFG.model.feature_k,
            },
            cfg.feature_names_path,
        )
        log.info("Saved feature names -> %s", cfg.feature_names_path)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the complete training pipeline end-to-end.

        Returns:
            Summary dict suitable for JSON serialisation, containing
            all model CV scores, best model name, and split sizes.
        """
        log.info("=" * 60)
        log.info("screen-recapture-detector  Solution A  Training Pipeline")
        log.info("=" * 60)

        # 1. Load features.
        X, y = self.load_features()

        # 2. Train / test split — stratified so class ratios are preserved.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CFG.data.test_size,
            random_state=CFG.data.random_state,
            stratify=y,
        )
        log.info("Split: %d train / %d test", len(X_train), len(X_test))

        # 3. Fit scaler (+ selector).
        X_train_proc, _ = self.fit_preprocessors(X_train, y_train)
        X_test_proc     = self.transform(X_test)

        # 4. Train all models.
        results = self.train_all_models(X_train_proc, y_train)

        # 5. Select best.
        self.select_best(results)

        # 6. Quick test-set evaluation (informational — full evaluation
        #    is delegated to evaluate.py).
        y_pred = self.best_model.predict(X_test_proc)
        from sklearn.metrics import classification_report
        report_str = classification_report(
            y_test, y_pred, target_names=["real", "screen"]
        )
        log.info("Test-set classification report:\n%s", report_str)

        # 7. Save artefacts.
        self.save()

        # 8. Build and save training summary.
        summary: Dict[str, Any] = {
            "best_model": self.best_model_name,
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "n_features_raw": int(X.shape[1]),
            "n_features_selected": int(X_train_proc.shape[1]),
            "models": {
                name: {
                    "mean_cv_f1": v["mean_cv_f1"],
                    "std_cv_f1":  v["std_cv_f1"],
                    "best_params": v["best_params"],
                    "fit_time_s":  v["fit_time_s"],
                }
                for name, v in results.items()
            },
        }
        ensure_dirs(CFG.output.reports_dir)
        save_json(summary, CFG.output.reports_dir / "training_summary.json")
        log.info("Training summary -> %s", CFG.output.reports_dir / "training_summary.json")
        log.info("Training complete.  Run: python evaluate.py")

        return summary
