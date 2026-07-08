"""Inference module for Solution A (classical CV pipeline).

Loads the trained artefacts once and exposes a clean prediction API.

Inference chain (mirrors training exactly)::

    image path
        -> preprocess()           BGR uint8, 256x256
        -> FeatureExtractor       float32 vector, shape (193,)
        -> StandardScaler         zero-mean unit-variance
        -> SelectKBest (optional) reduced dims if feature selection used
        -> model.predict_proba    probability of class 1 (screen)

Usage::

    from src.predictor import Predictor
    p = Predictor()
    prob = p.predict("photo.jpg")   # float in [0.0, 1.0]

predict.py calls ``src.logger.silence()`` before importing this module,
so no log output reaches stdout during command-line prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

from src.config import CFG
from src.feature_extractor import FeatureExtractor
from src.logger import get_logger
from src.preprocessing import preprocess
from src.utils import load_json

log = get_logger(__name__)


class Predictor:
    """Load trained artefacts and predict screen-capture probability.

    Artefacts are loaded once in ``__init__`` and reused across calls.
    This makes the class safe to instantiate once and call ``predict``
    many times without repeated disk I/O.

    Attributes:
        model: Fitted scikit-learn estimator.
        scaler: Fitted ``StandardScaler``.
        selector: Fitted ``SelectKBest``, or ``None`` if feature
            selection was not used during training.
        extractor: Stateless :class:`~src.feature_extractor.FeatureExtractor`.
        feature_names: Names of features after selection (informational).
    """

    def __init__(self) -> None:
        cfg = CFG.model
        self.model    = self._load(cfg.model_a_path,  "model")
        self.scaler   = self._load(cfg.scaler_a_path, "scaler")
        self.selector = self._load_optional(cfg.selector_a_path, "selector")
        self.extractor = FeatureExtractor()

        # Load feature names for logging / explainability.
        try:
            meta = load_json(cfg.feature_names_path)
            self.feature_names: List[str] = meta.get("feature_names", [])
            log.debug(
                "Loaded Predictor: model=%s  features=%d",
                meta.get("model_name", "?"),
                meta.get("n_features", "?"),
            )
        except FileNotFoundError:
            self.feature_names = []
            log.debug("feature_names.json not found — continuing without names.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str | Path) -> float:
        """Return the probability that *image_path* is a screen capture.

        Args:
            image_path: Path to any image supported by OpenCV.

        Returns:
            Float in ``[0.0, 1.0]``.  Values close to ``1.0`` indicate
            high confidence of a screen photo (class 1).  Values close
            to ``0.0`` indicate a genuine camera photo (class 0).

        Raises:
            FileNotFoundError: If the image does not exist.
            ValueError: If the image cannot be decoded.
        """
        vec    = self._extract(image_path)
        scaled = self._transform(vec)
        prob   = self._classify(scaled)
        log.debug("predict(%s) -> %.4f", Path(image_path).name, prob)
        return prob

    def predict_batch(
        self,
        image_paths: List[str | Path],
    ) -> List[float]:
        """Predict probabilities for a list of images.

        Failed images are assigned ``-1.0`` and a warning is logged.
        The return list has the same length and order as *image_paths*.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of floats.  ``-1.0`` marks prediction failures.
        """
        results: List[float] = []
        for path in image_paths:
            try:
                results.append(self.predict(path))
            except Exception as exc:
                log.warning("predict_batch: skipping %s — %s", path, exc)
                results.append(-1.0)
        return results

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _extract(self, image_path: str | Path) -> np.ndarray:
        """Preprocess image and extract raw feature vector.

        Args:
            image_path: Path to the source image.

        Returns:
            1-D float32 array of shape ``(193,)``.
        """
        img = preprocess(image_path)
        return self.extractor.extract(img)

    def _transform(self, vec: np.ndarray) -> np.ndarray:
        """Apply scaler (and optional selector) to a raw feature vector.

        Args:
            vec: Raw feature vector of shape ``(193,)``.

        Returns:
            Transformed 2-D array of shape ``(1, n_features)`` ready
            for ``model.predict_proba``.
        """
        x = self.scaler.transform(vec.reshape(1, -1))
        if self.selector is not None:
            x = self.selector.transform(x)
        return x

    def _classify(self, x: np.ndarray) -> float:
        """Run the classifier and return the class-1 probability.

        Handles both ``predict_proba`` (RF, LR, XGBoost, SVM with
        ``probability=True``) and ``decision_function`` (SVM without
        Platt scaling) via a sigmoid squash.

        Args:
            x: Transformed feature array of shape ``(1, n_features)``.

        Returns:
            Float in ``[0.0, 1.0]``.
        """
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(x)[0, 1])
        else:
            # Fallback: sigmoid of decision function score.
            score = float(self.model.decision_function(x)[0])
            prob  = float(1.0 / (1.0 + np.exp(-score)))

        # Clamp to [0, 1] — floating-point edge cases in Platt scaling
        # can produce values marginally outside this range.
        return float(np.clip(prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Artefact loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path, name: str) -> object:
        """Load a required Joblib artefact.

        Args:
            path: Path to the ``.joblib`` file.
            name: Human-readable name used in error messages.

        Returns:
            Deserialised object.

        Raises:
            FileNotFoundError: If *path* does not exist.
            RuntimeError: If Joblib cannot deserialise the file.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Trained {name} not found at:\n  {path}\n"
                "Run 'python train.py' first."
            )
        try:
            obj = joblib.load(path)
            log.debug("Loaded %s <- %s", name, path)
            return obj
        except Exception as exc:
            raise RuntimeError(f"Failed to load {name} from {path}: {exc}") from exc

    @staticmethod
    def _load_optional(path: Path, name: str) -> Optional[object]:
        """Load an optional Joblib artefact; return ``None`` if absent.

        Args:
            path: Path to the ``.joblib`` file.
            name: Human-readable name for debug logging.

        Returns:
            Deserialised object, or ``None`` if *path* does not exist.
        """
        if not path.exists():
            log.debug("Optional artefact %s not found at %s — skipping.", name, path)
            return None
        try:
            obj = joblib.load(path)
            log.debug("Loaded optional %s <- %s", name, path)
            return obj
        except Exception as exc:
            log.warning("Could not load optional %s from %s: %s", name, path, exc)
            return None
