"""Evaluation utilities for the screen-recapture-detector classifier.

Computes all classification metrics and generates every required plot.
All outputs are saved to ``outputs/plots/`` and ``outputs/reports/``.

No numbers are fabricated — every metric is derived from real
``model.predict`` / ``model.predict_proba`` calls on the held-out
test set passed in by the caller.

Generated artefacts
-------------------
Plots (outputs/plots/)
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
    calibration_curve.png
    feature_importance.png

Reports (outputs/reports/)
    metrics.json
    classification_report.json
    evaluation_summary.json

Usage::

    from src.evaluation import Evaluator
    ev = Evaluator()
    metrics = ev.run(model, scaler, selector, X_test_raw, y_test, feature_names)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on any server/OS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CFG
from src.logger import get_logger
from src.utils import ensure_dirs, save_json

log = get_logger(__name__)

# Class labels used in every plot.
_LABELS = ["Real (0)", "Screen (1)"]


class Evaluator:
    """Compute metrics and generate evaluation artefacts for a trained model.

    Args:
        plots_dir: Directory for PNG plots.  Defaults to
            ``CFG.output.plots_dir``.
        reports_dir: Directory for JSON reports.  Defaults to
            ``CFG.output.reports_dir``.
    """

    def __init__(
        self,
        plots_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
    ) -> None:
        self.plots_dir   = plots_dir   or CFG.output.plots_dir
        self.reports_dir = reports_dir or CFG.output.reports_dir
        ensure_dirs(self.plots_dir, self.reports_dir)

        # Apply a consistent plot style.
        try:
            plt.style.use(CFG.output.plot_style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model: Any,
        scaler: Any,
        selector: Optional[Any],
        X_test_raw: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Run the full evaluation suite and save all artefacts.

        Args:
            model: Fitted scikit-learn estimator.
            scaler: Fitted ``StandardScaler``.
            selector: Fitted ``SelectKBest``, or ``None``.
            X_test_raw: Raw (unscaled) test features, shape ``(N, 193)``.
            y_test: True test labels, shape ``(N,)``.
            feature_names: List of feature names after selection.

        Returns:
            Dict with keys ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``roc_auc``.
        """
        # Apply the same transform chain as training.
        X = scaler.transform(X_test_raw)
        if selector is not None:
            X = selector.transform(X)

        y_pred  = model.predict(X)
        y_prob  = self._get_probabilities(model, X)

        # 1. Scalar metrics.
        metrics = self._compute_metrics(y_test, y_pred, y_prob)

        # 2. Classification report.
        self._save_classification_report(y_test, y_pred)

        # 3. Plots.
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_prob)
        self.plot_precision_recall_curve(y_test, y_prob)
        self.plot_calibration_curve(y_test, y_prob)
        self.plot_feature_importance(model, X, y_test, feature_names)

        # 4. Full summary JSON.
        summary = {**metrics, "n_test": int(len(y_test))}
        save_json(summary, self.reports_dir / "evaluation_summary.json")
        log.info("Evaluation complete.  Artefacts in %s", self.plots_dir.parent)

        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """Compute and log the five standard binary-classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities for class 1.

        Returns:
            Dict with ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``roc_auc``.
        """
        metrics = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        }
        log.info(
            "Metrics — Accuracy=%.4f  Precision=%.4f  Recall=%.4f"
            "  F1=%.4f  ROC-AUC=%.4f",
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1"], metrics["roc_auc"],
        )
        save_json(metrics, self.reports_dir / "metrics.json")
        return metrics

    def _save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Save scikit-learn classification report as JSON and plain text.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
        """
        # Human-readable text version.
        report_str = classification_report(
            y_true, y_pred, target_names=["real", "screen"]
        )
        log.info("Classification report:\n%s", report_str)

        txt_path = self.reports_dir / "classification_report.txt"
        txt_path.write_text(report_str, encoding="utf-8")

        # Machine-readable JSON version.
        report_dict = classification_report(
            y_true, y_pred,
            target_names=["real", "screen"],
            output_dict=True,
        )
        save_json(report_dict, self.reports_dir / "classification_report.json")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """Save a colour-annotated confusion matrix.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            save_path: Override default save location.
        """
        save_path = save_path or self.plots_dir / "confusion_matrix.png"
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=_LABELS, yticklabels=_LABELS,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved confusion matrix -> %s", save_path)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """Save an ROC curve with AUC annotated.

        Args:
            y_true: Ground-truth labels.
            y_prob: Predicted probabilities for class 1.
            save_path: Override default save location.
        """
        save_path = save_path or self.plots_dir / "roc_curve.png"
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved ROC curve -> %s", save_path)

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """Save a Precision-Recall curve with AUC annotated.

        Why include this alongside ROC?
            ROC curves are optimistic on imbalanced datasets because the
            large number of true negatives keeps FPR low.  The PR curve
            focuses entirely on the positive class and better reflects
            performance when screen captures are rare.

        Args:
            y_true: Ground-truth labels.
            y_prob: Predicted probabilities for class 1.
            save_path: Override default save location.
        """
        save_path = save_path or self.plots_dir / "precision_recall_curve.png"
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        baseline = float(y_true.sum()) / len(y_true)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2, color="#4CAF50",
                label=f"PR curve (AUC = {pr_auc:.4f})")
        ax.axhline(baseline, color="gray", linestyle="--", lw=1,
                   label=f"Baseline (prevalence = {baseline:.2f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved PR curve -> %s", save_path)

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[Path] = None,
        n_bins: int = 10,
    ) -> None:
        """Save a calibration (reliability) curve.

        A well-calibrated model has points near the diagonal: when it
        predicts 70% probability, ~70% of those predictions are actually
        positive.  Poor calibration means the probability scores are not
        directly interpretable as probabilities.

        Args:
            y_true: Ground-truth labels.
            y_prob: Predicted probabilities for class 1.
            save_path: Override default save location.
            n_bins: Number of equally-spaced probability bins.
        """
        save_path = save_path or self.plots_dir / "calibration_curve.png"

        # Need at least 2 samples per bin; fall back to fewer bins.
        n_bins = min(n_bins, max(2, len(y_true) // 5))

        fraction_of_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        ax.plot(mean_pred, fraction_of_pos, "s-", lw=2, color="#FF5722",
                markersize=7, label="Model calibration")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        ax.set_ylabel("Fraction of positives", fontsize=11)
        ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13,
                     fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved calibration curve -> %s", save_path)

    def plot_feature_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        save_path: Optional[Path] = None,
        top_n: int = 30,
    ) -> None:
        """Save a horizontal bar chart of the top-N feature importances.

        Strategy by model type:
        * RandomForest / XGBoost: ``model.feature_importances_``
          (MDI — mean decrease in impurity).
        * LogisticRegression: ``|model.coef_[0]|`` (absolute weight).
        * Linear SVM: ``|model.coef_[0]|``.
        * RBF SVM (no direct importance): permutation importance on test
          set — adds a small evaluation cost but gives correct results.

        Args:
            model: Fitted estimator.
            X_test: Transformed test features.
            y_test: True test labels (needed for permutation importance).
            feature_names: List of feature name strings.
            save_path: Override default save location.
            top_n: Number of top features to show.
        """
        save_path = save_path or self.plots_dir / "feature_importance.png"

        importances, importance_label = self._get_importances(
            model, X_test, y_test
        )
        if importances is None:
            log.warning("No importances available for model type %s — skipping plot.",
                        type(model).__name__)
            return

        # Align with feature names (handles selector-reduced dimensions).
        n = min(len(importances), len(feature_names))
        importances = importances[:n]
        names       = feature_names[:n]

        # Select top-N.
        indices = np.argsort(importances)[-top_n:]
        top_names        = [names[i] for i in indices]
        top_importances  = importances[indices]

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
        colors = plt.cm.viridis(  # type: ignore[attr-defined]
            np.linspace(0.3, 0.9, len(indices))
        )
        bars = ax.barh(range(len(indices)), top_importances,
                       color=colors, edgecolor="none")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel(importance_label, fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=13,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved feature importance -> %s", save_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
        """Return class-1 probabilities, handling SVM without Platt scaling.

        Args:
            model: Fitted estimator.
            X: Transformed feature matrix.

        Returns:
            1-D array of probabilities for class 1.
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        # SVM decision_function -> sigmoid squash.
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    @staticmethod
    def _get_importances(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Extract feature importances appropriate to the model type.

        Returns:
            Tuple ``(importances_array, label_string)``.
            ``importances_array`` is ``None`` if no method is available.
        """
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_, "Importance (MDI)"

        if hasattr(model, "coef_"):
            coef = model.coef_
            imp  = np.abs(coef[0] if coef.ndim > 1 else coef)
            return imp, "Absolute Coefficient"

        # Fallback: permutation importance (model-agnostic, correct).
        log.info("Using permutation importance for %s (may be slow).",
                 type(model).__name__)
        try:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=5, random_state=CFG.model.random_state, n_jobs=1,
            )
            return result.importances_mean, "Permutation Importance"
        except Exception as exc:
            log.warning("Permutation importance failed: %s", exc)
            return None, ""
