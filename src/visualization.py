"""Feature and dataset visualisation for screen-recapture-detector.

Generates four analytical plots that help understand the dataset and
the discriminative power of the hand-crafted feature set.

These plots are about the **data and features**, not model performance.
Model-performance plots (confusion matrix, ROC, etc.) live in
``src/evaluation.py``.

Generated plots (outputs/plots/)
----------------------------------
feature_distributions.png
    Overlaid histograms (real vs screen) for the top-20 most
    discriminative features.

correlation_matrix.png
    Feature–feature Pearson correlation heatmap for the top-40
    features by variance.  Identifies redundant feature pairs.

class_separation_pca.png
    2-D PCA projection coloured by class label.  Visual check of
    whether the feature space is linearly separable.

feature_group_importance.png
    Mean feature importance aggregated by the 19 named feature groups
    (Sharpness, FFT, LBP, …).  More actionable than a 193-bar chart.

Usage::

    from src.visualization import FeatureAnalyzer
    analyzer = FeatureAnalyzer()
    analyzer.run(X, y, feature_names, importances)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.logger import get_logger
from src.utils import ensure_dirs

log = get_logger(__name__)

# Colour palette — consistent across all plots.
_PALETTE = {"real": "#2196F3", "screen": "#F44336"}
_CLASS_NAMES = {0: "real", 1: "screen"}


class FeatureAnalyzer:
    """Generate feature and dataset visualisation plots.

    Args:
        plots_dir: Output directory for PNG files.  Defaults to
            ``CFG.output.plots_dir``.
    """

    def __init__(self, plots_dir: Optional[Path] = None) -> None:
        self.plots_dir = plots_dir or CFG.output.plots_dir
        ensure_dirs(self.plots_dir)
        try:
            plt.style.use(CFG.output.plot_style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        importances: Optional[np.ndarray] = None,
    ) -> None:
        """Generate and save all four analysis plots.

        Args:
            X: Feature matrix of shape ``(N, D)``.  Should be the raw
                (unscaled) feature values for interpretable axis labels.
            y: Class labels of shape ``(N,)``.  Values 0 (real) or 1 (screen).
            feature_names: List of D feature name strings.
            importances: Optional 1-D array of D importance scores from
                the trained model.  If ``None``, variance is used as a
                proxy for feature importance.
        """
        log.info("Generating feature analysis plots (%d samples, %d features) ...",
                 X.shape[0], X.shape[1])

        self.plot_feature_distributions(X, y, feature_names, importances)
        self.plot_correlation_matrix(X, feature_names)
        self.plot_class_separation_pca(X, y)
        self.plot_feature_group_importance(feature_names, importances)

        log.info("All feature analysis plots saved to %s", self.plots_dir)

    # ------------------------------------------------------------------
    # Plot 1 — Feature distributions
    # ------------------------------------------------------------------

    def plot_feature_distributions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        importances: Optional[np.ndarray] = None,
        top_n: int = 20,
        save_path: Optional[Path] = None,
    ) -> None:
        """Overlaid histograms for the top-N most discriminative features.

        Selection criterion: if *importances* is provided, pick the top-N
        by importance score.  Otherwise, use the absolute mean-difference
        between the two classes (a model-free separation score).

        Args:
            X: Raw feature matrix ``(N, D)``.
            y: Class labels ``(N,)``.
            feature_names: Feature name list.
            importances: Optional importance scores.
            top_n: Number of features to display.
            save_path: Override default file path.
        """
        save_path = save_path or self.plots_dir / "feature_distributions.png"

        # Select top features.
        indices = self._top_feature_indices(X, y, feature_names, importances, top_n)
        n_cols = 4
        n_rows = (len(indices) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 4, n_rows * 2.8))
        axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]

        X_real   = X[y == 0]
        X_screen = X[y == 1]

        for plot_idx, feat_idx in enumerate(indices):
            ax = axes_flat[plot_idx]
            vals_real   = X_real[:, feat_idx]
            vals_screen = X_screen[:, feat_idx]

            # Clip extreme outliers for readability (1st–99th percentile).
            all_vals = np.concatenate([vals_real, vals_screen])
            lo, hi   = np.percentile(all_vals, [1, 99])
            bins = np.linspace(lo, hi, 30)

            ax.hist(vals_real,   bins=bins, alpha=0.6,
                    color=_PALETTE["real"],   label="Real",   density=True)
            ax.hist(vals_screen, bins=bins, alpha=0.6,
                    color=_PALETTE["screen"], label="Screen", density=True)

            # Truncate long feature names.
            name = feature_names[feat_idx] if feat_idx < len(feature_names) else str(feat_idx)
            ax.set_title(name[:22], fontsize=8, pad=3)
            ax.tick_params(labelsize=6)
            ax.set_yticks([])

        # Add a shared legend on the last visible axes.
        axes_flat[0].legend(fontsize=7, loc="upper right")

        # Hide unused subplots.
        for ax in axes_flat[len(indices):]:
            ax.set_visible(False)

        fig.suptitle(f"Feature Distributions — Top {top_n} by Discriminability",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved feature distributions -> %s", save_path)

    # ------------------------------------------------------------------
    # Plot 2 — Correlation matrix
    # ------------------------------------------------------------------

    def plot_correlation_matrix(
        self,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 40,
        save_path: Optional[Path] = None,
    ) -> None:
        """Pearson correlation heatmap for the top-N features by variance.

        High off-diagonal values indicate redundant feature pairs —
        candidates for removal by feature selection.

        Args:
            X: Feature matrix ``(N, D)``.
            feature_names: Feature name list.
            top_n: Number of features to include (sorted by variance).
            save_path: Override default file path.
        """
        save_path = save_path or self.plots_dir / "correlation_matrix.png"

        # Select top features by variance.
        variances = X.var(axis=0)
        top_n     = min(top_n, X.shape[1])
        indices   = np.argsort(variances)[-top_n:][::-1]

        X_sub  = X[:, indices]
        names  = [feature_names[i][:18] if i < len(feature_names) else str(i)
                  for i in indices]

        corr_matrix = np.corrcoef(X_sub.T)
        # Clip to [-1, 1] to handle floating-point edge cases.
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        fig_size = max(10, top_n * 0.35)
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.9))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3,
            xticklabels=names, yticklabels=names,
            cbar_kws={"shrink": 0.7, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)
        ax.set_title(
            f"Feature Correlation Matrix — Top {top_n} by Variance",
            fontsize=13, fontweight="bold", pad=12,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved correlation matrix -> %s", save_path)

    # ------------------------------------------------------------------
    # Plot 3 — PCA class separation
    # ------------------------------------------------------------------

    def plot_class_separation_pca(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> None:
        """2-D PCA scatter plot coloured by class label.

        If the two class clouds are visually separable in the first two
        principal components, the feature set is likely to support a
        high-accuracy classifier.  The explained variance ratio is
        annotated on each axis.

        Args:
            X: Feature matrix ``(N, D)``.
            y: Class labels ``(N,)``.
            save_path: Override default file path.
        """
        save_path = save_path or self.plots_dir / "class_separation_pca.png"

        # StandardScale before PCA — PCA is sensitive to feature scale.
        X_scaled = StandardScaler().fit_transform(X)
        pca      = PCA(n_components=2, random_state=CFG.model.random_state)
        X_2d     = pca.fit_transform(X_scaled)
        ev       = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(7, 6))
        for label, name in _CLASS_NAMES.items():
            mask = y == label
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=_PALETTE[name], label=name.capitalize(),
                alpha=0.55, s=25, edgecolors="none",
            )

        ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% variance)", fontsize=11)
        ax.set_title("Class Separation — PCA Projection (2D)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, markerscale=1.5)
        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved PCA class separation -> %s", save_path)

    # ------------------------------------------------------------------
    # Plot 4 — Feature group importance
    # ------------------------------------------------------------------

    def plot_feature_group_importance(
        self,
        feature_names: List[str],
        importances: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Bar chart of mean importance aggregated by feature group.

        Groups are determined by the naming convention established in
        :meth:`~src.feature_extractor.FeatureExtractor.get_feature_names`:
        each name begins with a group prefix (e.g. ``sharp_``, ``fft_``,
        ``lbp_``, ``hist_``).

        If *importances* is None, uses uniform weights (all features
        equally weighted), which still reveals group size.

        Args:
            feature_names: List of feature name strings.
            importances: Optional importance scores (same length as names).
            save_path: Override default file path.
        """
        save_path = save_path or self.plots_dir / "feature_group_importance.png"

        if importances is None:
            importances = np.ones(len(feature_names))

        # Truncate or pad importances to match feature_names length.
        n = min(len(feature_names), len(importances))
        importances  = np.array(importances[:n])
        feature_names = feature_names[:n]

        groups: Dict[str, float] = {}
        for name, imp in zip(feature_names, importances):
            prefix = self._group_prefix(name)
            groups[prefix] = groups.get(prefix, 0.0) + float(imp)

        # Sort by total group importance descending.
        sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
        labels = [g[0] for g in sorted_groups]
        values = [g[1] for g in sorted_groups]

        # Normalise to [0, 1] for readability.
        total = sum(values) or 1.0
        values_norm = [v / total for v in values]

        cmap   = plt.cm.viridis  # type: ignore[attr-defined]
        colors = [cmap(i / len(labels)) for i in range(len(labels))]

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
        bars = ax.barh(range(len(labels)), values_norm,
                       color=colors, edgecolor="none")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Normalised Group Importance", fontsize=11)
        ax.set_title("Feature Importance by Group", fontsize=13,
                     fontweight="bold")

        # Annotate bars with percentage.
        for i, (bar, val) in enumerate(zip(bars, values_norm)):
            ax.text(val + 0.002, i, f"{val*100:.1f}%",
                    va="center", fontsize=8, color="black")

        fig.tight_layout()
        fig.savefig(save_path, dpi=CFG.output.plot_dpi)
        plt.close(fig)
        log.info("Saved feature group importance -> %s", save_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _top_feature_indices(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        importances: Optional[np.ndarray],
        top_n: int,
    ) -> np.ndarray:
        """Return indices of the top-N most discriminative features.

        If *importances* is provided, sort by importance.
        Otherwise, use |mean(real) - mean(screen)| / (std + eps) —
        a model-free effect-size score similar to Cohen's d.

        Args:
            X: Feature matrix.
            y: Class labels.
            feature_names: Feature name list (unused, kept for signature).
            importances: Optional model importance scores.
            top_n: Number of indices to return.

        Returns:
            Array of ``top_n`` feature indices, sorted by score descending.
        """
        top_n = min(top_n, X.shape[1])

        if importances is not None and len(importances) == X.shape[1]:
            scores = np.array(importances)
        else:
            X_real   = X[y == 0]
            X_screen = X[y == 1]
            mean_diff = np.abs(X_real.mean(axis=0) - X_screen.mean(axis=0))
            pooled_std = (X.std(axis=0) + 1e-9)
            scores = mean_diff / pooled_std

        return np.argsort(scores)[-top_n:][::-1]

    @staticmethod
    def _group_prefix(name: str) -> str:
        """Extract the feature group from a feature name.

        Names follow the convention ``<group>_<detail>``.  The group is
        everything before the first underscore, mapped to a human-readable
        label.

        Args:
            name: Feature name string (e.g. ``"fft_mag_mean"``).

        Returns:
            Human-readable group label (e.g. ``"FFT Frequency"``).
        """
        _GROUP_MAP = {
            "sharp":  "Sharpness",
            "fft":    "FFT Frequency",
            "moire":  "Moire Detection",
            "grad":   "Gradient",
            "edge":   "Edge Density",
            "lbp":    "LBP Texture",
            "entropy":"Entropy",
            "bright": "Brightness",
            "contrast":"Contrast",
            "sat":    "Saturation",
            "hsv":    "HSV Stats",
            "lab":    "LAB Stats",
            "hist":   "Color Histogram",
            "glare":  "Glare/Reflection",
            "border": "Screen Border",
            "persp":  "Perspective",
            "noise":  "Noise",
            "tex":    "Texture",
        }
        prefix = name.split("_")[0].lower()
        return _GROUP_MAP.get(prefix, prefix.capitalize())
