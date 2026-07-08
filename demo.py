"""Streamlit demo for screen-recapture-detector — Solution A.

Single-page app: drag-and-drop an image, get the screen-capture
probability, a confidence gauge, and a feature-group breakdown bar chart
showing which signal groups drove the prediction.

Usage::

    streamlit run demo.py

Requirements:
    - Trained model artefacts in models/  (run python train.py first)
    - streamlit >= 1.28.0
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Silence all logging before any project import so no stray output
# interferes with Streamlit's stdout capture.
from src.logger import silence
silence()

import numpy as np
import streamlit as st
from PIL import Image

from src.config import CFG
from src.feature_extractor import FeatureExtractor
from src.predictor import Predictor
from src.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spot Fake Photo",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def _load_predictor() -> Predictor | None:
    """Load and cache the Predictor.  Returns None if artefacts are missing."""
    try:
        return Predictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def _load_extractor() -> FeatureExtractor:
    return FeatureExtractor()


# ---------------------------------------------------------------------------
# Feature group mapping (mirrors visualization.py)
# ---------------------------------------------------------------------------

_GROUP_MAP: Dict[str, str] = {
    "sharp":    "Sharpness",
    "fft":      "FFT Frequency",
    "moire":    "Moire Detection",
    "grad":     "Gradient",
    "edge":     "Edge Density",
    "lbp":      "LBP Texture",
    "entropy":  "Entropy",
    "bright":   "Brightness",
    "contrast": "Contrast",
    "sat":      "Saturation",
    "hsv":      "HSV Stats",
    "lab":      "LAB Stats",
    "hist":     "Color Histogram",
    "glare":    "Glare/Reflection",
    "border":   "Screen Border",
    "persp":    "Perspective",
    "noise":    "Noise",
    "tex":      "Texture",
}


def _group_prefix(name: str) -> str:
    prefix = name.split("_")[0].lower()
    return _GROUP_MAP.get(prefix, prefix.capitalize())


def _group_mean_magnitude(
    feature_vec: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Return mean |feature value| aggregated by group.

    The raw (unscaled) vector is used so values are interpretable.
    We take the absolute value because the sign carries no meaning for
    the magnitude contribution.

    Args:
        feature_vec: Raw feature vector of shape ``(193,)``.
        feature_names: Parallel list of feature names.

    Returns:
        Dict mapping group label to mean absolute feature value.
    """
    groups: Dict[str, List[float]] = {}
    n = min(len(feature_vec), len(feature_names))
    for i in range(n):
        group = _group_prefix(feature_names[i])
        groups.setdefault(group, []).append(abs(float(feature_vec[i])))
    return {g: float(np.mean(vals)) for g, vals in groups.items()}


def _predict_from_upload(
    uploaded_file,
    predictor: Predictor,
    extractor: FeatureExtractor,
) -> Tuple[float, np.ndarray]:
    """Save upload to a temp file, preprocess, extract features, predict.

    Args:
        uploaded_file: Streamlit ``UploadedFile`` object.
        predictor: Loaded :class:`~src.predictor.Predictor`.
        extractor: Loaded :class:`~src.feature_extractor.FeatureExtractor`.

    Returns:
        ``(probability, raw_feature_vector)``

    Raises:
        ValueError: If the image cannot be decoded or features extracted.
    """
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        img = preprocess(tmp_path)
        vec = extractor.extract(img)
    finally:
        tmp_path.unlink(missing_ok=True)

    scaled = predictor._transform(vec)
    prob = predictor._classify(scaled)
    return prob, vec


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _verdict_badge(prob: float) -> Tuple[str, str]:
    """Return ``(label, colour)`` based on probability threshold.

    Args:
        prob: Screen-capture probability in ``[0, 1]``.

    Returns:
        Tuple of display label and CSS colour string.
    """
    if prob >= 0.75:
        return "SCREEN PHOTO", "#F44336"
    if prob >= 0.45:
        return "UNCERTAIN", "#FF9800"
    return "REAL PHOTO", "#4CAF50"


def _probability_bar_html(prob: float, colour: str) -> str:
    """Build a simple HTML progress bar for the probability.

    Args:
        prob: Value in ``[0, 1]``.
        colour: CSS colour for the filled portion.

    Returns:
        HTML string safe to pass to ``st.markdown(..., unsafe_allow_html=True)``.
    """
    pct = int(prob * 100)
    return (
        f"<div style='background:#e0e0e0;border-radius:6px;height:22px;width:100%;'>"
        f"<div style='background:{colour};width:{pct}%;height:100%;border-radius:6px;"
        f"display:flex;align-items:center;justify-content:flex-end;padding-right:6px;"
        f"color:white;font-size:13px;font-weight:600;'>{pct}%</div></div>"
    )


def _group_chart(group_scores: Dict[str, float]) -> None:
    """Render a horizontal bar chart of feature-group magnitudes via Streamlit.

    Uses ``st.bar_chart`` (Altair under the hood) so no matplotlib is needed
    in the demo — keeping the runtime dependency footprint small.

    Args:
        group_scores: Dict mapping group name to mean absolute feature value.
    """
    import pandas as pd

    sorted_items = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_items, columns=["Feature Group", "Mean |value|"])
    df = df.set_index("Feature Group")
    st.bar_chart(df, height=320)


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Header ----
    st.title("🔍 Spot Fake Photo")
    st.markdown(
        "Upload any image to find out if it is a **genuine camera photo** "
        "or a **photo of a screen / recapture**.  "
        "The classifier uses a 193-dimensional hand-crafted feature vector "
        "(sharpness, FFT, Moiré, LBP texture, colour statistics, and more)."
    )
    st.divider()

    # ---- Model loading ----
    predictor = _load_predictor()
    extractor = _load_extractor()
    feature_names = extractor.get_feature_names()

    if predictor is None:
        st.error(
            "**Trained model not found.**  "
            "Run `python train.py` from the project root, then restart the demo."
        )
        st.stop()

    # ---- Upload widget ----
    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    )

    if uploaded is None:
        st.info("No image uploaded yet.  Waiting …")
        return

    # ---- Two-column layout: image | result ----
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Input image")
        pil_img = Image.open(io.BytesIO(uploaded.getbuffer()))
        st.image(pil_img, use_container_width=True)
        st.caption(
            f"{uploaded.name}  •  "
            f"{pil_img.width}×{pil_img.height}  •  "
            f"{uploaded.size / 1024:.1f} KB"
        )

    with col_result:
        st.subheader("Prediction")
        with st.spinner("Analysing …"):
            try:
                prob, raw_vec = _predict_from_upload(uploaded, predictor, extractor)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        label, colour = _verdict_badge(prob)

        st.markdown(
            f"<h2 style='color:{colour};margin-bottom:4px;'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:#555;margin-top:0;'>Screen-capture probability</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _probability_bar_html(prob, colour),
            unsafe_allow_html=True,
        )
        st.metric(label="Probability", value=f"{prob:.2f}")

        # Threshold guidance
        st.markdown(
            "<small style='color:#888;'>"
            "≥ 0.75 → screen photo &nbsp;|&nbsp; "
            "0.45–0.74 → uncertain &nbsp;|&nbsp; "
            "< 0.45 → real photo"
            "</small>",
            unsafe_allow_html=True,
        )

    # ---- Feature group breakdown ----
    st.divider()
    st.subheader("Feature-group signal breakdown")
    st.markdown(
        "Mean absolute feature value per group — higher bars indicate "
        "stronger signal from that feature family in this image."
    )
    group_scores = _group_mean_magnitude(raw_vec, feature_names)
    _group_chart(group_scores)

    # ---- Expandable raw feature values ----
    with st.expander("Raw feature vector (193 dimensions)"):
        import pandas as pd
        n = min(len(raw_vec), len(feature_names))
        df = pd.DataFrame({
            "Feature": feature_names[:n],
            "Value":   [f"{v:.6f}" for v in raw_vec[:n]],
        })
        st.dataframe(df, use_container_width=True, height=300)

    # ---- Footer ----
    st.divider()
    st.caption(
        "screen-recapture-detector · Solution A · Classical CV Pipeline · "
        "193-dim feature vector · "
        "Model: " + str(CFG.model.model_a_path.name)
    )


if __name__ == "__main__":
    main()
