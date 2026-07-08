"""Image pre-processing pipeline for screen-recapture-detector.

Every function is a pure transformation: ``np.ndarray -> np.ndarray``.
The top-level :func:`preprocess` chains them in the correct order.

Design principles
-----------------
* **Stay uint8.**  OpenCV's Laplacian, Canny, LBP etc. all accept uint8.
  Converting to float32 doubles memory with zero accuracy benefit.
  Normalisation to [0, 1] is deferred to individual feature calculations.

* **Letterbox, don't squash.**  Squashing a portrait photo to a square
  distorts aspect ratio, corrupting perspective-distortion features and
  the border/centre edge ratio.  Letterboxing preserves geometry.

* **Conservative denoising.**  h=5 (not the OpenCV default of 10)
  preserves micro-texture that LBP relies on.  A heavier filter would
  erase the signal we are trying to measure.

* **CLAHE on LAB-L only.**  Global histogram equalisation over-amplifies
  screen glare.  CLAHE clips the local contrast gain and leaves the
  hue/saturation channels untouched — no colour shift.

Pipeline order
--------------
load -> validate -> letterbox-resize -> denoise -> CLAHE
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.config import CFG

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_image(path: Path | str) -> np.ndarray:
    """Load an image from disk and return it as a BGR uint8 array.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        BGR ``uint8`` array of shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
        ValueError: If the file exists but OpenCV cannot decode it
            (e.g. corrupt file, unsupported format).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"OpenCV could not decode image: {path}\n"
            "Ensure the file is a valid, non-corrupt image."
        )

    log.debug("Loaded %s  shape=%s  dtype=%s", path.name, img.shape, img.dtype)
    return img


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_image(img: np.ndarray) -> np.ndarray:
    """Ensure *img* is a valid BGR uint8 array with 3 channels.

    Handles two common edge-cases before they silently corrupt downstream
    features:

    * **Greyscale** (2-D array): converted to 3-channel BGR.
    * **BGRA** (4-channel): alpha channel stripped.

    Args:
        img: Raw image array from :func:`load_image`.

    Returns:
        3-channel BGR uint8 array.

    Raises:
        ValueError: If *img* is empty or has an unexpected number of
            channels.
    """
    if img is None or img.size == 0:
        raise ValueError("Received an empty image array.")

    if img.ndim == 2:
        # Greyscale — convert to BGR so the rest of the pipeline is uniform.
        log.debug("Greyscale image detected; converting to BGR.")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # BGRA — drop the alpha channel.
        log.debug("BGRA image detected; dropping alpha channel.")
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass  # Already BGR — nothing to do.
    else:
        raise ValueError(
            f"Unexpected image shape {img.shape}. "
            "Expected (H, W), (H, W, 3), or (H, W, 4)."
        )

    if img.dtype != np.uint8:
        log.debug("Converting dtype %s -> uint8", img.dtype)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# Resize (letterbox)
# ---------------------------------------------------------------------------

def letterbox_resize(
    img: np.ndarray,
    target_size: Tuple[int, int] | None = None,
    pad_value: int | None = None,
) -> np.ndarray:
    """Resize *img* to *target_size* while preserving the aspect ratio.

    The image is scaled so that it fits entirely within the target canvas,
    then padded symmetrically with *pad_value* to reach the exact target
    dimensions.

    Why letterbox instead of squash?
        Squashing changes the aspect ratio, which:
        * Distorts the border/centre edge-density ratio.
        * Corrupts perspective-distortion feature calculations.
        * Changes the LBP texture statistics (circles become ellipses).

    Args:
        img: BGR uint8 image.
        target_size: ``(width, height)`` of the output canvas.  Defaults
            to ``CFG.preprocessing.target_size``.
        pad_value: Greyscale fill value for padding (0–255).  Defaults
            to ``CFG.preprocessing.pad_value`` (black).

    Returns:
        BGR uint8 image of exactly ``(target_h, target_w, 3)``.
    """
    if target_size is None:
        target_size = CFG.preprocessing.target_size
    if pad_value is None:
        pad_value = CFG.preprocessing.pad_value

    target_w, target_h = target_size
    src_h, src_w = img.shape[:2]

    if src_h == target_h and src_w == target_w:
        return img

    # Scale factor that fits the image inside the target canvas.
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    # Use INTER_AREA when shrinking (best anti-aliasing), INTER_LINEAR
    # when enlarging (preserves sharpness better than INTER_CUBIC at
    # the cost of very minor blurring).
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Create the target canvas and paste the resized image centred.
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

    log.debug(
        "Letterbox: (%d,%d) -> (%d,%d)  scale=%.3f  pad=(%d,%d)",
        src_w, src_h, target_w, target_h, scale, pad_top, pad_left,
    )
    return canvas


# ---------------------------------------------------------------------------
# Denoise
# ---------------------------------------------------------------------------

def denoise(img: np.ndarray) -> np.ndarray:
    """Apply fast non-local means denoising to *img*.

    Parameter choices:
        h=5, hColor=5 — conservative filter strength.  The default of
        h=10 in OpenCV over-smooths fine textures that LBP measures.

        templateWindowSize=7, searchWindowSize=21 — standard values
        that balance quality vs. runtime (~3 ms on 256×256 CPU).

    Args:
        img: BGR uint8 image.

    Returns:
        Denoised BGR uint8 image.
    """
    cfg = CFG.preprocessing
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=cfg.denoise_h,
        hColor=cfg.denoise_h_color,
        templateWindowSize=cfg.denoise_template_win,
        searchWindowSize=cfg.denoise_search_win,
    )
    log.debug("Denoising applied (h=%d, hColor=%d)", cfg.denoise_h, cfg.denoise_h_color)
    return denoised


# ---------------------------------------------------------------------------
# CLAHE histogram equalisation
# ---------------------------------------------------------------------------

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE histogram equalisation to the L channel of LAB.

    Why LAB instead of BGR?
        Operating on the Lightness channel only means we sharpen local
        contrast without touching colour — hue and saturation remain
        correct for the saturation/colour-histogram features.

    Why CLAHE instead of global equalisation?
        Global equalisation over-amplifies screen glare regions.  CLAHE
        clips the gain per tile (clip_limit=2.0), so bright spots cannot
        dominate the histogram.

    Args:
        img: BGR uint8 image.

    Returns:
        Contrast-enhanced BGR uint8 image.
    """
    cfg = CFG.preprocessing
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid,
    )
    l_eq = clahe.apply(l_ch)

    enhanced = cv2.merge([l_eq, a_ch, b_ch])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    log.debug(
        "CLAHE applied (clip=%.1f, tile=%s)",
        cfg.clahe_clip_limit, cfg.clahe_tile_grid,
    )
    return result


# ---------------------------------------------------------------------------
# Colour space conversions (stateless helpers used by feature_extractor)
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel greyscale uint8.

    Args:
        img: BGR uint8 image.

    Returns:
        Greyscale uint8 array of shape ``(H, W)``.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV uint8.

    Args:
        img: BGR uint8 image.

    Returns:
        HSV uint8 array of shape ``(H, W, 3)``.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_lab(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to CIE LAB float32.

    Uses ``cv2.COLOR_BGR2LAB`` which outputs:
    * L in [0, 100]
    * a in [-127, 127]
    * b in [-127, 127]

    Args:
        img: BGR uint8 image.

    Returns:
        LAB float32 array of shape ``(H, W, 3)``.
    """
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(path: Path | str) -> np.ndarray:
    """Run the complete pre-processing chain on a single image file.

    Chain::

        load -> validate -> letterbox_resize -> [denoise] -> [CLAHE]

    Square brackets denote steps that can be toggled in
    :class:`~src.config.PreprocessingConfig`.

    Args:
        path: Path to the source image.

    Returns:
        Pre-processed BGR uint8 image of shape
        ``(target_h, target_w, 3)`` ready for feature extraction.

    Raises:
        FileNotFoundError: If the image does not exist on disk.
        ValueError: If the file cannot be decoded or has an unexpected
            channel count.
    """
    cfg = CFG.preprocessing

    img = load_image(path)
    img = validate_image(img)
    img = letterbox_resize(img, cfg.target_size, cfg.pad_value)

    if cfg.apply_denoise:
        img = denoise(img)

    if cfg.apply_clahe:
        img = apply_clahe(img)

    return img
