"""General-purpose utility helpers for the screen-recapture-detector pipeline.

Responsibilities
----------------
* File-system operations: directory creation, image-path discovery.
* Dataset loading: pair image paths with binary labels from directory layout.
* JSON serialisation / deserialisation with Path-aware encoding.
* Reproducibility: global random-seed setter.
* Timing context manager for quick wall-clock measurements.

What does NOT belong here
--------------------------
Feature logic → feature_extractor.py
Model logic   → trainer.py / predictor.py
Plotting      → visualization.py
Logging setup → logger.py
Configuration → config.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Supported image extensions (lower-case).  Defined here as a fallback;
# the canonical definition lives in DataConfig.image_extensions.
_IMG_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create *path* (and any missing parents) if it does not exist.

    Args:
        path: Directory to create.

    Returns:
        The same *path* object (allows chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(*paths: Path) -> None:
    """Create multiple directories in one call.

    Args:
        *paths: One or more :class:`~pathlib.Path` objects to create.
    """
    for p in paths:
        ensure_dir(p)


def get_image_paths(
    directory: Path,
    extensions: frozenset[str] | None = None,
) -> List[Path]:
    """Return a sorted list of image file paths inside *directory*.

    Only direct children are returned (no recursive search).  Files whose
    suffix does not match *extensions* are silently ignored.

    Args:
        directory: Directory to scan.
        extensions: Allowed lower-case suffixes.  Defaults to
            :data:`_IMG_EXTS`.

    Returns:
        Sorted list of :class:`~pathlib.Path` objects.

    Raises:
        FileNotFoundError: If *directory* does not exist.
        NotADirectoryError: If *directory* is a file.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    exts = extensions or _IMG_EXTS
    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    log.debug("Found %d images in %s", len(paths), directory)
    return paths


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    real_dir: Path,
    screen_dir: Path,
    extensions: frozenset[str] | None = None,
) -> Tuple[List[Path], List[int]]:
    """Collect image paths and binary labels from two class directories.

    Label convention
    ----------------
    * ``0`` → real photograph (``real_dir``)
    * ``1`` → photo of a screen (``screen_dir``)

    Args:
        real_dir: Directory containing genuine camera images.
        screen_dir: Directory containing screen-capture images.
        extensions: Allowed image extensions.

    Returns:
        Tuple ``(paths, labels)`` where both lists have the same length
        and ``labels[i]`` is the class of ``paths[i]``.

    Raises:
        ValueError: If either directory contains no images.
    """
    real_paths = get_image_paths(real_dir, extensions)
    screen_paths = get_image_paths(screen_dir, extensions)

    if not real_paths:
        raise ValueError(
            f"No images found in real directory: {real_dir}\n"
            "Add images and re-run."
        )
    if not screen_paths:
        raise ValueError(
            f"No images found in screen directory: {screen_dir}\n"
            "Add images and re-run."
        )

    all_paths = real_paths + screen_paths
    all_labels = [0] * len(real_paths) + [1] * len(screen_paths)

    log.info(
        "Dataset loaded: %d real + %d screen = %d total",
        len(real_paths),
        len(screen_paths),
        len(all_paths),
    )
    return all_paths, all_labels


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

class _PathEncoder(json.JSONEncoder):
    """JSON encoder that serialises :class:`~pathlib.Path` as strings."""

    def default(self, obj: object) -> object:  # type: ignore[override]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, path: Path) -> None:
    """Serialise *data* to a JSON file at *path*.

    Parent directories are created automatically.

    Args:
        data: A JSON-serialisable dictionary (``Path`` objects are
            automatically converted to strings).
        path: Destination file path.

    Raises:
        TypeError: If *data* contains an object that cannot be serialised.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, cls=_PathEncoder)
    log.debug("Saved JSON -> %s", path)


def load_json(path: Path) -> dict:
    """Load and parse a JSON file.

    Args:
        path: Path to a ``.json`` file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    log.debug("Loaded JSON <- %s", path)
    return data


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch.

    Should be called once near the top of every entry-point script
    that trains a model.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore[import]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch is optional (Solution B only)

    log.debug("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager that logs wall-clock elapsed time.

    Usage::

        with timer("feature extraction"):
            vec = extractor.extract(img)
        # Logs: "feature extraction: 4.23 ms"

    Args:
        label: Human-readable name for the timed block.

    Yields:
        Nothing — use as a ``with`` block.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        log.debug("%s: %.2f ms", label or "elapsed", elapsed_ms)


def format_bytes(n: int) -> str:
    """Return a human-readable byte size string.

    Args:
        n: Size in bytes.

    Returns:
        String such as ``"4.2 MB"`` or ``"512 KB"``.

    Examples::

        format_bytes(1_048_576)  # "1.0 MB"
        format_bytes(2_500)      # "2.4 KB"
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"
