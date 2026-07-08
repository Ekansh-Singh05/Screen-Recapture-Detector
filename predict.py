"""Inference entry point for screen-recapture-detector.

Prints **exactly one float** to stdout — nothing else.  All log output
is suppressed so the output is safe to pipe or capture::

    python predict.py photo.jpg
    0.93

    prob=$(python predict.py photo.jpg)   # bash capture

Exit codes:
    0 — prediction succeeded; float printed to stdout.
    1 — wrong argument count; usage message printed to stderr.
    2 — image not found or cannot be decoded; error printed to stderr.
    3 — trained model artefacts missing; hint printed to stderr.

Usage::

    python predict.py <image_path>
"""
# ``silence()`` must be called before importing any src.* module so that
# logger handlers are never attached to the root logger.  Any import
# that happens after this point will call ``get_logger(__name__)`` and
# receive a no-op NullHandler logger.
import sys

# Silence logging before any project import.
from src.logger import silence
silence()

from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python predict.py <image_path>\n")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    try:
        from src.predictor import Predictor
        predictor = Predictor()
    except FileNotFoundError as exc:
        sys.stderr.write(f"Model not found: {exc}\nRun: python train.py\n")
        sys.exit(3)
    except Exception as exc:
        sys.stderr.write(f"Failed to load model: {exc}\n")
        sys.exit(3)

    try:
        prob = predictor.predict(image_path)
    except FileNotFoundError:
        sys.stderr.write(f"Image not found: {image_path}\n")
        sys.exit(2)
    except ValueError as exc:
        sys.stderr.write(f"Invalid image: {exc}\n")
        sys.exit(2)
    except Exception as exc:
        sys.stderr.write(f"Prediction failed: {exc}\n")
        sys.exit(2)

    # Print probability — exactly 2 decimal places, no trailing newline extras.
    print(f"{prob:.2f}")


if __name__ == "__main__":
    main()
