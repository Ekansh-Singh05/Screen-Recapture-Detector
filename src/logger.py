"""Centralised logging configuration for screen-recapture-detector.

All modules call ``logging.getLogger(__name__)`` as usual.  Call
:func:`setup` once at the top of every entry-point script (``train.py``,
``evaluate.py``, etc.) to configure formatting and optional file output.

:func:`silence` is provided for ``predict.py``, which must emit only a
single probability value to stdout — no log noise.

Design rationale
----------------
* stdlib ``logging`` only — zero extra dependencies.
* One function call from entry points; modules themselves stay clean.
* Optional rotating file sink for production/debugging use.
* Console handler writes to ``sys.stdout`` so it plays nicely with
  output redirection and pytest capsys.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------
_CONSOLE_FMT = "%(levelname)-8s | %(asctime)s | %(name)s:%(lineno)d | %(message)s"
_FILE_FMT    = "%(levelname)-8s | %(asctime)s | %(name)s:%(lineno)d | %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"


def setup(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
) -> None:
    """Configure the root logger with a console handler and optional file handler.

    Call this exactly once at the start of each entry-point script.
    Subsequent calls are no-ops (idempotent).

    Args:
        level: Logging level for both handlers.  Defaults to ``INFO``.
        log_dir: If supplied, a file handler is added that writes to
            ``log_dir/<log_filename>``.  The directory is created if it
            does not exist.
        log_filename: Name of the log file.  Defaults to
            ``run_YYYYMMDD_HHMMSS.log``.

    Example::

        from src.logger import setup
        setup(level=logging.DEBUG, log_dir=Path("logs"))
    """
    root = logging.getLogger()

    # Idempotency guard — avoid duplicate handlers on repeated calls.
    if root.handlers:
        return

    root.setLevel(level)

    # Console handler — stdout so pytest capsys and shell pipes work.
    # Force UTF-8 recoding on Windows where the default cp1252 would crash
    # on any non-ASCII character in a log message.
    import io
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    console = logging.StreamHandler(utf8_stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # Optional file handler.
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if log_filename is None:
            log_filename = f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        root.addHandler(file_handler)


def silence() -> None:
    """Disable all log output below CRITICAL level.

    Called by ``predict.py`` before importing any other module so that
    only the probability value reaches stdout.

    Example::

        from src.logger import silence
        silence()
    """
    logging.disable(logging.CRITICAL)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Thin wrapper so modules can do::

        from src.logger import get_logger
        log = get_logger(__name__)

    instead of importing ``logging`` directly.  Either style is fine;
    this just keeps imports consistent across the codebase.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
