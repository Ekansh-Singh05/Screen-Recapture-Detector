"""Benchmarking entry point for screen-recapture-detector — Solution A.

Measures end-to-end prediction latency, memory, CPU utilisation, and
model artefact size, then estimates deployment cost for six platforms
using the *measured* median latency.

Outputs
-------
outputs/reports/benchmark_report.json   full results
stdout                                  human-readable summary table

Usage::

    python benchmark.py [--n-images N] [--log-level {DEBUG,INFO,WARNING}]

Flags:
    --n-images    Number of images to benchmark.  Defaults to all images
                  found in data/real/ and data/screen/ (capped at 100 for
                  speed).  Warm-up runs use the first image in the list.
    --log-level   Console log verbosity.  Default: INFO.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import CFG
from src.logger import setup as setup_logging, get_logger
from src.utils import ensure_dirs, load_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the trained screen-recapture-detector Solution-A model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Max images to use.  Default: all (capped at 100).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return p.parse_args()


def _collect_images(n_max: int | None, log: logging.Logger):
    """Return a list of image paths from both class directories.

    Args:
        n_max: Cap on the number of images returned.  ``None`` uses the
            default cap of 100.
        log: Logger for status messages.

    Returns:
        List of :class:`~pathlib.Path` objects.

    Exits with code 1 if no images are found or the model is missing.
    """
    cap = n_max if n_max is not None else 100

    # Verify model artefacts exist before spending time on image loading.
    if not CFG.model.model_a_path.exists():
        log.error(
            "Trained model not found: %s\nRun: python train.py",
            CFG.model.model_a_path,
        )
        sys.exit(1)

    try:
        paths, _ = load_dataset(CFG.data.real_dir, CFG.data.screen_dir)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    selected = paths[:cap]
    log.info("Using %d images for benchmarking (total available: %d).",
             len(selected), len(paths))
    return selected


def _print_summary(report: dict) -> None:
    """Print a formatted benchmark summary to stdout.

    Args:
        report: Dict returned by :meth:`~src.benchmark.BenchmarkRunner.run`.
    """
    lat = report.get("latency_ms", {})
    mem = report.get("memory", {})
    cpu = report.get("cpu_percent", {})
    sz  = report.get("model_sizes", {})
    hw  = report.get("hardware", {})
    cost = report.get("cost_per_1m_images", {})

    sep = "=" * 48

    print(f"\n{sep}")
    print("  Benchmark Results — screen-recapture-detector Solution A")
    print(sep)

    print(f"\n  Hardware")
    print(f"    CPU   : {hw.get('cpu', '?')}")
    print(f"    Cores : {hw.get('cpu_physical_cores', '?')} physical / "
          f"{hw.get('cpu_logical_cores', '?')} logical")
    print(f"    RAM   : {hw.get('ram_total', '?')}")
    print(f"    OS    : {hw.get('os', '?')}")

    print(f"\n  Latency  (n={lat.get('n_measured', '?')} runs)")
    for label, key in [
        ("avg",    "avg_ms"),
        ("median", "median_ms"),
        ("p95",    "p95_ms"),
        ("p99",    "p99_ms"),
        ("min",    "min_ms"),
        ("max",    "max_ms"),
    ]:
        val = lat.get(key)
        if val is not None:
            print(f"    {label:<8s}: {val:>8.2f} ms")

    print(f"\n  Memory")
    print(f"    tracemalloc peak : {mem.get('tracemalloc_peak_mb', '?'):>6} MB")
    print(f"    process RSS      : {mem.get('process_rss_total_mb', '?'):>6} MB")

    print(f"\n  CPU utilisation")
    print(f"    process : {cpu.get('process_cpu_percent', '?')} %")
    print(f"    system  : {cpu.get('system_cpu_percent', '?')} %")

    print(f"\n  Model artefact sizes")
    for name, size in sz.items():
        print(f"    {name:<12s}: {size}")

    print(f"\n  Estimated cost per 1 M images")
    skip = {"assumptions", "recommendation"}
    for platform, info in cost.items():
        if platform in skip or not isinstance(info, dict):
            continue
        print(f"    {platform:<35s}: {info.get('cost_usd', 'N/A')}")

    rec = cost.get("recommendation", "")
    if rec:
        print(f"\n  Recommendation")
        # Word-wrap at 60 chars.
        words, line = rec.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 60:
                print(f"    {line}")
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            print(f"    {line}")

    print(f"\n  Report -> {CFG.output.reports_dir / 'benchmark_report.json'}")
    print(f"{sep}\n")


def main() -> None:
    args = _parse_args()

    ensure_dirs(CFG.output.logs_dir, CFG.output.reports_dir)
    setup_logging(
        level=getattr(logging, args.log_level),
        log_dir=CFG.output.logs_dir,
        log_filename="benchmark.log",
    )
    log = get_logger(__name__)
    log.info("=" * 60)
    log.info("screen-recapture-detector  Solution A  Benchmark")
    log.info("=" * 60)

    image_paths = _collect_images(args.n_images, log)

    from src.benchmark import BenchmarkRunner
    runner = BenchmarkRunner()
    report = runner.run(image_paths)

    _print_summary(report)
    log.info("Benchmark complete.")


if __name__ == "__main__":
    main()
