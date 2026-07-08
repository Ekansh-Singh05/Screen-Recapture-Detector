"""Benchmarking utilities for the screen-recapture-detector pipeline.

Measures end-to-end prediction latency, memory usage, CPU utilisation,
and model disk size.  Generates a deployment cost analysis for five
platforms using the *measured* median latency — not assumed numbers.

Outputs
-------
``outputs/reports/benchmark_report.json``
    Complete benchmark results including hardware specs, latency
    statistics, memory, and cost analysis.

Usage::

    from src.benchmark import BenchmarkRunner
    runner = BenchmarkRunner()
    report = runner.run(image_paths)
"""
from __future__ import annotations

import platform
import tracemalloc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

from src.config import CFG
from src.logger import get_logger
from src.predictor import Predictor
from src.utils import ensure_dirs, format_bytes, save_json

log = get_logger(__name__)


class BenchmarkRunner:
    """Run end-to-end latency, memory, CPU, and cost benchmarks.

    Args:
        predictor: A pre-loaded :class:`~src.predictor.Predictor`.
            If ``None``, one is instantiated automatically.
        warmup_runs: Number of warm-up passes to discard before timing.
        benchmark_runs: Number of timed passes per image.
    """

    def __init__(
        self,
        predictor: Optional[Predictor] = None,
        warmup_runs: int | None = None,
        benchmark_runs: int | None = None,
    ) -> None:
        self.predictor     = predictor or Predictor()
        self.warmup_runs   = warmup_runs   or CFG.benchmark.warmup_runs
        self.benchmark_runs = benchmark_runs or CFG.benchmark.benchmark_runs
        ensure_dirs(CFG.output.reports_dir)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Execute the full benchmark suite and save the report.

        Args:
            image_paths: List of image paths to benchmark.  Should be
                representative of production input (mix of real and screen
                images, varied resolutions).

        Returns:
            Complete benchmark report as a nested dictionary.

        Raises:
            ValueError: If *image_paths* is empty.
        """
        if not image_paths:
            raise ValueError("No images provided for benchmarking.")

        log.info("Benchmarking on %d images  (warmup=%d, runs=%d) ...",
                 len(image_paths), self.warmup_runs, self.benchmark_runs)

        hardware    = self._hardware_info()
        latency     = self._measure_latency(image_paths)
        memory      = self._measure_memory(image_paths)
        cpu_pct     = self._measure_cpu(image_paths)
        model_sizes = self._model_sizes()
        cost        = self._cost_analysis(latency["median_ms"])

        report: Dict[str, Any] = {
            "hardware":    hardware,
            "latency_ms":  latency,
            "memory":      memory,
            "cpu_percent": cpu_pct,
            "model_sizes": model_sizes,
            "cost_per_1m_images": cost,
            "n_images_benchmarked": len(image_paths),
            "warmup_runs":   self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
        }

        save_json(report, CFG.output.reports_dir / "benchmark_report.json")
        self._log_summary(report)
        return report

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def _measure_latency(self, image_paths: List[Path]) -> Dict[str, float]:
        """Time end-to-end prediction for each image.

        The first ``warmup_runs`` calls are discarded.  The remaining
        calls are timed with ``time.perf_counter`` (monotonic, highest
        resolution available on the OS).

        Args:
            image_paths: List of image paths.

        Returns:
            Dict with ``avg_ms``, ``median_ms``, ``p95_ms``, ``p99_ms``,
            ``min_ms``, ``max_ms``, ``std_ms``, ``n_measured``.
        """
        # Warm-up — fill Python/OpenCV/NumPy internal caches.
        warmup_img = image_paths[0]
        for _ in range(self.warmup_runs):
            try:
                self.predictor.predict(warmup_img)
            except Exception:
                pass

        latencies: List[float] = []
        # Cycle through images so we don't only measure one file.
        n = min(self.benchmark_runs, len(image_paths) * 10)
        for i in range(n):
            path = image_paths[i % len(image_paths)]
            t0 = time.perf_counter()
            try:
                self.predictor.predict(path)
            except Exception as exc:
                log.debug("Benchmark skip %s: %s", path.name, exc)
                continue
            latencies.append((time.perf_counter() - t0) * 1_000.0)

        if not latencies:
            return {}

        arr = np.array(latencies)
        return {
            "avg_ms":    round(float(arr.mean()), 3),
            "median_ms": round(float(np.median(arr)), 3),
            "p95_ms":    round(float(np.percentile(arr, 95)), 3),
            "p99_ms":    round(float(np.percentile(arr, 99)), 3),
            "min_ms":    round(float(arr.min()), 3),
            "max_ms":    round(float(arr.max()), 3),
            "std_ms":    round(float(arr.std()), 3),
            "n_measured": len(latencies),
        }

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _measure_memory(self, image_paths: List[Path]) -> Dict[str, Any]:
        """Measure peak memory allocated during a batch of predictions.

        Uses ``tracemalloc`` which tracks Python-level allocations.
        Note: C-extension allocations (NumPy, OpenCV) may not be fully
        captured.  The ``psutil`` RSS figure is also included as a
        complementary measure.

        Args:
            image_paths: Images to run through the predictor.

        Returns:
            Dict with ``tracemalloc_peak_mb``, ``process_rss_mb``.
        """
        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        tracemalloc.start()
        for path in image_paths[:20]:   # 20 images is enough to see peak
            try:
                self.predictor.predict(path)
            except Exception:
                pass
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_after = proc.memory_info().rss

        return {
            "tracemalloc_peak_mb": round(peak_bytes / 1024 / 1024, 2),
            "process_rss_delta_mb": round((rss_after - rss_before) / 1024 / 1024, 2),
            "process_rss_total_mb": round(rss_after / 1024 / 1024, 2),
        }

    # ------------------------------------------------------------------
    # CPU utilisation
    # ------------------------------------------------------------------

    def _measure_cpu(self, image_paths: List[Path]) -> Dict[str, float]:
        """Measure per-process CPU utilisation during inference.

        ``psutil.Process.cpu_percent()`` returns the CPU time used by
        this process as a percentage of one logical CPU core.  Values
        above 100% are possible on multi-core systems if multiple threads
        are used (e.g. NumPy BLAS operations).

        Args:
            image_paths: Images to run through the predictor.

        Returns:
            Dict with ``process_cpu_percent`` and
            ``system_cpu_percent_avg``.
        """
        proc = psutil.Process()
        proc.cpu_percent()          # initialise (first call always 0.0)
        psutil.cpu_percent()        # initialise system-level counter

        n = min(20, len(image_paths))
        for path in image_paths[:n]:
            try:
                self.predictor.predict(path)
            except Exception:
                pass

        process_cpu = round(proc.cpu_percent(), 1)
        system_cpu  = round(psutil.cpu_percent(), 1)

        return {
            "process_cpu_percent": process_cpu,
            "system_cpu_percent":  system_cpu,
        }

    # ------------------------------------------------------------------
    # Model sizes
    # ------------------------------------------------------------------

    def _model_sizes(self) -> Dict[str, str]:
        """Report the disk size of every saved model artefact.

        Returns:
            Dict mapping artefact name to a human-readable size string
            (e.g. ``"2.3 MB"``).
        """
        cfg = CFG.model
        paths = {
            "model_a":    cfg.model_a_path,
            "scaler_a":   cfg.scaler_a_path,
            "selector_a": cfg.selector_a_path,
        }
        sizes: Dict[str, str] = {}
        for name, path in paths.items():
            if path.exists():
                sizes[name] = format_bytes(path.stat().st_size)
            else:
                sizes[name] = "not found"

        total = sum(
            p.stat().st_size for p in paths.values() if p.exists()
        )
        sizes["total"] = format_bytes(total)
        return sizes

    # ------------------------------------------------------------------
    # Deployment cost analysis
    # ------------------------------------------------------------------

    def _cost_analysis(self, median_latency_ms: float) -> Dict[str, Any]:
        """Estimate deployment cost per 1 million images.

        All calculations use the *measured* median latency so the
        report reflects actual hardware performance.

        Pricing (us-east-1, as of 2025-Q3 — verify before production use)
        ------------------------------------------------------------------
        EC2 t3.medium  : $0.0416 / hr  (2 vCPU, 4 GB)
        EC2 c5.xlarge  : $0.170  / hr  (4 vCPU, 8 GB)
        Lambda 512 MB  : $0.0000166667 / GB-s + $0.20 / 1M requests
        Cloud Run 1cpu : $0.000024 / vCPU-s + $0.0000025 / GB-s
        Azure Functions: $0.000016 / GB-s + $0.20 / 1M executions

        Args:
            median_latency_ms: Measured median end-to-end latency in ms.

        Returns:
            Nested dict with one entry per platform and an
            ``assumptions`` block documenting all pricing inputs.
        """
        n = CFG.benchmark.n_images_cost
        lat_s = median_latency_ms / 1_000.0

        # Total single-thread CPU seconds for 1 M images.
        cpu_s_single = lat_s * n

        # ---- EC2 t3.medium (single-threaded) ----
        ec2_t3_rate  = 0.0416           # USD/hr
        ec2_t3_hrs   = cpu_s_single / 3600.0
        ec2_t3_cost  = ec2_t3_hrs * ec2_t3_rate

        # ---- EC2 c5.xlarge (4 parallel workers) ----
        ec2_c5_rate  = 0.170
        ec2_c5_hrs   = (cpu_s_single / 4.0) / 3600.0
        ec2_c5_cost  = ec2_c5_hrs * ec2_c5_rate

        # ---- AWS Lambda (512 MB, billed duration = latency) ----
        lambda_gb    = 0.512
        lambda_rate  = 0.0000166667    # USD / GB-s
        lambda_req   = 0.20            # USD / 1M requests
        lambda_cost  = (lat_s * lambda_gb * lambda_rate * n) + lambda_req

        # ---- Google Cloud Run (1 vCPU, 256 MB) ----
        cr_cpu_rate  = 0.000024        # USD / vCPU-s
        cr_mem_rate  = 0.0000025       # USD / GB-s
        cr_mem_gb    = 0.256
        cr_cost      = (lat_s * cr_cpu_rate * n) + (lat_s * cr_mem_gb * cr_mem_rate * n)

        # ---- Azure Functions (512 MB, consumption plan) ----
        az_gb        = 0.512
        az_rate      = 0.000016        # USD / GB-s
        az_exec_rate = 0.20            # USD / 1M executions
        az_cost      = (lat_s * az_gb * az_rate * n) + az_exec_rate

        def _fmt(cost: float) -> str:
            return f"${cost:.4f}"

        return {
            "assumptions": {
                "measured_median_latency_ms": round(median_latency_ms, 3),
                "n_images":  n,
                "note": (
                    "Costs are estimates based on measured median latency on the "
                    "benchmarking machine.  Cold-start penalties, network I/O, "
                    "storage, and data-transfer costs are excluded.  Prices are "
                    "approximate as of 2025-Q3 — verify current rates before "
                    "production planning."
                ),
            },
            "on_device": {
                "cost_usd": "$0.0000",
                "notes": (
                    "Zero cloud spend.  Model runs on-device (mobile / edge CPU). "
                    "Requires ~50 MB RAM.  One-time model distribution cost only. "
                    "Recommended for high-throughput real-time liveness checks."
                ),
            },
            "aws_ec2_t3_medium": {
                "cost_usd": _fmt(ec2_t3_cost),
                "hourly_rate": f"${ec2_t3_rate}/hr",
                "cpu_hours":   round(ec2_t3_hrs, 2),
                "notes": "Single-threaded.  Use n_jobs=-1 to parallelize.",
            },
            "aws_ec2_c5_xlarge_parallel": {
                "cost_usd": _fmt(ec2_c5_cost),
                "hourly_rate": f"${ec2_c5_rate}/hr",
                "cpu_hours":   round(ec2_c5_hrs, 2),
                "notes": "4 parallel workers.  Best for batch processing jobs.",
            },
            "aws_lambda_512mb": {
                "cost_usd": _fmt(lambda_cost),
                "duration_charge": _fmt(lat_s * lambda_gb * lambda_rate * n),
                "request_charge":  _fmt(lambda_req),
                "notes": (
                    "Scales to zero.  Cold-start adds 200-800ms.  "
                    "Best for sporadic traffic < 50 req/s."
                ),
            },
            "google_cloud_run": {
                "cost_usd": _fmt(cr_cost),
                "notes": (
                    "1 vCPU, 256 MB.  Scales to zero.  "
                    "Best for REST API with variable traffic."
                ),
            },
            "azure_functions_512mb": {
                "cost_usd": _fmt(az_cost),
                "duration_charge": _fmt(lat_s * az_gb * az_rate * n),
                "execution_charge": _fmt(az_exec_rate),
                "notes": "Consumption plan.  Similar trade-offs to Lambda.",
            },
            "recommendation": (
                "On-device for mobile liveness detection (zero cost, lowest latency). "
                "Cloud Run or Lambda for server-side REST API with variable load. "
                "EC2 c5.xlarge for high-throughput batch processing jobs."
            ),
        }

    # ------------------------------------------------------------------
    # Hardware info
    # ------------------------------------------------------------------

    @staticmethod
    def _hardware_info() -> Dict[str, Any]:
        """Capture hardware specifications for benchmark context.

        Returns:
            Dict with CPU model, core count, RAM, OS, and Python version.
        """
        import sys
        mem = psutil.virtual_memory()
        return {
            "cpu":          platform.processor() or platform.machine(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores":  psutil.cpu_count(logical=True),
            "ram_total":    format_bytes(mem.total),
            "ram_available": format_bytes(mem.available),
            "os":           f"{platform.system()} {platform.release()}",
            "python":       sys.version.split()[0],
        }

    # ------------------------------------------------------------------
    # Logging summary
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(report: Dict[str, Any]) -> None:
        """Log a concise human-readable benchmark summary.

        Args:
            report: Full benchmark report dict.
        """
        lat = report.get("latency_ms", {})
        mem = report.get("memory", {})
        cpu = report.get("cpu_percent", {})
        sz  = report.get("model_sizes", {})
        log.info(
            "Benchmark summary:\n"
            "  Latency  avg=%.1f ms  median=%.1f ms  p95=%.1f ms  p99=%.1f ms\n"
            "  Memory   tracemalloc_peak=%.1f MB  rss_total=%.1f MB\n"
            "  CPU      process=%.1f%%  system=%.1f%%\n"
            "  Model    total=%s",
            lat.get("avg_ms", 0),    lat.get("median_ms", 0),
            lat.get("p95_ms", 0),    lat.get("p99_ms", 0),
            mem.get("tracemalloc_peak_mb", 0),
            mem.get("process_rss_total_mb", 0),
            cpu.get("process_cpu_percent", 0),
            cpu.get("system_cpu_percent", 0),
            sz.get("total", "?"),
        )
