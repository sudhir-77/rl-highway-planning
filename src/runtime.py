"""Thread caps and peak-memory sampling."""

from __future__ import annotations

import os
import platform
import threading

import psutil


def limit_threads(n: int = 4) -> None:
    """Cap BLAS and Torch threads. Safe to call more than once."""
    value = str(n)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass


class PeakRSS:
    """Sample this process and its children until ``stop``."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="peak-rss", daemon=True)

    def start(self) -> "PeakRSS":
        self._sample()
        self._thread.start()
        return self

    def _sample(self) -> None:
        proc = psutil.Process()
        total = int(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except psutil.Error:
                continue
        if total > self.peak_bytes:
            self.peak_bytes = total

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def stop(self) -> float:
        self._sample()
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak_bytes / (1024.0**3)


def hardware_info(peak_rss_gb: float, device: str) -> dict:
    gpu = None
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except ImportError:
            gpu = None
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024.0**3), 2),
        "gpu": gpu,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": round(float(peak_rss_gb), 4),
    }
