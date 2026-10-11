"""Run-time helpers: timestamped progress logging and a GPU check.

Real-model runs on a CPU are silently hundreds of times slower, so loading a
model or verifier refuses to fall back to the CPU unless RDGEN_ALLOW_CPU=1.
"""
from __future__ import annotations

import os
import sys
import time

_T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line on stderr (lands in the job's log file)."""
    el = int(time.time() - _T0)
    print(f"[{time.strftime('%H:%M:%S')} +{el // 3600}h{el % 3600 // 60:02d}m] {msg}",
          file=sys.stderr, flush=True)


def pick_device(what: str) -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("RDGEN_ALLOW_CPU") == "1":
        log(f"WARNING: no GPU visible; running {what} on CPU (RDGEN_ALLOW_CPU=1). This is very slow.")
        return "cpu"
    raise SystemExit(
        f"rdgen: no GPU visible to PyTorch while loading {what} "
        f"(torch {torch.__version__}, torch.cuda.is_available() is False). "
        "Install a CUDA build of PyTorch, or set RDGEN_ALLOW_CPU=1 to run on CPU anyway.")


class Progress:
    """Logs '<label>: i/n (x%)' about every 10% of a long loop."""

    def __init__(self, label: str, total: int, min_total: int = 20):
        self.label, self.total = label, total
        self.on = total >= min_total
        self.step = max(1, total // 10)
        self.next = self.step
        self.t0 = time.time()

    def update(self, done: int) -> None:
        if self.on and (done >= self.next or done == self.total):
            rate = done / max(time.time() - self.t0, 1e-9)
            eta = (self.total - done) / rate if rate > 0 else 0
            log(f"  {self.label}: {done}/{self.total} ({100 * done // self.total}%), "
                f"about {int(eta // 60)} min left")
            while self.next <= done:
                self.next += self.step
