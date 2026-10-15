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
    raise SystemExit(f"rdgen: no GPU available while loading {what}.\n" + _gpu_diagnosis(torch)
                     + "\nTo run on CPU anyway (very slow), set RDGEN_ALLOW_CPU=1.")


def _gpu_diagnosis(torch) -> str:
    """Say why CUDA is unavailable, instead of just that it is."""
    if torch.version.cuda is None:
        return (f"  Cause: this PyTorch ({torch.__version__}) is a CPU-only build.\n"
                "  Fix: install a CUDA build, e.g. pip install torch --index-url "
                "https://download.pytorch.org/whl/cu128")
    lines = [f"  PyTorch {torch.__version__} is a CUDA {torch.version.cuda} build, so reinstalling "
             "it will not help; the GPU is not reachable from this process."]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1"):
        lines.append(f"  Cause: CUDA_VISIBLE_DEVICES='{cvd}' hides all GPUs. Fix: unset CUDA_VISIBLE_DEVICES")
    try:
        torch.cuda.init()
    except Exception as e:  # the real reason is.available() swallows
        lines.append(f"  CUDA error: {str(e).strip().splitlines()[0]}")
    if not os.path.exists("/dev/nvidia-uvm"):
        lines.append("  /dev/nvidia-uvm is missing (common after laptop suspend). "
                     "Fix: sudo modprobe nvidia_uvm, or reboot")
    lines.append("  Other fixes: reboot; check the driver with nvidia-smi; "
                 "add yourself to the video and render groups.")
    return "\n".join(lines)


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
