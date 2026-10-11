"""Pre-registration lock (rigour protocol 12.1-12.2).

``lock`` hashes the pre-registration file and writes ``prereg.lock``.
``require_lock`` refuses to evaluate the shifted test unless the lock exists
and still matches the file - so S2 cannot be looked at before the hypotheses,
metrics and analysis are fixed, and any later edit to them is detectable.
Overriding is possible but recorded in the run's provenance.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class PreregError(RuntimeError):
    pass


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock(prereg_path: str | Path, lock_path: str | Path = "prereg.lock") -> dict:
    rec = {"prereg": str(prereg_path), "sha256": file_hash(prereg_path),
           "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(lock_path).write_text(json.dumps(rec, indent=2))
    return rec


def require_lock(prereg_path: str | Path, lock_path: str | Path = "prereg.lock",
                 override: bool = False) -> dict:
    lp = Path(lock_path)
    if not lp.exists():
        if override:
            return {"status": "override_no_lock"}
        raise PreregError("Shifted-test evaluation requires a pre-registration lock; run "
                          "`rdgen prereg-lock` first (or pass --override-prereg, which is logged).")
    rec = json.loads(lp.read_text())
    if rec["sha256"] != file_hash(prereg_path):
        if override:
            return {"status": "override_hash_mismatch", **rec}
        raise PreregError("Pre-registration file changed after locking. Revert it or re-lock and "
                          "report the amendment.")
    return {"status": "locked", **rec}
