"""Capture training-run provenance: git state, env, command, data hashes.

Every NPZ produced by ``cross_sample_train.py`` gets a sidecar
``<basename>.manifest.json`` written next to it.  A single line is also
appended to ``outputs/cross_sample/RUN_LEDGER.jsonl`` so a full inventory
of every run that produced an artefact in the repo is one ``wc -l`` away.

The manifest is the first thing the audit subagent or a future reviewer
should consult to answer "which exact code state produced this number?"
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "outputs" / "cross_sample" / "RUN_LEDGER.jsonl"


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def _git_state() -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"])
    dirty = bool(_git(["status", "--porcelain"]))
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    diff = _git(["diff", "HEAD"]) if dirty else ""
    diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else ""
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "diff_hash": diff_hash,
    }


def _env() -> dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = ""
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    return info


def _hash_array(arr) -> str:
    """sha256 of an ndarray's bytes; safe for the ints/floats we hash here.

    Uses a canonical contiguous form so dtype changes (float32 vs float64)
    do not silently change the hash.
    """
    import numpy as np
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


@dataclass
class ProvenanceContext:
    """Mutable provenance accumulated during a run; finalised at save time.

    The training driver builds one of these at the top of ``run()``,
    fills ``data_hash`` / ``test_hash`` after dataloaders are constructed,
    and calls ``write()`` once per produced NPZ.
    """
    npz_path: Path
    config: dict[str, Any]
    data_hash: str = ""
    test_hash: str = ""
    started_at: float = field(default_factory=time.time)
    extras: dict[str, Any] = field(default_factory=dict)

    def write(self) -> Path:
        """Write <npz>.manifest.json and append one line to RUN_LEDGER.jsonl.

        Returns the manifest path.  Idempotent on the manifest file
        (overwrites) but always appends to the ledger; the ledger is the
        chronological record of every save call.
        """
        record = {
            "npz_path": str(self.npz_path.relative_to(REPO_ROOT))
                if self.npz_path.is_relative_to(REPO_ROOT) else str(self.npz_path),
            "git": _git_state(),
            "env": _env(),
            "command": sys.argv,
            "config": self.config,
            "data_hash": self.data_hash,
            "test_hash": self.test_hash,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "wallclock_seconds": round(time.time() - self.started_at, 3),
            **self.extras,
        }
        manifest_path = self.npz_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(record, indent=2, default=str))
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return manifest_path


def make_context(npz_path: Path, config: dict[str, Any]) -> ProvenanceContext:
    """Construct a provenance context for a soon-to-be-saved NPZ.

    Pass the path the NPZ will live at and a dict of the run-defining
    config (dataset, method, K, lam, canonical_data_seed, train_seed,
    epochs, ...).  Fill ``ctx.data_hash`` and ``ctx.test_hash`` after
    you have the canonical pool / test indices.  Call ``ctx.write()``
    immediately after ``np.savez_compressed``.
    """
    return ProvenanceContext(npz_path=Path(npz_path), config=dict(config))


__all__ = ["ProvenanceContext", "make_context", "_hash_array"]
