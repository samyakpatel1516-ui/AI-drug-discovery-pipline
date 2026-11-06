"""Run registry: provenance, artifact tracking, reproducibility.

Every stage writes into a single run directory and appends to a manifest.
If you cannot say which code, config and data produced a candidate list, the
list is not auditable -- and in a regulated setting that matters as much as
the modelling.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger("dd_pipeline")


def set_global_seed(seed: int) -> None:
    """Seed the RNGs this pipeline touches.

    Note what this *cannot* do: PYTHONHASHSEED only takes effect at
    interpreter start, so setting it here would not make string hashing
    deterministic in this process -- it would only mislead. Code that must be
    reproducible therefore has to avoid depending on set/dict iteration order
    of strings rather than relying on a seed. `BRICSRecombiner._decompose`
    sorts its fragment set for exactly this reason.
    """
    random.seed(seed)
    np.random.seed(seed)


def file_digest(path: str | Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_digest(df: pd.DataFrame) -> str:
    """Content hash of a dataframe, stable across column order."""
    h = hashlib.sha256()
    for col in sorted(df.columns):
        h.update(str(col).encode())
        h.update(pd.util.hash_pandas_object(df[col], index=False).values.tobytes())
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class Run:
    """A single pipeline execution rooted at one directory."""

    root: Path
    project: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    @classmethod
    def create(cls, run_root: str | Path, project: str, run_id: str | None = None) -> "Run":
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        root = Path(run_root) / f"{project}_{run_id}"
        for sub in ("data", "models", "reports", "logs", "figures"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        run = cls(root=root, project=project)
        run._write_env()
        return run

    def _write_env(self) -> None:
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "packages": _package_versions(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (self.root / "environment.json").write_text(json.dumps(env, indent=2))

    # ---- artifact helpers -------------------------------------------------
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_table(self, df: pd.DataFrame, name: str, stage: str) -> Path:
        out = self.path("data", f"{name}.parquet")
        try:
            df.to_parquet(out, index=False)
        except Exception:  # pyarrow missing -> csv fallback, still auditable
            out = self.path("data", f"{name}.csv")
            df.to_csv(out, index=False)
        self.artifacts.append(
            {
                "name": name,
                "stage": stage,
                "path": str(out.relative_to(self.root)),
                "rows": int(len(df)),
                "columns": list(map(str, df.columns)),
                "sha256": file_digest(out),
                "content_hash": frame_digest(df),
            }
        )
        LOG.info("wrote %s (%d rows) -> %s", name, len(df), out)
        return out

    def save_json(self, obj: Any, name: str, stage: str = "misc") -> Path:
        out = self.path("reports", f"{name}.json")
        out.write_text(json.dumps(obj, indent=2, default=str))
        self.artifacts.append(
            {"name": name, "stage": stage, "path": str(out.relative_to(self.root)),
             "sha256": file_digest(out)}
        )
        return out

    def log_metrics(self, stage: str, metrics: dict[str, Any]) -> None:
        self.metrics.setdefault(stage, {}).update(metrics)
        LOG.info("metrics[%s]: %s", stage, _fmt(metrics))

    def finalize(self, config_dump: dict[str, Any]) -> Path:
        manifest = {
            "project": self.project,
            "run_dir": str(self.root),
            "duration_s": round(time.time() - self.started, 2),
            "config": config_dump,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        out = self.root / "manifest.json"
        out.write_text(json.dumps(manifest, indent=2, default=str))
        LOG.info("run manifest -> %s", out)
        return out


def read_run_table(run_root: str | Path, name: str) -> pd.DataFrame | None:
    """Read a table a stage wrote into a run directory, parquet or csv.

    Reading an artifact back by name (rather than threading every intermediate
    frame through function signatures) keeps the run directory as the single
    source of truth for what a stage actually produced.
    """
    for ext in (".parquet", ".csv"):
        p = Path(run_root) / "data" / f"{name}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return None


def _package_versions() -> dict[str, str]:
    mods = ["numpy", "pandas", "sklearn", "scipy", "rdkit", "xgboost", "pydantic"]
    vers: dict[str, str] = {}
    for m in mods:
        try:
            mod = __import__(m)
            vers[m] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[m] = "absent"
    return vers


def _fmt(d: dict[str, Any]) -> str:
    bits = []
    for k, v in d.items():
        bits.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(bits)


def setup_logging(run: Run | None = None, level: int = logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run is not None:
        handlers.append(logging.FileHandler(run.root / "logs" / "pipeline.log"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    logging.getLogger("rdkit").setLevel(logging.ERROR)
