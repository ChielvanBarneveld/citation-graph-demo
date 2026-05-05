"""manifest.py - reproducibility helper for FORAS thesis runs.

Each training/simulation run writes a small JSON manifest beside its output
so we can reconstruct exactly what produced each artifact. The Streamlit
sidebar's "Run history" panel reads from MANIFEST_DIR.

Usage from any train/sim script:

    from manifest import write_manifest
    write_manifest(
        kind="gcn_train",
        output_path="outputs/gnn_leerlab/scores.parquet",
        params={"epochs": 100, "hidden": 64, "lr": 0.01, "label": "TIAB"},
        metrics={"test_recall_at_10pct": 0.579},
    )

Manifest schema:
    {
      "manifest_id": "<kind>_<utc-ts>",
      "kind": "asreview_sim" | "gcn_train" | "drift_run" | "ppr_baseline" | ...,
      "generated_at": "<ISO8601 UTC>",
      "git_commit": "<short hash or 'no-git'>",
      "branch_label": "<feat/* if applicable>",
      "params": {...},
      "input_hashes": {"<input_name>": "<sha256>"},
      "metrics": {...},
      "output_path": "<relative path>",
      "runtime_seconds": <int or null>,
      "notes": "<free text>"
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "outputs" / "manifests"


def _git_commit() -> str:
    """Return short git commit hash, or 'no-git' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no-git"


def _sha256_file(path: Path, max_bytes: int = 50_000_000) -> str:
    """SHA-256 of a file, capped at `max_bytes` to keep it fast on big CSVs."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while read < max_bytes:
            chunk = f.read(min(1_048_576, max_bytes - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return f"sha256:{h.hexdigest()[:16]}" + ("" if read < max_bytes else f":truncated@{max_bytes}")


def write_manifest(
    *,
    kind: str,
    output_path: str | os.PathLike,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    inputs: dict[str, str | os.PathLike] | None = None,
    branch_label: str = "",
    notes: str = "",
    runtime_seconds: float | None = None,
) -> Path:
    """Write a manifest JSON next to `output_path` and append to MANIFEST_DIR."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_id = f"{kind}_{ts}"

    inp_hashes = {}
    if inputs:
        for name, path in inputs.items():
            try:
                inp_hashes[name] = _sha256_file(Path(path))
            except Exception as exc:
                inp_hashes[name] = f"hash-err: {exc}"

    manifest = {
        "manifest_id": manifest_id,
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "branch_label": branch_label,
        "params": params or {},
        "input_hashes": inp_hashes,
        "metrics": metrics or {},
        "output_path": str(output_path),
        "runtime_seconds": float(runtime_seconds) if runtime_seconds is not None else None,
        "notes": notes,
    }
    target = MANIFEST_DIR / f"{manifest_id}.json"
    target.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return target


def list_manifests(limit: int | None = None) -> list[dict[str, Any]]:
    """Read all manifests, newest first."""
    if not MANIFEST_DIR.exists():
        return []
    out = []
    for p in sorted(MANIFEST_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
        if limit is not None and len(out) >= limit:
            break
    return out


class StopWatch:
    """Tiny context-manager for runtime tracking."""
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0


if __name__ == "__main__":
    # Self-test
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true",
                   help="Write a demo manifest to confirm the system works.")
    p.add_argument("--list", action="store_true",
                   help="List existing manifests.")
    args = p.parse_args()
    if args.demo:
        target = write_manifest(
            kind="self_test",
            output_path="outputs/manifests/_self_test.txt",
            params={"hello": "world"},
            metrics={"answer": 42},
            notes="Demo manifest from `python code/manifest.py --demo`.",
        )
        print(f"wrote demo manifest: {target}")
    if args.list:
        for m in list_manifests(limit=20):
            print(f"{m['generated_at']} | {m['kind']:20} | "
                  f"{m.get('git_commit','?'):8} | {m['output_path']}")
