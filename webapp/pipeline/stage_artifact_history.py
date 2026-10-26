"""
Immutable, content-addressed history for OCR job output artifacts.

This module does not modify OCR stage scripts. The Django runner calls
begin_stage_capture() before a stage and finish_stage_capture() afterward.
Only files beneath the job's output directory are captured.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_paths(job: dict[str, Any]) -> tuple[Path, Path, Path]:
    job_dir = Path(job["job_dir"]).resolve()
    output_dir = Path(job.get("output_dir", job_dir / "output")).resolve()
    history_dir = job_dir / "stage_history"
    objects_dir = history_dir / "objects"
    return output_dir, history_dir, objects_dir


def _manifest_path(history_dir: Path) -> Path:
    return history_dir / "manifest.json"


def _load_manifest(history_dir: Path) -> dict[str, Any]:
    path = _manifest_path(history_dir)
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": history_dir.parent.name,
            "created_at": _utc_now(),
            "baseline": None,
            "runs": [],
            "latest_state": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported stage history manifest: {path}")
    data.setdefault("runs", [])
    data.setdefault("latest_state", {})
    return data


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _snapshot_output(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output_dir, history_dir, objects_dir = _job_paths(job)
    if not output_dir.exists():
        return {}
    objects_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, dict[str, Any]] = {}

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        # Defensive exclusion if history is ever configured under output.
        try:
            path.resolve().relative_to(history_dir.resolve())
            continue
        except ValueError:
            pass

        relative = path.relative_to(output_dir).as_posix()
        digest, size = _hash_file(path)
        object_path = objects_dir / digest[:2] / digest
        if not object_path.exists():
            object_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy to a temporary file and atomically publish the immutable object.
            fd, temp_name = tempfile.mkstemp(prefix=digest + ".", suffix=".tmp", dir=object_path.parent)
            os.close(fd)
            try:
                shutil.copyfile(path, temp_name)
                copied_digest, copied_size = _hash_file(Path(temp_name))
                if copied_digest != digest or copied_size != size:
                    raise OSError(f"Artifact changed while being captured: {path}")
                try:
                    os.link(temp_name, object_path)
                except FileExistsError:
                    pass
                except OSError:
                    if not object_path.exists():
                        os.replace(temp_name, object_path)
                        temp_name = ""
            finally:
                if temp_name:
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass

        state[relative] = {
            "sha256": digest,
            "size_bytes": size,
            "object": (Path("objects") / digest[:2] / digest).as_posix(),
        }
    return state


def _diff_states(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    before_paths, after_paths = set(before), set(after)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    modified = sorted(
        path for path in before_paths & after_paths
        if before[path]["sha256"] != after[path]["sha256"]
    )
    unchanged = sorted(
        path for path in before_paths & after_paths
        if before[path]["sha256"] == after[path]["sha256"]
    )
    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "unchanged": unchanged,
    }


def begin_stage_capture(job: dict[str, Any], stage: int) -> dict[str, Any]:
    """
    Capture the pre-stage output state before a stage can overwrite files.
    Returns a token that must be passed to finish_stage_capture().
    """
    _, history_dir, _ = _job_paths(job)
    history_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(history_dir)
    before = _snapshot_output(job)
    if manifest.get("baseline") is None:
        manifest["baseline"] = {
            "captured_at": _utc_now(),
            "reason": "first stage capture; preserves files existing before tracking",
            "state": before,
        }
    token = {
        "stage": int(stage),
        "started_at": _utc_now(),
        "before_state": before,
    }
    _atomic_write_json(_manifest_path(history_dir), manifest)
    return token


def finish_stage_capture(
    job: dict[str, Any],
    token: dict[str, Any],
    *,
    return_code: int,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Capture post-stage output state and append an immutable stage-run record.
    Failed/interrupted stages are recorded too, so partial outputs are not lost.
    """
    _, history_dir, _ = _job_paths(job)
    manifest = _load_manifest(history_dir)
    before = token.get("before_state", {})
    after = _snapshot_output(job)
    status = "succeeded" if return_code == 0 else "failed"
    record = {
        "stage": int(token["stage"]),
        "started_at": token["started_at"],
        "finished_at": _utc_now(),
        "status": status,
        "return_code": int(return_code),
        "error": error,
        "changes": _diff_states(before, after),
        "before_state": before,
        "after_state": after,
    }
    manifest["runs"].append(record)
    manifest["latest_state"] = after
    _atomic_write_json(_manifest_path(history_dir), manifest)
    return record


def load_stage_history(job_dir: str | Path) -> dict[str, Any]:
    """Load the history manifest for inspection or comparison."""
    path = Path(job_dir) / "stage_history" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No stage history manifest found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Invalid or unsupported stage history manifest: {path}")
    return data


def read_artifact(job_dir: str | Path, artifact_ref: dict[str, Any]) -> bytes:
    """Read a captured artifact using a manifest state entry."""
    job_dir = Path(job_dir)
    relative = Path(str(artifact_ref["object"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid artifact object reference")
    path = job_dir / "stage_history" / relative
    content = path.read_bytes()
    expected = str(artifact_ref["sha256"])
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Artifact hash verification failed: {path}")
    return content
