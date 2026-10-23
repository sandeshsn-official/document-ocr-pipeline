"""Read-only learning inspection and guarded JSON updates.

Stage 9 is not executed here. Updates are explicitly requested by callers
and use the cross-process locking and atomic-write utility.
"""

import json
from pathlib import Path

from .safe_json import update_json_safely


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_learning_paths(project_root=None):
    """Return canonical paths for the shared learning files."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    learning_dir = root / "output" / "learning"

    return {
        "learning_dir": learning_dir,
        "decisions_file": learning_dir / "manual_review_decisions.json",
        "memory_file": learning_dir / "ocr_learning_memory.json",
    }


def read_json_file(path):
    """Read JSON without modifying the source file."""
    path = Path(path)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def inspect_learning_ledger(project_root=None):
    """Return a read-only summary of the shared learning files."""
    paths = get_learning_paths(project_root)
    decisions_data = read_json_file(paths["decisions_file"])
    memory_data = read_json_file(paths["memory_file"])

    decisions = (
        decisions_data.get("decisions", [])
        if isinstance(decisions_data, dict)
        else []
    )
    outcomes = (
        decisions_data.get("review_outcomes", [])
        if isinstance(decisions_data, dict)
        else []
    )

    if not isinstance(decisions, list):
        decisions = []
    if not isinstance(outcomes, list):
        outcomes = []

    trusted_types = {
        "USER_CONFIRMED_ORIGINAL",
        "USER_CONFIRMED_SOURCE",
        "USER_ENTERED_CORRECTION",
        "USER_CORRECTION",
    }

    trusted_count = sum(
        1
        for item in decisions
        if isinstance(item, dict)
        and item.get("decision") in trusted_types
    )

    return {
        "decisions_file_exists": paths["decisions_file"].is_file(),
        "memory_file_exists": paths["memory_file"].is_file(),
        "decision_count": len(decisions),
        "review_outcome_count": len(outcomes),
        "trusted_decision_count": trusted_count,
        "memory_version": (
            memory_data.get("version")
            if isinstance(memory_data, dict)
            else None
        ),
        "read_only": True,
    }


def update_learning_json(path, updater, project_root=None, timeout=30.0):
    """Safely update one canonical learning JSON file.

    The target must be either the shared decisions ledger or learning memory.
    This function does not decide what should be learned and never runs Stage 9.
    """
    paths = get_learning_paths(project_root)
    target = Path(path).resolve()

    allowed_targets = {
        paths["decisions_file"].resolve(),
        paths["memory_file"].resolve(),
    }

    if target not in allowed_targets:
        raise ValueError(
            "Updates are restricted to the canonical learning JSON files."
        )

    return update_json_safely(
        target,
        updater,
        timeout=timeout,
    )
