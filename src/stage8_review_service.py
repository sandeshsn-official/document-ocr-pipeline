"""
Django-facing service adapter for the existing Stage 08 OCR review engine.

The Stage 08 decision logic itself is NOT reimplemented here.
This adapter loads the existing 08_manual_visual_review.py backend and
delegates human decisions to it.

Job-specific OCR/review files are read from:
    output/jobs/<job_id>/

The learning memory remains shared at:
    output/learning/
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE8_FILE = PROJECT_ROOT / "src" / "08_manual_visual_review.py"

_stage8 = None


def _load_stage8():
    """Load Stage 08 without starting its Streamlit UI."""
    global _stage8

    if _stage8 is not None:
        return _stage8

    if not STAGE8_FILE.exists():
        raise FileNotFoundError(
            f"Stage 08 source not found: {STAGE8_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "ocr_stage08_backend",
        STAGE8_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Stage 08 module: {STAGE8_FILE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _stage8 = module
    return module


def _job_paths(job_dir: Path) -> dict[str, Path]:
    """
    Build job-local paths while keeping learning memory global.
    """

    job_dir = Path(job_dir).resolve()

    return {
        "job_dir": job_dir,
        "input_dir": job_dir / "input",
        "output_dir": job_dir / "output",
        "verified_dir": job_dir / "output" / "verified_data",
        "review_dir": job_dir / "output" / "verified_data" / "review_required",
        "review_crops_dir": job_dir / "output" / "verified_data" / "review_crops",
        "pages_dir": job_dir / "output" / "pages",

        # GLOBAL — intentionally shared between jobs.
        "learning_dir": PROJECT_ROOT / "output" / "learning",
        "decisions_file": PROJECT_ROOT / "output" / "learning" / "manual_review_decisions.json",
        "memory_file": PROJECT_ROOT / "output" / "learning" / "ocr_learning_memory.json",
    }


def _configure_stage8_for_job(job_dir: Path):
    """
    Temporarily redirect Stage 08's path globals to the requested job.

    Learning memory remains global.
    """

    stage8 = _load_stage8()
    paths = _job_paths(job_dir)

    stage8.BASE_DIR = paths["job_dir"]

    stage8.INPUT_DIR = paths["input_dir"]
    stage8.VERIFIED_DIR = paths["verified_dir"]
    stage8.REVIEW_DIR = paths["review_dir"]
    stage8.REVIEW_CROPS_DIR = paths["review_crops_dir"]
    stage8.PAGES_DIR = paths["pages_dir"]

    # Global learning / decision storage.
    stage8.LEARNING_DIR = paths["learning_dir"]
    stage8.DECISIONS_FILE = paths["decisions_file"]
    stage8.MEMORY_FILE = paths["memory_file"]

    # Stage 08 may expose additional path constants.
    if hasattr(stage8, "SKIPS_FILE"):
        stage8.SKIPS_FILE = (
            paths["verified_dir"] / "manual_review_skips.json"
        )

    return stage8


def _review_candidates(
    job_dir: Path,
    document_name: str,
) -> list[dict[str, Any]]:
    """Build the canonical unresolved Stage 08 review queue."""

    stage8 = _configure_stage8_for_job(job_dir)

    review_file = stage8._visual_review_file_for(document_name)

    historical_items: list[dict[str, Any]] = []

    if review_file is not None:
        review_data = stage8.load_json(review_file)

        if isinstance(review_data, dict):
            raw_items = review_data.get("items", [])

            if isinstance(raw_items, list):
                historical_items = [
                    dict(x)
                    for x in raw_items
                    if isinstance(x, dict)
                ]

    verified_file = (
        stage8.VERIFIED_DIR /
        f"{document_name}.json"
    )

    if not verified_file.exists() and review_file is not None:
        verified_file = (
            stage8.VERIFIED_DIR /
            f"{review_file.stem}.json"
        )

    if not verified_file.exists():
        return []

    final_data = stage8.load_json(verified_file)

    if isinstance(final_data, dict) and not final_data.get("document"):
        final_data["document"] = document_name

    items_by_key: dict[Any, dict[str, Any]] = {}

    for raw in historical_items:
        item = dict(raw)
        item.setdefault("document", document_name)

        if stage8._visual_item_requires_current_review(
            final_data,
            item,
        ):
            items_by_key[
                stage8.decision_key(item)
            ] = item

    canonical_unresolved = (
        stage8._visual_collect_canonical_unresolved(
            final_data,
            document_name,
        )
    )

    for key, item in canonical_unresolved.items():
        items_by_key.setdefault(key, item)

    return [
        dict(item)
        for item in items_by_key.values()
        if isinstance(item, dict)
    ]

def _run_stage8_auto_reuse(
    job_dir: Path,
    document_name: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run the existing Stage 08 automatic reuse chain.

    Candidates already resolved by an existing human decision, exact
    evidence, learned evidence, or another Stage 08 automatic mechanism
    are removed from the Django human-review queue.

    IMPORTANT:
    Do NOT rebuild the queue from Stage 06 after auto-processing.
    Stage 06 review_required.json is a historical/current-review source,
    not the final Django queue after Stage 08 automatic reuse.
    """
    if not candidates:
        return []

    (
        stage8,
        review_file,
        verified_file,
        final_data,
        existing_decisions,
        existing_lookup,
        same_content_lookup,
        memory,
    ) = _load_state(
        job_dir,
        document_name,
    )

    unresolved = []

    for raw_item in candidates:
        if not isinstance(raw_item, dict):
            continue

        item = dict(raw_item)
        item.setdefault("document", document_name)

        handled, final_data, _kind = stage8._visual_process_auto_item(
            item,
            final_data,
            document_name,
            review_file,
            existing_decisions,
            existing_lookup,
            same_content_lookup,
            memory,
        )

        if not handled:
            unresolved.append(item)

    return unresolved


def get_documents(job_dir: Path) -> list[str]:
    """Return documents belonging to this job only."""

    stage8 = _configure_stage8_for_job(job_dir)

    return sorted(
        stage8.get_active_document_names()
    )


def get_review_queue(
    job_dir: Path,
    document_name: str,
) -> list[dict[str, Any]]:
    """
    Return unresolved review items for this job/document.

    Existing Stage 08 automatic reuse is applied before the queue is
    returned. Persistent skips remain job-local.
    """
    stage8 = _configure_stage8_for_job(job_dir)

    candidates = _review_candidates(
        job_dir,
        document_name,
    )

    candidates = _run_stage8_auto_reuse(
        job_dir,
        document_name,
        candidates,
    )

    skipped_keys = set()

    if stage8.SKIPS_FILE.exists():
        try:
            skip_data = stage8.load_json(
                stage8.SKIPS_FILE
            )

            if isinstance(skip_data, dict):
                raw_items = skip_data.get(
                    "skipped_items",
                    [],
                )

                if isinstance(raw_items, list):
                    for item in raw_items:
                        if isinstance(item, dict):
                            skipped_keys.add(
                                stage8._visual_skip_key(item)
                            )

            elif isinstance(skip_data, list):
                for item in skip_data:
                    if isinstance(item, dict):
                        skipped_keys.add(
                            stage8._visual_skip_key(item)
                        )

        except Exception:
            pass

    if not skipped_keys:
        return candidates

    return [
        item
        for item in candidates
        if stage8._visual_skip_key(item)
        not in skipped_keys
    ]

def get_review_item(
    job_dir: Path,
    document_name: str,
    index: int,
) -> dict[str, Any] | None:
    """Return one unresolved item by zero-based queue index."""

    queue = get_review_queue(
        job_dir,
        document_name,
    )

    if index < 0 or index >= len(queue):
        return None

    return queue[index]


def _load_state(
    job_dir: Path,
    document_name: str,
):
    """Load the exact Stage 08 state required by its decision handler."""

    stage8 = _configure_stage8_for_job(job_dir)

    review_file = stage8._visual_review_file_for(
        document_name
    )

    if review_file is None:
        raise FileNotFoundError(
            f"No Stage 08 review file for: {document_name}"
        )

    verified_file = (
        stage8.VERIFIED_DIR /
        f"{review_file.stem}.json"
    )

    if not verified_file.exists():
        verified_file = (
            stage8.VERIFIED_DIR /
            f"{document_name}.json"
        )

    if not verified_file.exists():
        raise FileNotFoundError(
            f"Verified JSON not found for: {document_name}"
        )

    final_data = stage8.load_json(
        verified_file
    )

    if (
        isinstance(final_data, dict)
        and not final_data.get("document")
    ):
        final_data["document"] = document_name

    (
        existing_decisions,
        existing_lookup,
        same_content_lookup,
    ) = stage8._visual_load_decision_state()

    memory = stage8.load_memory()

    return (
        stage8,
        review_file,
        verified_file,
        final_data,
        existing_decisions,
        existing_lookup,
        same_content_lookup,
        memory,
    )


def _apply_human_decision(
    job_dir: Path,
    document_name: str,
    item: dict[str, Any],
    verified_text: str,
    decision: str,
):
    """
    Delegate the actual human decision to Stage 08 unchanged.
    """

    (
        stage8,
        review_file,
        verified_file,
        final_data,
        existing_decisions,
        existing_lookup,
        same_content_lookup,
        memory,
    ) = _load_state(
        job_dir,
        document_name,
    )

    saved = stage8._visual_record_human_decision(
        item,
        final_data,
        review_file,
        verified_file,
        document_name,
        existing_decisions,
        existing_lookup,
        same_content_lookup,
        memory,
        verified_text,
        decision,
    )

    if saved is None:
        raise ValueError(
            "Stage 08 could not resolve this review item "
            "to its canonical target."
        )

    stage8._visual_record_review_outcome(
        document_name,
        item,
        decision,
        verified_text,
        "HUMAN",
    )

    return saved


def confirm_original(
    job_dir: Path,
    document_name: str,
    item: dict[str, Any],
):
    text = str(
        item.get("original_text", "")
    )

    return _apply_human_decision(
        job_dir,
        document_name,
        item,
        text,
        "USER_CONFIRMED_ORIGINAL",
    )


def confirm_source(
    job_dir: Path,
    document_name: str,
    item: dict[str, Any],
):
    text = str(
        item.get("source_text", "")
    )

    return _apply_human_decision(
        job_dir,
        document_name,
        item,
        text,
        "USER_CONFIRMED_SOURCE",
    )


def save_correction(
    job_dir: Path,
    document_name: str,
    item: dict[str, Any],
    correction: str,
):
    correction = str(correction).strip()

    if not correction:
        raise ValueError(
            "Correction cannot be empty."
        )

    return _apply_human_decision(
        job_dir,
        document_name,
        item,
        correction,
        "USER_ENTERED_CORRECTION",
    )


def skip_item(
    job_dir: Path,
    document_name: str,
    item: dict[str, Any],
):
    """
    Record audit-only USER_SKIPPED and persist the skip
    inside this job.
    """

    stage8 = _configure_stage8_for_job(
        job_dir
    )

    result = stage8._visual_record_review_outcome(
        document_name,
        item,
        "USER_SKIPPED",
        None,
        "HUMAN",
    )

    skip_file = stage8.SKIPS_FILE

    if skip_file.exists():
        skip_data = stage8.load_json(
            skip_file
        )
    else:
        skip_data = {
            "version": 2,
            "total_items": 0,
            "skipped_items": [],
        }

    if not isinstance(skip_data, dict):
        skip_data = {
            "version": 2,
            "total_items": 0,
            "skipped_items": [],
        }

    skipped_items = skip_data.get(
        "skipped_items",
        [],
    )

    if not isinstance(skipped_items, list):
        skipped_items = []

    skip_key = stage8._visual_skip_key(item)

    skip_record = dict(item)
    skip_record["document"] = document_name
    skip_record["skip_key"] = skip_key

    existing_keys = {
        str(existing.get("skip_key", ""))
        for existing in skipped_items
        if isinstance(existing, dict)
    }

    if skip_key not in existing_keys:
        skipped_items.append(skip_record)

    skip_data["version"] = 2
    skip_data["skipped_items"] = skipped_items
    skip_data["total_items"] = len(skipped_items)

    stage8.save_json(
        skip_file,
        skip_data,
    )

    return result


def get_source_page(
    job_dir: Path,
    document_name: str,
    page_number: Any,
):
    """Return this job's source page image."""

    stage8 = _configure_stage8_for_job(
        job_dir
    )

    return stage8._visual_page_image(
        document_name,
        page_number,
    )


def get_progress(
    job_dir: Path,
    document_name: str,
) -> dict[str, int]:
    """Return current review progress."""

    stage8 = _configure_stage8_for_job(
        job_dir
    )

    queue = get_review_queue(
    job_dir,
    document_name,
)

    human = 0
    skipped = 0

    if stage8.DECISIONS_FILE.exists():
        try:
            data = stage8.load_json(
                stage8.DECISIONS_FILE
            )

            if isinstance(data, dict):
                outcomes = data.get(
                    "review_outcomes",
                    [],
                )

                if isinstance(outcomes, list):
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue

                        if (
                            str(
                                outcome.get(
                                    "document",
                                    "",
                                )
                            ).strip()
                            != document_name
                        ):
                            continue

                        kind = str(
                            outcome.get(
                                "outcome",
                                "",
                            )
                        ).strip().upper()

                        if kind in {
                            "USER_CONFIRMED_ORIGINAL",
                            "USER_CONFIRMED_SOURCE",
                            "USER_ENTERED_CORRECTION",
                        }:
                            human += 1

                        elif kind == "USER_SKIPPED":
                            skipped += 1

        except Exception:
            pass

    return {
        "unresolved": len(
            get_review_queue(
                job_dir,
                document_name,
            )
        ),
        "human_actions": human,
        "skipped": skipped,
    }