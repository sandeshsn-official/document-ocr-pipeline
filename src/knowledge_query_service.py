"""
Django-facing service adapter for the Stage 13 knowledge/query layer.

This adapter does not reimplement OCR or Stage 13 decision logic.

It reads the existing Stage 13F cross-document index and the Stage 13A
knowledge index, then delegates deterministic querying to Stage 13G.

Designed for later use by Django views/API endpoints.

Stages 1-12 and Stage 13A-13G remain untouched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE13G_FILE = PROJECT_ROOT / "src" / "13g_generic_knowledge_query.py"


_stage13g = None


def _load_stage13g():
    """Load Stage 13G without executing its CLI entry point."""
    global _stage13g

    if _stage13g is not None:
        return _stage13g

    if not STAGE13G_FILE.exists():
        raise FileNotFoundError(
            f"Stage 13G source not found: {STAGE13G_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "ocr_stage13g_backend",
        STAGE13G_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Stage 13G module: {STAGE13G_FILE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _stage13g = module
    return module


def _job_knowledge_dir(job_id: str) -> Path:
    """
    Return the knowledge directory for a specific OCR job.

    Stage 13 knowledge remains job-local:
        output/jobs/<job_id>/output/knowledge/
    """
    return (
        PROJECT_ROOT
        / "output"
        / "jobs"
        / str(job_id)
        / "output"
        / "knowledge"
    )


def _load_cross_document_index(job_id: str) -> dict[str, Any]:
    """
    Load the authoritative Stage 13F derived index for the requested job.
    """
    path = _job_knowledge_dir(job_id) / "cross_document_query_index.json"

    if not path.exists():
        raise FileNotFoundError(
            "Stage 13F index not found for job "
            f"{job_id}: {path}"
        )

    stage13g = _load_stage13g()
    return stage13g.load_json(path)


def _load_knowledge_index(job_id: str) -> dict[str, Any]:
    """
    Load the Stage 13A structural knowledge index for the requested job.

    Stage 13G uses this index for exact generic value lookup when the
    query does not resolve as an entity.
    """
    path = _job_knowledge_dir(job_id) / "knowledge_index.json"

    if not path.exists():
        raise FileNotFoundError(
            "Stage 13A knowledge index not found for job "
            f"{job_id}: {path}"
        )

    stage13g = _load_stage13g()
    return stage13g.load_json(path)


def query_knowledge(
    job_id: str,
    query: str,
) -> dict[str, Any]:
    """
    Execute the deterministic Stage 13G query against a job.

    Stage 13G first attempts exact entity lookup through the Stage 13F
    cross-document index. If no entity matches, it performs exact generic
    value lookup through the Stage 13A knowledge index.

    Returns a JSON-serializable dictionary suitable for a Django
    JsonResponse.
    """
    if not str(job_id).strip():
        raise ValueError("job_id is required.")

    if not str(query).strip():
        raise ValueError("query is required.")

    stage13g = _load_stage13g()

    index = _load_cross_document_index(job_id)
    knowledge_index = _load_knowledge_index(job_id)

    return stage13g.query_index(
        index,
        knowledge_index,
        query,
    )


def get_entity(
    job_id: str,
    entity_id: str,
) -> dict[str, Any]:
    """
    Retrieve one entity by its Stage 13C entity ID.

    This is an exact ID lookup. No fuzzy or semantic matching is performed.
    """
    if not str(job_id).strip():
        raise ValueError("job_id is required.")

    if not str(entity_id).strip():
        raise ValueError("entity_id is required.")

    index = _load_cross_document_index(job_id)

    entities = index.get("entities", [])

    if isinstance(entities, dict):
        entity = entities.get(str(entity_id))
        if isinstance(entity, dict):
            return {
                "match_type": "EXACT_ENTITY_ID",
                "match_count": 1,
                "result": entity,
            }

    elif isinstance(entities, list):
        for entity in entities:
            if (
                isinstance(entity, dict)
                and str(entity.get("entity_id", "")) == str(entity_id)
            ):
                return {
                    "match_type": "EXACT_ENTITY_ID",
                    "match_count": 1,
                    "result": entity,
                }

    return {
        "match_type": "NO_MATCH",
        "match_count": 0,
        "result": None,
    }


def knowledge_index_status(job_id: str) -> dict[str, Any]:
    """
    Return lightweight status information for the Stage 13F knowledge index.
    """
    index = _load_cross_document_index(job_id)

    statistics = index.get("statistics", {})

    return {
        "job_id": str(job_id),
        "available": True,
        "schema_version": index.get("schema_version"),
        "stage": index.get("stage"),
        "stage_name": index.get("stage_name"),
        "statistics": statistics,
    }


__all__ = [
    "query_knowledge",
    "get_entity",
    "knowledge_index_status",
]