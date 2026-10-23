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
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE13OLLAMA_FILE = PROJECT_ROOT / "src" / "ollama_natural_language_query.py"


_stage13ollama = None


def _load_stage13ollama():
    """Load the Ollama query layer without executing its CLI entry point."""
    global _stage13ollama

    if _stage13ollama is not None:
        return _stage13ollama

    if not STAGE13OLLAMA_FILE.exists():
        raise FileNotFoundError(
            f"Ollama query source not found: {STAGE13OLLAMA_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "ocr_stage13ollama_backend",
        STAGE13OLLAMA_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Ollama query module: {STAGE13OLLAMA_FILE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _stage13ollama = module
    return module


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required knowledge index not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


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

    return _load_json(path)


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

    return _load_json(path)


def query_knowledge(
    job_id: str,
    query: str,
) -> dict[str, Any]:
    """
    Interpret and answer a natural-language question with local Ollama.

    Ollama interprets the question; Stage 13A/13F remain the evidence sources.
    The OCR indexes and provenance are read-only.

    Returns a JSON-serializable dictionary suitable for a Django
    JsonResponse.
    """
    if not str(job_id).strip():
        raise ValueError("job_id is required.")

    if not str(query).strip():
        raise ValueError("query is required.")

    ollama_query = _load_stage13ollama()
    index = _load_cross_document_index(job_id)
    knowledge_index = _load_knowledge_index(job_id)

    return ollama_query.query_knowledge_with_ollama(
        query,
        index,
        knowledge_index,
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