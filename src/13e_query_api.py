"""
Stage 13E - QUERY / RETRIEVAL API

Purpose
-------
Provide a reusable deterministic query interface over the Stage 13D
entity retrieval index.

This stage does NOT:
    - modify OCR data
    - perform OCR
    - perform semantic/LLM reasoning
    - use Ollama
    - use embeddings/vector search
    - invent relationships

It performs exact normalized entity lookup and returns the complete records,
fields, and provenance already stored by Stages 13-13D.

CLI examples
------------
Query one entity:

    py -3.11 src\\13e_query_api.py ^
        --job-id 20261008T035618Z_d9e536a1 ^
        --query A-01

Write result to a file:

    py -3.11 src\\13e_query_api.py ^
        --job-id 20261008T035618Z_d9e536a1 ^
        --query A-01 ^
        --output output\\query_A-01.json

The core functions in this file can later be imported directly by Django.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "13e.1"


def normalize_value(value: Any) -> str:
    """Use the same deterministic normalization used by Stage 13D."""
    if value is None:
        return ""

    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def find_job_root(job_id: str) -> Path:
    cwd = Path.cwd().resolve()

    candidates = [
        cwd / "output" / "jobs" / job_id,
        cwd.parent / "output" / "jobs" / job_id,
        Path(__file__).resolve().parent.parent
        / "output"
        / "jobs"
        / job_id,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the job directory:\n"
        f"  <project_root>/output/jobs/{job_id}"
    )


def load_retrieval_index(job_id: str) -> tuple[dict[str, Any], Path]:
    job_root = find_job_root(job_id)
    path = (
        job_root
        / "output"
        / "knowledge"
        / "entity_retrieval_index.json"
    )

    return load_json(path), path


def build_entity_lookup(
    retrieval_index: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Build an in-memory lookup using both:
      - exact normalized canonical/display value
      - entity_id

    The normalized-value lookup is exact; it is not fuzzy matching.
    """
    lookup: dict[str, dict[str, Any]] = {}

    entities = retrieval_index.get("entities", [])

    if not isinstance(entities, list):
        raise ValueError("retrieval_index.entities must be a list")

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_id = entity.get("entity_id")

        if entity_id:
            lookup[f"id:{entity_id}"] = entity

        for value in (
            entity.get("canonical_value"),
            entity.get("display_value"),
        ):
            normalized = normalize_value(value)

            if normalized:
                lookup[f"value:{normalized}"] = entity

    return lookup


def query_entity(
    retrieval_index: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    """
    Query one entity using exact normalized-value matching.

    Returns a stable result object suitable for:
        - CLI use
        - Django service code
        - REST API serialization later
    """
    original_query = "" if query is None else str(query)
    normalized_query = normalize_value(original_query)

    if not normalized_query:
        return {
            "schema_version": SCHEMA_VERSION,
            "query": original_query,
            "normalized_query": "",
            "match_type": "none",
            "found": False,
            "result_count": 0,
            "entities": [],
        }

    lookup = build_entity_lookup(retrieval_index)

    entity = lookup.get(f"value:{normalized_query}")

    if entity is None:
        # Allow a caller that already knows the internal entity ID.
        entity = lookup.get(f"id:{original_query.strip()}")

    if entity is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "query": original_query,
            "normalized_query": normalized_query,
            "match_type": "none",
            "found": False,
            "result_count": 0,
            "entities": [],
        }

    # Return the stored entity structure without changing its source data.
    return {
        "schema_version": SCHEMA_VERSION,
        "query": original_query,
        "normalized_query": normalized_query,
        "match_type": "exact_normalized_value",
        "found": True,
        "result_count": 1,
        "entities": [entity],
    }


def query_entities(
    retrieval_index: dict[str, Any],
    queries: list[str],
) -> dict[str, Any]:
    """
    Execute multiple independent exact queries.

    This is useful later for Django/API requests containing multiple entity
    names.
    """
    results = []

    for query in queries:
        results.append(query_entity(retrieval_index, query))

    return {
        "schema_version": SCHEMA_VERSION,
        "query_count": len(queries),
        "results": results,
    }


def summarize_result(result: dict[str, Any]) -> str:
    """Human-readable CLI summary."""
    lines = []

    if not result.get("found"):
        lines.append(
            f"Query: {result.get('query', '')!r} -> NOT FOUND"
        )
        return "\n".join(lines)

    entities = result.get("entities", [])

    lines.append(
        f"Query: {result.get('query', '')!r} -> EXACT MATCH"
    )
    lines.append(f"Entities: {len(entities)}")

    for entity in entities:
        lines.append("")
        lines.append(
            f"Entity: {entity.get('display_value')}"
        )
        lines.append(
            f"Entity ID: {entity.get('entity_id')}"
        )
        lines.append(
            f"Link status: {entity.get('link_status')}"
        )
        lines.append(
            f"Documents: {entity.get('document_count', 0)}"
        )
        lines.append(
            f"Occurrences: {entity.get('occurrence_count', 0)}"
        )

        records = entity.get("records", [])

        for record in records:
            lines.append("")
            lines.append(
                "  "
                f"Document={record.get('document_id')} | "
                f"Page={record.get('page')} | "
                f"Table={record.get('table_index')} | "
                f"Row={record.get('source_row_index')} | "
                f"Source status={record.get('source_status', 'UNKNOWN')}"
            )

            fields = record.get("fields", [])

            for field in fields:
                lines.append(
                    "    "
                    f"{field.get('header')!r} = "
                    f"{field.get('value')!r} "
                    f"(column={field.get('column_index')})"
                )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query the Stage 13D deterministic retrieval index."
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="OCR job ID",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Entity value to query, e.g. A-01",
    )

    parser.add_argument(
        "--output",
        help="Optional JSON output path",
    )

    args = parser.parse_args()

    try:
        retrieval_index, index_path = load_retrieval_index(
            args.job_id
        )

        result = query_entity(
            retrieval_index,
            args.query,
        )

        print("=" * 80)
        print("STAGE 13E - QUERY / RETRIEVAL API")
        print("=" * 80)
        print(f"Job:                         {args.job_id}")
        print(f"Query:                       {args.query!r}")
        print(f"Index:                       {index_path}")
        print(f"Match type:                  {result['match_type']}")
        print(f"Found:                       {result['found']}")
        print(f"Result count:                {result['result_count']}")
        print("=" * 80)
        print(summarize_result(result))

        if args.output:
            output_path = Path(args.output)

            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    result,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write("\n")

            print("=" * 80)
            print(f"JSON result:                 {output_path}")

        return 0 if result["found"] else 2

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
