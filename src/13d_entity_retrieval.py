"""
Stage 13D - DETERMINISTIC ENTITY RETRIEVAL

Consumes:
    - Stage 13 knowledge_index.json
    - Stage 13C entity_links.json

Important:
    Stage 13 records store their complete row data under:
        record["fields"]["column_N"]

This stage therefore reads the actual Stage 13 schema directly. It does not
assume a "cells" array exists inside each record.

No Ollama / LLM / embeddings / semantic inference are used.

Output:
    output/jobs/<job_id>/output/knowledge/entity_retrieval_index.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "13d.3"


def normalize_value(value: Any) -> str:
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


def build_record_map(
    knowledge_index: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = knowledge_index.get("records", [])

    if not isinstance(records, list):
        raise ValueError("knowledge_index.records must be a list")

    result: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        record_id = record.get("record_id")
        if record_id:
            result[str(record_id)] = record

    return result


def extract_record_fields(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract every field from the real Stage 13 record schema:

        record["fields"]["column_1"]
        record["fields"]["column_2"]
        ...

    No column count or header name is hardcoded.
    """
    fields = record.get("fields")

    if not isinstance(fields, dict):
        return []

    extracted: list[dict[str, Any]] = []

    for field_key, field in fields.items():
        if not isinstance(field, dict):
            continue

        column_index = field.get("column_index")

        if column_index is None:
            match = re.search(r"(\d+)$", str(field_key))
            if match:
                column_index = int(match.group(1))

        value = (
            field.get("value")
            if field.get("value") is not None
            else field.get("final_verified_text")
            if field.get("final_verified_text") is not None
            else field.get("verified_text")
            if field.get("verified_text") is not None
            else field.get("text")
            if field.get("text") is not None
            else ""
        )

        provenance = field.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        extracted.append(
            {
                "field_key": str(field_key),
                "column_index": column_index,
                "header": field.get("header"),
                "value": value,
                "normalized_value": (
                    field.get("normalized_value")
                    or normalize_value(value)
                ),
                "value_type": field.get("value_type"),
                "role_candidates": field.get(
                    "role_candidates",
                    {},
                ),
                "bbox": provenance.get("bbox"),
                "verification_status": provenance.get(
                    "verification_status"
                ),
                "source_ocr_confidence": provenance.get(
                    "source_ocr_confidence"
                ),
                "verification_reason": provenance.get(
                    "verification_reason"
                ),
                "provenance": provenance,
            }
        )

    extracted.sort(
        key=lambda item: (
            item["column_index"]
            if isinstance(item["column_index"], (int, float))
            else 10**9,
            item["field_key"],
        )
    )

    return extracted


def build_retrieval_entities(
    entity_links: dict[str, Any],
    record_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entities = entity_links.get("entities", [])

    if not isinstance(entities, list):
        raise ValueError("entity_links.entities must be a list")

    retrieval_entities: list[dict[str, Any]] = []
    exact_lookup: dict[str, str] = {}

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_id = entity.get("entity_id")
        canonical = normalize_value(entity.get("canonical_value"))

        if not entity_id or not canonical:
            continue

        entity_id = str(entity_id)
        exact_lookup[canonical] = entity_id

        records: list[dict[str, Any]] = []
        seen_records: set[str] = set()

        for occurrence in entity.get("occurrences", []) or []:
            if not isinstance(occurrence, dict):
                continue

            record_id = occurrence.get("record_id")
            if not record_id:
                continue

            record_id = str(record_id)

            if record_id in seen_records:
                continue

            seen_records.add(record_id)

            source_record = record_map.get(record_id)

            if source_record is None:
                # Preserve the entity occurrence even if the source record
                # cannot be resolved, but do not invent row fields.
                source_record = {}

            fields = extract_record_fields(source_record)

            document_id = (
                occurrence.get("document_id")
                or source_record.get("document_id")
            )

            page = occurrence.get("page")
            if page is None:
                page = source_record.get("page")

            table_id = occurrence.get("table_id")
            if table_id is None:
                table_id = source_record.get("table_id")

            table_index = occurrence.get("table_index")
            if table_index is None:
                table_index = source_record.get("table_index")

            source_row_index = occurrence.get("source_row_index")
            if source_row_index is None:
                source_row_index = source_record.get(
                    "source_row_index"
                )

            records.append(
    {
        "record_id": record_id,
        "document_id": document_id,
        "source_status": (
            source_record.get("source_status")
            or occurrence.get("source_status")
            or "UNKNOWN"
        ),
        "page": page,
        "table_id": table_id,
        "table_index": table_index,
        "source_row_index": source_row_index,
        "source_row_value": (
            occurrence.get("source_row_value")
            if occurrence.get("source_row_value") is not None
            else source_record.get("source_row_value")
        ),
        "identifier_candidates": (
            occurrence.get("identifier_candidates")
            if occurrence.get("identifier_candidates") is not None
            else source_record.get(
                "identifier_candidates",
                [],
            )
        ),
        "entity_column_index": occurrence.get(
            "column_index"
        ),
        "entity_header": occurrence.get("header"),
        "entity_value": occurrence.get("value"),
        "fields": fields,
        "field_count": len(fields),
        "provenance": occurrence.get(
            "provenance",
            {},
        ),
    }
)

        retrieval_entities.append(
            {
                "entity_id": entity_id,
                "canonical_value": canonical,
                "display_value": (
                    entity.get("display_value")
                    or entity.get("canonical_value")
                ),
                "link_status": entity.get("link_status"),
                "link_method": entity.get("link_method"),
                "candidate_score": entity.get("candidate_score"),
                "occurrence_count": entity.get(
                    "occurrence_count",
                    0,
                ),
                "document_count": entity.get(
                    "document_count",
                    0,
                ),
                "document_ids": entity.get(
                    "document_ids",
                    [],
                ),
                "records": records,
            }
        )

    retrieval_entities.sort(
        key=lambda item: (
            normalize_value(item.get("display_value")),
            item["entity_id"],
        )
    )

    return retrieval_entities, exact_lookup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic Stage 13D entity retrieval index."
    )
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    try:
        job_root = find_job_root(args.job_id)
        knowledge_dir = job_root / "output" / "knowledge"

        knowledge_path = knowledge_dir / "knowledge_index.json"
        links_path = knowledge_dir / "entity_links.json"
        output_path = knowledge_dir / "entity_retrieval_index.json"

        knowledge_index = load_json(knowledge_path)
        entity_links = load_json(links_path)

        record_map = build_record_map(knowledge_index)

        retrieval_entities, exact_lookup = build_retrieval_entities(
            entity_links,
            record_map,
        )

        total_records = sum(
            len(entity["records"])
            for entity in retrieval_entities
        )

        records_with_fields = sum(
            1
            for entity in retrieval_entities
            for record in entity["records"]
            if record.get("field_count", 0) > 0
        )

        total_fields = sum(
            record.get("field_count", 0)
            for entity in retrieval_entities
            for record in entity["records"]
        )

        output = {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "type": "stage_13c_entity_links",
                "job_id": args.job_id,
                "knowledge_index": str(knowledge_path),
                "entity_links": str(links_path),
            },
            "method": {
                "type": "deterministic_entity_retrieval",
                "llm_used": False,
                "semantic_search_used": False,
                "matching": "exact_normalized_entity_value",
                "record_source": "stage_13_knowledge_index",
                "field_source": "record.fields",
            },
            "statistics": {
                "entities_indexed": len(retrieval_entities),
                "exact_lookup_keys": len(exact_lookup),
                "records_retrievable": total_records,
                "records_with_fields": records_with_fields,
                "retrieved_fields": total_fields,
            },
            "lookup": {
                "exact_normalized_value": exact_lookup,
            },
            "entities": retrieval_entities,
        }

        knowledge_dir.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                output,
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

        print("=" * 80)
        print("STAGE 13D - DETERMINISTIC ENTITY RETRIEVAL")
        print("=" * 80)
        print(f"Job:                         {args.job_id}")
        print(f"Entities indexed:            {len(retrieval_entities)}")
        print(f"Exact lookup keys:           {len(exact_lookup)}")
        print(f"Records retrievable:         {total_records}")
        print(f"Records with fields:         {records_with_fields}")
        print(f"Retrieved fields:            {total_fields}")
        print(f"Source links:                {links_path}")
        print(f"Output:                      {output_path}")
        print("=" * 80)
        print("LOOKUP KEYS")

        for entity in retrieval_entities[:25]:
            field_count = sum(
                record.get("field_count", 0)
                for record in entity["records"]
            )
            print(
                f"  {entity['display_value']!r} | "
                f"records={len(entity['records'])} | "
                f"documents={entity['document_count']} | "
                f"fields={field_count}"
            )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
