"""
Stage 13F - Cross-Document Query Aggregation

Purpose
-------
Aggregate deterministic entity retrieval results across all documents in a job.

This stage does NOT perform:
- OCR
- fuzzy matching
- semantic matching
- LLM reasoning
- embedding/vector search
- identity invention

It consumes:
    output/jobs/<job_id>/output/knowledge/entity_links.json
    output/jobs/<job_id>/output/knowledge/entity_retrieval_index.json

and produces:
    output/jobs/<job_id>/output/knowledge/cross_document_query_index.json

The implementation is schema-driven. It does not assume a fixed number of
documents, tables, rows, columns, or fields.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "13f.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_knowledge_dir(job_id: str, project_root: Path) -> Path:
    return (
        project_root
        / "output"
        / "jobs"
        / job_id
        / "output"
        / "knowledge"
    )


def build_entity_map(entity_links: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Accept the current 13C entity_links schema without assuming that every
    producer field is present. The entity ID remains the authoritative key.
    """
    result: dict[str, dict[str, Any]] = {}

    entities = entity_links.get("entities", [])
    if isinstance(entities, dict):
        entities = list(entities.values())

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_id = entity.get("entity_id") or entity.get("entity_candidate_id")
        if not entity_id:
            continue

        canonical = (
            entity.get("canonical_value")
            or entity.get("display_value")
            or entity.get("value")
            or ""
        )

        result[str(entity_id)] = {
            "entity_id": str(entity_id),
            "canonical_value": str(canonical),
            "display_value": str(entity.get("display_value") or canonical),
            "normalized_value": normalize(canonical),
            "link_status": entity.get("link_status", "unknown"),
            "document_count": entity.get("document_count", 0),
            "occurrence_count": entity.get("occurrence_count", 0),
        }

    return result


def iter_retrieval_entities(retrieval_index: dict[str, Any]):
    """
    Yield entity-level retrieval objects while tolerating the current
    retrieval index's list/dict representation.
    """
    entities = retrieval_index.get("entities", [])

    if isinstance(entities, dict):
        for entity_id, entity in entities.items():
            if isinstance(entity, dict):
                item = dict(entity)
                item.setdefault("entity_id", entity_id)
                yield item
        return

    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict):
                yield entity


def extract_records(entity: dict[str, Any]) -> list[dict[str, Any]]:
    records = entity.get("records", [])

    if isinstance(records, dict):
        return [v for v in records.values() if isinstance(v, dict)]

    if isinstance(records, list):
        return [v for v in records if isinstance(v, dict)]

    return []


def extract_entity_occurrences(entity: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Prefer explicit occurrences if the retrieval index contains them.
    Otherwise derive occurrences from records.
    """
    occurrences = entity.get("occurrences")

    if isinstance(occurrences, list):
        return [x for x in occurrences if isinstance(x, dict)]

    derived = []
    for record in extract_records(entity):
        occurrence = {
            "record_id": record.get("record_id"),
            "document_id": record.get("document_id"),
            "page": record.get("page"),
            "table_id": record.get("table_id"),
            "table_index": record.get("table_index"),
            "source_row_index": record.get("source_row_index"),
            "source_row_value": record.get("source_row_value"),
            "identifier_candidates": record.get("identifier_candidates", []),
            "fields": record.get("fields", {}),
        }
        derived.append(occurrence)

    return derived


def compact_field(field: dict[str, Any]) -> dict[str, Any]:
    """
    Preserve the field and its provenance rather than flattening it to text.
    """
    result = dict(field)

    # Explicitly retain the important retrieval/provenance components when
    # present. No field names are hardcoded beyond metadata keys.
    if "provenance" in field:
        result["provenance"] = field["provenance"]

    return result


def compact_occurrence(
    occurrence: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve an occurrence and all dynamically defined fields.

    Stage 13D may represent ``fields`` as either:
        1. a dictionary keyed by field name, or
        2. a list of field objects containing ``field_key``.

    Stage 13F normalizes both representations into a dictionary without
    assuming any fixed number of columns or field names.
    """

    fields = occurrence.get(
        "fields",
        {},
    )

    compact_fields: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Representation 1:
    # fields is already a dictionary.
    # ------------------------------------------------------------------
    if isinstance(fields, dict):

        for field_key, field in fields.items():

            if not isinstance(field, dict):
                continue

            compact_fields[str(field_key)] = compact_field(
                field
            )

    # ------------------------------------------------------------------
    # Representation 2:
    # fields is a list of field objects.
    #
    # This is the representation currently produced by Stage 13D.
    # ------------------------------------------------------------------
    elif isinstance(fields, list):

        for position, field in enumerate(fields):

            if not isinstance(field, dict):
                continue

            field_key = (
                field.get("field_key")
                or field.get("key")
            )

            # If a producer does not provide a field key, preserve
            # the field deterministically using its position.
            if field_key is None:
                field_key = f"field_{position + 1}"

            compact_fields[str(field_key)] = compact_field(
                field
            )

    result = {
        "record_id": occurrence.get(
            "record_id"
        ),
        "document_id": occurrence.get(
            "document_id"
        ),
        "page": occurrence.get(
            "page"
        ),
        "table_id": occurrence.get(
            "table_id"
        ),
        "table_index": occurrence.get(
            "table_index"
        ),
        "source_row_index": occurrence.get(
            "source_row_index"
        ),
        "source_row_value": occurrence.get(
            "source_row_value"
        ),
        "identifier_candidates": occurrence.get(
            "identifier_candidates",
            [],
        ),
        "fields": compact_fields,
    }

    # Preserve any additional occurrence metadata without
    # inventing semantics or discarding information.
    for key, value in occurrence.items():

        if key not in result and key != "fields":
            result[key] = value

    return result


def build_cross_document_index(
    job_id: str,
    entity_links: dict[str, Any],
    retrieval_index: dict[str, Any],
) -> dict[str, Any]:

    entity_map = build_entity_map(entity_links)

    aggregated: dict[str, dict[str, Any]] = {}

    # Entity metadata from 13C.
    for entity_id, entity in entity_map.items():
        aggregated[entity_id] = {
            "entity_id": entity_id,
            "canonical_value": entity["canonical_value"],
            "display_value": entity["display_value"],
            "normalized_value": entity["normalized_value"],
            "link_status": entity["link_status"],
            "occurrences": [],
        }

    # Retrieval records from 13D.
    for retrieval_entity in iter_retrieval_entities(retrieval_index):
        entity_id = retrieval_entity.get("entity_id")

        if not entity_id:
            # A retrieval object may identify the entity through another field.
            # Only use it if that field is unambiguous.
            entity_id = retrieval_entity.get("entity_candidate_id")

        if not entity_id:
            continue

        entity_id = str(entity_id)

        if entity_id not in aggregated:
            canonical = (
                retrieval_entity.get("canonical_value")
                or retrieval_entity.get("display_value")
                or retrieval_entity.get("value")
                or ""
            )
            aggregated[entity_id] = {
                "entity_id": entity_id,
                "canonical_value": str(canonical),
                "display_value": str(retrieval_entity.get("display_value") or canonical),
                "normalized_value": normalize(canonical),
                "link_status": retrieval_entity.get("link_status", "unknown"),
                "occurrences": [],
            }

        for occurrence in extract_entity_occurrences(retrieval_entity):
            aggregated[entity_id]["occurrences"].append(
                compact_occurrence(occurrence)
            )

    # Deduplicate exact occurrences while preserving order.
    for entity in aggregated.values():
        seen = set()
        unique = []

        for occurrence in entity["occurrences"]:
            key = (
                occurrence.get("record_id"),
                occurrence.get("document_id"),
                occurrence.get("page"),
                occurrence.get("table_id"),
                occurrence.get("source_row_index"),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(occurrence)

        entity["occurrences"] = unique

        document_ids = []
        table_keys = []

        for occurrence in unique:
            document_id = occurrence.get("document_id")
            if document_id is not None and document_id not in document_ids:
                document_ids.append(document_id)

            table_key = (
                occurrence.get("document_id"),
                occurrence.get("table_id"),
                occurrence.get("table_index"),
            )
            if table_key not in table_keys:
                table_keys.append(table_key)

        entity["statistics"] = {
            "occurrence_count": len(unique),
            "document_count": len(document_ids),
            "table_count": len(table_keys),
            "documents": document_ids,
        }

        # This is a structural fact, not a semantic judgment.
        entity["cross_document"] = len(document_ids) > 1

    cross_document_entities = [
        entity
        for entity in aggregated.values()
        if entity["cross_document"]
    ]

    single_document_entities = [
        entity
        for entity in aggregated.values()
        if not entity["cross_document"]
    ]

    # Exact normalized query index. Multiple entities may not normally share
    # a normalized value, but keep a list so collisions are represented rather
    # than silently overwritten.
    exact_query_index: dict[str, list[str]] = defaultdict(list)

    for entity_id, entity in aggregated.items():
        normalized = entity.get("normalized_value", "")
        if normalized and entity_id not in exact_query_index[normalized]:
            exact_query_index[normalized].append(entity_id)

    # Deterministic ordering.
    entity_list = sorted(
        aggregated.values(),
        key=lambda x: (
            x.get("normalized_value", ""),
            x.get("entity_id", ""),
        ),
    )

    exact_query_index = {
        key: sorted(value)
        for key, value in sorted(exact_query_index.items())
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "13F",
        "stage_name": "Cross-Document Query Aggregation",
        "generated_at_utc": utc_now(),
        "job_id": job_id,
        "method": "deterministic_cross_document_aggregation",
        "llm_used": False,
        "semantic_linking_performed": False,
        "source": {
            "entity_links": "entity_links.json",
            "entity_retrieval_index": "entity_retrieval_index.json",
        },
        "design_principles": [
            "Cross-document aggregation is performed only over entities already linked by Stage 13C.",
            "Exact normalized identity is preserved; no fuzzy or semantic identity is invented.",
            "All retrieved records retain document/page/table/row/field provenance.",
            "The implementation is independent of document count, table count, row count, and column count.",
            "Derived aggregation does not modify authoritative verified OCR data.",
            "An entity is cross-document only when its linked occurrences contain more than one document.",
        ],
        "statistics": {
            "entities": len(entity_list),
            "cross_document_entities": len(cross_document_entities),
            "single_document_entities": len(single_document_entities),
            "linked_occurrences": sum(
                e["statistics"]["occurrence_count"] for e in entity_list
            ),
            "documents_represented": len({
                doc
                for e in entity_list
                for doc in e["statistics"]["documents"]
            }),
        },
        "entities": entity_list,
        "exact_query_index": exact_query_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 13F - Cross-document deterministic query aggregation"
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--project-root",
        default=".",
        help="OCR project root containing output/jobs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    knowledge_dir = get_knowledge_dir(args.job_id, project_root)

    entity_links_path = knowledge_dir / "entity_links.json"
    retrieval_path = knowledge_dir / "entity_retrieval_index.json"

    entity_links = load_json(entity_links_path)
    retrieval_index = load_json(retrieval_path)

    result = build_cross_document_index(
        job_id=args.job_id,
        entity_links=entity_links,
        retrieval_index=retrieval_index,
    )

    output_path = (
        Path(args.output).resolve()
        if args.output
        else knowledge_dir / "cross_document_query_index.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    stats = result["statistics"]

    print("=" * 72)
    print("STAGE 13F - CROSS-DOCUMENT QUERY AGGREGATION")
    print("=" * 72)
    print(f"Job:                         {args.job_id}")
    print(f"Entities:                    {stats['entities']}")
    print(f"Cross-document entities:     {stats['cross_document_entities']}")
    print(f"Single-document entities:    {stats['single_document_entities']}")
    print(f"Linked occurrences:          {stats['linked_occurrences']}")
    print(f"Documents represented:       {stats['documents_represented']}")
    print(f"Output:                      {output_path}")
    print()

    for entity in result["entities"]:
        print(
            f"{entity['display_value']} | "
            f"documents={entity['statistics']['document_count']} | "
            f"occurrences={entity['statistics']['occurrence_count']} | "
            f"cross_document={entity['cross_document']}"
        )

    print()
    print("13F completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
