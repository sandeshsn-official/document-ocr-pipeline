"""
Stage 13C - ENTITY LINKING

Purpose
-------
Build a deterministic entity-link layer from:
    13 Knowledge Index
    13B Entity Candidates

Output
------
output/jobs/<job_id>/output/knowledge/entity_links.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "13c.1"


def normalize_value(value: Any) -> str:
    """Deterministic normalization used only for exact identity matching."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at: {path}")

    return data


def find_job_root(job_id: str) -> Path:
    """
    Resolve the project root from the current working directory.

    Expected execution:
        project_root> py -3.11 src\\13c_entity_linking.py --job-id ...

    Also supports execution from inside the project root's src directory.
    """
    cwd = Path.cwd().resolve()

    candidates = [
        cwd / "output" / "jobs" / job_id,
        cwd.parent / "output" / "jobs" / job_id,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # If the script itself is inside project/src, derive project root.
    script_project_root = Path(__file__).resolve().parent.parent
    candidate = script_project_root / "output" / "jobs" / job_id
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Could not locate the job directory. Expected:\n"
        f"  <project_root>/output/jobs/{job_id}"
    )


def build_occurrence_key(occurrence: dict[str, Any]) -> tuple[Any, ...]:
    """
    Stable occurrence identity.

    Prefer the source record ID plus column. Fall back to document/table/row/
    column provenance when a record ID is unavailable.
    """
    record_id = occurrence.get("record_id")
    if record_id:
        return (
            "record",
            str(record_id),
            occurrence.get("column_index"),
        )

    provenance = occurrence.get("provenance") or {}

    return (
        "source",
        provenance.get("document_id", occurrence.get("document_id")),
        provenance.get("page", occurrence.get("page")),
        provenance.get("table_index", occurrence.get("table_index")),
        provenance.get("row", occurrence.get("source_row_index")),
        provenance.get("column", occurrence.get("column_index")),
    )


def collect_candidate_occurrences(
    entity_candidates: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Convert Stage 13B candidates into a deterministic normalized-value map.

    The map is keyed by normalized canonical value. No semantic inference is
    performed here.
    """
    grouped: dict[str, dict[str, Any]] = {}

    candidates = entity_candidates.get("entity_candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("entity_candidates must be a list")

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        canonical = normalize_value(candidate.get("canonical_value"))
        if not canonical:
            continue

        bucket = grouped.setdefault(
            canonical,
            {
                "display_value": candidate.get("display_value")
                or candidate.get("canonical_value")
                or canonical,
                "candidate_ids": [],
                "occurrences": [],
            },
        )

        candidate_id = candidate.get("entity_candidate_id")
        if candidate_id and candidate_id not in bucket["candidate_ids"]:
            bucket["candidate_ids"].append(candidate_id)

        for occurrence in candidate.get("occurrences", []) or []:
            if isinstance(occurrence, dict):
                bucket["occurrences"].append(occurrence)

    return grouped


def deduplicate_occurrences(
    occurrences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []

    for occurrence in occurrences:
        key = build_occurrence_key(occurrence)
        if key in seen:
            continue
        seen.add(key)
        result.append(occurrence)

    return result


def collect_linked_records(
    knowledge_index: dict[str, Any],
    normalized_value: str,
) -> list[dict[str, Any]]:
    """
    Independently verify the candidate occurrence against the Stage 13
    knowledge index.

    This is intentionally exact normalized-value matching. It prevents the
    linking stage from depending solely on the derived 13B file.
    """
    matches: list[dict[str, Any]] = []

    records = knowledge_index.get("records", [])
    if not isinstance(records, list):
        return matches

    for record in records:
        if not isinstance(record, dict):
            continue

        cells = record.get("cells", [])
        if not isinstance(cells, list):
            continue

        for cell in cells:
            if not isinstance(cell, dict):
                continue

            value = (
                cell.get("final_verified_text")
                or cell.get("verified_text")
                or cell.get("text")
                or ""
            )

            if normalize_value(value) != normalized_value:
                continue

            matches.append(
                {
                    "record_id": record.get("record_id"),
                    "document_id": record.get("document_id"),
                    "page": record.get("page"),
                    "table_id": record.get("table_id"),
                    "table_index": record.get("table_index"),
                    "source_row_index": record.get("source_row_index"),
                    "column_index": cell.get("column"),
                    "header": cell.get("header"),
                    "value": value,
                    "bbox": cell.get("bbox"),
                    "verification_status": (
                        cell.get("final_verification_status")
                        or cell.get("verification_status")
                    ),
                    "source_ocr_confidence": cell.get(
                        "source_ocr_confidence"
                    ),
                }
            )

    return matches


def build_entities(
    knowledge_index: dict[str, Any],
    entity_candidates: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped = collect_candidate_occurrences(entity_candidates)
    entities: list[dict[str, Any]] = []

    for normalized_value in sorted(grouped):
        bucket = grouped[normalized_value]

        candidate_occurrences = deduplicate_occurrences(
            bucket["occurrences"]
        )

        verified_index_matches = collect_linked_records(
            knowledge_index,
            normalized_value,
        )

        # Candidate extraction is the eligibility layer. The knowledge index
        # is the source-of-truth verification layer. Use the union only when
        # the exact normalized value matches.
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}

        for occurrence in candidate_occurrences:
            key = build_occurrence_key(occurrence)
            merged[key] = {
                "record_id": occurrence.get("record_id"),
                "document_id": occurrence.get("document_id"),
                "page": occurrence.get("page"),
                "table_id": occurrence.get("table_id"),
                "table_index": occurrence.get("table_index"),
                "source_row_index": occurrence.get("source_row_index"),
                "column_index": occurrence.get("column_index"),
                "header": occurrence.get("header"),
                "value": occurrence.get("value"),
                "normalized_value": occurrence.get("normalized_value")
                or normalized_value,
                "role_candidates": occurrence.get("role_candidates", {}),
                "provenance": occurrence.get("provenance", {}),
            }

        for match in verified_index_matches:
            key = build_occurrence_key(match)
            if key not in merged:
                merged[key] = {
                    "record_id": match.get("record_id"),
                    "document_id": match.get("document_id"),
                    "page": match.get("page"),
                    "table_id": match.get("table_id"),
                    "table_index": match.get("table_index"),
                    "source_row_index": match.get("source_row_index"),
                    "column_index": match.get("column_index"),
                    "header": match.get("header"),
                    "value": match.get("value"),
                    "normalized_value": normalized_value,
                    "role_candidates": {},
                    "provenance": {
                        "document_id": match.get("document_id"),
                        "page": match.get("page"),
                        "table_index": match.get("table_index"),
                        "row": match.get("source_row_index"),
                        "column": match.get("column_index"),
                        "bbox": match.get("bbox"),
                        "verification_status": match.get(
                            "verification_status"
                        ),
                        "source_ocr_confidence": match.get(
                            "source_ocr_confidence"
                        ),
                    },
                }

        occurrences = list(merged.values())

        document_ids = sorted(
            {
                str(item.get("document_id"))
                for item in occurrences
                if item.get("document_id") is not None
            }
        )

        table_keys = {
            (
                item.get("document_id"),
                item.get("table_index"),
            )
            for item in occurrences
        }

        candidate_scores = []
        for candidate in entity_candidates.get("entity_candidates", []):
            if not isinstance(candidate, dict):
                continue
            if normalize_value(candidate.get("canonical_value")) != normalized_value:
                continue
            score = candidate.get("candidate_score")
            if isinstance(score, (int, float)):
                candidate_scores.append(float(score))

        mean_score = (
            sum(candidate_scores) / len(candidate_scores)
            if candidate_scores
            else None
        )

        if len(document_ids) > 1:
            link_status = "cross_document"
        else:
            link_status = "single_document"

        entities.append(
            {
                "entity_id": stable_id("entity", normalized_value),
                "canonical_value": normalized_value,
                "display_value": bucket["display_value"],
                "link_status": link_status,
                "link_method": "exact_normalized_value",
                "semantic_linking_performed": False,
                "candidate_score": max(candidate_scores)
                if candidate_scores
                else None,
                "mean_candidate_score": mean_score,
                "candidate_ids": bucket["candidate_ids"],
                "occurrence_count": len(occurrences),
                "document_count": len(document_ids),
                "table_count": len(table_keys),
                "document_ids": document_ids,
                "occurrences": occurrences,
            }
        )

    return entities


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic Stage 13C entity links."
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="OCR job ID, e.g. 20261008T035618Z_d9e536a1",
    )
    args = parser.parse_args()

    try:
        job_root = find_job_root(args.job_id)
        knowledge_dir = job_root / "output" / "knowledge"

        knowledge_path = knowledge_dir / "knowledge_index.json"
        candidates_path = knowledge_dir / "entity_candidates.json"
        output_path = knowledge_dir / "entity_links.json"

        knowledge_index = load_json(knowledge_path)
        entity_candidates = load_json(candidates_path)

        entities = build_entities(
            knowledge_index,
            entity_candidates,
        )

        cross_document = [
            entity
            for entity in entities
            if entity.get("link_status") == "cross_document"
        ]

        total_occurrences = sum(
            int(entity.get("occurrence_count", 0))
            for entity in entities
        )

        output = {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "type": "stage_13_knowledge_index_plus_13b_candidates",
                "job_id": args.job_id,
                "knowledge_index": str(knowledge_path),
                "entity_candidates": str(candidates_path),
            },
            "method": {
                "type": "deterministic_exact_normalized_value_linking",
                "llm_used": False,
                "semantic_linking_performed": False,
                "identity_rule": (
                    "Entities are linked only when candidate values have "
                    "the same deterministic normalized representation."
                ),
            },
            "statistics": {
                "entities": len(entities),
                "cross_document_entities": len(cross_document),
                "single_document_entities": (
                    len(entities) - len(cross_document)
                ),
                "total_occurrences": total_occurrences,
            },
            "entities": entities,
        }

        knowledge_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        print("=" * 80)
        print("STAGE 13C - ENTITY LINKING")
        print("=" * 80)
        print(f"Job:                         {args.job_id}")
        print(f"Entities:                    {len(entities)}")
        print(f"Cross-document entities:     {len(cross_document)}")
        print(
            "Single-document entities:   "
            f"{len(entities) - len(cross_document)}"
        )
        print(f"Linked occurrences:          {total_occurrences}")
        print(f"Source index:                {knowledge_path}")
        print(f"Candidate source:            {candidates_path}")
        print(f"Output:                      {output_path}")
        print("=" * 80)

        if entities:
            print("ENTITIES")
            for entity in entities[:25]:
                print(
                    f"  {entity['display_value']!r} | "
                    f"status={entity['link_status']} | "
                    f"occurrences={entity['occurrence_count']} | "
                    f"documents={entity['document_count']}"
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
