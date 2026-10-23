"""
Stage 13B - Generic Entity Candidate Extraction

Consumes the deterministic Stage 13 knowledge index and extracts possible
entity values without assuming a domain, a fixed identifier column, or a
specific table schema.

This stage does NOT perform semantic entity linking and does NOT use an LLM.
It produces candidates only. Exact provenance is preserved so a later linking
or retrieval layer can decide how candidates should be connected.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = BASE_DIR / "output" / "jobs"


# ---------------------------------------------------------------------------
# Generic text normalization
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def canonical_text(value: Any) -> str:
    text = normalize_text(value)
    return text.casefold()


def is_numeric_like(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    cleaned = value.replace(",", "")
    cleaned = re.sub(r"[%°±Ωμλ∆]", "", cleaned)
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def is_temporal_like(value: str) -> bool:
    value = value.strip()
    patterns = [
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[- ]\d{4}$",
        r"^\d{4}[-/]\d{1,2}$",
    ]
    lowered = value.casefold()
    return any(re.match(pattern, lowered) for pattern in patterns)


def looks_identifier_like(value: str) -> bool:
    """Conservative structural heuristic; never a semantic declaration."""
    value = value.strip()
    if not value or is_numeric_like(value) or is_temporal_like(value):
        return False

    # Mixed alphanumeric codes such as A-01, SN_102, TEST42, X12-B.
    has_alpha = bool(re.search(r"[A-Za-z]", value))
    has_digit = bool(re.search(r"\d", value))
    if has_alpha and has_digit and len(value) <= 80:
        return True

    # Explicitly delimited compact codes that contain no whitespace.
    if re.fullmatch(r"[A-Za-z]{1,12}[-_/][A-Za-z0-9]{1,24}", value):
        return True

    return False


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def candidate_score(
    value: str,
    role_scores: dict[str, Any],
    value_frequency: int,
    column_frequency: int,
    document_frequency: int,
) -> tuple[float, list[str]]:
    """Return a deterministic structural score and evidence list."""

    evidence: list[str] = []
    score = 0.0

    identifier = float(role_scores.get("identifier_candidate", 0.0) or 0.0)
    temporal = float(role_scores.get("temporal_candidate", 0.0) or 0.0)
    measurement = float(role_scores.get("measurement_candidate", 0.0) or 0.0)
    categorical = float(role_scores.get("categorical_candidate", 0.0) or 0.0)
    text = float(role_scores.get("text_candidate", 0.0) or 0.0)
    parameter = float(role_scores.get("parameter_candidate", 0.0) or 0.0)
    status = float(role_scores.get("status_candidate", 0.0) or 0.0)

    # Strongest evidence: the Stage 13 column-role classifier already found
    # the value's column structurally identifier-like.
    if identifier >= 0.60:
        score += 0.65 * identifier
        evidence.append("identifier_role")

    if looks_identifier_like(value):
        score += 0.20
        evidence.append("identifier_like_value")

    # Repetition across records/documents is useful for entity discovery, but
    # is deliberately weaker than explicit identifier evidence.
    if value_frequency >= 2:
        score += min(0.10, 0.025 * value_frequency)
        evidence.append("repeated_value")

    if column_frequency >= 2:
        score += 0.05
        evidence.append("repeated_column_context")

    if document_frequency >= 2:
        score += 0.10
        evidence.append("cross_document_occurrence")

    # Suppress obvious non-entity values.
    if temporal >= 0.70 or is_temporal_like(value):
        score -= 0.35
        evidence.append("temporal_suppression")

    if measurement >= 0.70 or is_numeric_like(value):
        score -= 0.45
        evidence.append("measurement_suppression")

    # A status/categorical/text field may still contain an entity label, but
    # it should not become an entity merely because it is ordinary text.
    if status >= 0.80 and identifier < 0.60 and not looks_identifier_like(value):
        score -= 0.20
        evidence.append("status_suppression")

    if parameter >= 0.80 and identifier < 0.60 and not looks_identifier_like(value):
        score -= 0.10
        evidence.append("parameter_suppression")

    if categorical >= 0.80 and identifier < 0.60 and not looks_identifier_like(value):
        score -= 0.05
        evidence.append("categorical_penalty")

    if text >= 0.80 and identifier < 0.60 and not looks_identifier_like(value):
        score -= 0.05
        evidence.append("text_penalty")

    return max(0.0, min(1.0, score)), evidence


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_candidates(index: dict[str, Any]) -> dict[str, Any]:
    records = index.get("records", [])

    # Global frequency of canonical values.
    value_frequency: Counter[str] = Counter()
    value_documents: defaultdict[str, set[str]] = defaultdict(set)
    value_contexts: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)

    for record in records:
        document_id = normalize_text(record.get("document_id"))
        for field in record.get("fields", {}).values():
            value = normalize_text(field.get("value"))
            canonical = canonical_text(value)
            if not canonical:
                continue
            value_frequency[canonical] += 1
            value_documents[canonical].add(document_id)
            value_contexts[canonical].add(
                (
                    document_id,
                    int(field.get("column_index", -1)),
                )
            )

    grouped: dict[str, dict[str, Any]] = {}

    for record in records:
        document_id = normalize_text(record.get("document_id"))

        for field in record.get("fields", {}).values():
            value = normalize_text(field.get("value"))
            canonical = canonical_text(value)
            if not canonical:
                continue

            role_scores = field.get("role_candidates") or {}
            frequency = value_frequency[canonical]
            doc_frequency = len(value_documents[canonical])
            context_frequency = len(value_contexts[canonical])

            score, evidence = candidate_score(
                value=value,
                role_scores=role_scores,
                value_frequency=frequency,
                column_frequency=context_frequency,
                document_frequency=doc_frequency,
            )

            # Candidate threshold is intentionally conservative. We want the
            # entity layer to avoid inventing entities; later stages can add
            # broader semantic candidates.
            if score < 0.60:
                continue

            bucket = grouped.setdefault(
                canonical,
                {
                    "canonical_value": canonical,
                    "display_values": Counter(),
                    "candidate_scores": [],
                    "evidence": set(),
                    "occurrences": [],
                    "documents": set(),
                    "tables": set(),
                },
            )

            bucket["display_values"][value] += 1
            bucket["candidate_scores"].append(score)
            bucket["evidence"].update(evidence)
            bucket["documents"].add(document_id)
            bucket["tables"].add(
                (
                    document_id,
                    normalize_text(record.get("table_id")),
                )
            )

            bucket["occurrences"].append(
                {
                    "record_id": record.get("record_id"),
                    "document_id": document_id,
                    "page": record.get("page"),
                    "table_id": record.get("table_id"),
                    "table_index": record.get("table_index"),
                    "source_row_index": record.get("source_row_index"),
                    "column_index": field.get("column_index"),
                    "header": field.get("header"),
                    "value": value,
                    "normalized_value": canonical,
                    "role_candidates": role_scores,
                    "provenance": field.get("provenance"),
                }
            )

    entities: list[dict[str, Any]] = []

    for canonical, bucket in grouped.items():
        scores = bucket["candidate_scores"]
        display_value = bucket["display_values"].most_common(1)[0][0]
        max_score = max(scores)
        mean_score = sum(scores) / len(scores)

        entities.append(
            {
                "entity_candidate_id": f"entity_{len(entities) + 1:06d}",
                "canonical_value": canonical,
                "display_value": display_value,
                "candidate_score": round(max_score, 6),
                "mean_candidate_score": round(mean_score, 6),
                "occurrence_count": len(bucket["occurrences"]),
                "document_count": len(bucket["documents"]),
                "table_count": len(bucket["tables"]),
                "evidence": sorted(bucket["evidence"]),
                "occurrences": bucket["occurrences"],
            }
        )

    # Stable ordering: strongest candidates first, then canonical value.
    entities.sort(
        key=lambda item: (
            -item["candidate_score"],
            -item["occurrence_count"],
            item["canonical_value"],
        )
    )

    # Re-number after sorting so IDs are deterministic.
    for index_number, entity in enumerate(entities, start=1):
        entity["entity_candidate_id"] = f"entity_{index_number:06d}"

    return {
        "schema_version": "13b.1",
        "source": {
            "type": "stage_13_knowledge_index",
            "job_id": index.get("job_id"),
        },
        "method": {
            "type": "deterministic_structural_candidate_extraction",
            "llm_used": False,
            "semantic_linking_performed": False,
        },
        "statistics": {
            "records_examined": len(records),
            "entity_candidates": len(entities),
            "cross_document_candidates": sum(
                1 for entity in entities if entity["document_count"] > 1
            ),
            "total_candidate_occurrences": sum(
                entity["occurrence_count"] for entity in entities
            ),
        },
        "entity_candidates": entities,
    }


# ---------------------------------------------------------------------------
# Job discovery / I/O
# ---------------------------------------------------------------------------


def find_job(job_id: str | None) -> Path:
    if job_id:
        path = JOBS_DIR / job_id / "output" / "knowledge" / "knowledge_index.json"
        if not path.exists():
            raise FileNotFoundError(f"Knowledge index not found: {path}")
        return path

    candidates = sorted(
        JOBS_DIR.glob("*/output/knowledge/knowledge_index.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No Stage 13 knowledge_index.json found under output/jobs."
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deterministic entity candidates from Stage 13."
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Job ID. If omitted, the newest knowledge index is used.",
    )
    args = parser.parse_args()

    source_path = find_job(args.job_id)
    with source_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    result = extract_candidates(index)

    output_path = source_path.parent / "entity_candidates.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("STAGE 13B - ENTITY CANDIDATE EXTRACTION")
    print("=" * 80)
    print(f"Job:                         {index.get('job_id')}")
    print(f"Records examined:            {result['statistics']['records_examined']}")
    print(f"Entity candidates:           {result['statistics']['entity_candidates']}")
    print(
        "Cross-document candidates:   "
        f"{result['statistics']['cross_document_candidates']}"
    )
    print(
        "Candidate occurrences:       "
        f"{result['statistics']['total_candidate_occurrences']}"
    )
    print(f"Source:                      {source_path}")
    print(f"Output:                      {output_path}")
    print("=" * 80)

    if result["entity_candidates"]:
        print("TOP CANDIDATES")
        for entity in result["entity_candidates"][:20]:
            print(
                f"  {entity['display_value']!r} | "
                f"score={entity['candidate_score']:.3f} | "
                f"occurrences={entity['occurrence_count']} | "
                f"documents={entity['document_count']} | "
                f"evidence={','.join(entity['evidence'])}"
            )


if __name__ == "__main__":
    main()
