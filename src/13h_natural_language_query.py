"""
Stage 13H - Deterministic Natural-Language Question Layer

Adds conservative question parsing over the existing Stage 13A and 13F
indexes. It does not modify Stage 13A-13G or the OCR pipeline.

Supported pattern:
    What is the temperature of SEN-104 in February?

Design:
- Resolve an entity only when its indexed display/canonical value appears
  explicitly in the question.
- Select fields by lexical overlap with actual indexed field headers.
- Apply month filtering using document IDs and date-like field values.
- Return source-linked evidence; do not invent an answer when ambiguous.
- No LLM, embeddings, fuzzy entity matching, or domain-specific sensor logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from calendar import month_name
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "13h.1"
MONTHS = {name.casefold(): number for number, name in enumerate(month_name) if name}
MONTHS.update({name[:3].casefold(): number for number, name in enumerate(month_name) if name})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def knowledge_dir(job_id: str, project_root: Path) -> Path:
    return project_root / "output" / "jobs" / job_id / "output" / "knowledge"


def load_stage13g(project_root: Path):
    path = project_root / "src" / "13g_generic_knowledge_query.py"
    if not path.exists():
        raise FileNotFoundError(f"Stage 13G source not found: {path}")
    spec = importlib.util.spec_from_file_location("ocr_stage13g_for_13h", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage 13G: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_entities(index: dict[str, Any]):
    entities = index.get("entities", [])
    if isinstance(entities, dict):
        for entity_id, entity in entities.items():
            if isinstance(entity, dict):
                item = dict(entity)
                item.setdefault("entity_id", entity_id)
                yield item
    elif isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict):
                yield entity


def resolve_entity(question: str, cross_index: dict[str, Any]) -> dict[str, Any] | None:
    q = normalize(question)
    candidates = []
    for entity in iter_entities(cross_index):
        values = {
            normalize(entity.get("display_value")),
            normalize(entity.get("canonical_value")),
            normalize(entity.get("normalized_value")),
        }
        values.discard("")
        for value in values:
            # Require token-like boundaries so a short identifier does not
            # accidentally match as a substring inside a longer identifier.
            pattern = rf"(?<![\w.-]){re.escape(value)}(?![\w.-])"
            if re.search(pattern, q, flags=re.IGNORECASE):
                candidates.append((len(value), entity))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    longest = candidates[0][0]
    best = [entity for length, entity in candidates if length == longest]
    if len({str(e.get("entity_id")) for e in best}) != 1:
        return None
    return best[0]


def words(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", normalize(value))
        if len(token) > 1 and token not in {
            "what", "which", "when", "where", "show", "give", "tell",
            "find", "get", "the", "for", "from", "with", "about", "does",
            "did", "was", "were", "are", "is", "of", "in", "on", "to",
            "and", "me", "please", "value", "record", "document",
        }
    }


def get_field_header(field_key: Any, field: dict[str, Any]) -> str:
    return str(field.get("header") or field.get("field_name") or field.get("label") or field_key)


def extract_requested_field(question: str, occurrences: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    question_words = words(question)
    header_candidates: dict[str, tuple[int, str]] = {}
    for occurrence in occurrences:
        fields = occurrence.get("fields", {})
        if isinstance(fields, dict):
            iterable = fields.items()
        elif isinstance(fields, list):
            iterable = [
                (field.get("field_key") or field.get("key") or f"field_{i+1}", field)
                for i, field in enumerate(fields) if isinstance(field, dict)
            ]
        else:
            continue
        for key, field in iterable:
            if not isinstance(field, dict):
                continue
            header = get_field_header(key, field)
            header_words = words(header)
            overlap = len(question_words & header_words)
            if overlap:
                normalized_header = normalize(header)
                current = header_candidates.get(normalized_header)
                score = overlap
                if current is None or score > current[0]:
                    header_candidates[normalized_header] = (score, header)

    if not header_candidates:
        return None, []
    best_score = max(item[0] for item in header_candidates.values())
    best_headers = [header for score, header in header_candidates.values() if score == best_score]
    if len(best_headers) != 1:
        return None, []
    return best_headers[0], [item for item in occurrences if isinstance(item, dict)]


def month_from_question(question: str) -> int | None:
    tokens = set(re.findall(r"[a-z]+", normalize(question)))
    found = {number for name, number in MONTHS.items() if name in tokens}
    return next(iter(found)) if len(found) == 1 else None


def occurrence_month(occurrence: dict[str, Any]) -> int | None:
    # First use ISO-like dates from any indexed field, irrespective of field name.
    fields = occurrence.get("fields", {})
    values = []
    if isinstance(fields, dict):
        for field in fields.values():
            if isinstance(field, dict):
                values.extend([field.get("value"), field.get("text")])
    elif isinstance(fields, list):
        for field in fields:
            if isinstance(field, dict):
                values.extend([field.get("value"), field.get("text")])
    for value in values:
        if value is None:
            continue
        match = re.search(r"\b\d{4}[-/](\d{1,2})[-/]\d{1,2}\b", str(value))
        if match:
            month = int(match.group(1))
            if 1 <= month <= 12:
                return month

    # Fall back to a month name in the source document identifier.
    doc = normalize(occurrence.get("document_id"))
    found = {number for name, number in MONTHS.items() if re.search(rf"\b{re.escape(name)}\b", doc)}
    return next(iter(found)) if len(found) == 1 else None


def field_value(field: dict[str, Any]) -> Any:
    return field.get("value", field.get("text"))


def field_resolution_state(question: str, occurrences: list[dict[str, Any]]) -> str:
    """Return RESOLVED, NOT_FOUND, or AMBIGUOUS using indexed headers only."""
    question_words = words(question)
    scores: dict[str, int] = {}
    for occurrence in occurrences:
        fields = occurrence.get("fields", {})
        if isinstance(fields, dict):
            iterable = fields.items()
        elif isinstance(fields, list):
            iterable = [
                (field.get("field_key") or field.get("key") or f"field_{i+1}", field)
                for i, field in enumerate(fields) if isinstance(field, dict)
            ]
        else:
            continue
        for key, field in iterable:
            if isinstance(field, dict):
                header = get_field_header(key, field)
                overlap = len(question_words & words(header))
                if overlap:
                    norm_header = normalize(header)
                    scores[norm_header] = max(scores.get(norm_header, 0), overlap)
    if not scores:
        # Generic nouns such as "value" do not identify an indexed field.
        # Treat these as underspecified questions, not proof that the field
        # is absent from the index. Keep genuinely unknown field names as
        # NOT_FOUND.
        generic_field_terms = {"value", "measurement", "reading", "data", "information"}
        question_tokens = set(re.findall(r"[a-z0-9]+", normalize(question)))
        if question_tokens & generic_field_terms:
            return "AMBIGUOUS"
        return "NOT_FOUND"
    best_score = max(scores.values())
    return "RESOLVED" if sum(score == best_score for score in scores.values()) == 1 else "AMBIGUOUS"


def answer_question(question: str, cross_index: dict[str, Any]) -> dict[str, Any]:
    entity = resolve_entity(question, cross_index)
    if entity is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "13H",
            "stage_name": "Deterministic Natural-Language Question Layer",
            "generated_at_utc": utc_now(),
            "status": "NO_ENTITY_OR_AMBIGUOUS_ENTITY",
            "answer": None,
            "message": "No unique indexed entity was explicitly identified in the question.",
            "question": question,
            "evidence": [],
        }

    all_occurrences = entity.get("occurrences", [])
    if not isinstance(all_occurrences, list):
        all_occurrences = []
    all_occurrences = [o for o in all_occurrences if isinstance(o, dict)]
    requested_month = month_from_question(question)

    # Resolve the field against all indexed occurrences first. Filtering by month
    # before field resolution incorrectly made absent months look like unknown fields.
    resolution = field_resolution_state(question, all_occurrences)
    requested_header, _ = extract_requested_field(question, all_occurrences)
    if resolution != "RESOLVED" or not requested_header:
        status = "FIELD_NOT_FOUND" if resolution == "NOT_FOUND" else "FIELD_AMBIGUOUS"
        message = (
            "The requested field is not present in the indexed headers for this entity."
            if status == "FIELD_NOT_FOUND" else
            "The question overlaps multiple indexed fields; no unique field was selected."
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "13H",
            "stage_name": "Deterministic Natural-Language Question Layer",
            "generated_at_utc": utc_now(),
            "status": status,
            "answer": None,
            "message": message,
            "question": question,
            "entity": entity.get("display_value") or entity.get("canonical_value"),
            "requested_field": None,
            "month_filter": requested_month,
            "evidence": [],
        }

    occurrences = all_occurrences
    if requested_month is not None:
        occurrences = [o for o in occurrences if occurrence_month(o) == requested_month]
        if not occurrences:
            return {
                "schema_version": SCHEMA_VERSION,
                "stage": "13H",
                "stage_name": "Deterministic Natural-Language Question Layer",
                "generated_at_utc": utc_now(),
                "status": "NO_MATCHING_DATA",
                "answer": None,
                "message": "The entity and field are indexed, but no occurrence matches the requested month.",
                "question": question,
                "entity": entity.get("display_value") or entity.get("canonical_value"),
                "requested_field": requested_header,
                "month_filter": requested_month,
                "evidence": [],
            }

    evidence = []
    for occurrence in occurrences:
        fields = occurrence.get("fields", {})
        if isinstance(fields, dict):
            iterable = fields.items()
        elif isinstance(fields, list):
            iterable = [
                (field.get("field_key") or field.get("key") or f"field_{i+1}", field)
                for i, field in enumerate(fields) if isinstance(field, dict)
            ]
        else:
            continue
        for key, field in iterable:
            if not isinstance(field, dict):
                continue
            header = get_field_header(key, field)
            if normalize(header) != normalize(requested_header):
                continue
            evidence.append({
                "document_id": occurrence.get("document_id"),
                "page": occurrence.get("page"),
                "table_id": occurrence.get("table_id"),
                "table_index": occurrence.get("table_index"),
                "source_row_index": occurrence.get("source_row_index"),
                "field_header": header,
                "value": field_value(field),
                # Field provenance contains the original cell-level OCR evidence.
                "provenance": field.get("provenance") or occurrence.get("provenance", {}),
                "verification_status": (
                    field.get("verification_status")
                    or field.get("verification", {}).get("status")
                    or (field.get("provenance") or {}).get("verification_status")
                ),
                "source_ocr_confidence": (
                    field.get("source_ocr_confidence")
                    if field.get("source_ocr_confidence") is not None
                    else (field.get("provenance") or {}).get("source_ocr_confidence")
                ),
                "source_ocr": field.get("source_ocr") or (field.get("provenance") or {}).get("source_ocr"),
                "source_ocr_variants": field.get("source_ocr_variants") or (field.get("provenance") or {}).get("source_ocr_variants"),
                "verification_reason": field.get("verification_reason") or (field.get("provenance") or {}).get("verification_reason"),
                "review_crop": field.get("review_crop") or (field.get("provenance") or {}).get("review_crop"),
                "bbox": field.get("bbox") or (field.get("provenance") or {}).get("bbox"),
            })


    unique_values = {normalize(item.get("value")) for item in evidence}

    has_numeric_disagreement = any(
        normalize(item.get("verification_status")) == "numeric_disagreement"
        or normalize(
            (item.get("provenance") or {}).get("verification_status")
        ) == "numeric_disagreement"
        for item in evidence
    )

    if evidence and len(unique_values) > 1:
        # Different readings across records are ambiguous;
        # do not label the entire query an OCR disagreement.
        status = "AMBIGUOUS_MULTIPLE_VALUES"
        answer = None
    elif has_numeric_disagreement:
        # One distinct candidate exists, but source verification is unresolved.
        status = "UNVERIFIED_SOURCE_DISAGREEMENT"
        answer = evidence[0]["value"] if evidence else None
    elif len(evidence) == 1:
        status = "ANSWERED"
        answer = evidence[0]["value"]
    elif evidence and len(unique_values) == 1:
        status = "CONSISTENT_MULTIPLE_EVIDENCE"
        answer = evidence[0]["value"]
    else:
        status = "NO_EVIDENCE"
        answer = None


    
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "13H",
        "stage_name": "Deterministic Natural-Language Question Layer",
        "generated_at_utc": utc_now(),
        "status": status,
        "answer": answer,
        "method": "explicit_entity_resolution_field_header_overlap_optional_month_filter",
        "llm_used": False,
        "semantic_search_performed": False,
        "question": question,
        "entity": entity.get("display_value") or entity.get("canonical_value"),
        "entity_id": entity.get("entity_id"),
        "requested_field": requested_header,
        "month_filter": requested_month,
        "evidence_count": len(evidence),
        "message": "Source OCR disagreement is recorded for this value; it is not returned as a verified answer." if status == "UNVERIFIED_SOURCE_DISAGREEMENT" else None,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 13H - Deterministic natural-language question layer")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    directory = knowledge_dir(args.job_id, root)
    cross_index = load_json(directory / "cross_document_query_index.json")
    result = answer_question(args.query, cross_index)

    print("=" * 72)
    print("STAGE 13H - DETERMINISTIC NATURAL-LANGUAGE QUESTION")
    print("=" * 72)
    print(f"Question: {args.query}")
    print(f"Status:   {result['status']}")
    print(f"Entity:   {result.get('entity')}")
    print(f"Field:    {result.get('requested_field')}")
    print(f"Answer:   {result.get('answer')}")
    if result.get("message"):
        print(result["message"])
    for item in result.get("evidence", []):
        print(
            f"Source: {item.get('document_id')} | Page={item.get('page')} | "
            f"Table={item.get('table_index')} | Row={item.get('source_row_index')} | "
            f"{item.get('field_header')}={item.get('value')!r}"
        )

    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
