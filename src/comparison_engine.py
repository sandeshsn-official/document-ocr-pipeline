"""
Deterministic comparison primitives for OCR knowledge-index tables.

This module is deliberately independent of Django, Ollama, and the OCR stages.
It consumes the existing Stage 13 knowledge-index schema and returns structured
comparison results while preserving cell-level provenance and verification data.

Initial scope:
- compare two table objects or two knowledge-index table collections;
- align columns by normalized header, not physical column position;
- align rows using an explicitly selected key or a conservative inferred key;
- report changed, unchanged, added, removed, missing, and ambiguous values;
- never silently discard provenance or treat uncertain OCR as verified.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable
import re


UNCERTAIN_STATUSES = {
    "HIGH_RISK_AGREEMENT_REVIEW",
    "REQUIRES_VISUAL_REVIEW",
    "AMBIGUOUS",
    "CONFLICTING_EVIDENCE",
    "NUMERIC_DISAGREEMENT",
    "UNVERIFIED",
    "LOW_CONFIDENCE",
}
MISSING_STATUSES = {"", "MISSING", "NOT_FOUND", "NULL", "NONE"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_label(value: Any) -> str:
    """Normalize a header/label for matching without changing source text."""
    value = _text(value).casefold()
    value = value.replace("°", " degree ")
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def normalize_value(value: Any) -> str:
    """Conservative value normalization; preserves numeric precision and units."""
    return " ".join(_text(value).casefold().split())


def _provenance(cell: dict[str, Any]) -> dict[str, Any]:
    provenance = cell.get("provenance")
    return dict(provenance) if isinstance(provenance, dict) else {}


def _status(cell: dict[str, Any]) -> str:
    return _text(_provenance(cell).get("verification_status")).upper()


def _is_uncertain(cell: dict[str, Any]) -> bool:
    status = _status(cell)
    reason = _text(_provenance(cell).get("verification_reason")).casefold()
    return status in UNCERTAIN_STATUSES or "must be visually reviewed" in reason


def _cell_value(cell: dict[str, Any]) -> str:
    # The source text is authoritative for display. normalized_text is only a
    # matching aid; it must not replace the displayed OCR value.
    return _text(cell.get("text", cell.get("value", "")))


def _cell_header(cell: dict[str, Any], table: dict[str, Any]) -> str:
    column_number = cell.get("column")
    columns = table.get("header", [])
    if isinstance(columns, list) and isinstance(column_number, int):
        if 1 <= column_number <= len(columns):
            return _text(columns[column_number - 1])
    return _text(cell.get("header", f"Column {column_number or '?'}"))


def _table_cells(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten data rows into cell records, retaining the original row/cell."""
    flattened: list[dict[str, Any]] = []
    for row in table.get("rows", []) or []:
        if not isinstance(row, dict) or _text(row.get("row_type")).casefold() == "header":
            continue
        row_index = row.get("source_row_index", row.get("source_row_value"))
        for cell in row.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            item = dict(cell)
            item["_row_index"] = row_index
            item["_row_type"] = _text(row.get("row_type")) or "data"
            item["_header"] = _cell_header(item, table)
            item["_value"] = _cell_value(item)
            item["_norm_header"] = normalize_label(item["_header"])
            item["_norm_value"] = normalize_value(
                item.get("normalized_text", item["_value"])
            )
            flattened.append(item)
    return flattened


def _table_identity(table: dict[str, Any]) -> dict[str, Any]:
    provenance = table.get("table_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return {
        "table_id": table.get("table_id"),
        "table_index": table.get("table_index"),
        "document_id": provenance.get("document_id", table.get("document_id")),
        "page": provenance.get("page", table.get("page")),
        "header": list(table.get("header", []) or []),
    }


def _record_identifier_values(index: dict[str, Any], table_id: Any) -> dict[Any, dict[str, str]]:
    """Map source row index to candidate identifier fields from index records."""
    result: dict[Any, dict[str, str]] = {}
    for record in index.get("records", []) or []:
        if record.get("table_id") != table_id:
            continue
        row_idx = record.get("source_row_index")
        identifiers: dict[str, str] = {}
        for candidate in record.get("identifier_candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            score = candidate.get("score", 0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0
            header = normalize_label(candidate.get("header"))
            value = normalize_value(candidate.get("normalized_value", candidate.get("value")))
            if header and value and score >= 0.8:
                # Keep only strong identifier evidence.
                identifiers[header] = value
        if identifiers:
            result[row_idx] = identifiers
    return result


def _row_map(table: dict[str, Any], index: dict[str, Any], row_key: str | None, side_tag: str):
    rows: dict[Any, dict[str, Any]] = {}
    duplicate_keys: set[Any] = set()
    identifier_values = _record_identifier_values(index, table.get("table_id"))
    for row in table.get("rows", []) or []:
        if not isinstance(row, dict) or _text(row.get("row_type")).casefold() == "header":
            continue
        row_index = row.get("source_row_index", row.get("source_row_value"))
        cells = [c for c in (row.get("cells", []) or []) if isinstance(c, dict)]
        by_header = {}
        for cell in cells:
            header = _cell_header(cell, table)
            by_header[normalize_label(header)] = cell

        key = None
        key_header = None
        if row_key:
            key_header = normalize_label(row_key)
            key_cell = by_header.get(key_header)
            if key_cell is not None:
                key = normalize_value(key_cell.get("normalized_text", key_cell.get("text")))
        else:
            candidates = identifier_values.get(row_index, {})
            # Use an identifier only when the row has exactly one strong
            # candidate. Multiple candidates need an explicit key choice.
            if len(candidates) == 1:
                key_header, key = next(iter(candidates.items()))
            elif len(candidates) > 1:
                # Prefer an identifier candidate that maps to a visible cell.
                usable = [(h, v) for h, v in candidates.items() if h in by_header]
                if len(usable) == 1:
                    key_header, key = usable[0]

        if key:
            key = (key_header, key)
        else:
            # No safe key: keep row identity local, and do not imply a match.
            key = ("__unmatched_row__", side_tag, row_index)

        if key in rows:
            duplicate_keys.add(key)
        rows[key] = {"row": row, "cells": by_header, "row_index": row_index}
    return rows, duplicate_keys



def _cell_result(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    header: str,
    row_key: Any,
) -> dict[str, Any]:
    left_value = _cell_value(left) if left else None
    right_value = _cell_value(right) if right else None
    left_uncertain = _is_uncertain(left) if left else False
    right_uncertain = _is_uncertain(right) if right else False

    # Describe the value relationship independently of verification.
    if left is None or not normalize_value(left_value):
        if right is None or not normalize_value(right_value):
            value_relation = "MISSING"
        else:
            value_relation = "RIGHT_ONLY"
    elif right is None or not normalize_value(right_value):
        value_relation = "LEFT_ONLY"
    elif normalize_value(left_value) == normalize_value(right_value):
        value_relation = "MATCH"
    else:
        value_relation = "DIFFERENT"

    # Preserve the existing status semantics and safeguards.
    if left is None:
        status = "ADDED"
    elif right is None:
        status = "REMOVED"
    elif left_uncertain or right_uncertain:
        status = "AMBIGUOUS"
    elif normalize_value(left_value) == normalize_value(right_value):
        status = "UNCHANGED"
    else:
        status = "CHANGED"

    def evidence(cell):
        if cell is None:
            return None
        return {
            "value": _cell_value(cell),
            "normalized_value": _text(cell.get("normalized_text")),
            "value_type": cell.get("value_type"),
            "provenance": _provenance(cell),
            "verification_status": _status(cell) or None,
            "verification_uncertain": _is_uncertain(cell),
        }

    return {
        "row_key": row_key,
        "field": header,
        "status": status,
        "value_relation": value_relation,
        "left": evidence(left),
        "right": evidence(right),
    }



def compare_tables(
    left_table: dict[str, Any],
    right_table: dict[str, Any],
    *,
    left_index: dict[str, Any] | None = None,
    right_index: dict[str, Any] | None = None,
    row_key: str | None = None,
    fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Compare two tables from the existing knowledge-index schema.

    `row_key` is an optional column header such as "Sensor ID". If omitted,
    the engine uses only a single high-confidence identifier candidate from
    Stage 13 records. Rows without a safe key remain unmatched rather than
    being paired by their physical position.
    """
    left_index = left_index or {}
    right_index = right_index or {}
    left_rows, left_duplicates = _row_map(left_table, left_index, row_key, "left")
    right_rows, right_duplicates = _row_map(right_table, right_index, row_key, "right")

    requested = {normalize_label(f) for f in fields} if fields is not None else None
    left_headers = {
        normalize_label(h): _text(h) for h in (left_table.get("header", []) or [])
    }
    right_headers = {
        normalize_label(h): _text(h) for h in (right_table.get("header", []) or [])
    }
    shared_headers = set(left_headers) & set(right_headers)
    all_headers = set(left_headers) | set(right_headers)
    if requested is not None:
        all_headers = {h for h in all_headers if h in requested}

    row_keys = set(left_rows) | set(right_rows)
    comparisons = []
    for key in sorted(row_keys, key=lambda x: str(x)):
        left_entry = left_rows.get(key)
        right_entry = right_rows.get(key)

        # Unkeyed rows are not positionally matched. Report as added/removed
        # rows; no cell-level value equivalence is inferred.
        if key[0] == "__unmatched_row__":
            row_status = "AMBIGUOUS"
            if left_entry and not right_entry:
                row_status = "REMOVED"
            elif right_entry and not left_entry:
                row_status = "ADDED"
            comparisons.append({
                "row_key": None,
                "row_status": row_status,
                "left_row_index": left_entry["row_index"] if left_entry else None,
                "right_row_index": right_entry["row_index"] if right_entry else None,
                "cells": [],
                "note": "No reliable row identifier was available; rows were not matched by position.",
            })
            continue

        cell_results = []
        for header in sorted(all_headers):
            left_cell = left_entry["cells"].get(header) if left_entry else None
            right_cell = right_entry["cells"].get(header) if right_entry else None
            # A header absent on one side means field added/removed; a missing
            # cell in a present column is still reported explicitly.
            display_header = left_headers.get(header) or right_headers.get(header) or header
            result = _cell_result(left_cell, right_cell, display_header, key[1])
            cell_results.append(result)

        if left_entry is None:
            row_status = "ADDED"
        elif right_entry is None:
            row_status = "REMOVED"
        elif key in left_duplicates or key in right_duplicates:
            row_status = "AMBIGUOUS"
        elif any(c["status"] in {"CHANGED", "ADDED", "REMOVED"} for c in cell_results):
            row_status = "CHANGED"
        elif any(c["status"] == "AMBIGUOUS" for c in cell_results):
            row_status = "AMBIGUOUS"
        else:
            row_status = "UNCHANGED"

        comparisons.append({
            "row_key": key[1],
            "row_key_field": key[0],
            "row_status": row_status,
            "left_row_index": left_entry["row_index"] if left_entry else None,
            "right_row_index": right_entry["row_index"] if right_entry else None,
            "cells": cell_results,
        })

    counts = {name: 0 for name in ("CHANGED", "UNCHANGED", "ADDED", "REMOVED", "AMBIGUOUS")}
    for row in comparisons:
        if row["row_status"] in counts:
            counts[row["row_status"]] += 1
    cell_counts = {name: 0 for name in counts}
    for row in comparisons:
        for cell in row["cells"]:
            if cell["status"] in cell_counts:
                cell_counts[cell["status"]] += 1

    return {
        "schema_version": "comparison.1",
        "status": "OK",
        "left_table": _table_identity(left_table),
        "right_table": _table_identity(right_table),
        "row_key_requested": row_key,
        "matched_headers": sorted(shared_headers),
        "left_only_headers": sorted(set(left_headers) - set(right_headers)),
        "right_only_headers": sorted(set(right_headers) - set(left_headers)),
        "row_summary": counts,
        "cell_summary": cell_counts,
        "rows": comparisons,
        "notes": [
            "Rows without a reliable key are not matched by physical row position.",
            "Numeric strings are compared conservatively; this function does not convert units or round values.",
            "Uncertain OCR evidence is surfaced as AMBIGUOUS rather than silently accepted.",
        ],
    }


def compare_knowledge_indexes(
    left_index: dict[str, Any],
    right_index: dict[str, Any],
    *,
    left_document_id: str | None = None,
    right_document_id: str | None = None,
    row_key: str | None = None,
    fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Compare tables in two knowledge indexes.

    Tables are paired using normalized-header similarity. Ambiguous pairings
    are reported instead of arbitrarily selecting a table. This initial
    implementation intentionally compares two index snapshots at a time;
    multi-index orchestration belongs in the chatbot query router.
    """
    def select_tables(index, document_id):
        tables = index.get("tables", []) or []
        if document_id is None:
            return list(tables)
        return [
            table for table in tables
            if _table_identity(table).get("document_id") == document_id
        ]

    left_tables = select_tables(left_index, left_document_id)
    right_tables = select_tables(right_index, right_document_id)
    pairs = []
    used_right = set()
    ambiguous = []

    def headers(table):
        return {normalize_label(h) for h in (table.get("header", []) or []) if normalize_label(h)}

    for left in left_tables:
        lh = headers(left)
        scores = []
        for idx, right in enumerate(right_tables):
            if idx in used_right:
                continue
            rh = headers(right)
            if not lh or not rh:
                score = 0.0
            else:
                overlap = len(lh & rh) / len(lh | rh)
                sequence = SequenceMatcher(
                    None, " | ".join(sorted(lh)), " | ".join(sorted(rh))
                ).ratio()
                score = 0.75 * overlap + 0.25 * sequence
            scores.append((score, idx, right))

        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores or scores[0][0] < 0.5:
            pairs.append({
                "status": "REMOVED",
                "left_table": _table_identity(left),
                "right_table": None,
                "score": scores[0][0] if scores else 0.0,
                "comparison": None,
            })
            continue

        best_score, best_idx, best_right = scores[0]
        if len(scores) > 1 and abs(best_score - scores[1][0]) < 0.05:
            ambiguous.append({
                "left_table": _table_identity(left),
                "candidate_right_tables": [
                    _table_identity(scores[0][2]), _table_identity(scores[1][2])
                ],
                "reason": "Multiple tables have near-identical header similarity.",
            })
            continue

        used_right.add(best_idx)
        pairs.append({
            "status": "MATCHED",
            "left_table": _table_identity(left),
            "right_table": _table_identity(best_right),
            "score": round(best_score, 4),
            "comparison": compare_tables(
                left, best_right,
                left_index=left_index,
                right_index=right_index,
                row_key=row_key,
                fields=fields,
            ),
        })

    for idx, right in enumerate(right_tables):
        if idx not in used_right:
            pairs.append({
                "status": "ADDED",
                "left_table": None,
                "right_table": _table_identity(right),
                "score": None,
                "comparison": None,
            })

    return {
        "schema_version": "comparison.1",
        "status": "OK",
        "left_job_id": left_index.get("job_id"),
        "right_job_id": right_index.get("job_id"),
        "table_pairs": pairs,
        "ambiguous_table_matches": ambiguous,
        "summary": {
            "left_tables": len(left_tables),
            "right_tables": len(right_tables),
            "matched_tables": sum(p["status"] == "MATCHED" for p in pairs),
            "added_tables": sum(p["status"] == "ADDED" for p in pairs),
            "removed_tables": sum(p["status"] == "REMOVED" for p in pairs),
            "ambiguous_table_matches": len(ambiguous),
        },
        "notes": [
            "Header similarity proposes table matches; it does not prove semantic equivalence.",
            "Review ambiguous table matches before treating them as authoritative.",
            "This module does not mutate source indexes or OCR output.",
        ],
    }
