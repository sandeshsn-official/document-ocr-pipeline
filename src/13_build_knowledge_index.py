#!/usr/bin/env python3
"""
STAGE 13 — KNOWLEDGE INDEX BUILDER

Generic structural/knowledge indexing layer for verified OCR output.

Stages 1–12 are authoritative and are NOT modified by this stage.

Stage 13A:
- detect likely table headers using structural evidence
- separate header rows from data rows
- profile columns from observed data
- infer multiple role candidates without domain-specific hardcoding

Stage 13B preparation:
- identify structurally plausible identifier values
- preserve provenance
- create deterministic records and value index

Important:
This stage does not invent semantic entities. An identifier candidate is only
an evidence-backed candidate for later entity linking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / "output" / "jobs"


# ============================================================================
# BASIC HELPERS
# ============================================================================

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def canonical_text(value: Any) -> str:
    return normalize_text(value).casefold()


def stable_id(*parts: Any) -> str:
    raw = "\x1f".join(normalize_text(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def bbox_list(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return list(value)
    return None


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================================
# VALUE CHARACTERIZATION
# ============================================================================

NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?$"
)

DATE_PATTERNS = (
    re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"),
    re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"),
    re.compile(
        r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[- /]\d{2,4}$",
        re.I,
    ),
)

BOOLEAN_VALUES = {
    "yes",
    "no",
    "true",
    "false",
    "y",
    "n",
}


def is_number_like(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False

    cleaned = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace(",", "")
        .strip()
    )

    if NUMBER_RE.fullmatch(cleaned):
        return True

    return bool(
        re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][+-]?\d+)?\s*[^\d\s]{1,8}",
            cleaned,
        )
    )


def is_date_like(text: str) -> bool:
    text = normalize_text(text)
    return bool(
        text and any(
            pattern.fullmatch(text)
            for pattern in DATE_PATTERNS
        )
    )


def is_boolean_like(text: str) -> bool:
    return canonical_text(text) in BOOLEAN_VALUES


def identifier_shape_score(text: str) -> float:
    """
    Generic code-shape evidence.

    This intentionally does not know what an identifier means in a particular
    domain. It only recognizes compact structured values.
    """
    text = normalize_text(text)

    if not text or len(text) > 120:
        return 0.0

    if is_number_like(text) or is_date_like(text):
        return 0.0

    if re.fullmatch(
        r"[A-Za-z0-9]+(?:[-_/.:][A-Za-z0-9]+)+",
        text,
    ):
        return 1.0

    if re.fullmatch(
        r"[A-Za-z0-9]{2,32}",
        text,
    ):
        has_digit = any(c.isdigit() for c in text)
        all_upper = text.upper() == text

        if has_digit and all_upper:
            return 0.9
        if has_digit:
            return 0.75
        if all_upper and len(text) <= 12:
            return 0.55

    return 0.0


def value_type(text: str) -> str:
    text = normalize_text(text)

    if not text:
        return "empty"

    if is_date_like(text):
        return "date"

    if is_number_like(text):
        return "number"

    if is_boolean_like(text):
        return "boolean"

    if identifier_shape_score(text) >= 0.75:
        return "identifier_like"

    return "text"


# ============================================================================
# SOURCE SELECTION
# ============================================================================

def newest_job() -> Path:
    if not JOBS_DIR.exists():
        raise FileNotFoundError(
            f"Jobs directory not found: {JOBS_DIR}"
        )

    jobs = [
        p
        for p in JOBS_DIR.iterdir()
        if p.is_dir()
        and (p / "output" / "verified_data").is_dir()
    ]

    if not jobs:
        raise FileNotFoundError(
            "No completed jobs with verified_data found under "
            f"{JOBS_DIR}"
        )

    return max(
        jobs,
        key=lambda p: p.stat().st_mtime,
    )


def verified_files(job_dir: Path) -> list[Path]:
    directory = (
        job_dir
        / "output"
        / "verified_data"
    )

    return sorted(
        p
        for p in directory.glob("*.json")
        if p.is_file()
        and p.name != "review_required.json"
        and not p.name.startswith("review_")
    )


# ============================================================================
# SOURCE CELL HANDLING
# ============================================================================

def cell_text(cell: dict[str, Any]) -> str:
    for key in (
        "final_verified_text",
        "verified_text",
        "text",
    ):
        text = normalize_text(
            cell.get(key)
        )
        if text:
            return text

    return ""


def cell_provenance(
    document: dict[str, Any],
    page: dict[str, Any],
    table_index: int,
    row_index: int,
    column_index: int,
    cell: dict[str, Any],
) -> dict[str, Any]:
    return {
        "document_id": (
            document.get("document_id")
            or document.get("document")
            or ""
        ),
        "document": document.get("document"),
        "source_pdf": document.get("source_pdf"),
        "page": page.get("page"),
        "table_index": table_index,
        "row": row_index,
        "column": column_index,
        "bbox": bbox_list(
            cell.get("bbox")
        ),
        "verification_status": (
            cell.get("final_verification_status")
            or cell.get("verification_status")
        ),
        "source_ocr_confidence": safe_float(
            cell.get("source_ocr_confidence")
        ),
        # Preserve source-image OCR evidence so downstream query results
        # can explain disagreements instead of silently losing their origin.
        "source_ocr": cell.get("source_ocr"),
        "source_ocr_variants": cell.get("source_ocr_variants"),
        "review_crop": cell.get("review_crop"),
        "verification_reason": cell.get(
            "verification_reason"
        ),
    }


# ============================================================================
# ROW CHARACTERISTICS
# ============================================================================

def row_nonempty_ratio(
    cells: list[dict[str, Any]],
) -> float:
    if not cells:
        return 0.0

    return sum(
        bool(
            normalize_text(
                c.get("text")
            )
        )
        for c in cells
    ) / len(cells)


def row_text_ratio(
    cells: list[dict[str, Any]],
) -> float:
    nonempty = [
        c
        for c in cells
        if normalize_text(
            c.get("text")
        )
    ]

    if not nonempty:
        return 0.0

    return sum(
        value_type(
            c.get("text", "")
        ) == "text"
        for c in nonempty
    ) / len(nonempty)


def row_numeric_ratio(
    cells: list[dict[str, Any]],
) -> float:
    nonempty = [
        c
        for c in cells
        if normalize_text(
            c.get("text")
        )
    ]

    if not nonempty:
        return 0.0

    return sum(
        value_type(
            c.get("text", "")
        ) == "number"
        for c in nonempty
    ) / len(nonempty)


def row_identifier_ratio(
    cells: list[dict[str, Any]],
) -> float:
    nonempty = [
        c
        for c in cells
        if normalize_text(
            c.get("text")
        )
    ]

    if not nonempty:
        return 0.0

    return sum(
        identifier_shape_score(
            c.get("text", "")
        ) >= 0.75
        for c in nonempty
    ) / len(nonempty)


def row_type_vector(
    cells: list[dict[str, Any]],
) -> list[str]:
    return [
        value_type(
            c.get("text", "")
        )
        for c in cells
    ]


def comparable_data_consistency(
    candidate_row: list[dict[str, Any]],
    following_rows: list[list[dict[str, Any]]],
) -> float:
    """
    Measures how strongly the rows below a candidate header form stable
    column-wise data types.

    This is deliberately vocabulary-free.
    """
    if not following_rows or not candidate_row:
        return 0.0

    width = len(candidate_row)
    if width == 0:
        return 0.0

    comparisons = 0
    matches = 0

    for column_index in range(width):
        observed = []

        for row in following_rows[:5]:
            if column_index >= len(row):
                continue

            text = normalize_text(
                row[column_index].get("text", "")
            )

            if text:
                observed.append(
                    value_type(text)
                )

        if len(observed) < 2:
            continue

        comparisons += 1

        # Numeric/date/boolean/identifier columns are especially stable.
        if len(set(observed)) == 1:
            matches += 1
            continue

        # A mixture of text and identifier-like values can be a legitimate
        # textual field; give it weak partial consistency.
        if set(observed).issubset(
            {"text", "identifier_like"}
        ):
            matches += 0.5

    if comparisons == 0:
        return 0.0

    return matches / comparisons


def row_header_label_evidence(
    cells: list[dict[str, Any]],
) -> float:
    """
    Weak structural evidence that a row contains labels.

    No domain vocabulary is used.
    """
    nonempty = [
        normalize_text(c.get("text"))
        for c in cells
        if normalize_text(
            c.get("text")
        )
    ]

    if not nonempty:
        return 0.0

    score = 0.0

    # Labels tend to be relatively short.
    short_ratio = sum(
        len(v) <= 40
        for v in nonempty
    ) / len(nonempty)

    score += 0.25 * short_ratio

    # Numeric-looking values are weak header evidence.
    non_numeric_ratio = sum(
        not is_number_like(v)
        for v in nonempty
    ) / len(nonempty)

    score += 0.25 * non_numeric_ratio

    # Identifier-shaped values are less likely to be labels.
    non_identifier_ratio = sum(
        identifier_shape_score(v) < 0.75
        for v in nonempty
    ) / len(nonempty)

    score += 0.20 * non_identifier_ratio

    return min(
        1.0,
        score,
    )


# ============================================================================
# HEADER DETECTION
# ============================================================================

def candidate_header_score(
    rows: list[list[dict[str, Any]]],
    index: int,
) -> dict[str, Any]:
    """
    Score one possible header row.

    A candidate must explain the rows below it. A later data row is therefore
    penalized if treating it as a header does not improve column consistency.
    """
    candidate = rows[index]
    following = rows[index + 1:]

    if not candidate:
        return {
            "score": 0.0,
            "label_evidence": 0.0,
            "data_consistency": 0.0,
            "coverage": 0.0,
        }

    coverage = row_nonempty_ratio(
        candidate
    )

    label_evidence = row_header_label_evidence(
        candidate
    )

    data_consistency = comparable_data_consistency(
        candidate,
        following,
    )

    text_ratio = row_text_ratio(
        candidate
    )

    identifier_ratio = row_identifier_ratio(
        candidate
    )

    # A strong header:
    # - is populated
    # - looks label-like
    # - is followed by structurally consistent data
    # - is mostly textual
    # - does not itself look like a normal identifier row
    score = (
        0.20 * coverage
        + 0.20 * label_evidence
        + 0.40 * data_consistency
        + 0.15 * text_ratio
        + 0.05 * (1.0 - identifier_ratio)
    )

    # Penalize candidate rows that are themselves strongly data-like.
    numeric_ratio = row_numeric_ratio(
        candidate
    )

    if numeric_ratio >= 0.50:
        score *= 0.55

    return {
        "score": round(
            min(1.0, score),
            4,
        ),
        "label_evidence": round(
            label_evidence,
            4,
        ),
        "data_consistency": round(
            data_consistency,
            4,
        ),
        "coverage": round(
            coverage,
            4,
        ),
    }


def detect_header_row(
    rows: list[list[dict[str, Any]]],
) -> tuple[int | None, float, str]:
    """
    Detect a header only when the evidence is strong.

    Important safeguards:
    1. Prefer the earliest candidate when scores are close.
    2. Require meaningful evidence from following rows.
    3. Reject candidates whose score is driven only by textual values.
    4. Do not force a header on an ambiguous table.
    """
    if not rows:
        return (
            None,
            0.0,
            "NO_ROWS",
        )

    candidates = []

    # Conventional headers occur near the top. We inspect at most the first
    # three rows to avoid promoting arbitrary later data rows.
    for index in range(
        min(3, len(rows))
    ):
        evidence = candidate_header_score(
            rows,
            index,
        )

        candidates.append(
            (
                index,
                evidence,
            )
        )

    # Prefer the earliest row when scores are nearly tied.
    best_index, best_evidence = max(
        candidates,
        key=lambda item: (
            item[1]["score"],
            -item[0],
        ),
    )

    best_score = best_evidence[
        "score"
    ]

    # Need actual data-consistency evidence.
    if (
        best_score >= 0.50
        and best_evidence[
            "data_consistency"
        ] >= 0.25
    ):
        return (
            best_index,
            best_score,
            "STRUCTURAL_HEADER_EVIDENCE",
        )

    return (
        None,
        best_score,
        "HEADER_UNCERTAIN",
    )


# ============================================================================
# COLUMN PROFILING
# ============================================================================

def unique_ratio(
    values: list[str],
) -> float:
    nonempty = [
        canonical_text(v)
        for v in values
        if normalize_text(v)
    ]

    if not nonempty:
        return 0.0

    return (
        len(set(nonempty))
        / len(nonempty)
    )


def repeated_value_ratio(
    values: list[str],
) -> float:
    nonempty = [
        canonical_text(v)
        for v in values
        if normalize_text(v)
    ]

    if not nonempty:
        return 0.0

    counts = defaultdict(int)

    for value in nonempty:
        counts[value] += 1

    repeated = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return repeated / len(
        nonempty
    )


def infer_column_roles(
    rows: list[list[dict[str, Any]]],
    header_index: int | None,
) -> list[dict[str, Any]]:

    if not rows:
        return []

    header = (
        rows[header_index]
        if header_index is not None
        else None
    )

    data_rows = [
        row
        for index, row in enumerate(rows)
        if index != header_index
    ]

    width = max(
        (len(row) for row in rows),
        default=0,
    )

    results = []

    for column_index in range(width):

        header_text = ""

        if (
            header is not None
            and column_index < len(header)
        ):
            header_text = normalize_text(
                header[
                    column_index
                ].get("text", "")
            )

        values = []

        for row in data_rows:
            if column_index >= len(row):
                continue

            value = normalize_text(
                row[
                    column_index
                ].get("text", "")
            )

            if value:
                values.append(value)

        # Headerless tables still get profiled.
        if not values:
            for row in rows:
                if column_index >= len(row):
                    continue

                value = normalize_text(
                    row[
                        column_index
                    ].get("text", "")
                )

                if value:
                    values.append(value)

        types = [
            value_type(value)
            for value in values
        ]

        count = len(values)

        numeric_ratio = (
            sum(
                t == "number"
                for t in types
            ) / count
            if count
            else 0.0
        )

        date_ratio = (
            sum(
                t == "date"
                for t in types
            ) / count
            if count
            else 0.0
        )

        boolean_ratio = (
            sum(
                t == "boolean"
                for t in types
            ) / count
            if count
            else 0.0
        )

        identifier_ratio = (
            sum(
                identifier_shape_score(
                    value
                ) >= 0.75
                for value in values
            ) / count
            if count
            else 0.0
        )

        text_ratio = (
            sum(
                t == "text"
                for t in types
            ) / count
            if count
            else 0.0
        )

        uniqueness = unique_ratio(
            values
        )

        repeated_ratio = repeated_value_ratio(
            values
        )

        # ---------------------------------------------------------------
        # Identifier candidate
        # ---------------------------------------------------------------
        #
        # Strong only when:
        # - values actually have code-like structure
        # - values are mostly unique
        #
        # This prevents ordinary textual columns from becoming identifiers
        # merely because their values are unique.
        identifier_score = min(
            1.0,
            (
                0.70 * identifier_ratio
                + 0.30 * uniqueness
            ),
        )

        # ---------------------------------------------------------------
        # Temporal candidate
        # ---------------------------------------------------------------
        temporal_score = min(
            1.0,
            (
                0.85 * date_ratio
                + 0.10 * uniqueness
                + 0.05 * text_ratio
            ),
        )

        # ---------------------------------------------------------------
        # Measurement candidate
        # ---------------------------------------------------------------
        #
        # Numeric homogeneity is the strongest structural evidence.
        measurement_score = min(
            1.0,
            (
                0.85 * numeric_ratio
                + 0.10 * uniqueness
                + 0.05 * (
                    1.0 - boolean_ratio
                )
            ),
        )

        # ---------------------------------------------------------------
        # Categorical candidate
        # ---------------------------------------------------------------
        #
        # Text values with repetition are more plausibly categorical than
        # text values where every value is unique.
        categorical_score = min(
            1.0,
            (
                0.60 * text_ratio
                + 0.30 * repeated_ratio
                + 0.10 * (
                    1.0 - identifier_ratio
                )
            ),
        )

        # ---------------------------------------------------------------
        # Parameter candidate
        # ---------------------------------------------------------------
        #
        # Do NOT infer "parameter" merely from textual uniqueness.
        # It is only a weak candidate for textual columns whose values are
        # short and structured. Later semantic/entity layers may strengthen
        # this interpretation.
        short_text_ratio = (
            sum(
                len(value) <= 40
                for value in values
                if value
            )
            / count
            if count
            else 0.0
        )

        parameter_score = min(
            1.0,
            (
                0.55 * text_ratio
                + 0.25 * short_text_ratio
                + 0.20 * (
                    1.0 - identifier_ratio
                )
            ),
        )

        # Parameter candidates are suppressed when the data clearly behaves
        # as a numeric measurement or temporal series.
        if numeric_ratio >= 0.75:
            parameter_score *= 0.20

        if date_ratio >= 0.75:
            parameter_score *= 0.20

        # ---------------------------------------------------------------
        # Status candidate
        # ---------------------------------------------------------------
        status_score = min(
            1.0,
            (
                0.70 * boolean_ratio
                + 0.30 * categorical_score
            ),
        )

        roles = {
            "identifier_candidate": round(
                max(
                    0.0,
                    identifier_score,
                ),
                4,
            ),
            "temporal_candidate": round(
                max(
                    0.0,
                    temporal_score,
                ),
                4,
            ),
            "measurement_candidate": round(
                max(
                    0.0,
                    measurement_score,
                ),
                4,
            ),
            "categorical_candidate": round(
                max(
                    0.0,
                    categorical_score,
                ),
                4,
            ),
            "parameter_candidate": round(
                max(
                    0.0,
                    parameter_score,
                ),
                4,
            ),
            "status_candidate": round(
                max(
                    0.0,
                    status_score,
                ),
                4,
            ),
        }

        ranked = sorted(
            roles.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results.append(
            {
                "column_index": (
                    column_index + 1
                ),
                "header": (
                    header_text or None
                ),
                "observed_values": values,
                "value_count": count,
                "unique_value_count": len(
                    set(
                        canonical_text(v)
                        for v in values
                    )
                ),
                "uniqueness_ratio": round(
                    uniqueness,
                    4,
                ),
                "repeated_value_ratio": round(
                    repeated_ratio,
                    4,
                ),
                "value_type_ratios": {
                    "number": round(
                        numeric_ratio,
                        4,
                    ),
                    "date": round(
                        date_ratio,
                        4,
                    ),
                    "boolean": round(
                        boolean_ratio,
                        4,
                    ),
                    "identifier_like": round(
                        identifier_ratio,
                        4,
                    ),
                    "text": round(
                        text_ratio,
                        4,
                    ),
                },
                "role_candidates": roles,
                "primary_role_candidate": (
                    ranked[0][0]
                    if ranked
                    else None
                ),
                "primary_role_score": (
                    ranked[0][1]
                    if ranked
                    else 0.0
                ),
            }
        )

    return results


# ============================================================================
# TABLE EXTRACTION
# ============================================================================

def normalize_source_rows(
    document: dict[str, Any],
    page: dict[str, Any],
    table: dict[str, Any],
    table_index: int,
) -> list[list[dict[str, Any]]]:

    raw_rows = table.get(
        "rows",
        [],
    )

    if not isinstance(
        raw_rows,
        list,
    ):
        return []

    rows = []

    for physical_row_index, raw_row in enumerate(
        raw_rows,
        start=1,
    ):

        if not isinstance(
            raw_row,
            dict,
        ):
            continue

        raw_cells = raw_row.get(
            "cells",
            [],
        )

        if not isinstance(
            raw_cells,
            list,
        ):
            raw_cells = []

        cells = []

        for physical_column_index, raw_cell in enumerate(
            raw_cells,
            start=1,
        ):

            if not isinstance(
                raw_cell,
                dict,
            ):
                continue

            text = cell_text(
                raw_cell
            )

            cells.append(
                {
                    "column": physical_column_index,
"source_column": raw_cell.get(
    "column",
    physical_column_index,
),
                    "text": text,
                    "normalized_text": canonical_text(
                        text
                    ),
                    "value_type": value_type(
                        text
                    ),
                    "provenance": cell_provenance(
                        document,
                        page,
                        table_index,
                        physical_row_index,
                        physical_column_index,
                        raw_cell,
                    ),
                }
            )

        rows.append(
            cells
        )

    return rows


def extract_table(
    document: dict[str, Any],
    page: dict[str, Any],
    table: dict[str, Any],
    table_index: int,
) -> dict[str, Any]:

    rows = normalize_source_rows(
        document,
        page,
        table,
        table_index,
    )

    (
        header_index,
        header_confidence,
        header_reason,
    ) = detect_header_row(
        rows
    )

    columns = infer_column_roles(
        rows,
        header_index,
    )

    table_id = stable_id(
        document.get(
            "document_id"
        )
        or document.get(
            "document"
        ),
        page.get("page"),
        table_index,
    )

    header = None

    if header_index is not None:
        header = [
            cell.get(
                "text",
                "",
            )
            for cell in rows[
                header_index
            ]
        ]

    normalized_rows = []

    source_rows = table.get(
        "rows",
        [],
    )

    for source_index, cells in enumerate(
        rows
    ):

        source_row_value = (
            source_index + 1
        )

        if (
            source_index < len(
                source_rows
            )
            and isinstance(
                source_rows[
                    source_index
                ],
                dict,
            )
        ):
            source_row_value = (
                source_rows[
                    source_index
                ].get(
                    "row",
                    source_index + 1,
                )
            )

        normalized_rows.append(
            {
                "source_row_index": (
                    source_index + 1
                ),
                "source_row_value": (
                    source_row_value
                ),
                "row_type": (
                    "header"
                    if source_index
                    == header_index
                    else "data"
                ),
                "cells": cells,
            }
        )

    data_rows = [
        row
        for row in normalized_rows
        if row["row_type"] == "data"
    ]

    return {
        "table_id": table_id,
        "table_index": table_index,
        "row_count": len(rows),
        "data_row_count": len(
            data_rows
        ),
        "cell_count": sum(
            len(row["cells"])
            for row in normalized_rows
        ),
        "header": header,
        "header_detection": {
            "detected": (
                header_index is not None
            ),
            "source_row_index": (
                header_index + 1
                if header_index is not None
                else None
            ),
            "confidence": (
                header_confidence
            ),
            "reason": (
                header_reason
            ),
        },
        "columns": columns,
        "rows": normalized_rows,
        "table_provenance": {
            "document_id": (
                document.get(
                    "document_id"
                )
                or document.get(
                    "document"
                )
            ),
            "page": page.get(
                "page"
            ),
            "table_index": table_index,
            "bbox": bbox_list(
                table.get(
                    "bbox"
                )
            ),
            "status": table.get(
                "status"
            ),
            "source": table.get(
                "source"
            ),
        },
    }


# ============================================================================
# RECORD DISCOVERY
# ============================================================================

def discover_records(
    document: dict[str, Any],
    table_info: dict[str, Any],
) -> list[dict[str, Any]]:

    records = []

    columns = table_info.get(
        "columns",
        [],
    )

    for row in table_info.get(
        "rows",
        [],
    ):

        # A detected header is metadata, never a data record.
        if row.get(
            "row_type"
        ) != "data":
            continue

        fields = {}

        for cell in row.get(
            "cells",
            [],
        ):

            column_index = cell.get(
                "column"
            )

            key = (
                f"column_{column_index}"
            )

            column_definition = next(
                (
                    column
                    for column in columns
                    if column.get(
                        "column_index"
                    )
                    == column_index
                ),
                None,
            )

            fields[key] = {
                "column_index": (
                    column_index
                ),
                "header": (
                    column_definition.get(
                        "header"
                    )
                    if column_definition
                    else None
                ),
                "value": cell.get(
                    "text",
                    "",
                ),
                "normalized_value": (
                    cell.get(
                        "normalized_text",
                        "",
                    )
                ),
                "value_type": (
                    cell.get(
                        "value_type"
                    )
                ),
                "role_candidates": (
                    column_definition.get(
                        "role_candidates",
                        {},
                    )
                    if column_definition
                    else {}
                ),
                "provenance": (
                    cell.get(
                        "provenance"
                    )
                ),
            }

        # These are structural candidates only. They are NOT semantic
        # entities. Entity linking is a later Stage 13B operation.
        identifier_candidates = []

        for column_definition in columns:

            score = column_definition.get(
                "role_candidates",
                {},
            ).get(
                "identifier_candidate",
                0.0,
            )

            if score < 0.60:
                continue

            column_index = column_definition.get(
                "column_index"
            )

            for cell in row.get(
                "cells",
                [],
            ):

                if cell.get(
                    "column"
                ) != column_index:
                    continue

                value = normalize_text(
                    cell.get(
                        "text"
                    )
                )

                if not value:
                    continue

                identifier_candidates.append(
                    {
                        "column_index": (
                            column_index
                        ),
                        "header": (
                            column_definition.get(
                                "header"
                            )
                        ),
                        "value": value,
                        "normalized_value": (
                            canonical_text(
                                value
                            )
                        ),
                        "score": score,
                        "provenance": (
                            cell.get(
                                "provenance"
                            )
                        ),
                    }
                )

        document_id = (
            document.get(
                "document_id"
            )
            or document.get(
                "document"
            )
        )

        record_id = stable_id(
            document_id,
            table_info.get(
                "table_id"
            ),
            row.get(
                "source_row_index"
            ),
        )

        records.append(
            {
                "record_id": record_id,
                "document_id": document_id,
                "page": table_info.get(
                    "table_provenance",
                    {},
                ).get(
                    "page"
                ),
                "table_id": table_info.get(
                    "table_id"
                ),
                "table_index": table_info.get(
                    "table_index"
                ),
                "source_row_index": (
                    row.get(
                        "source_row_index"
                    )
                ),
                "source_row_value": (
                    row.get(
                        "source_row_value"
                    )
                ),
                "identifier_candidates": (
                    identifier_candidates
                ),
                "fields": fields,
            }
        )

    return records


# ============================================================================
# VALUE INDEX
# ============================================================================

def add_value_index(
    value_index: dict[
        str,
        list[dict[str, Any]]
    ],
    value: str,
    occurrence: dict[str, Any],
) -> None:

    normalized = canonical_text(
        value
    )

    if not normalized:
        return

    value_index[
        normalized
    ].append(
        occurrence
    )


# ============================================================================
# BUILD INDEX
# ============================================================================

def build_index(
    job_dir: Path,
) -> dict[str, Any]:

    source_files = verified_files(
        job_dir
    )

    if not source_files:
        raise FileNotFoundError(
            "No verified JSON files found in "
            f"{job_dir / 'output' / 'verified_data'}"
        )

    documents = []
    tables = []
    records = []

    value_index: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    for source_path in source_files:

        document = json.loads(
            source_path.read_text(
                encoding="utf-8"
            )
        )

        document_id = (
            document.get(
                "document_id"
            )
            or document.get(
                "document"
            )
            or source_path.stem
        )

        document_entry = {
            "document_id": document_id,
            "document": document.get(
                "document"
            ),
            "source_pdf": document.get(
                "source_pdf"
            ),
            "page_count": document.get(
                "page_count"
            ),
            "source_file": str(
                source_path.relative_to(
                    job_dir
                )
            ),
            "pages": [],
        }

        pages = document.get(
            "pages",
            [],
        )

        if not isinstance(
            pages,
            list,
        ):
            pages = []

        for page in pages:

            if not isinstance(
                page,
                dict,
            ):
                continue

            page_number = page.get(
                "page"
            )

            page_entry = {
                "page": page_number,
                "tables": [],
            }

            raw_tables = page.get(
                "tables",
                [],
            )

            if not isinstance(
                raw_tables,
                list,
            ):
                raw_tables = []

            for table_index, raw_table in enumerate(
                raw_tables,
                start=1,
            ):

                if not isinstance(
                    raw_table,
                    dict,
                ):
                    continue

                table_info = extract_table(
                    document,
                    page,
                    raw_table,
                    table_index,
                )

                table_records = discover_records(
                    document,
                    table_info,
                )

                tables.append(
                    table_info
                )

                records.extend(
                    table_records
                )

                page_entry[
                    "tables"
                ].append(
                    table_info[
                        "table_id"
                    ]
                )

                # Index all non-empty cells, including headers.
                for row in table_info[
                    "rows"
                ]:

                    for cell in row[
                        "cells"
                    ]:

                        value = normalize_text(
                            cell.get(
                                "text"
                            )
                        )

                        if not value:
                            continue

                        add_value_index(
                            value_index,
                            value,
                            {
                                "kind": "cell",
                                "document_id": (
                                    document_id
                                ),
                                "page": (
                                    page_number
                                ),
                                "table_id": (
                                    table_info[
                                        "table_id"
                                    ]
                                ),
                                "table_index": (
                                    table_index
                                ),
                                "source_row_index": (
                                    row[
                                        "source_row_index"
                                    ]
                                ),
                                "row_type": (
                                    row[
                                        "row_type"
                                    ]
                                ),
                                "column": (
                                    cell.get(
                                        "column"
                                    )
                                ),
                                "value": value,
                                "provenance": (
                                    cell.get(
                                        "provenance"
                                    )
                                ),
                            },
                        )

            document_entry[
                "pages"
            ].append(
                page_entry
            )

        documents.append(
            document_entry
        )

    deduped_value_index = {}

    for value, occurrences in (
        value_index.items()
    ):

        seen = set()
        unique = []

        for occurrence in occurrences:

            key = json.dumps(
                occurrence,
                sort_keys=True,
                ensure_ascii=False,
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(
                occurrence
            )

        deduped_value_index[
            value
        ] = unique

    header_count = sum(
        1
        for table in tables
        if table[
            "header_detection"
        ][
            "detected"
        ]
    )

    knowledge = {
        "schema_version": "2.1",
        "stage": 13,
        "stage_name": (
            "knowledge_index"
        ),
        "generated_at_utc": now_utc(),
        "job_id": job_dir.name,

        "source": {
            "verified_data_directory": str(
                job_dir
                / "output"
                / "verified_data"
            ),
            "source_of_truth": (
                "Stages 1–12 verified/reconstructed OCR data"
            ),
        },

        "design_principles": {
            "domain_specific_hardcoding": False,
            "fixed_identifier_column": False,
            "fixed_table_schema": False,
            "semantic_entity_assumption": False,
            "header_row_removed_from_records": True,
            "multi_role_column_inference": True,
            "provenance_preserved": True,
            "derived_index_only": True,
            "ambiguous_headers_are_not_forced": True,
            "identifier_requires_structural_evidence": True,
        },

        "statistics": {
            "documents": len(
                documents
            ),
            "pages": sum(
                len(
                    document[
                        "pages"
                    ]
                )
                for document in documents
            ),
            "tables": len(
                tables
            ),
            "tables_with_detected_headers": (
                header_count
            ),
            "tables_without_confident_headers": (
                len(tables)
                - header_count
            ),
            "records": len(
                records
            ),
            "cells": sum(
                table[
                    "cell_count"
                ]
                for table in tables
            ),
            "indexed_unique_values": len(
                deduped_value_index
            ),
        },

        "documents": documents,
        "tables": tables,
        "records": records,
        "value_index": (
            deduped_value_index
        ),
    }

    return knowledge


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build the generic Stage 13 knowledge index."
        )
    )

    parser.add_argument(
        "--job-id",
        default=None,
        help=(
            "Existing output/jobs/<job-id> "
            "to process."
        ),
    )

    return parser.parse_args()


def main() -> int:

    args = parse_args()

    if args.job_id:

        job_dir = (
            JOBS_DIR
            / args.job_id
        )

        if not job_dir.is_dir():
            raise FileNotFoundError(
                "Job directory does not exist: "
                f"{job_dir}"
            )

    else:
        job_dir = newest_job()

    knowledge = build_index(
        job_dir
    )

    output_path = (
        job_dir
        / "output"
        / "knowledge"
        / "knowledge_index.json"
    )

    json_dump(
        output_path,
        knowledge,
    )

    stats = knowledge[
        "statistics"
    ]

    print("=" * 78)
    print(
        "STAGE 13 — KNOWLEDGE INDEX BUILDER"
    )
    print("=" * 78)
    print(
        f"Job:                         {job_dir.name}"
    )
    print(
        f"Documents:                   {stats['documents']}"
    )
    print(
        f"Pages:                       {stats['pages']}"
    )
    print(
        f"Tables:                      {stats['tables']}"
    )
    print(
        "Tables with detected headers:"
        f" {stats['tables_with_detected_headers']}"
    )
    print(
        "Tables without confident headers:"
        f" {stats['tables_without_confident_headers']}"
    )
    print(
        f"Records:                     {stats['records']}"
    )
    print(
        f"Cells:                       {stats['cells']}"
    )
    print(
        "Unique indexed values:       "
        f"{stats['indexed_unique_values']}"
    )
    print()
    print(
        f"Knowledge index:             {output_path}"
    )
    print()
    print(
        "Stages 1–12 were not modified."
    )
    print(
        "No domain-specific entity/column assumptions were used."
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
