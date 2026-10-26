
"""Normalize raw Stage 2/3 OCR JSON into comparison_engine table schema.

Preserves source OCR text, confidence, coordinates, and provenance.
Fallback OCR remains UNVERIFIED until independently verified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _cell_text(cell: dict[str, Any]) -> str:
    value = cell.get("text", cell.get("value", ""))
    return "" if value is None else str(value).strip()


def _table_candidates(page: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer reconstructed grid tables; use ordinary tables if absent."""
    reconstructed = [
        item for item in _as_list(page.get("reconstructed_tables"))
        if isinstance(item, dict)
    ]
    if reconstructed:
        return reconstructed

    return [
        item for item in _as_list(page.get("tables"))
        if isinstance(item, dict)
    ]


def _paragraph_geometry_table_candidates(
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a conservative table candidate from aligned OCR text blocks."""
    paragraph = _as_dict(page.get("paragraph"))
    blocks = [
        block for block in _as_list(paragraph.get("text_blocks"))
        if isinstance(block, dict)
    ]

    positioned: list[dict[str, Any]] = []

    for block in blocks:
        bbox = block.get("bbox")
        text = _cell_text(block)

        if (
            not text
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue

        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        positioned.append({
            "text": text,
            "confidence": block.get("confidence"),
            "bbox": list(bbox),
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
        })

    if not positioned:
        return []

    positioned.sort(key=lambda block: (block["y"], block["x"]))

    # Cluster blocks into approximate text lines using their vertical centers.
    lines: list[list[dict[str, Any]]] = []

    for block in positioned:
        if not lines:
            lines.append([block])
            continue

        current_center = (
            sum(item["y"] for item in lines[-1]) / len(lines[-1])
        )

        if abs(block["y"] - current_center) <= 45:
            lines[-1].append(block)
        else:
            lines.append([block])

    for line in lines:
        line.sort(key=lambda block: block["x"])

    # A candidate header needs at least three cells and at least one
    # following line with the same cell count. Never invent missing cells.
    for header_index, header_line in enumerate(lines):
        column_count = len(header_line)

        if column_count < 3:
            continue

        data_lines: list[list[dict[str, Any]]] = []

        for line in lines[header_index + 1:]:
            if len(line) == column_count:
                data_lines.append(line)
            elif data_lines:
                break

        if not data_lines:
            continue

        rows = []

        for row_number, line in enumerate(
            [header_line, *data_lines],
            start=1,
        ):
            cells = []

            for column_number, block in enumerate(line, start=1):
                cells.append({
                    "column": column_number,
                    "text": block["text"],
                    "confidence": block["confidence"],
                    "bbox": block["bbox"],
                    "status": "FALLBACK_OCR",
                })

            rows.append({
                "row": row_number,
                "cells": cells,
            })

        return [{
            "rows": rows,
            "status": "FALLBACK_OCR",
            "source": "paragraph_text_blocks_geometry",
        }]

    return []


def adapt_stage_document(
    source: dict[str, Any],
    *,
    stage: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Normalize a Stage 2/3 document or page JSON.

    Accepts document JSON containing pages, a page JSON, or an object
    containing table arrays. The result is compatible with
    comparison_engine.compare_tables.
    """
    if not isinstance(source, dict):
        raise TypeError("source must be a JSON object/dict")

    document_id = str(
        source.get("document") or source.get("document_id") or ""
    ).strip()

    pages = [
        page for page in _as_list(source.get("pages"))
        if isinstance(page, dict)
    ]

    if not pages and (
        "page" in source
        or "reconstructed_tables" in source
        or "tables" in source
        or "paragraph" in source
    ):
        pages = [source]

    if not pages:
        raise ValueError(
            "No pages or table arrays found in the supplied OCR JSON"
        )

    output_tables: list[dict[str, Any]] = []

    for page_position, page in enumerate(pages, start=1):
        page_number = page.get("page", page_position)
        page_document = str(
            page.get("document") or document_id or ""
        ).strip()

        # IMPORTANT: prefer existing tables. Use paragraph geometry only
        # when neither reconstructed_tables nor tables contains candidates.
        raw_tables = _table_candidates(page)

        if not raw_tables:
            raw_tables = _paragraph_geometry_table_candidates(page)

        for table_position, raw_table in enumerate(raw_tables, start=1):
            raw_rows = [
                row for row in _as_list(raw_table.get("rows"))
                if isinstance(row, dict)
            ]

            if not raw_rows:
                # Do not guess a grid from flat cell arrays.
                continue

            header_cells = _as_list(raw_rows[0].get("cells"))
            headers = [
                _cell_text(_as_dict(cell)) or f"Column {index + 1}"
                for index, cell in enumerate(header_cells)
            ]

            source_status = str(
                raw_table.get("status") or "UNKNOWN"
            ).strip().upper()

            verification_status = (
                "UNVERIFIED"
                if source_status in {
                    "FALLBACK_OCR",
                    "LOW_CONFIDENCE",
                    "UNKNOWN",
                    "",
                }
                else source_status
            )

            normalized_rows = []

            for row_position, raw_row in enumerate(raw_rows, start=1):
                raw_cells = [
                    cell
                    for cell in _as_list(raw_row.get("cells"))
                    if isinstance(cell, dict)
                ]

                row_number = raw_row.get(
                    "row",
                    raw_row.get("source_row_index", row_position),
                )
                is_header = row_position == 1
                cells = []

                for column_position, raw_cell in enumerate(
                    raw_cells,
                    start=1,
                ):
                    text = _cell_text(raw_cell)
                    confidence = raw_cell.get("confidence")
                    bbox = raw_cell.get("bbox")
                    col_number = raw_cell.get(
                        "column", column_position
                    )

                    cell_source_status = str(
                        raw_cell.get("status") or source_status
                    ).strip().upper()

                    cell_verification = (
                        "UNVERIFIED"
                        if cell_source_status in {
                            "FALLBACK_OCR",
                            "LOW_CONFIDENCE",
                            "UNKNOWN",
                            "",
                        }
                        else cell_source_status
                    )

                    provenance = {
                        "document_id": page_document,
                        "stage": stage or None,
                        "page": page_number,
                        "table_index": table_position,
                        "table_source": raw_table.get("source"),
                        "row": row_number,
                        "column": col_number,
                        "bbox": bbox,
                        "source_ocr_confidence": confidence,
                        "source_status": cell_source_status,
                        "verification_status": cell_verification,
                        "source_path": source_path,
                    }

                    cells.append({
                        "column": col_number,
                        "text": text,
                        "normalized_text": " ".join(
                            text.casefold().split()
                        ),
                        "confidence": confidence,
                        "bbox": bbox,
                        "provenance": provenance,
                    })

                normalized_rows.append({
                    "source_row_index": row_number,
                    "source_row_value": row_number,
                    "row_type": "header" if is_header else "data",
                    "cells": cells,
                })

            table_id = (
                f"{stage}:{page_document or 'unknown'}:"
                f"p{page_number}:t{table_position}"
            )

            output_tables.append({
                "table_id": table_id,
                "table_index": table_position,
                "header": headers,
                "rows": normalized_rows,
                "table_provenance": {
                    "document_id": page_document,
                    "stage": stage,
                    "page": page_number,
                    "source_path": source_path,
                    "source_status": source_status,
                    "verification_status": verification_status,
                },
            })

    return {
        "stage": stage,
        "document": document_id or (
            pages[0].get("document") if pages else None
        ),
        "source_path": source_path,
        "tables": output_tables,
        "table_count": len(output_tables),
    }


def load_and_adapt_stage_json(
    path: str | Path,
    *,
    stage: str | None = None,
) -> dict[str, Any]:
    """Read an OCR JSON file and adapt it without modifying the source."""
    path = Path(path)

    with path.open("r", encoding="utf-8-sig") as handle:
        source = json.load(handle)

    return adapt_stage_document(
        source,
        stage=stage,
        source_path=str(path),
    )
