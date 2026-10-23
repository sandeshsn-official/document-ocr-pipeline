import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "output"
    / "final_data"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)




# ============================================================
# EMPTY-CELL EVIDENCE
# ============================================================

def _safe_bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_overlap_ratio(a, b):
    a = _safe_bbox(a)
    b = _safe_bbox(b)
    if a is None or b is None:
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _column_boundaries(table, rows):
    centers = table.get("column_centers", [])
    try:
        centers = sorted(float(x) for x in centers)
    except Exception:
        centers = []

    if len(centers) >= 2:
        boundaries = []
        for i, center in enumerate(centers):
            left = float("-inf") if i == 0 else (centers[i - 1] + center) / 2.0
            right = float("inf") if i == len(centers) - 1 else (center + centers[i + 1]) / 2.0
            boundaries.append((left, right))
        return boundaries

    # Fallback when a table does not persist its inferred column centers.
    inferred = []
    for row in rows:
        for cell in row.get("cells", []):
            if not isinstance(cell, dict) or not str(cell.get("text", "")).strip():
                continue
            bbox = _safe_bbox(cell.get("bbox"))
            if bbox is not None:
                inferred.append((bbox[0] + bbox[2]) / 2.0)

    if len(inferred) < 2:
        return []

    inferred.sort()
    span = max(inferred) - min(inferred)
    tolerance = max(15.0, 0.04 * max(1.0, span))
    clusters = []
    for x in inferred:
        if not clusters:
            clusters.append([x])
            continue
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(x - center) <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    centers = [sum(c) / len(c) for c in clusters]
    boundaries = []
    for i, center in enumerate(centers):
        left = float("-inf") if i == 0 else (centers[i - 1] + center) / 2.0
        right = float("inf") if i == len(centers) - 1 else (center + centers[i + 1]) / 2.0
        boundaries.append((left, right))
    return boundaries


def _row_y_band(row):
    boxes = []
    for cell in row.get("cells", []):
        if not isinstance(cell, dict) or not str(cell.get("text", "")).strip():
            continue
        bbox = _safe_bbox(cell.get("bbox"))
        if bbox is not None:
            boxes.append(bbox)

    if not boxes:
        return None

    y1 = min(b[1] for b in boxes)
    y2 = max(b[3] for b in boxes)
    height = max(1.0, y2 - y1)
    padding = max(4.0, 0.20 * height)
    return [y1 - padding, y2 + padding]


def _table_spatial_bbox(table):
    """Return the strongest available physical bbox for a reconstructed table.

    Validation must not use OCR blocks from another table or another page
    region as evidence for an empty cell.  Prefer the source-image region
    recorded by reconstruction; fall back to a table-level bbox when present.
    """
    if not isinstance(table, dict):
        return None

    candidates = []

    direct = _safe_bbox(table.get("bbox"))
    if direct is not None:
        candidates.append(direct)

    evidence = table.get("detection_evidence", {})
    if isinstance(evidence, dict):
        for key in ("horizontal_rule_region", "image_grid_region"):
            item = evidence.get(key)
            if isinstance(item, dict):
                bbox = _safe_bbox(item.get("bbox"))
                if bbox is not None:
                    candidates.append(bbox)

        for key in ("table_bbox", "region_bbox", "source_bbox"):
            bbox = _safe_bbox(evidence.get(key))
            if bbox is not None:
                candidates.append(bbox)

    if not candidates:
        return None

    # Prefer the most specific physical region available.  Smaller area is
    # generally the table region rather than a page-sized fallback.
    candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return candidates[0]


def _bbox_center_inside(bbox, region, margin=0.0):
    if bbox is None or region is None:
        return True
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return (
        region[0] - margin <= cx <= region[2] + margin
        and region[1] - margin <= cy <= region[3] + margin
    )


def _ocr_evidence_for_empty_cell(page, table, row, column_index):
    """
    Check whether the source OCR already contains text in the spatial slot
    occupied by an empty reconstructed cell.

    This is deliberately a validation-only check. It does not insert text,
    change a cell, or infer a value. It only distinguishes a structurally
    empty cell from a cell whose source OCR geometry suggests content may have
    been missed during reconstruction.
    """
    paragraph = page.get("paragraph", {}) if isinstance(page, dict) else {}
    blocks = paragraph.get("text_blocks", []) if isinstance(paragraph, dict) else []
    if not isinstance(blocks, list) or not blocks:
        return False, []

    rows = table.get("rows", [])
    boundaries = _column_boundaries(table, rows)
    table_region = _table_spatial_bbox(table)
    if column_index < 0 or column_index >= len(boundaries):
        return False, []

    y_band = _row_y_band(row)
    if y_band is None:
        return False, []

    left, right = boundaries[column_index]
    evidence = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text", "")).strip()
        bbox = _safe_bbox(block.get("bbox"))
        if not text or bbox is None:
            continue

        # Never borrow OCR evidence from a different physical table on the
        # same page.  A page can contain several tables/figures side by side.
        if table_region is not None and not _bbox_center_inside(bbox, table_region):
            continue

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if cx < left or cx > right:
            continue
        if cy < y_band[0] or cy > y_band[1]:
            continue

        slot_left = left if left != float("-inf") else bbox[0] - 1.0e6
        slot_right = right if right != float("inf") else bbox[2] + 1.0e6
        overlap = max(0.0, min(bbox[2], slot_right) - max(bbox[0], slot_left))
        overlap_ratio = overlap / max(1.0, bbox[2] - bbox[0])
        if overlap_ratio < 0.35:
            continue

        evidence.append({
            "text": text,
            "bbox": bbox,
            "overlap_ratio": round(overlap_ratio, 4),
        })

    return bool(evidence), evidence


def classify_empty_cell(page, table, row, cell):
    """Classify an empty cell without altering its extracted content."""
    if str(cell.get("text", "")).strip():
        return "NOT_EMPTY", []

    explicit_bbox = _safe_bbox(cell.get("bbox"))
    if explicit_bbox is not None:
        paragraph = page.get("paragraph", {}) if isinstance(page, dict) else {}
        blocks = paragraph.get("text_blocks", []) if isinstance(paragraph, dict) else []
        evidence = []
        table_region = _table_spatial_bbox(table)
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text", "")).strip()
            bbox = _safe_bbox(block.get("bbox"))
            if not text or bbox is None:
                continue
            if table_region is not None and not _bbox_center_inside(bbox, table_region):
                continue
            if _bbox_overlap_ratio(bbox, explicit_bbox) >= 0.20:
                evidence.append({"text": text, "bbox": bbox})
        if evidence:
            return "POSSIBLE_MISSING_OCR", evidence

    has_evidence, evidence = _ocr_evidence_for_empty_cell(
        page,
        table,
        row,
        int(cell.get("column", 0)) - 1,
    )
    if has_evidence:
        return "POSSIBLE_MISSING_OCR", evidence

    return "STRUCTURAL_EMPTY", []


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FINAL DATA VALIDATION")
print("PER-PDF JSON VALIDATION")
print("=" * 70)


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not INPUT_DIR.exists():

    print()
    print(
        "ERROR: Final data directory does not exist:"
    )

    print(
        INPUT_DIR
    )

    raise SystemExit(1)


# ============================================================
# FIND INDIVIDUAL DOCUMENT JSON FILES
# ============================================================

json_files = sorted(
    [
        path
        for path in INPUT_DIR.glob("*.json")
        if path.name != "all_documents.json"
    ]
)


print()

print(
    f"Individual JSON files found: "
    f"{len(json_files)}"
)


for json_file in json_files:

    print(
        f"  {json_file.name}"
    )


if not json_files:

    print()
    print(
        "ERROR: No individual document JSON files found."
    )

    raise SystemExit(1)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

total_documents = 0
total_pages = 0
total_paragraph_chars = 0
total_text_blocks = 0
total_tables = 0
total_rows = 0
total_cells = 0

total_missing_paragraphs = 0
total_missing_tables = 0
total_missing_cell_text = 0
total_structural_empty_cells = 0
total_suspicious_empty_cells = 0
total_low_confidence_cells = 0

global_verification_counts = {}

documents_with_problems = []


# ============================================================
# PROCESS EACH PDF JSON INDEPENDENTLY
# ============================================================

for input_file in json_files:

    print()
    print("=" * 70)

    print(
        f"DOCUMENT JSON: "
        f"{input_file.name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD DOCUMENT
    # --------------------------------------------------------

    try:

        data = load_json(
            input_file
        )

    except Exception as error:

        print()

        print(
            "ERROR reading JSON:"
        )

        print(
            error
        )

        documents_with_problems.append(
            input_file.stem
        )

        continue


    # --------------------------------------------------------
    # DOCUMENT STRUCTURE
    # --------------------------------------------------------

    if not isinstance(data, dict):
        print("ERROR: JSON root must be an object/dictionary.")
        documents_with_problems.append(input_file.stem)
        continue

    # --------------------------------------------------------
    # DOCUMENT INFORMATION
    # --------------------------------------------------------

    document_id = data.get(
        "document_id",
        input_file.stem
    )

    pages = data.get(
        "pages",
        []
    )


    if not isinstance(
        pages,
        list
    ):

        pages = []


    total_documents += 1


    print()

    print(
        f"Document: "
        f"{document_id}"
    )

    print(
        f"Pages: "
        f"{len(pages)}"
    )


    # --------------------------------------------------------
    # DOCUMENT-LEVEL STATISTICS
    # --------------------------------------------------------

    document_pages = 0
    document_paragraph_chars = 0
    document_text_blocks = 0
    document_tables = 0
    document_rows = 0
    document_cells = 0

    document_missing_paragraphs = 0
    document_missing_tables = 0
    document_malformed_pages = 0
    document_missing_cell_text = 0
    document_structural_empty_cells = 0
    document_suspicious_empty_cells = 0
    document_suspicious_empty_details = []
    document_low_confidence_cells = 0

    document_verification_counts = {}


    # ========================================================
    # PAGES
    # ========================================================

    for page in pages:

        if not isinstance(page, dict):
            document_malformed_pages += 1
            print(f"WARNING: Ignoring malformed page entry: {page!r}")
            continue


        page_number = page.get(
            "page",
            "UNKNOWN"
        )


        document_pages += 1
        total_pages += 1


        paragraph = page.get(
            "paragraph",
            {}
        )


        if not isinstance(
            paragraph,
            dict
        ):

            paragraph = {}


        paragraph_text = paragraph.get(
            "text",
            ""
        )


        if not isinstance(
            paragraph_text,
            str
        ):

            paragraph_text = str(
                paragraph_text
            )


        text_blocks = paragraph.get(
            "text_blocks",
            []
        )


        if not isinstance(
            text_blocks,
            list
        ):

            text_blocks = []


        tables = page.get(
            "tables",
            []
        )


        if not isinstance(
            tables,
            list
        ):

            tables = []


        # ----------------------------------------------------
        # PARAGRAPH
        # ----------------------------------------------------

        if not paragraph_text.strip():

            document_missing_paragraphs += 1
            total_missing_paragraphs += 1


        document_paragraph_chars += len(
            paragraph_text
        )

        total_paragraph_chars += len(
            paragraph_text
        )


        document_text_blocks += len(
            text_blocks
        )

        total_text_blocks += len(
            text_blocks
        )


        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        if not tables:

            document_missing_tables += 1
            total_missing_tables += 1


        document_tables += len(
            tables
        )

        total_tables += len(
            tables
        )


        print()

        print(
            f"PAGE {page_number}"
        )

        print(
            f"  Paragraph characters : "
            f"{len(paragraph_text)}"
        )

        print(
            f"  Text blocks          : "
            f"{len(text_blocks)}"
        )

        print(
            f"  Tables               : "
            f"{len(tables)}"
        )


        # ====================================================
        # TABLE VALIDATION
        # ====================================================

        for table_index, table in enumerate(
            tables,
            start=1
        ):

            if not isinstance(
                table,
                dict
            ):

                continue


            rows = table.get(
                "rows",
                []
            )


            if not isinstance(
                rows,
                list
            ):

                rows = []


            document_rows += len(
                rows
            )

            total_rows += len(
                rows
            )


            print(
                f"    Table {table_index}: "
                f"{len(rows)} rows"
            )


            # ------------------------------------------------
            # ROWS
            # ------------------------------------------------

            for row_index, row in enumerate(rows, start=1):

                if not isinstance(
                    row,
                    dict
                ):

                    continue


                cells = row.get(
                    "cells",
                    []
                )


                if not isinstance(
                    cells,
                    list
                ):

                    cells = []


                document_cells += len(
                    cells
                )

                total_cells += len(
                    cells
                )


                # --------------------------------------------
                # CELLS
                # --------------------------------------------

                for cell_index, cell in enumerate(cells, start=1):

                    if not isinstance(
                        cell,
                        dict
                    ):

                        continue


                    text = cell.get(
                        "text",
                        ""
                    )


                    if not isinstance(
                        text,
                        str
                    ):

                        text = str(
                            text
                        )


                    confidence = cell.get(
                        "paddle_confidence",
                        0.0
                    )


                    status = cell.get(
                        "verification_status",
                        ""
                    )


                    # ----------------------------------------
                    # EMPTY TEXT — EVIDENCE BASED
                    # ----------------------------------------

                    if not text.strip():

                        document_missing_cell_text += 1
                        total_missing_cell_text += 1

                        classification, evidence = classify_empty_cell(
                            page,
                            table,
                            row,
                            cell,
                        )

                        if classification == "POSSIBLE_MISSING_OCR":
                            document_suspicious_empty_cells += 1
                            total_suspicious_empty_cells += 1

                            document_suspicious_empty_details.append({
                                "page": page_number,
                                "table": table_index,
                                "row": row_index,
                                "column": cell.get("column", cell_index),
                                "cell_bbox": cell.get("bbox"),
                                "evidence": evidence,
                            })
                        else:
                            document_structural_empty_cells += 1
                            total_structural_empty_cells += 1


                    # ----------------------------------------
                    # LOW CONFIDENCE
                    # ----------------------------------------

                    try:

                        confidence = float(
                            confidence
                        )

                    except Exception:

                        confidence = 0.0


                    if (
                        confidence > 0
                        and confidence < 0.80
                    ):

                        document_low_confidence_cells += 1
                        total_low_confidence_cells += 1


                    # ----------------------------------------
                    # VERIFICATION STATUS
                    # ----------------------------------------

                    if status:

                        document_verification_counts[
                            status
                        ] = (
                            document_verification_counts.get(
                                status,
                                0
                            )
                            + 1
                        )


                        global_verification_counts[
                            status
                        ] = (
                            global_verification_counts.get(
                                status,
                                0
                            )
                            + 1
                        )


    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    print()

    print(
        "-" * 70
    )

    print(
        "DOCUMENT SUMMARY"
    )

    print(
        "-" * 70
    )

    print()

    print(
        f"Pages                  : "
        f"{document_pages}"
    )

    print(
        f"Paragraph characters   : "
        f"{document_paragraph_chars}"
    )

    print(
        f"Text blocks            : "
        f"{document_text_blocks}"
    )

    print(
        f"Tables                 : "
        f"{document_tables}"
    )

    print(
        f"Rows                   : "
        f"{document_rows}"
    )

    print(
        f"Cells                  : "
        f"{document_cells}"
    )


    print()

    print(
        "Potential problems:"
    )

    print(
        f"  Pages with no paragraph : "
        f"{document_missing_paragraphs}"
    )

    print(
        f"  Pages with no table     : "
        f"{document_missing_tables}"
    )

    print(
        f"  Cells with empty text   : "
        f"{document_missing_cell_text}"
    )

    print(
        f"  Structural empty cells  : "
        f"{document_structural_empty_cells}"
    )

    print(
        f"  Possible missing OCR    : "
        f"{document_suspicious_empty_cells}"
    )

    print(
        f"  Low-confidence cells    : "
        f"{document_low_confidence_cells}"
    )


    # --------------------------------------------------------
    # POSSIBLE MISSING OCR DETAILS
    # --------------------------------------------------------

    if document_suspicious_empty_details:

        print()
        print("Possible missing OCR details:")

        for item_index, item in enumerate(
            document_suspicious_empty_details,
            start=1
        ):

            print()
            print(
                f"  Item {item_index}:"
            )
            print(
                f"    Page       : {item.get('page')}"
            )
            print(
                f"    Table      : {item.get('table')}"
            )
            print(
                f"    Row        : {item.get('row')}"
            )
            print(
                f"    Column     : {item.get('column')}"
            )
            print(
                f"    Cell bbox  : {item.get('cell_bbox')}"
            )

            evidence = item.get("evidence", [])

            if evidence:

                print("    OCR evidence:")

                for evidence_item in evidence:

                    print(
                        f"      Text       : {evidence_item.get('text', '')}"
                    )
                    print(
                        f"      OCR bbox   : {evidence_item.get('bbox')}"
                    )

                    if "overlap_ratio" in evidence_item:
                        print(
                            f"      Overlap    : {evidence_item.get('overlap_ratio')}"
                        )

            else:

                print(
                    "    OCR evidence: unavailable"
                )


    # --------------------------------------------------------
    # VERIFICATION STATUS
    # --------------------------------------------------------

    print()

    print(
        "OCR verification status:"
    )


    if document_verification_counts:

        for status, count in sorted(
            document_verification_counts.items()
        ):

            print(
                f"  {status:25s}: "
                f"{count}"
            )

    else:

        print(
            "  No verification statuses found."
        )


    # ========================================================
    # DOCUMENT CHECK
    # ========================================================

    document_problems = []


    if document_pages == 0:
        document_problems.append("No valid pages")

    if document_malformed_pages > 0:
        document_problems.append(
            f"{document_malformed_pages} malformed page entries were skipped"
        )

    if document_missing_paragraphs > 0:
        document_problems.append(
            f"{document_missing_paragraphs} pages have no paragraph text"
        )

    if document_missing_tables > 0:
        document_problems.append(
            f"{document_missing_tables} pages have no reconstructed tables"
        )

    if document_paragraph_chars == 0:

        document_problems.append(
            "No paragraph text"
        )


    if document_cells == 0:

        document_problems.append(
            "No table cells"
        )


    if document_suspicious_empty_cells > 0:

        document_problems.append(
            f"{document_suspicious_empty_cells} table cells may contain missing OCR"
        )


    if document_problems:

        documents_with_problems.append(
            document_id
        )

        print()

        print(
            "DOCUMENT STATUS: NEEDS REVIEW"
        )

        for problem in document_problems:

            print(
                f"  - {problem}"
            )

    else:

        print()

        print(
            "DOCUMENT STATUS: BASIC VALIDATION PASSED"
        )


# ============================================================
# GLOBAL VALIDATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("ALL INDIVIDUAL JSON FILES — SUMMARY")
print("=" * 70)

print()

print(
    f"Documents              : "
    f"{total_documents}"
)

print(
    f"Pages                  : "
    f"{total_pages}"
)

print(
    f"Paragraph characters   : "
    f"{total_paragraph_chars}"
)

print(
    f"Text blocks            : "
    f"{total_text_blocks}"
)

print(
    f"Tables                 : "
    f"{total_tables}"
)

print(
    f"Rows                   : "
    f"{total_rows}"
)

print(
    f"Cells                  : "
    f"{total_cells}"
)


# ============================================================
# GLOBAL PROBLEMS
# ============================================================

print()
print("=" * 70)
print("GLOBAL POTENTIAL PROBLEMS")
print("=" * 70)

print()

print(
    f"Pages with no paragraph : "
    f"{total_missing_paragraphs}"
)

print(
    f"Pages with no table     : "
    f"{total_missing_tables}"
)

print(
    f"Cells with empty text   : "
    f"{total_missing_cell_text}"
)

print(
    f"Structural empty cells  : "
    f"{total_structural_empty_cells}"
)

print(
    f"Possible missing OCR    : "
    f"{total_suspicious_empty_cells}"
)

print(
    f"Low-confidence cells    : "
    f"{total_low_confidence_cells}"
)


# ============================================================
# GLOBAL VERIFICATION COUNTS
# ============================================================

print()
print("=" * 70)
print("GLOBAL OCR VERIFICATION STATUS")
print("=" * 70)


if global_verification_counts:

    for status, count in sorted(
        global_verification_counts.items()
    ):

        print(
            f"{status:25s}: "
            f"{count}"
        )

else:

    print(
        "No verification statuses found."
    )


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)


if documents_with_problems:

    print()

    print(
        "STATUS: NEEDS REVIEW"
    )

    print()

    print(
        "Documents requiring attention:"
    )

    for document in documents_with_problems:

        print(
            f"  - {document}"
        )

else:

    print()

    print(
        "STATUS: BASIC VALIDATION PASSED"
    )


print()
print("=" * 70)

# Return a non-zero exit status when any document requires attention.
# The report above remains available for detailed diagnostics.
if documents_with_problems:
    raise SystemExit(1)
