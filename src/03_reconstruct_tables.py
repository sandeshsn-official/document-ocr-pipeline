from pathlib import Path
import json
import math


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("output/final")
OUTPUT_DIR = Path("output/reconstructed")

ROW_TOLERANCE = 25

# Fallback table detection
MIN_BLOCKS_PER_ROW = 3
MIN_TABLE_ROWS = 3
MAX_ROW_GAP = 150
COLUMN_X_TOLERANCE = 80


# ============================================================
# BASIC HELPERS
# ============================================================

def get_box_center(bbox):
    """
    Accept:
        [x1, y1, x2, y2]

    or:
        [[x1,y1], [x2,y2], ...]
    """

    if not bbox:
        return None

    try:

        # Rectangle
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(v, (int, float))
                for v in bbox
            )
        ):
            x1, y1, x2, y2 = bbox

            return (
                (x1 + x2) / 2,
                (y1 + y2) / 2
            )

        # Polygon
        if isinstance(bbox, list):

            points = []

            for point in bbox:

                if (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                ):
                    points.append(
                        (
                            float(point[0]),
                            float(point[1])
                        )
                    )

            if points:

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                return (
                    (min(xs) + max(xs)) / 2,
                    (min(ys) + max(ys)) / 2
                )

    except Exception:
        pass

    return None


def normalize_bbox(bbox):

    if not bbox:
        return None

    try:

        # Rectangle
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(v, (int, float))
                for v in bbox
            )
        ):
            x1, y1, x2, y2 = bbox

            return [
                float(min(x1, x2)),
                float(min(y1, y2)),
                float(max(x1, x2)),
                float(max(y1, y2))
            ]

        # Polygon
        if isinstance(bbox, list):

            points = []

            for point in bbox:

                if (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                ):
                    points.append(
                        (
                            float(point[0]),
                            float(point[1])
                        )
                    )

            if points:

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                return [
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys)
                ]

    except Exception:
        pass

    return None


# ============================================================
# EXISTING TABLE RECONSTRUCTION
# ============================================================

def reconstruct_table(table):

    cells = table.get("cells", [])

    if not cells:

        return {
            "rows": [],
            "row_count": 0,
            "cell_count": 0,
            "status": table.get("status", "")
        }

    positioned_cells = []

    for cell in cells:

        bbox = cell.get("bbox")

        center = get_box_center(bbox)

        if center is None:
            continue

        x_center, y_center = center

        positioned_cells.append(
            {
                "cell": cell,
                "x": x_center,
                "y": y_center
            }
        )

    if not positioned_cells:

        return {
            "rows": [],
            "row_count": 0,
            "cell_count": 0,
            "status": table.get("status", "")
        }

    # --------------------------------------------------------
    # IMPORTANT:
    # Sort by actual vertical position first
    # --------------------------------------------------------

    positioned_cells.sort(
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    rows = []

    for item in positioned_cells:

        placed = False

        for row in rows:

            row_y = sum(
                cell["y"]
                for cell in row
            ) / len(row)

            heights = []

            for existing in row:

                bbox = existing["cell"].get(
                    "bbox"
                )

                if bbox:

                    heights.append(
                        abs(
                            bbox[3] - bbox[1]
                        )
                    )

            current_bbox = item["cell"].get(
                "bbox"
            )

            if current_bbox:

                current_height = abs(
                    current_bbox[3]
                    - current_bbox[1]
                )
            else:

                current_height = 25

            if heights:

                average_height = (
                    sum(heights)
                    / len(heights)
                )

            else:

                average_height = current_height

            adaptive_tolerance = max(
                ROW_TOLERANCE,
                average_height * 0.7,
                current_height * 0.7
            )

            if abs(
                item["y"] - row_y
            ) <= adaptive_tolerance:

                row.append(item)
                placed = True
                break

        if not placed:

            rows.append(
                [item]
            )

    # --------------------------------------------------------
    # Sort rows TOP → BOTTOM
    # --------------------------------------------------------

    rows.sort(
        key=lambda row: sum(
            item["y"]
            for item in row
        ) / len(row)
    )

    reconstructed_rows = []

    for row_index, row in enumerate(
        rows,
        start=1
    ):

        # ----------------------------------------------------
        # Sort cells LEFT → RIGHT
        # ----------------------------------------------------

        row.sort(
            key=lambda item: item["x"]
        )

        reconstructed_cells = []

        for column_index, item in enumerate(
            row,
            start=1
        ):

            cell = item["cell"]

            reconstructed_cells.append(
                {
                    "column": column_index,
                    "text": cell.get(
                        "text",
                        ""
                    ),
                    "bbox": cell.get(
                        "bbox"
                    ),
                    "confidence": cell.get(
                        "confidence",
                        None
                    ),
                    "status": cell.get(
                        "status",
                        ""
                    )
                }
            )

        reconstructed_rows.append(
            {
                "row": row_index,
                "cells": reconstructed_cells
            }
        )

    return {
        "rows": reconstructed_rows,
        "row_count": len(
            reconstructed_rows
        ),
        "cell_count": sum(
            len(row["cells"])
            for row in reconstructed_rows
        ),
        "status": table.get(
            "status",
            ""
        )
    }


# ============================================================
# GET OCR PARAGRAPH BLOCKS
# ============================================================

def get_paragraph_blocks(page_data):

    paragraph = page_data.get(
        "paragraph",
        {}
    )

    if not isinstance(
        paragraph,
        dict
    ):
        return []

    blocks = paragraph.get(
        "text_blocks",
        []
    )

    if not isinstance(
        blocks,
        list
    ):
        return []

    valid_blocks = []

    for block in blocks:

        if not isinstance(
            block,
            dict
        ):
            continue

        text = str(
            block.get(
                "text",
                ""
            )
        ).strip()

        bbox = normalize_bbox(
            block.get(
                "bbox"
            )
        )

        if not text:
            continue

        if bbox is None:
            continue

        center = get_box_center(
            bbox
        )

        if center is None:
            continue

        valid_blocks.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": block.get(
                    "confidence",
                    None
                ),
                "x": center[0],
                "y": center[1]
            }
        )

    return valid_blocks


# ============================================================
# GROUP OCR BLOCKS INTO VISUAL ROWS
# ============================================================

def group_blocks_into_visual_rows(blocks):

    if not blocks:
        return []

    # TOP → BOTTOM
    blocks = sorted(
        blocks,
        key=lambda b: (
            b["y"],
            b["x"]
        )
    )

    rows = []

    for block in blocks:

        placed = False

        for row in rows:

            average_y = sum(
                item["y"]
                for item in row
            ) / len(row)

            heights = []

            for item in row:

                bbox = item["bbox"]

                heights.append(
                    abs(
                        bbox[3] - bbox[1]
                    )
                )

            current_bbox = block["bbox"]

            current_height = abs(
                current_bbox[3]
                - current_bbox[1]
            )

            if heights:

                average_height = (
                    sum(heights)
                    / len(heights)
                )

            else:

                average_height = current_height

            adaptive_tolerance = max(
                ROW_TOLERANCE,
                average_height * 0.7,
                current_height * 0.7
            )

            if abs(
                block["y"] - average_y
            ) <= adaptive_tolerance:

                row.append(block)

                placed = True

                break

        if not placed:

            rows.append(
                [block]
            )

    # --------------------------------------------------------
    # LEFT → RIGHT
    # --------------------------------------------------------

    for row in rows:

        row.sort(
            key=lambda b: b["x"]
        )

    # --------------------------------------------------------
    # TOP → BOTTOM
    # --------------------------------------------------------

    rows.sort(
        key=lambda row: sum(
            b["y"]
            for b in row
        ) / len(row)
    )

    return rows


# ============================================================
# FIND CANDIDATE TABLE REGIONS
# ============================================================

def find_candidate_table_regions(rows):

    candidate_rows = [
        row
        for row in rows
        if len(row) >= MIN_BLOCKS_PER_ROW
    ]

    if not candidate_rows:
        return []

    regions = []

    current_region = [
        candidate_rows[0]
    ]

    for row in candidate_rows[1:]:

        previous_y = sum(
            b["y"]
            for b in current_region[-1]
        ) / len(
            current_region[-1]
        )

        current_y = sum(
            b["y"]
            for b in row
        ) / len(row)

        gap = current_y - previous_y

        if gap <= MAX_ROW_GAP:

            current_region.append(
                row
            )

        else:

            if len(current_region) >= MIN_TABLE_ROWS:

                regions.append(
                    current_region
                )

            current_region = [
                row
            ]

    if len(current_region) >= MIN_TABLE_ROWS:

        regions.append(
            current_region
        )

    return regions


# ============================================================
# CHECK COLUMN CONSISTENCY
# ============================================================

def determine_common_columns(region):

    if not region:
        return False, []

    reference_row = max(
        region,
        key=len
    )

    reference_x = sorted(
        block["x"]
        for block in reference_row
    )

    if len(reference_x) < MIN_BLOCKS_PER_ROW:

        return False, []

    matching_row_count = 0

    for row in region:

        row_x = sorted(
            block["x"]
            for block in row
        )

        matches = 0

        for ref_x in reference_x:

            closest_distance = min(
                abs(
                    ref_x - x
                )
                for x in row_x
            )

            if closest_distance <= COLUMN_X_TOLERANCE:

                matches += 1

        if matches >= MIN_BLOCKS_PER_ROW:

            matching_row_count += 1

    required_rows = max(
        MIN_TABLE_ROWS,
        math.ceil(
            len(region) * 0.60
        )
    )

    if matching_row_count < required_rows:

        return False, []

    return True, reference_x


# ============================================================
# HEADER DETECTION
# ============================================================

def looks_like_header(row):

    """
    Detect a likely header using common table-header words.

    This does NOT modify OCR text.
    It only helps move a visually misplaced header row
    to the beginning when OCR geometry produces a strange
    ordering.
    """

    header_words = {
        "id",
        "sensor",
        "reading",
        "range",
        "unit",
        "flag",
        "channel",
        "step",
        "action",
        "expected",
        "observed",
        "result",
        "time",
        "parameter",
        "reference",
        "decision",
        "comment",
        "mass",
        "pressure",
        "temperature",
        "impedance",
        "particle",
        "size"
    }

    score = 0

    for block in row:

        text = (
            str(
                block.get(
                    "text",
                    ""
                )
            )
            .strip()
            .lower()
        )

        words = text.replace(
            "/",
            " "
        ).replace(
            "_",
            " "
        ).split()

        for word in words:

            if word in header_words:

                score += 1

    return score >= 2


def move_header_to_front(region):

    if not region:
        return region

    # Already a header
    if looks_like_header(
        region[0]
    ):
        return region

    for index in range(
        1,
        len(region)
    ):

        if looks_like_header(
            region[index]
        ):

            header = region[index]

            remaining = (
                region[:index]
                + region[index + 1:]
            )

            return [
                header
            ] + remaining

    return region


# ============================================================
# BUILD FALLBACK TABLE
# ============================================================

def build_fallback_table(region):

    if not region:
        return None

    looks_like_table, reference_x = (
        determine_common_columns(
            region
        )
    )

    if not looks_like_table:
        return None

    # --------------------------------------------------------
    # Put likely header first
    # --------------------------------------------------------

    region = move_header_to_front(
        region
    )

    cells = []

    for row_index, row in enumerate(
        region,
        start=1
    ):

        row = sorted(
            row,
            key=lambda b: b["x"]
        )

        for column_index, block in enumerate(
            row,
            start=1
        ):

            cells.append(
                {
                    "text": block["text"],
                    "bbox": block["bbox"],
                    "confidence": block.get(
                        "confidence",
                        None
                    ),
                    "status": "FALLBACK_OCR",
                    "row": row_index,
                    "column": column_index
                }
            )

    if len(cells) < (
        MIN_TABLE_ROWS
        * MIN_BLOCKS_PER_ROW
    ):

        return None

    return {
        "cells": cells,
        "status": "FALLBACK_OCR",
        "source": "paragraph_text_blocks",
        "confidence_method": "OCR_BLOCK_CONFIDENCE"
    }


# ============================================================
# FALLBACK DETECTOR
# ============================================================

def detect_tables_from_paragraph_blocks(
    page_data
):

    blocks = get_paragraph_blocks(
        page_data
    )

    minimum_blocks = (
        MIN_TABLE_ROWS
        * MIN_BLOCKS_PER_ROW
    )

    if len(blocks) < minimum_blocks:

        return []

    rows = group_blocks_into_visual_rows(
        blocks
    )

    regions = find_candidate_table_regions(
        rows
    )

    fallback_tables = []

    for region in regions:

        table = build_fallback_table(
            region
        )

        if table is not None:

            fallback_tables.append(
                table
            )

    return fallback_tables


# ============================================================
# PROCESS PAGE
# ============================================================

def process_page(page_data):

    reconstructed_tables = []

    # --------------------------------------------------------
    # PRIMARY TABLES
    # --------------------------------------------------------

    tables = page_data.get(
        "tables",
        []
    )

    if isinstance(
        tables,
        list
    ):

        for table in tables:

            if not isinstance(
                table,
                dict
            ):
                continue

            reconstructed = reconstruct_table(
                table
            )

            if reconstructed.get(
                "cell_count",
                0
            ) > 0:

                reconstructed_tables.append(
                    reconstructed
                )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not reconstructed_tables:

        fallback_tables = (
            detect_tables_from_paragraph_blocks(
                page_data
            )
        )

        for table in fallback_tables:

            reconstructed = reconstruct_table(
                table
            )

            if reconstructed.get(
                "cell_count",
                0
            ) > 0:

                reconstructed_tables.append(
                    reconstructed
                )

    # --------------------------------------------------------
    # PRESERVE ORIGINAL PAGE DATA
    # --------------------------------------------------------

    result = dict(
        page_data
    )

    result[
        "reconstructed_tables"
    ] = reconstructed_tables

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TABLE RECONSTRUCTION + OCR FALLBACK")
    print("=" * 80)

    if not INPUT_DIR.exists():

        print(
            f"ERROR: Input directory does not exist: "
            f"{INPUT_DIR}"
        )

        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    document_dirs = [
        d
        for d in INPUT_DIR.iterdir()
        if d.is_dir()
    ]

    if not document_dirs:

        print(
            "No document directories found."
        )

        return

    total_documents = 0
    total_pages = 0
    total_tables = 0
    total_cells = 0

    # ========================================================
    # DOCUMENTS
    # ========================================================

    for document_dir in sorted(
        document_dirs
    ):

        document_name = (
            document_dir.name
        )

        print()
        print(
            f"Document: {document_name}"
        )

        output_document_dir = (
            OUTPUT_DIR
            / document_name
        )

        output_document_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        page_files = sorted(
            document_dir.glob(
                "page_*.json"
            )
        )

        if not page_files:

            print(
                "  No page JSON files found."
            )

            continue

        document_pages = []

        document_table_count = 0
        document_cell_count = 0

        # ====================================================
        # PAGES
        # ====================================================

        for page_file in page_files:

            try:

                with open(
                    page_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    page_data = json.load(f)

            except Exception as e:

                print(
                    f"  ERROR reading "
                    f"{page_file.name}: {e}"
                )

                continue

            processed_page = process_page(
                page_data
            )

            reconstructed_tables = (
                processed_page.get(
                    "reconstructed_tables",
                    []
                )
            )

            table_count = len(
                reconstructed_tables
            )

            cell_count = sum(
                table.get(
                    "cell_count",
                    0
                )
                for table in reconstructed_tables
            )

            fallback_count = sum(
                1
                for table in reconstructed_tables
                if table.get(
                    "status"
                ) == "FALLBACK_OCR"
            )

            print(
                f"  {page_file.name}: "
                f"Tables={table_count}, "
                f"Cells={cell_count}"
                + (
                    f", Fallback={fallback_count}"
                    if fallback_count
                    else ""
                )
            )

            output_page_file = (
                output_document_dir
                / page_file.name
            )

            with open(
                output_page_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    processed_page,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            document_pages.append(
                processed_page
            )

            document_table_count += (
                table_count
            )

            document_cell_count += (
                cell_count
            )

            total_pages += 1

        # ====================================================
        # DOCUMENT JSON
        # ====================================================

        document_output = {
            "document": document_name,
            "page_count": len(
                document_pages
            ),
            "table_count": document_table_count,
            "cell_count": document_cell_count,
            "pages": document_pages
        }

        document_json_path = (
            output_document_dir
            / "document.json"
        )

        with open(
            document_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                document_output,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"  Document totals: "
            f"Tables={document_table_count}, "
            f"Cells={document_cell_count}"
        )

        total_documents += 1

        total_tables += (
            document_table_count
        )

        total_cells += (
            document_cell_count
        )

    # ========================================================
    # ALL DOCUMENTS
    # ========================================================

    all_documents = []

    for document_dir in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if not document_dir.is_dir():
            continue

        document_json = (
            document_dir
            / "document.json"
        )

        if not document_json.exists():
            continue

        try:

            with open(
                document_json,
                "r",
                encoding="utf-8"
            ) as f:

                all_documents.append(
                    json.load(f)
                )

        except Exception as e:

            print(
                f"WARNING: Could not read "
                f"{document_json}: {e}"
            )

    all_documents_output = {
        "status": "COMPLETED",
        "statistics": {
            "documents": total_documents,
            "pages": total_pages,
            "tables": total_tables,
            "cells": total_cells
        },
        "documents": all_documents
    }

    all_documents_path = (
        OUTPUT_DIR
        / "all_documents.json"
    )

    with open(
        all_documents_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_documents_output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("TABLE RECONSTRUCTION COMPLETE")
    print("=" * 80)

    print(
        f"Documents processed:  {total_documents}"
    )

    print(
        f"Pages processed:      {total_pages}"
    )

    print(
        f"Tables reconstructed:  {total_tables}"
    )

    print(
        f"Cells reconstructed:   {total_cells}"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"All documents: "
        f"{all_documents_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()