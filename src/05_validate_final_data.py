import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "output/final_data/all_documents.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL DATA VALIDATION")
print("=" * 70)

print()
print(f"Input: {INPUT_FILE}")


if not INPUT_FILE.exists():

    print()
    print("ERROR: Final JSON file does not exist.")

    raise SystemExit(1)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


# ============================================================
# BASIC VALIDATION
# ============================================================

documents = data.get(
    "documents",
    []
)


print()
print(
    f"Documents found: "
    f"{len(documents)}"
)


if not documents:

    print()
    print("ERROR: No documents found.")

    raise SystemExit(1)


# ============================================================
# STATISTICS
# ============================================================

total_pages = 0
total_paragraph_chars = 0
total_text_blocks = 0
total_tables = 0
total_rows = 0
total_cells = 0

missing_paragraphs = 0
missing_tables = 0
missing_cell_text = 0
low_confidence_cells = 0

verification_counts = {}


# ============================================================
# DOCUMENTS
# ============================================================

for document in documents:

    document_id = document.get(
        "document_id",
        "UNKNOWN"
    )

    pages = document.get(
        "pages",
        []
    )


    print()
    print("=" * 70)
    print(
        f"DOCUMENT: {document_id}"
    )
    print("=" * 70)

    print()
    print(
        f"Pages: {len(pages)}"
    )


    # ========================================================
    # PAGES
    # ========================================================

    for page in pages:

        page_number = page.get(
            "page",
            "UNKNOWN"
        )


        total_pages += 1


        paragraph = page.get(
            "paragraph",
            {}
        )


        paragraph_text = paragraph.get(
            "text",
            ""
        )


        text_blocks = paragraph.get(
            "text_blocks",
            []
        )


        tables = page.get(
            "tables",
            []
        )


        # ----------------------------------------------------
        # PARAGRAPH
        # ----------------------------------------------------

        if not paragraph_text.strip():

            missing_paragraphs += 1


        total_paragraph_chars += len(
            paragraph_text
        )


        total_text_blocks += len(
            text_blocks
        )


        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        if not tables:

            missing_tables += 1


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

            rows = table.get(
                "rows",
                []
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

            for row in rows:

                cells = row.get(
                    "cells",
                    []
                )


                total_cells += len(
                    cells
                )


                # --------------------------------------------
                # CELLS
                # --------------------------------------------

                for cell in cells:

                    text = cell.get(
                        "text",
                        ""
                    )


                    paddle_text = cell.get(
                        "paddle_text",
                        ""
                    )


                    tesseract_text = cell.get(
                        "tesseract_text",
                        ""
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
                    # EMPTY TEXT
                    # ----------------------------------------

                    if not text.strip():

                        missing_cell_text += 1


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

                        low_confidence_cells += 1


                    # ----------------------------------------
                    # VERIFICATION STATUS
                    # ----------------------------------------

                    if status:

                        verification_counts[
                            status
                        ] = (
                            verification_counts.get(
                                status,
                                0
                            )
                            + 1
                        )


# ============================================================
# VALIDATION REPORT
# ============================================================

print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print()

print(
    f"Documents              : "
    f"{len(documents)}"
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
# PROBLEMS
# ============================================================

print()
print("=" * 70)
print("POTENTIAL PROBLEMS")
print("=" * 70)

print()

print(
    f"Pages with no paragraph : "
    f"{missing_paragraphs}"
)

print(
    f"Pages with no table     : "
    f"{missing_tables}"
)

print(
    f"Cells with empty text   : "
    f"{missing_cell_text}"
)

print(
    f"Low-confidence cells    : "
    f"{low_confidence_cells}"
)


# ============================================================
# VERIFICATION COUNTS
# ============================================================

print()
print("=" * 70)
print("OCR VERIFICATION STATUS")
print("=" * 70)

if verification_counts:

    for status, count in sorted(
        verification_counts.items()
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

problems = []


if len(documents) == 0:

    problems.append(
        "No documents"
    )


if total_pages == 0:

    problems.append(
        "No pages"
    )


if total_paragraph_chars == 0:

    problems.append(
        "No paragraph text"
    )


if total_cells == 0:

    problems.append(
        "No table cells"
    )


if missing_cell_text > 0:

    problems.append(
        f"{missing_cell_text} empty table cells"
    )


if problems:

    print()
    print(
        "STATUS: NEEDS REVIEW"
    )

    print()

    for problem in problems:

        print(
            f"  - {problem}"
        )

else:

    print()
    print(
        "STATUS: BASIC VALIDATION PASSED"
    )


print()
print("=" * 70)