import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FINAL_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "all_documents_final_verified.json"
)

DECISIONS_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "manual_review_decisions.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def normalize_text(text):
    if text is None:
        return ""

    return str(text).strip()


def find_cell(
    final_data,
    document_name,
    page_number,
    table_number,
    row_number,
    column_number
):
    for document in final_data.get(
        "documents",
        []
    ):

        current_name = str(
            document.get(
                "document",
                document.get(
                    "name",
                    ""
                )
            )
        )

        if current_name != document_name:
            continue

        for page in document.get(
            "pages",
            []
        ):

            try:
                current_page = int(
                    page.get(
                        "page",
                        0
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                current_page = 0

            if current_page != page_number:
                continue

            tables = page.get(
                "tables",
                []
            )

            table_index = table_number - 1

            if not (
                0 <= table_index < len(tables)
            ):
                return None

            rows = tables[
                table_index
            ].get(
                "rows",
                []
            )

            row_index = row_number - 1

            if not (
                0 <= row_index < len(rows)
            ):
                return None

            cells = rows[
                row_index
            ].get(
                "cells",
                []
            )

            column_index = column_number - 1

            if not (
                0 <= column_index < len(cells)
            ):
                return None

            return cells[
                column_index
            ]

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CORRECT PREVIOUS OCR REVIEW")
    print("=" * 70)

    if not FINAL_FILE.exists():
        raise FileNotFoundError(
            f"Final verified data not found:\n"
            f"{FINAL_FILE}"
        )

    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(
            f"Manual decision log not found:\n"
            f"{DECISIONS_FILE}"
        )

    final_data = load_json(
        FINAL_FILE
    )

    decision_data = load_json(
        DECISIONS_FILE
    )

    decisions = decision_data.get(
        "decisions",
        []
    )

    if not decisions:
        print()
        print("No previous decisions found.")
        return

    print()
    print(
        f"Previous decisions: "
        f"{len(decisions)}"
    )

    print()
    print(
        "Enter the exact location of the "
        "decision you want to correct."
    )

    print()

    document_name = input(
        "Document: "
    ).strip()

    try:
        page_number = int(
            input("Page: ").strip()
        )

        table_number = int(
            input("Table: ").strip()
        )

        row_number = int(
            input("Row: ").strip()
        )

        column_number = int(
            input("Column: ").strip()
        )

    except ValueError:
        print()
        print(
            "ERROR: Page, table, row, and "
            "column must be numbers."
        )
        return

    target_decision = None

    for decision in decisions:

        try:
            same_location = (
                str(
                    decision.get(
                        "document",
                        ""
                    )
                ) == document_name
                and
                int(
                    decision.get(
                        "page",
                        0
                    )
                ) == page_number
                and
                int(
                    decision.get(
                        "table",
                        0
                    )
                ) == table_number
                and
                int(
                    decision.get(
                        "row",
                        0
                    )
                ) == row_number
                and
                int(
                    decision.get(
                        "column",
                        0
                    )
                ) == column_number
            )

        except (
            TypeError,
            ValueError
        ):
            same_location = False

        if same_location:
            target_decision = decision
            break

    if target_decision is None:

        print()
        print(
            "No previous decision exists "
            "at that location."
        )
        return

    print()
    print("=" * 70)
    print("CURRENT DECISION")
    print("=" * 70)

    print()
    print(
        f"Original OCR : "
        f"{target_decision.get('original_text', '')}"
    )

    print(
        f"Source OCR   : "
        f"{target_decision.get('source_text', '')}"
    )

    print(
        f"Current value: "
        f"{target_decision.get('verified_text', '')}"
    )

    print(
        f"Decision     : "
        f"{target_decision.get('decision', '')}"
    )

    print()

    new_value = input(
        "Enter the CORRECT final value exactly: "
    )

    new_value = normalize_text(
        new_value
    )

    if not new_value:
        print()
        print(
            "ERROR: Correct value cannot be empty."
        )
        return

    old_value = target_decision.get(
        "verified_text",
        ""
    )

    old_decision = target_decision.get(
        "decision",
        ""
    )

    # --------------------------------------------------------
    # UPDATE DECISION HISTORY
    # --------------------------------------------------------

    target_decision[
        "verified_text"
    ] = new_value

    target_decision[
        "decision"
    ] = "USER_CORRECTION"

    target_decision[
        "corrected_from"
    ] = old_value

    target_decision[
        "previous_decision"
    ] = old_decision

    # --------------------------------------------------------
    # UPDATE FINAL CELL
    # --------------------------------------------------------

    cell = find_cell(
        final_data,
        document_name,
        page_number,
        table_number,
        row_number,
        column_number
    )

    if cell is None:

        print()
        print(
            "ERROR: Could not locate the "
            "cell in final verified data."
        )
        return

    cell[
        "final_verified_text"
    ] = new_value

    cell[
        "final_verification_status"
    ] = "USER_CORRECTION"

    cell[
        "manual_review_original_ocr"
    ] = target_decision.get(
        "original_text",
        ""
    )

    cell[
        "manual_review_source_ocr"
    ] = target_decision.get(
        "source_text",
        ""
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_json(
        DECISIONS_FILE,
        {
            "total_items":
                len(decisions),
            "decisions":
                decisions,
        }
    )

    save_json(
        FINAL_FILE,
        final_data
    )

    print()
    print("=" * 70)
    print("CORRECTION SAVED")
    print("=" * 70)

    print()
    print(
        f"Previous value : {old_value}"
    )

    print(
        f"Corrected value: {new_value}"
    )

    print()
    print(
        "Updated:"
    )

    print(
        DECISIONS_FILE
    )

    print(
        FINAL_FILE
    )

    print()
    print(
        "IMPORTANT: Run Step 09 afterward "
        "to rebuild learning memory."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()