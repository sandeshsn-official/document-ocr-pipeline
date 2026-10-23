import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FINAL_VERIFIED_DIR = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "final_verified"
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def normalize_text(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_document_name(value):
    name = normalize_text(value)

    if name.lower().endswith(".json"):
        name = name[:-5]

    return name


def document_file_path(document_name):
    name = normalize_document_name(document_name)

    if not name:
        return None

    path = FINAL_VERIFIED_DIR / f"{name}.json"

    if path.exists():
        return path

    return None


def get_pages(final_data):
    """
    Return pages from one per-PDF final JSON.
    """
    if not isinstance(final_data, dict):
        return []

    pages = final_data.get("pages")

    if isinstance(pages, list):
        return pages

    # Legacy compatibility only.
    documents = final_data.get("documents")

    if isinstance(documents, list):
        for document in documents:
            if isinstance(document, dict):
                pages = document.get("pages", [])

                if isinstance(pages, list):
                    return pages

    return []


def int_value(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def coordinates(decision):
    return (
        int_value(decision.get("page", 0)),
        int_value(decision.get("table", 0)),
        int_value(decision.get("row", 0)),
        int_value(decision.get("column", 0))
    )


def same_document(decision, document_name):
    return (
        normalize_document_name(
            decision.get("document", "")
        )
        ==
        normalize_document_name(document_name)
    )


def evidence_key(decision):
    """
    IMPORTANT:
    Duplicate review records are grouped using the actual OCR
    evidence and stored location only.

    object_type is deliberately NOT part of this key.

    This prevents duplicate records for the same reviewed item
    from appearing as separate correction choices.
    """
    page, table, row, column = coordinates(
        decision
    )

    return (
        page,
        table,
        row,
        column,
        normalize_text(
            decision.get("original_text", "")
        ),
        normalize_text(
            decision.get("source_text", "")
        )
    )


def group_document_decisions(
    decisions,
    document_name
):
    """
    Group duplicate records representing the same evidence.
    """
    groups = {}

    for decision in decisions:

        if not same_document(
            decision,
            document_name
        ):
            continue

        key = evidence_key(
            decision
        )

        groups.setdefault(
            key,
            []
        ).append(decision)

    return list(groups.values())


def collect_current_unresolved(final_data, document_name):
    """
    Collect cells that are still marked as unresolved REVIEW items in the
    current final JSON.

    This is deliberately generic: it uses only the stored verification
    status and table coordinates. It never uses document names, page numbers,
    expected values, or hard-coded OCR corrections.
    """
    unresolved = []

    for page_data in get_pages(final_data):
        page_number = int_value(page_data.get("page", 0))

        for table_index, table_data in enumerate(page_data.get("tables", []), start=1):
            for row_index, row_data in enumerate(table_data.get("rows", []), start=1):
                for column_index, cell in enumerate(row_data.get("cells", []), start=1):
                    final_status = normalize_text(cell.get("final_verification_status", ""))
                    verification_status = normalize_text(cell.get("verification_status", ""))
                    status = final_status or verification_status

                    # The specific safety case handled here is a recovered OCR
                    # candidate that was written into final_verified_text while
                    # still carrying SOURCE_RECOVERED_REVIEW. Such a value is
                    # evidence, not verified data, and must be surfaced for
                    # human confirmation/correction. Other review classes are
                    # left to the normal Stage 8 workflow.
                    if "SOURCE_RECOVERED_REVIEW" not in status:
                        continue

                    original_text = normalize_text(cell.get("text", ""))
                    source_text = normalize_text(cell.get("source_ocr", ""))
                    current_value = normalize_text(cell.get("final_verified_text", ""))

                    # Empty recovered candidates are still unresolved and
                    # must remain visible to a human. Stage 10 is a fallback
                    # safety net; Stage 08 should normally receive these via
                    # the review queue first.
                    unresolved.append({
                        "document": document_name,
                        "page": page_number,
                        "table": table_index,
                        "row": row_index,
                        "column": column_index,
                        "original_text": original_text,
                        "source_text": source_text,
                        "verified_text": current_value,
                        "decision": status,
                        "bbox": cell.get("bbox"),
                        "_current_unresolved": True,
                    })

    return unresolved


def merge_current_unresolved(groups, current_items):
    """Merge current unresolved cells into the historical evidence groups."""
    by_key = {}
    for group in groups:
        if group:
            by_key[evidence_key(group[0])] = group

    for item in current_items:
        key = evidence_key(item)
        if key in by_key:
            by_key[key].append(item)
        else:
            group = [item]
            groups.append(group)
            by_key[key] = group

    # Current unresolved items are intentionally shown first so they cannot be
    # mistaken for merely historical decisions.
    current_groups = [g for g in groups if any(x.get("_current_unresolved") for x in g)]
    historical_groups = [g for g in groups if not any(x.get("_current_unresolved") for x in g)]
    return current_groups + historical_groups


def effective_decision(group):
    """
    Select the authoritative existing decision for display/update.

    Preference:
      1. USER_CORRECTION
      2. USER_ENTERED_CORRECTION
      3. USER_CONFIRMED_ORIGINAL
      4. USER_CONFIRMED_SOURCE
      5. another non-empty decision
      6. empty decision

    Historical duplicate records are preserved.
    """
    priorities = {
        "USER_CORRECTION": 5,
        "USER_ENTERED_CORRECTION": 4,
        "USER_CONFIRMED_ORIGINAL": 3,
        "USER_CONFIRMED_SOURCE": 3
    }

    best = None
    best_score = -1

    for decision in group:

        decision_type = normalize_text(
            decision.get("decision", "")
        )

        # A current unresolved record represents the state that exists in
        # final_verified/*.json NOW. It must outrank an older historical
        # correction at the same coordinates/evidence group; otherwise Stage
        # 10 can display a stale historical decision instead of the unresolved
        # current candidate that still needs attention.
        if decision.get("_current_unresolved"):
            score = 100
        else:
            score = priorities.get(
                decision_type,
                1 if decision_type else 0
            )

        if normalize_text(
            decision.get("verified_text", "")
        ):
            score += 1

        if best is None or score > best_score:
            best = decision
            best_score = score

    return best if best is not None else group[0]


def find_cell(
    final_data,
    page_number,
    table_number,
    row_number,
    column_number
):
    """
    Locate one table cell in one per-PDF JSON.
    """
    for page in get_pages(final_data):

        if int_value(
            page.get("page", 0)
        ) != page_number:
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


def cell_original_values(cell):
    return {
        normalize_text(
            cell.get("original_ocr", "")
        ),
        normalize_text(
            cell.get("original_text", "")
        ),
        normalize_text(
            cell.get("ocr_text", "")
        ),
        normalize_text(
            cell.get("text", "")
        )
    }


def cell_source_values(cell):
    return {
        normalize_text(
            cell.get("source_ocr", "")
        ),
        normalize_text(
            cell.get("source_text", "")
        )
    }


def find_cell_by_evidence(
    final_data,
    decision
):
    """
    Locate the selected cell.

    First:
        stored coordinates + OCR evidence

    Second:
        stored coordinates

    Third:
        unique OCR evidence match anywhere in this PDF.
    """
    page, table, row, column = coordinates(
        decision
    )

    target_original = normalize_text(
        decision.get(
            "original_text",
            ""
        )
    )

    target_source = normalize_text(
        decision.get(
            "source_text",
            ""
        )
    )

    # --------------------------------------------------------
    # 1. Coordinates + evidence
    # --------------------------------------------------------

    cell = find_cell(
        final_data,
        page,
        table,
        row,
        column
    )

    if cell is not None:

        originals = cell_original_values(
            cell
        )

        sources = cell_source_values(
            cell
        )

        original_match = (
            not target_original
            or target_original in originals
        )

        source_match = (
            not target_source
            or target_source in sources
        )

        if (
            original_match
            and source_match
        ):
            return cell

    # --------------------------------------------------------
    # 2. Coordinates fallback
    # --------------------------------------------------------

    if cell is not None:
        return cell

    # --------------------------------------------------------
    # 3. Unique evidence match
    # --------------------------------------------------------

    matches = []

    for page_data in get_pages(
        final_data
    ):

        for table_data in page_data.get(
            "tables",
            []
        ):

            for row_data in table_data.get(
                "rows",
                []
            ):

                for candidate in row_data.get(
                    "cells",
                    []
                ):

                    originals = cell_original_values(
                        candidate
                    )

                    sources = cell_source_values(
                        candidate
                    )

                    original_match = (
                        not target_original
                        or target_original in originals
                    )

                    source_match = (
                        not target_source
                        or target_source in sources
                    )

                    if (
                        original_match
                        and source_match
                    ):
                        matches.append(
                            candidate
                        )

    if len(matches) == 1:
        return matches[0]

    return None


def update_authoritative_decision(
    decision,
    new_value,
    old_value,
    old_decision
):
    """
    Update ONLY the authoritative decision record.

    Other duplicate records remain untouched as historical
    audit evidence.
    """
    decision[
        "verified_text"
    ] = new_value

    decision[
        "decision"
    ] = "USER_CORRECTION"

    decision[
        "corrected_from"
    ] = old_value

    decision[
        "previous_decision"
    ] = old_decision


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CORRECT PREVIOUS OCR REVIEW")
    print("PER-PDF EVIDENCE-BASED CORRECTION")
    print("TRUE DUPLICATE-SAFE SELECTION")
    print("=" * 70)

    if not FINAL_VERIFIED_DIR.exists():
        raise FileNotFoundError(
            "Per-PDF final verified directory not found:\n"
            f"{FINAL_VERIFIED_DIR}"
        )

    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(
            "Manual decision log not found:\n"
            f"{DECISIONS_FILE}"
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
        print(
            "No previous decisions found."
        )
        return

    print()
    print(
        f"Previous decision records: "
        f"{len(decisions)}"
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    print()

    document_name = normalize_document_name(
        input("Document: ").strip()
    )

    final_file = document_file_path(
        document_name
    )

    if final_file is None:

        print()
        print(
            "ERROR: Per-PDF final verified JSON "
            "was not found."
        )

        print()
        print(
            "Expected:"
        )

        print(
            FINAL_VERIFIED_DIR
            / f"{document_name}.json"
        )

        return

    print()
    print(
        "Final verified JSON:"
    )

    print(final_file)

    # --------------------------------------------------------
    # LOAD CURRENT FINAL JSON
    # --------------------------------------------------------

    final_data = load_json(
        final_file
    )

    # --------------------------------------------------------
    # GROUP HISTORICAL + CURRENT UNRESOLVED EVIDENCE
    # --------------------------------------------------------

    groups = group_document_decisions(
        decisions,
        document_name
    )

    current_unresolved = collect_current_unresolved(
        final_data,
        document_name
    )

    groups = merge_current_unresolved(
        groups,
        current_unresolved
    )

    if not groups:

        print()
        print(
            "No previous decisions exist "
            f"for document: {document_name}"
        )

        return

    print()
    print(
        f"Unique evidence items for "
        f"{document_name}: {len(groups)}"
    )

    print()
    print(
        "Duplicate records are grouped only "
        "for selection. No history is deleted."
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    for index, group in enumerate(
        groups,
        start=1
    ):

        representative = effective_decision(
            group
        )

        page, table, row, column = coordinates(
            representative
        )

        print()
        print(
            f"[{index}] "
            f"Page={page} "
            f"Table={table} "
            f"Row={row} "
            f"Column={column}"
        )

        print(
            f"    Original OCR : "
            f"{representative.get('original_text', '')}"
        )

        print(
            f"    Source OCR   : "
            f"{representative.get('source_text', '')}"
        )

        print(
            f"    Current value: "
            f"{normalize_text(representative.get('verified_text', '')) or '(empty)'}"
        )

        print(
            f"    Decision     : "
            f"{normalize_text(representative.get('decision', '')) or '(empty)'}"
        )

        if len(group) > 1:
            print(
                f"    History records grouped: "
                f"{len(group)}"
            )

    # --------------------------------------------------------
    # SELECTION
    # --------------------------------------------------------

    print()

    try:
        selection = int(
            input(
                "Decision number: "
            ).strip()
        )

    except ValueError:

        print()
        print(
            "ERROR: Decision number must be "
            "a number."
        )

        return

    if not (
        1 <= selection <= len(groups)
    ):

        print()
        print(
            "ERROR: Invalid decision number."
        )

        return

    selected_group = groups[
        selection - 1
    ]

    target_decision = effective_decision(
        selected_group
    )

    # --------------------------------------------------------
    # LOCATE SELECTED CELL IN CURRENT JSON
    # --------------------------------------------------------

    cell = find_cell_by_evidence(
        final_data,
        target_decision
    )

    if cell is None:

        print()
        print(
            "ERROR: Could not safely locate "
            "the selected evidence in the "
            "per-PDF final verified JSON."
        )

        print()
        print(
            "The correction was NOT saved."
        )

        return

    # --------------------------------------------------------
    # SHOW SELECTED EVIDENCE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTED EVIDENCE")
    print("=" * 70)

    page, table, row, column = coordinates(
        target_decision
    )

    print()
    print(
        f"Document: {document_name}"
    )

    print(
        f"Page: {page}"
    )

    print(
        f"Table: {table}"
    )

    print(
        f"Row: {row}"
    )

    print(
        f"Column: {column}"
    )

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
        f"{normalize_text(target_decision.get('verified_text', '')) or '(empty)'}"
    )

    print(
        f"Decision     : "
        f"{normalize_text(target_decision.get('decision', '')) or '(empty)'}"
    )

    print(
        f"History records in group: "
        f"{len(selected_group)}"
    )

    print()

    # --------------------------------------------------------
    # NEW VALUE
    # --------------------------------------------------------

    new_value = normalize_text(
        input(
            "Enter the CORRECT final value exactly: "
        )
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
    # CURRENT-UNRESOLVED ITEM → ADD TO AUDIT HISTORY
    # --------------------------------------------------------

    if target_decision.get("_current_unresolved"):
        target_decision.pop("_current_unresolved", None)
        decisions.append(target_decision)

    # --------------------------------------------------------
    # UPDATE ONLY AUTHORITATIVE RECORD
    # --------------------------------------------------------

    update_authoritative_decision(
        target_decision,
        new_value,
        old_value,
        old_decision
    )

    # --------------------------------------------------------
    # UPDATE PER-PDF FINAL JSON
    # --------------------------------------------------------

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
            "total_items": len(decisions),
            "decisions": decisions
        }
    )

    save_json(
        final_file,
        final_data
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CORRECTION SAVED")
    print("=" * 70)

    print()
    print(
        f"Document        : {document_name}"
    )

    print(
        f"Previous value  : {old_value}"
    )

    print(
        f"Corrected value : {new_value}"
    )

    print(
        f"History records preserved: "
        f"{len(selected_group)}"
    )

    print()
    print("Updated:")

    print(
        DECISIONS_FILE
    )

    print(
        final_file
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Run Step 09 afterward "
        "to rebuild learning memory."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
