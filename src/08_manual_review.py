import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__).resolve().parent.parent
)

VERIFIED_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "all_documents_verified.json"
)

REVIEW_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "review_required.json"
)

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

LEARNING_MEMORY_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "ocr_learning_memory.json"
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


def save_json(
    path,
    data
):

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


def normalize_choice(choice):

    return str(
        choice
    ).strip().lower()


def normalize_text(text):

    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def repair_mojibake(text):

    if not text:
        return ""

    replacements = {

        "Î©": "Ω",
        "Ï‰": "ω",

        "Â±": "±",

        "â‰¥": "≥",
        "â‰¤": "≤",

        "Î”": "Δ",
        "âˆ†": "∆",

        "Î¼": "μ",
        "Âµ": "µ",

        "Â°": "°",

        "â†’": "→",
        "â†”": "↔",

        "â‰ˆ": "≈",
        "â‰…": "≅",

        "Ã—": "×",
        "Ã·": "÷",

        "âˆž": "∞",
        "âˆ‘": "∑",
        "âˆš": "√",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return text


def normalize_for_learning(text):

    text = normalize_text(
        text
    )

    text = repair_mojibake(
        text
    )

    text = text.replace(
        "μm",
        "µm"
    )

    text = text.replace(
        "um",
        "µm"
    )

    text = text.replace(
        "°℃",
        "°C"
    )

    text = text.replace(
        "℃",
        "°C"
    )

    for dash in [
        "–",
        "—",
        "−",
        "‒",
        "﹘",
        "﹣",
        "－",
    ]:

        text = text.replace(
            dash,
            "-"
        )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

def normalize_for_exact_case(text):
    """
    Normalize text only for duplicate-review comparison.

    Actual OCR/final values are never modified.
    This comparison removes whitespace differences while
    preserving every non-whitespace character and its order.
    """
    text = normalize_for_learning(text)

    return re.sub(
        r"\s+",
        "",
        text
    )


# ============================================================
# FIND CELL
# ============================================================

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

        if current_name != str(
            document_name
        ):
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

            if current_page != int(
                page_number
            ):
                continue

            tables = page.get(
                "tables",
                []
            )

            table_index = (
                int(table_number) - 1
            )

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

            row_index = (
                int(row_number) - 1
            )

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

            column_index = (
                int(column_number) - 1
            )

            if not (
                0 <= column_index < len(cells)
            ):

                return None

            return cells[
                column_index
            ]

    return None


# ============================================================
# DISPLAY
# ============================================================

def display_review_item(
    index,
    total,
    item
):

    print()
    print("=" * 75)

    print(
        f"REVIEW ITEM "
        f"{index} OF {total}"
    )

    print("=" * 75)

    print(
        f"Document : "
        f"{item.get('document', '')}"
    )

    print(
        f"Page     : "
        f"{item.get('page', '')}"
    )

    print(
        f"Table    : "
        f"{item.get('table', '')}"
    )

    print(
        f"Row      : "
        f"{item.get('row', '')}"
    )

    print(
        f"Column   : "
        f"{item.get('column', '')}"
    )

    print()

    print(
        f"Original OCR : "
        f"{item.get('original_text', '')}"
    )

    print(
        f"Source OCR   : "
        f"{item.get('source_text', '')}"
    )

    verification = item.get(
        "verification",
        {}
    )

    print()

    print(
        f"Status       : "
        f"{verification.get('decision', '')}"
    )

    print(
        f"Reason       : "
        f"{verification.get('reason', '')}"
    )

    crop = item.get(
        "review_crop",
        ""
    )

    if crop:

        print()

        print(
            f"Image crop   : "
            f"{crop}"
        )

    print()


# ============================================================
# LEARNING-PATTERN DETECTION
# ============================================================

LEARNABLE_SYMBOLS = {
    "Ω",
    "ω",
    "±",
    "∓",
    "≥",
    "≤",
    "Δ",
    "∆",
    "μ",
    "µ",
    "°",
    "→",
    "↔",
    "≠",
    "≈",
    "≅",
    "×",
    "÷",
    "∞",
    "∑",
    "√",
    "λ",
    "θ",
    "φ",
    "σ",
    "τ",
    "α",
    "β",
    "γ",
    "δ",
    "ε",
    "η",
    "κ",
    "π",
    "χ",
    "ψ",
    "ξ",
    "ζ",
}


def extract_symbol_confusion(
    original,
    source,
    verified
):
    """
    Extract one technical-symbol confusion from a review item.

    Conditions:

    - original and source have equal length
    - exactly one character differs
    - verified equals the original OCR at that position
    - the differing character is a technical symbol
    """

    original = normalize_for_learning(
        original
    )

    source = normalize_for_learning(
        source
    )

    verified = normalize_for_learning(
        verified
    )

    if not original or not source or not verified:
        return None

    if not (
        len(original)
        ==
        len(source)
        ==
        len(verified)
    ):
        return None

    differences = []

    for index in range(
        len(original)
    ):

        if (
            original[index]
            !=
            source[index]
        ):

            differences.append(
                index
            )

    if len(differences) != 1:
        return None

    index = differences[0]

    original_char = original[
        index
    ]

    source_char = source[
        index
    ]

    verified_char = verified[
        index
    ]

    # Human must confirm the original character.
    if verified_char != original_char:
        return None

    # Do not learn identifier O/0/I/1 confusion here.
    if not (
        original_char in LEARNABLE_SYMBOLS
        or
        source_char in LEARNABLE_SYMBOLS
    ):
        return None

    return {
        "original_character":
            original_char,

        "source_character":
            source_char,

        "verified_character":
            verified_char,
    }


def make_learning_key(
    original,
    source,
    verified
):

    diff = extract_symbol_confusion(
        original,
        source,
        verified
    )

    if diff is None:
        return None

    return (
        diff["original_character"],
        diff["source_character"],
        diff["verified_character"],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("MANUAL OCR REVIEW")
    print("=" * 75)

    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    if not VERIFIED_FILE.exists():

        raise FileNotFoundError(
            f"Verified data not found:\n"
            f"{VERIFIED_FILE}"
        )

    if not REVIEW_FILE.exists():

        raise FileNotFoundError(
            f"Review file not found:\n"
            f"{REVIEW_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    final_data = load_json(
        VERIFIED_FILE
    )

    review_data = load_json(
        REVIEW_FILE
    )

    review_items = review_data.get(
        "items",
        []
    )

    # --------------------------------------------------------
    # NO REVIEW ITEMS
    # --------------------------------------------------------

    if not review_items:

        print()

        print(
            "No items require manual review."
        )

        save_json(
            FINAL_FILE,
            final_data
        )

        if not DECISIONS_FILE.exists():

            save_json(
                DECISIONS_FILE,
                {
                    "total_items": 0,
                    "decisions": []
                }
            )

        print()

        print(
            f"Saved final data: "
            f"{FINAL_FILE}"
        )

        return

    # ========================================================
    # LOAD EXISTING DECISIONS
    # ========================================================

    existing_decisions = []

    if DECISIONS_FILE.exists():

        try:

            existing_data = load_json(
                DECISIONS_FILE
            )

            existing_decisions = (
                existing_data.get(
                    "decisions",
                    []
                )
            )

            if not isinstance(
                existing_decisions,
                list
            ):

                existing_decisions = []

        except Exception:

            existing_decisions = []

    existing_lookup = {}

    for decision in existing_decisions:

        try:

            key = (
                str(
                    decision.get(
                        "document",
                        ""
                    )
                ),

                int(
                    decision.get(
                        "page",
                        0
                    )
                ),

                int(
                    decision.get(
                        "table",
                        0
                    )
                ),

                int(
                    decision.get(
                        "row",
                        0
                    )
                ),

                int(
                    decision.get(
                        "column",
                        0
                    )
                ),
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        existing_lookup[
            key
        ] = decision

    # ========================================================
    # EXACT OCR-CASE MEMORY
    # ========================================================

    exact_case_patterns = {}

    trusted_human_decisions = {
        "USER_CONFIRMED_ORIGINAL",
        "USER_CONFIRMED_SOURCE",
        "USER_ENTERED_CORRECTION",
    }

    for decision in existing_decisions:
        if decision.get("decision") not in trusted_human_decisions:
            continue

        original_key = normalize_for_exact_case(
            decision.get("original_text", "")
        )

        source_key = normalize_for_exact_case(
            decision.get("source_text", "")
        )

        verified_value = decision.get(
            "verified_text",
            ""
        )

        if original_key and source_key and verified_value:
            exact_case_key = (
                original_key,
                source_key,
            )

            exact_case_patterns[
                exact_case_key
            ] = verified_value

    # ========================================================
    # CURRENT-SESSION LEARNED SYMBOLS
    # ========================================================

    session_symbol_patterns = {}
	
    # ========================================================
    # CURRENT-SESSION LEARNED SYMBOLS
    # ========================================================
    #
    # This allows:
    #
    # Case 13 -> human verifies Ω
    # Case 18 -> same Ω/Q confusion
    #             -> automatically skipped
    #
    # Only technical symbols are supported.
    # Numeric values and identifier ambiguities remain manual.
    # ========================================================

    session_symbol_patterns = {}

    for decision in existing_decisions:

        symbol_key = make_learning_key(
            decision.get(
                "original_text",
                ""
            ),
            decision.get(
                "source_text",
                ""
            ),
            decision.get(
                "verified_text",
                ""
            )
        )

        if symbol_key:

            session_symbol_patterns[
                symbol_key
            ] = decision.get(
                "verified_text",
                ""
            )

    # --------------------------------------------------------
    # EXISTING PERMANENT MEMORY
    # --------------------------------------------------------

    permanent_symbol_patterns = {}

    if LEARNING_MEMORY_FILE.exists():

        try:

            memory = load_json(
                LEARNING_MEMORY_FILE
            )

            for pattern in memory.get(
                "reusable_symbol_patterns",
                []
            ):

                try:

                    repeat_count = int(
                        pattern.get(
                            "repeat_count",
                            0
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    repeat_count = 0

                if repeat_count < 2:
                    continue

                key = (
                    normalize_for_learning(
                        pattern.get(
                            "original_character",
                            ""
                        )
                    ),

                    normalize_for_learning(
                        pattern.get(
                            "source_character",
                            ""
                        )
                    ),

                    normalize_for_learning(
                        pattern.get(
                            "verified_character",
                            ""
                        )
                    ),
                )

                if all(key):

                    permanent_symbol_patterns[
                        key
                    ] = True

        except Exception:

            permanent_symbol_patterns = {}

    # ========================================================
    # PROCESS REVIEW ITEMS
    # ========================================================

    total = len(
        review_items
    )

    auto_skipped = 0

    newly_learned_in_session = 0

    for index, item in enumerate(
        review_items,
        start=1
    ):

        document_name = str(
            item.get(
                "document",
                ""
            )
        )

        try:

            page_number = int(
                item.get(
                    "page",
                    0
                )
            )

            table_number = int(
                item.get(
                    "table",
                    0
                )
            )

            row_number = int(
                item.get(
                    "row",
                    0
                )
            )

            column_number = int(
                item.get(
                    "column",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        key = (
            document_name,
            page_number,
            table_number,
            row_number,
            column_number,
        )

        # ----------------------------------------------------
        # ALREADY REVIEWED
        # ----------------------------------------------------

        if key in existing_lookup:

            previous = existing_lookup[key]

            previous_value = previous.get(
                "verified_text",
                ""
            )

            print()

            print(
                f"Item {index}/{total} already reviewed:"
            )

            print(
                f"  Decision: "
                f"{previous.get('decision', '')}"
            )

            print(
                f"  Value   : "
                f"{previous_value}"
            )

            # Re-apply the previous decision to the
            # freshly regenerated Step 06 data.
            cell = find_cell(
                final_data,
                document_name,
                page_number,
                table_number,
                row_number,
                column_number
            )

            if cell is not None and previous_value:

                cell[
                    "final_verified_text"
                ] = previous_value

                cell[
                    "final_verification_status"
                ] = previous.get(
                    "decision",
                    ""
                )

                cell[
                    "manual_review_original_ocr"
                ] = previous.get(
                    "original_text",
                    ""
                )

                cell[
                    "manual_review_source_ocr"
                ] = previous.get(
                    "source_text",
                    ""
                )

            save_json(
                FINAL_FILE,
                final_data
            )

            print(
                "  Previous verified value reapplied."
            )

            print(
                "  Skipping..."
            )

            continue

        original = str(
            item.get(
                "original_text",
                ""
            )
        )

        source = str(
            item.get(
                "source_text",
                ""
            )
        )

        # ----------------------------------------------------
        # CHECK EXACT PREVIOUSLY VERIFIED OCR CASE
        # ----------------------------------------------------

        exact_case_key = (
    normalize_for_exact_case(original),
    normalize_for_exact_case(source),
)
        if exact_case_key in exact_case_patterns:

            learned_value = exact_case_patterns[
                exact_case_key
            ]

            print()

            print("=" * 75)

            print(
                f"REVIEW ITEM {index} OF {total}"
            )

            print("=" * 75)

            print(
                "AUTOMATICALLY SKIPPED "
                "USING EXACT PREVIOUSLY "
                "VERIFIED OCR CASE"
            )

            print()

            print(
                f"Original OCR : {original}"
            )

            print(
                f"Source OCR   : {source}"
            )

            print(
                f"Final value  : {learned_value}"
            )

            decision_record = {

                "document":
                    document_name,

                "page":
                    page_number,

                "table":
                    table_number,

                "row":
                    row_number,

                "column":
                    column_number,

                "original_text":
                    original,

                "source_text":
                    source,

                "verified_text":
                    learned_value,

                "decision":
                    "EXACT_CASE_REUSED",

                "previous_status":
                    item.get(
                        "verification",
                        {}
                    ).get(
                        "decision",
                        ""
                    ),

                "review_crop":
                    item.get(
                        "review_crop",
                        ""
                    ),

                "learning_pattern":
                    {
                        "type":
                            "EXACT_OCR_CASE",

                        "original_text":
                            normalize_for_learning(
                                original
                            ),

                        "source_text":
                            normalize_for_learning(
                                source
                            ),
                    },
            }

            existing_decisions.append(
                decision_record
            )

            existing_lookup[
                key
            ] = decision_record

            cell = find_cell(
                final_data,
                document_name,
                page_number,
                table_number,
                row_number,
                column_number
            )

            if cell is not None:

                cell[
                    "final_verified_text"
                ] = learned_value

                cell[
                    "final_verification_status"
                ] = "EXACT_CASE_REUSED"

                cell[
                    "manual_review_original_ocr"
                ] = original

                cell[
                    "manual_review_source_ocr"
                ] = source

            save_json(
                DECISIONS_FILE,
                {
                    "total_items":
                        len(existing_decisions),

                    "decisions":
                        existing_decisions,
                }
            )

            save_json(
                FINAL_FILE,
                final_data
            )

            auto_skipped += 1

            continue

        # ----------------------------------------------------
        # CHECK CURRENT-SESSION SYMBOL LEARNING
        # ----------------------------------------------------

        # ----------------------------------------------------
        # CHECK CURRENT-SESSION SYMBOL LEARNING
        # ----------------------------------------------------

        session_key = None

        temp_verified_original = original

        symbol_diff = extract_symbol_confusion(
            original,
            source,
            original
        )

        if symbol_diff:

            session_key = (
                symbol_diff[
                    "original_character"
                ],

                symbol_diff[
                    "source_character"
                ],

                symbol_diff[
                    "verified_character"
                ],
            )

        if (
            session_key
            and
            (
                session_key
                in session_symbol_patterns
            )
        ):

            learned_value = (
                session_symbol_patterns[
                    session_key
                ]
            )

            print()

            print("=" * 75)

            print(
                f"REVIEW ITEM {index} OF {total}"
            )

            print("=" * 75)

            print(
                "AUTOMATICALLY SKIPPED "
                "USING CURRENT-SESSION "
                "VERIFIED SYMBOL PATTERN"
            )

            print()

            print(
                f"Pattern: "
                f"{session_key[0]} "
                f"-> "
                f"{session_key[1]} "
                f"-> "
                f"{session_key[2]}"
            )

            print(
                f"Value: "
                f"{original}"
            )

            print(
                f"Verified using previous "
                f"manual decision: "
                f"{learned_value}"
            )

            # We preserve a separate audit record.
            decision_record = {

                "document":
                    document_name,

                "page":
                    page_number,

                "table":
                    table_number,

                "row":
                    row_number,

                "column":
                    column_number,

                "original_text":
                    original,

                "source_text":
                    source,

                "verified_text":
                    learned_value,

                "decision":
                    "SESSION_LEARNED_SYMBOL",

                "previous_status":
                    item.get(
                        "verification",
                        {}
                    ).get(
                        "decision",
                        ""
                    ),

                "review_crop":
                    item.get(
                        "review_crop",
                        ""
                    ),

                "learning_pattern":
                    {
                        "original_character":
                            session_key[0],

                        "source_character":
                            session_key[1],

                        "verified_character":
                            session_key[2],
                    },
            }

            existing_decisions.append(
                decision_record
            )

            existing_lookup[
                key
            ] = decision_record

            # Update the actual cell.
            cell = find_cell(
                final_data,
                document_name,
                page_number,
                table_number,
                row_number,
                column_number
            )

            if cell is not None:

                cell[
                    "final_verified_text"
                ] = learned_value

                cell[
                    "final_verification_status"
                ] = (
                    "SESSION_LEARNED_SYMBOL"
                )

                cell[
                    "manual_review_original_ocr"
                ] = original

                cell[
                    "manual_review_source_ocr"
                ] = source

            save_json(
                DECISIONS_FILE,
                {
                    "total_items":
                        len(existing_decisions),

                    "decisions":
                        existing_decisions,
                }
            )

            save_json(
                FINAL_FILE,
                final_data
            )

            auto_skipped += 1

            continue

        # ----------------------------------------------------
        # DISPLAY NORMAL REVIEW
        # ----------------------------------------------------

        display_review_item(
            index,
            total,
            item
        )

        # ----------------------------------------------------
        # ASK USER
        # ----------------------------------------------------

        while True:

            print(
                "[O] Confirm original OCR"
            )

            print(
                "[S] Confirm source OCR"
            )

            print(
                "[C] Enter correct value manually"
            )

            print(
                "[Q] Quit and save progress"
            )

            print()

            choice = normalize_choice(
                input(
                    "Your choice: "
                )
            )

            if choice == "o":

                verified_text = original

                decision = (
                    "USER_CONFIRMED_ORIGINAL"
                )

                break

            if choice == "s":

                verified_text = source

                decision = (
                    "USER_CONFIRMED_SOURCE"
                )

                break

            if choice == "c":

                print()

                verified_text = input(
                    "Enter the correct value exactly: "
                )

                verified_text = (
                    verified_text.strip()
                )

                if not verified_text:

                    print(
                        "Value cannot be empty."
                    )

                    continue

                decision = (
                    "USER_ENTERED_CORRECTION"
                )

                break

            if choice == "q":

                print()

                print(
                    "Stopping review."
                )

                save_json(
                    DECISIONS_FILE,
                    {
                        "total_items":
                            len(existing_decisions),

                        "decisions":
                            existing_decisions,
                    }
                )

                save_json(
                    FINAL_FILE,
                    final_data
                )

                print()

                print(
                    f"Progress saved to:\n"
                    f"{DECISIONS_FILE}"
                )

                print()

                print(
                    "Run this script again to continue."
                )

                return

            print(
                "Invalid choice. "
                "Enter O, S, C, or Q."
            )

        # ====================================================
        # FIND CELL
        # ====================================================

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
                "ERROR: Could not locate the cell "
                "in the verified data."
            )

            print(
                "The decision was NOT recorded."
            )

            continue

        # ====================================================
        # UPDATE CELL
        # ====================================================

        cell[
            "final_verified_text"
        ] = verified_text

        cell[
            "final_verification_status"
        ] = decision

        cell[
            "manual_review_original_ocr"
        ] = original

        cell[
            "manual_review_source_ocr"
        ] = source

        # ====================================================
        # DECISION RECORD
        # ====================================================

        decision_record = {

            "document":
                document_name,

            "page":
                page_number,

            "table":
                table_number,

            "row":
                row_number,

            "column":
                column_number,

            "original_text":
                original,

            "source_text":
                source,

            "verified_text":
                verified_text,

            "decision":
                decision,

            "previous_status":
                item.get(
                    "verification",
                    {}
                ).get(
                    "decision",
                    ""
                ),

            "review_crop":
                item.get(
                    "review_crop",
                    ""
                ),
        }

        existing_decisions.append(
            decision_record
        )

        existing_lookup[
            key
        ] = decision_record

        # Remember this exact OCR case immediately.
        exact_case_key = (
            normalize_for_learning(original),
            normalize_for_learning(source),
        )

        if (
            exact_case_key[0]
            and exact_case_key[1]
        ):
            exact_case_patterns[
                exact_case_key
            ] = verified_text

        # ====================================================
        # IMMEDIATE SESSION LEARNING
        # ====================================================
        #
        # Only technical-symbol confusions are learned
        # immediately during this review session.
        #
        # Numeric differences and identifiers are NOT
        # learned automatically here.
        # ====================================================

        symbol_key = make_learning_key(
            original,
            source,
            verified_text
        )

        if symbol_key:

            session_symbol_patterns[
                symbol_key
            ] = verified_text

            newly_learned_in_session += 1

            print()

            print(
                "Session learning recorded:"
            )

            print(
                f"  {symbol_key[0]} "
                f"-> "
                f"{symbol_key[1]} "
                f"-> "
                f"{symbol_key[2]}"
            )

            print(
                "  Later matching symbol cases "
                "in this review session can be "
                "skipped automatically."
            )

        # ====================================================
        # SAVE AFTER EVERY DECISION
        # ====================================================

        save_json(
            DECISIONS_FILE,
            {
                "total_items":
                    len(existing_decisions),

                "decisions":
                    existing_decisions,
            }
        )

        save_json(
            FINAL_FILE,
            final_data
        )

        print()

        print(
            f"Recorded: "
            f"{decision}"
        )

        print(
            f"Final value: "
            f"{verified_text}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    save_json(
        DECISIONS_FILE,
        {
            "total_items":
                len(existing_decisions),

            "decisions":
                existing_decisions,
        }
    )

    save_json(
        FINAL_FILE,
        final_data
    )

    print()
    print("=" * 75)
    print("MANUAL REVIEW COMPLETE")
    print("=" * 75)

    print()

    print(
        f"Review items          : "
        f"{total}"
    )

    print(
        f"Session auto-skipped  : "
        f"{auto_skipped}"
    )

    print(
        f"New session patterns  : "
        f"{newly_learned_in_session}"
    )

    print(
        f"Decisions recorded    : "
        f"{len(existing_decisions)}"
    )

    print()

    print(
        "Final verified data:"
    )

    print(
        FINAL_FILE
    )

    print()

    print(
        "Manual decision log:"
    )

    print(
        DECISIONS_FILE
    )

    print()

    print(
        "Original OCR data was preserved."
    )

    print("=" * 75)


if __name__ == "__main__":
    main()