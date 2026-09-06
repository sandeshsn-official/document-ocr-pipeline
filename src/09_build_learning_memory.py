import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DECISIONS_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "manual_review_decisions.json"
)

MEMORY_FILE = (
    BASE_DIR
    / "output"
    / "verified_data"
    / "ocr_learning_memory.json"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_REPEATS = 2


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    if text is None:
        return ""

    return " ".join(
        str(text).strip().split()
    )


def repair_common_mojibake(text):
    """
    Normalize the common broken UTF-8 representations that have
    appeared in this OCR project.
    """

    if not text:
        return ""

    replacements = {
        "Î©": "Ω",
        "Ï‰": "ω",
        "Â±": "±",
        "âˆ“": "√",
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
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


def normalize_for_learning(text):
    """
    Normalize representation without destroying the identity
    of important symbols.
    """

    text = normalize(text)

    text = repair_common_mojibake(
        text
    )

    # Common harmless unit representations.
    text = text.replace(
        "um",
        "µm"
    )

    text = text.replace(
        "μm",
        "µm"
    )

    text = text.replace(
        "℃",
        "°C"
    )

    text = text.replace(
        "°℃",
        "°C"
    )

    # Normalize dash variants.
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


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(
            f
        )


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


# ============================================================
# SYMBOL / CHARACTER DIFFERENCE EXTRACTION
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
    "π",
}


def symbol_difference(
    original,
    source,
    verified
):
    """
    Extract a simple repeated symbol confusion.

    We only learn a symbol when:
      - original and source have equal length
      - verified has equal length
      - exactly one character position differs between
        original and source
      - the verified character matches the original at
        that position
      - the changed source character is different
      - the position is a technical/ambiguous symbol OR
        a classic O/0/I/1 ambiguity
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

    original_char = original[index]
    source_char = source[index]
    verified_char = verified[index]

    # --------------------------------------------------------
    # Human verification must support the original character.
    # --------------------------------------------------------

    if verified_char != original_char:
        return None

    # --------------------------------------------------------
    # Learn only known ambiguous/special characters.
    # --------------------------------------------------------

    allowed_ambiguous = (
        LEARNABLE_SYMBOLS
        |
        {
            "O",
            "o",
            "0",
            "I",
            "l",
            "1",
            "S",
            "s",
            "5",
            "B",
            "8",
        }
    )

    if (
        original_char not in allowed_ambiguous
        and
        source_char not in allowed_ambiguous
    ):
        return None

    return {
        "original_character":
            original_char,

        "source_character":
            source_char,

        "verified_character":
            verified_char,

        "position":
            index,
    }


# ============================================================
# CONTEXT SKELETON
# ============================================================

def remove_target_character(
    text,
    index
):
    """
    Replace the learned character with a placeholder.
    """

    return (
        text[:index]
        + "#"
        + text[index + 1:]
    )


def build_context_key(
    original,
    source,
    diff
):
    """
    Build a context key around the learned character.

    The exact character is removed, but surrounding text is kept.

    Example:

        48.00 Ω  -> 48.00 #
        48.00 Q  -> 48.00 #

    Context key:

        48.00 #
    """

    index = diff["position"]

    original_context = remove_target_character(
        original,
        index
    )

    source_context = remove_target_character(
        source,
        index
    )

    if (
        original_context
        !=
        source_context
    ):
        return None

    return original_context


# ============================================================
# DECISION CLASSIFICATION
# ============================================================

def is_human_verified(decision):
    return decision in {
        "USER_CONFIRMED_ORIGINAL",
        "USER_CONFIRMED_SOURCE",
        "USER_ENTERED_CORRECTION",
        "USER_CORRECTION",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCR LEARNING MEMORY BUILDER")
    print("=" * 70)

    if not DECISIONS_FILE.exists():

        raise FileNotFoundError(
            f"Manual decision file not found:\n"
            f"{DECISIONS_FILE}"
        )

    data = load_json(
        DECISIONS_FILE
    )

    decisions = data.get(
        "decisions",
        []
    )

    print()

    print(
        f"Manual decisions found : "
        f"{len(decisions)}"
    )

    # --------------------------------------------------------
    # GROUP WHOLE-CELL PATTERNS
    # --------------------------------------------------------

    whole_cell_groups = defaultdict(list)

    # --------------------------------------------------------
    # GROUP SYMBOL PATTERNS
    # --------------------------------------------------------

    symbol_groups = defaultdict(list)

    # --------------------------------------------------------
    # INSPECT DECISIONS
    # --------------------------------------------------------

    for decision in decisions:

        if not is_human_verified(
            decision.get(
                "decision",
                ""
            )
        ):
            continue

        original = normalize_for_learning(
            decision.get(
                "original_text",
                ""
            )
        )

        source = normalize_for_learning(
            decision.get(
                "source_text",
                ""
            )
        )

        verified = normalize_for_learning(
            decision.get(
                "verified_text",
                ""
            )
        )

        if not verified:
            continue

        if not original and not source:
            continue

        # ----------------------------------------------------
        # Whole-cell pattern
        # ----------------------------------------------------

        if (
            original
            and
            source
            and
            verified
            and
            (
                original != source
                or
                original != verified
            )
        ):

            whole_key = (
                original,
                source,
                verified
            )

            whole_cell_groups[
                whole_key
            ].append(
                decision
            )

        # ----------------------------------------------------
        # Symbol-level pattern
        # ----------------------------------------------------

        diff = symbol_difference(
            original,
            source,
            verified
        )

        if diff is None:
            continue

        context_key = build_context_key(
            original,
            source,
            diff
        )

        if context_key is None:
            continue

        symbol_key = (
            diff["original_character"],
            diff["source_character"],
            diff["verified_character"],
            context_key,
        )

        symbol_groups[
            symbol_key
        ].append(
            decision
        )

    # ========================================================
    # BUILD WHOLE-CELL REUSABLE PATTERNS
    # ========================================================

    reusable_text_patterns = []

    one_off_text_patterns = []

    for key, examples in (
        whole_cell_groups.items()
    ):

        original, source, verified = key

        repeat_count = len(
            examples
        )

        record = {

            "original_ocr":
                original,

            "source_ocr":
                source,

            "verified_text":
                verified,

            "repeat_count":
                repeat_count,

            "examples": [

                {
                    "document":
                        example.get(
                            "document",
                            ""
                        ),

                    "page":
                        example.get(
                            "page",
                            0
                        ),

                    "table":
                        example.get(
                            "table",
                            0
                        ),

                    "row":
                        example.get(
                            "row",
                            0
                        ),

                    "column":
                        example.get(
                            "column",
                            0
                        ),

                    "decision":
                        example.get(
                            "decision",
                            ""
                        ),
                }

                for example in examples
            ],
        }

        if repeat_count >= MIN_REPEATS:

            record[
                "confidence"
            ] = "REPEATED_HUMAN_VERIFIED"

            reusable_text_patterns.append(
                record
            )

        else:

            record[
                "reason"
            ] = (
                "Pattern occurred fewer than "
                f"{MIN_REPEATS} times."
            )

            one_off_text_patterns.append(
                record
            )

    # ========================================================
    # BUILD SYMBOL-LEVEL REUSABLE PATTERNS
    # ========================================================

    reusable_symbol_patterns = []

    one_off_symbol_patterns = []

    for key, examples in (
        symbol_groups.items()
    ):

        (
            original_character,
            source_character,
            verified_character,
            context_key,
        ) = key

        repeat_count = len(
            examples
        )

        record = {

            "original_character":
                original_character,

            "source_character":
                source_character,

            "verified_character":
                verified_character,

            "context":
                context_key,

            "repeat_count":
                repeat_count,

            "confidence":
                (
                    "REPEATED_HUMAN_VERIFIED"
                    if repeat_count >= MIN_REPEATS
                    else
                    "ONE_OFF"
                ),

            "examples": [

                {
                    "document":
                        example.get(
                            "document",
                            ""
                        ),

                    "page":
                        example.get(
                            "page",
                            0
                        ),

                    "table":
                        example.get(
                            "table",
                            0
                        ),

                    "row":
                        example.get(
                            "row",
                            0
                        ),

                    "column":
                        example.get(
                            "column",
                            0
                        ),

                    "decision":
                        example.get(
                            "decision",
                            ""
                        ),
                }

                for example in examples
            ],
        }

        if repeat_count >= MIN_REPEATS:

            reusable_symbol_patterns.append(
                record
            )

        else:

            one_off_symbol_patterns.append(
                record
            )

    # ========================================================
    # SAVE MEMORY
    # ========================================================

    memory = {

        "description": (
            "OCR learning memory built from human-verified "
            "manual review decisions. Repeated whole-cell "
            "patterns and repeated character-level OCR "
            "confusions are stored separately. One-off "
            "patterns are retained for audit but are not "
            "eligible for automatic reuse."
        ),

        "minimum_repeats_required":
            MIN_REPEATS,

        "total_manual_decisions":
            len(decisions),

        "reusable_text_patterns":
            reusable_text_patterns,

        "reusable_symbol_patterns":
            reusable_symbol_patterns,

        "one_off_text_patterns":
            one_off_text_patterns,

        "one_off_symbol_patterns":
            one_off_symbol_patterns,
    }

    save_json(
        MEMORY_FILE,
        memory
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("LEARNING MEMORY COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Reusable whole-cell patterns : "
        f"{len(reusable_text_patterns)}"
    )

    print(
        f"Reusable symbol patterns     : "
        f"{len(reusable_symbol_patterns)}"
    )

    print(
        f"One-off whole-cell patterns  : "
        f"{len(one_off_text_patterns)}"
    )

    print(
        f"One-off symbol patterns      : "
        f"{len(one_off_symbol_patterns)}"
    )

    print()

    if reusable_symbol_patterns:

        print(
            "REUSABLE SYMBOL PATTERNS"
        )

        print()

        for pattern in reusable_symbol_patterns:

            print(
                f"  "
                f"{pattern['original_character']!r}"
                f" -> "
                f"{pattern['source_character']!r}"
                f" -> "
                f"{pattern['verified_character']!r}"
                f" | context="
                f"{pattern['context']!r}"
                f" | repeated "
                f"{pattern['repeat_count']}x"
            )

    else:

        print(
            "No repeated symbol-level patterns "
            "qualified for automatic reuse."
        )

    print()

    if reusable_text_patterns:

        print(
            "REUSABLE WHOLE-CELL PATTERNS"
        )

        print()

        for pattern in reusable_text_patterns:

            print(
                f"  "
                f"{pattern['original_ocr']!r}"
                f" + "
                f"{pattern['source_ocr']!r}"
                f" -> "
                f"{pattern['verified_text']!r}"
                f" "
                f"(repeated "
                f"{pattern['repeat_count']}x)"
            )

    print()

    print(
        "Memory file:"
    )

    print(
        MEMORY_FILE
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()