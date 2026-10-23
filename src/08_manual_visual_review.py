import json
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

# Visual review dependencies (UI only; backend logic below is preserved)
import streamlit as st
from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


INPUT_DIR = (
    BASE_DIR
    / "input"
)

VERIFIED_DIR = (
    BASE_DIR
    / "output"
    / "verified_data"
)

# Persistent human-review and learning state lives in output/learning.
# verified_data is generated/canonical OCR output, not the authoritative
# cumulative human-evidence ledger.
LEARNING_DIR = (
    BASE_DIR
    / "output"
    / "learning"
)

REVIEW_DIR = (
    VERIFIED_DIR
    / "review_required"
)

FINAL_DIR = (
    VERIFIED_DIR
    / "final_verified"
)

DECISIONS_FILE = (
    LEARNING_DIR
    / "manual_review_decisions.json"
)

SKIPS_FILE = (
    LEARNING_DIR
    / "manual_review_skips.json"
)

LEARNING_MEMORY_FILE = (
    LEARNING_DIR
    / "ocr_learning_memory.json"
)

LEGACY_DECISIONS_FILE = (
    VERIFIED_DIR
    / "manual_review_decisions.json"
)

# ============================================================
# CONSTANTS
# ============================================================

TRUSTED_HUMAN_DECISIONS = {
    "USER_CONFIRMED_ORIGINAL",
    "USER_CONFIRMED_SOURCE",
    "USER_ENTERED_CORRECTION",
}

AUTO_DECISIONS = {
    "EXACT_CASE_REUSED",
    "CONTEXT_PATTERN_REUSED",
    "OCR_TRANSFORMATION_PATTERN_REUSED",
    "STRUCTURAL_PATTERN_REUSED",
    "LEARNED_GENERALIZED_TRANSFORMATION",
    "LEARNED_STRUCTURAL_CONTEXT",
    "AGREEMENT_PATTERN_REUSED",
    "LEARNED_CROSS_CONTEXT_TRANSFORMATION",
    "LEARNED_CROSS_CONTEXT_STRUCTURAL",
}

NUMBER_RE = re.compile(
    r"""
    [+-]?
    (?:
        \d+(?:[.,]\d+)*
        |
        \d+/\d+
    )
    (?:[eE][+-]?\d+)?
    """,
    re.VERBOSE,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def now_utc():
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _decision_value(decision, names):
    """Return the first non-empty value from current/legacy decision schemas."""
    if not isinstance(decision, dict):
        return ""
    for name in names:
        value = decision.get(name)
        if value not in (None, ""):
            return value
    for container_name in ("decision_record", "review_item", "item", "evidence"):
        container = decision.get(container_name)
        if isinstance(container, dict):
            for name in names:
                value = container.get(name)
                if value not in (None, ""):
                    return value
    return ""


def _decision_document(decision):
    return str(_decision_value(decision, ("document", "document_name", "doc"))).strip()


def _decision_page(decision):
    return str(_decision_value(decision, ("page", "page_number"))).strip()


def _decision_original(decision):
    return normalize_text(_decision_value(decision, ("original_text", "original_ocr", "ocr_text", "original", "ocr")))


def _decision_source(decision):
    return normalize_text(_decision_value(decision, ("source_text", "source_ocr", "reference_text", "reference_ocr", "source", "reference")))


def _decision_verified(decision):
    return str(_decision_value(decision, ("verified_text", "final_verified_text", "corrected_text", "final_text")) or "")


def _decision_kind(decision):
    return str(_decision_value(decision, ("decision", "status", "action"))).strip().upper()


def _is_automatic_decision(decision):
    kind = _decision_kind(decision)
    return kind in AUTO_DECISIONS or kind in {
        "EXACT_EVIDENCE_REUSED",
        "SESSION_LEARNED_SYMBOL",
        "AUTO",
        "AUTOMATIC",
    }


def _is_human_evidence_decision(decision):
    """
    Detect a real human confirmation even when the decision record was later
    wrapped by an automatic reuse record.

    Stage 08 may preserve the original human verification inside the nested
    ``verification`` object while the top-level ``decision`` becomes an
    automatic learning/reuse status.  That nested human verification is still
    authoritative for EXACT evidence reuse.
    """
    if not isinstance(decision, dict):
        return False

    # Explicit immutable human-evidence records are authoritative.
    # They store the actual human choice in `human_decision`, rather than
    # the generic top-level `decision` field used by automatic records.
    if decision.get("evidence_type") == "HUMAN_CONFIRMED_EVIDENCE":
        human_kind = str(
            decision.get("human_decision", "")
        ).strip().upper()
        return human_kind in TRUSTED_HUMAN_DECISIONS

    # First inspect the original verification object.  This MUST happen before
    # rejecting the top-level automatic status.
    verification = decision.get("verification")
    if isinstance(verification, dict):
        verification_kind = str(
            verification.get("decision", "")
        ).strip().upper()
        if verification_kind in TRUSTED_HUMAN_DECISIONS:
            return True
        if any(
            token in verification_kind
            for token in (
                "USER_CONFIRMED",
                "USER_ENTERED",
                "HUMAN_CONFIRMED",
                "MANUAL_CONFIRMED",
            )
        ):
            return True

    # Then inspect the top-level decision.  Automatic records themselves are
    # never treated as a source of new human evidence.
    if _is_automatic_decision(decision):
        return False

    kind = _decision_kind(decision)
    if kind in TRUSTED_HUMAN_DECISIONS:
        return True

    return any(
        token in kind
        for token in (
            "USER_CONFIRMED",
            "USER_ENTERED",
            "HUMAN_CONFIRMED",
            "MANUAL_CONFIRMED",
        )
    )


def _human_verification_kind(decision):
    """Return the underlying human decision, including nested verification."""
    if not isinstance(decision, dict):
        return ""

    # Explicit immutable human-evidence records store the actual choice here.
    if decision.get("evidence_type") == "HUMAN_CONFIRMED_EVIDENCE":
        return str(
            decision.get("human_decision", "")
        ).strip().upper()

    verification = decision.get("verification")
    if isinstance(verification, dict):
        kind = str(
            verification.get("decision", "")
        ).strip().upper()
        if kind:
            return kind

    return _decision_kind(decision)


def _canonical_evidence_page(page):
    """Canonicalize page identifiers so paragraph/table evidence can match.

    Paragraph and table review records can serialize the same page as values
    such as 1, "1", or 1.0.  Those are the same evidence location and must
    share the human-evidence namespace.
    """
    text = str(page).strip()
    if not text:
        return ""
    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
    except (TypeError, ValueError):
        pass
    return text


def _canonical_evidence_text(value):
    """Normalize harmless serialization differences without merging glyphs."""
    import unicodedata

    text = unicodedata.normalize("NFC", str(value))
    return normalize_text(text)


def _exact_evidence_key(document, page, original, source):
    return (
        str(document).strip(),
        _canonical_evidence_page(page),
        _canonical_evidence_text(original),
        _canonical_evidence_text(source),
    )


def _global_exact_evidence_key(original, source):
    """
    Evidence identity independent of document/object representation.

    This is intentionally based on the COMPLETE OCR/source pair, not on
    individual characters. Therefore it can reuse a human decision for an
    identical evidence pair appearing in another document or in a different
    paragraph/table representation, without creating a global glyph rule.
    """
    return (
        _canonical_evidence_text(original),
        _canonical_evidence_text(source),
    )


def _find_global_exact_human_evidence(existing_decisions, original, source):
    """
    Find a consistent HUMAN decision for an identical complete OCR/source
    evidence pair, regardless of document or object type.

    IMPORTANT:
    - USER_CONFIRMED_ORIGINAL and USER_CONFIRMED_SOURCE do not require a
      stored `verified_text`; their verified value is the current original
      or source text.
    - USER_ENTERED_CORRECTION requires an actual stored verified value.
    - Automatic reuse records are never accepted as the first human source.
    - Conflicting human outcomes disable automatic reuse.
    """
    wanted = _global_exact_evidence_key(original, source)
    matches = []

    for decision in existing_decisions:
        if not _is_human_evidence_decision(decision):
            continue

        d_original = _decision_original(decision)
        d_source = _decision_source(decision)

        if not d_original or not d_source:
            continue

        if _global_exact_evidence_key(d_original, d_source) != wanted:
            continue

        kind = _human_verification_kind(decision)

        if kind == "USER_CONFIRMED_ORIGINAL":
            verified_value = d_original
        elif kind == "USER_CONFIRMED_SOURCE":
            verified_value = d_source
        else:
            verified_value = _decision_verified(decision)

        if not verified_value:
            continue

        matches.append(
            (
                decision,
                verified_value,
                kind,
            )
        )

    if not matches:
        return None

    # The same complete evidence pair must always resolve to one human
    # verified value. Never guess if humans previously disagreed.
    values = {
        verified_value
        for _, verified_value, _ in matches
    }

    if len(values) != 1:
        return None

    return matches[-1][0]



def _find_unambiguous_human_confirmation_by_exact_text(existing_decisions, original):
    """Reuse only a prior human confirmation that this exact text is correct.

    This is cross-document reuse for unchanged text only. If any human record
    for the same original text confirms a different value, or contains a
    correction/source choice that does not equal the original, fail closed.
    """
    wanted = _canonical_evidence_text(original)
    if len(wanted) < 4 or not any(ch.isalpha() for ch in wanted):
        return None

    matches = []
    for decision in existing_decisions:
        if not _is_human_evidence_decision(decision):
            continue
        if _canonical_evidence_text(_decision_original(decision)) != wanted:
            continue

        kind = _human_verification_kind(decision)
        if kind == "USER_CONFIRMED_ORIGINAL":
            verified_value = _decision_original(decision)
        elif kind == "USER_CONFIRMED_SOURCE":
            verified_value = _decision_source(decision)
        else:
            verified_value = _decision_verified(decision)

        if _canonical_evidence_text(verified_value) != wanted:
            return None
        matches.append(decision)

    return matches[-1] if matches else None

def _bbox_tuple(value):
    """Return a normalized (x1, y1, x2, y2) bbox or None."""
    if value is None:
        return None
    if isinstance(value, dict):
        # Accept common bbox dictionary forms.
        if all(k in value for k in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(k in value for k in ("x", "y", "w", "h")):
            value = [value["x"], value["y"], value["x"] + value["w"], value["y"] + value["h"]]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[i]) for i in range(4)]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_intersection_over_smaller_area(a, b):
    """Measure how much of the smaller box is covered by their intersection."""
    a = _bbox_tuple(a)
    b = _bbox_tuple(b)
    if a is None or b is None:
        return 0.0

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


def _find_human_evidence_by_original_and_bbox(existing_decisions, item):
    """
    Reuse a human decision when the exact ORIGINAL OCR belongs to the same
    document/page and the stored evidence boxes refer to the same visual
    region, even if paragraph and table source OCR differ.

    This is deliberately NOT a glyph substitution rule. The original OCR
    must match exactly and the visual evidence must overlap strongly.
    """
    if not isinstance(item, dict):
        return None

    original = item.get("original_text", "")
    if not original:
        return None

    item_bbox = _bbox_tuple(item.get("bbox"))
    if item_bbox is None:
        return None

    document = str(item.get("document", "")).strip()
    page = _canonical_evidence_page(item.get("page", ""))
    wanted_original = _canonical_evidence_text(original)

    matches = []

    for decision in existing_decisions:
        if not _is_human_evidence_decision(decision):
            continue

        if str(_decision_document(decision)).strip() != document:
            continue
        if _canonical_evidence_page(_decision_page(decision)) != page:
            continue

        d_original = _decision_original(decision)
        if not d_original:
            continue
        if _canonical_evidence_text(d_original) != wanted_original:
            continue

        decision_bbox = _bbox_tuple(decision.get("bbox"))
        if decision_bbox is None:
            continue

        overlap = _bbox_intersection_over_smaller_area(item_bbox, decision_bbox)
        if overlap < 0.80:
            continue

        kind = _human_verification_kind(decision)
        if kind == "USER_CONFIRMED_ORIGINAL":
            verified_value = d_original
        elif kind == "USER_CONFIRMED_SOURCE":
            verified_value = _decision_source(decision)
        else:
            verified_value = _decision_verified(decision)

        if not verified_value:
            continue

        matches.append((decision, verified_value, overlap))

    if not matches:
        return None

    # If multiple human decisions cover the same visual evidence but resolve
    # to different values, never guess.
    values = {value for _, value, _ in matches}
    if len(values) != 1:
        return None

    matches.sort(key=lambda entry: entry[2], reverse=True)
    return matches[0][0]


def _find_legacy_human_evidence_by_unique_original(existing_decisions, item):
    """
    Conservative fallback for older HUMAN_CONFIRMED_EVIDENCE records that
    were created before bbox persistence existed.

    This is intentionally narrower than a global original-text rule:
    - current item must be a table cell;
    - prior human evidence must be a paragraph block;
    - document and page must match;
    - ORIGINAL OCR must match exactly after harmless normalization;
    - exactly ONE qualifying human paragraph record must exist.

    If more than one qualifying paragraph exists, reuse is disabled rather
    than guessing which visual occurrence the table cell represents.
    """
    if not isinstance(item, dict):
        return None

    current_type = object_type(item)
    if current_type not in {"table_cell", "table"}:
        return None

    document = str(item.get("document", "")).strip()
    page = _canonical_evidence_page(item.get("page", ""))
    wanted_original = _canonical_evidence_text(item.get("original_text", ""))
    if not document or not page or not wanted_original:
        return None

    matches = []
    for decision in existing_decisions:
        if not _is_human_evidence_decision(decision):
            continue

        if str(_decision_document(decision)).strip() != document:
            continue
        if _canonical_evidence_page(_decision_page(decision)) != page:
            continue

        d_original = _decision_original(decision)
        if _canonical_evidence_text(d_original) != wanted_original:
            continue

        d_type = str(decision.get("object_type", "")).strip().lower()
        if d_type not in {"paragraph_block", "paragraph"}:
            continue

        verified_value = _reuse_exact_evidence_value(
            decision,
            d_original,
            _decision_source(decision),
        )
        if not verified_value:
            continue

        matches.append(decision)

    # Ambiguous legacy evidence must never be guessed.
    if len(matches) != 1:
        return None

    return matches[0]


def _find_exact_human_evidence(existing_decisions, document, page, original, source):
    """Find a previously human-confirmed identical OCR/source evidence pair."""
    wanted = _exact_evidence_key(document, page, original, source)
    matches = []
    for decision in existing_decisions:
        if not _is_human_evidence_decision(decision):
            continue
        d_original = _decision_original(decision)
        d_source = _decision_source(decision)
        d_verified = _decision_verified(decision)
        if not d_original or not d_source or not d_verified:
            continue
        candidate = _exact_evidence_key(
            _decision_document(decision),
            _decision_page(decision),
            d_original,
            d_source,
        )
        if candidate == wanted:
            matches.append(decision)
    if not matches:
        return None
    # If humans gave conflicting verified values for the exact same evidence,
    # do not guess.
    values = { _decision_verified(d) for d in matches }
    if len(values) != 1:
        return None
    return matches[-1]


def _reuse_exact_evidence_value(decision, current_original, current_source):
    """Return the human-verified value for an identical evidence pair."""
    kind = _human_verification_kind(decision)

    if kind == "USER_CONFIRMED_ORIGINAL":
        return current_original

    if kind == "USER_CONFIRMED_SOURCE":
        return current_source

    # For USER_ENTERED_CORRECTION (and legacy human-confirmation records),
    # reuse the exact value the human actually verified.
    verified = _decision_verified(decision)

    # Some records store verified_text only inside verification.
    if not verified:
        verification = decision.get("verification", {})
        if isinstance(verification, dict):
            verified = str(
                verification.get("verified_text", "")
                or verification.get("final_verified_text", "")
                or verification.get("corrected_text", "")
                or ""
            )

    return verified or None


def normalize_choice(choice):
    return str(
        choice
    ).strip().lower()


# ============================================================
# UNICODE CHARACTER DESCRIPTION
# ============================================================

def unicode_name(character):
    try:
        return unicodedata.name(
            character
        )
    except ValueError:
        return "UNKNOWN"


def unicode_category(character):
    return unicodedata.category(
        character
    )


def is_letter(character):
    return unicodedata.category(
        character
    ).startswith("L")


def is_number(character):
    return unicodedata.category(
        character
    ).startswith("N")


def is_mark(character):
    return unicodedata.category(
        character
    ).startswith("M")


def is_symbol(character):
    category = unicodedata.category(
        character
    )

    return category.startswith("S")


def is_punctuation(character):
    category = unicodedata.category(
        character
    )

    return category.startswith("P")


def describe_character(character):
    return {
        "character": character,
        "unicode": (
            f"U+{ord(character):04X}"
        ),
        "name": unicode_name(
            character
        ),
        "category": unicode_category(
            character
        ),
    }


# ============================================================
# DYNAMIC SPECIAL CHARACTER DETECTION
# ============================================================

def detect_special_characters(text):
    result = []

    seen = set()

    for character in str(
        text or ""
    ):

        if (
            not is_symbol(character)
            and not (
                is_letter(character)
                and ord(character) > 127
            )
            and not (
                is_punctuation(character)
                and ord(character) > 127
            )
        ):
            continue

        if character in seen:
            continue

        seen.add(
            character
        )

        result.append(
            describe_character(
                character
            )
        )

    return result


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Tokenizes text without changing the original value.

    Numbers become NUM.

    ASCII/Unicode letters are represented by their
    Unicode category.

    Symbols/punctuation retain their actual Unicode
    code point in the learning fingerprint.
    """

    text = normalize_text(
        text
    )

    if not text:
        return []

    tokens = []

    i = 0

    while i < len(text):

        character = text[i]

        if character.isspace():
            i += 1
            continue

        number_match = NUMBER_RE.match(
            text,
            i,
        )

        if number_match:

            tokens.append(
                {
                    "type": "NUMBER",
                    "value": "<NUM>",
                }
            )

            i = number_match.end()

            continue

        category = unicode_category(
            character
        )

        # ----------------------------------------------------
        # Letters
        # ----------------------------------------------------

        if category.startswith("L"):

            tokens.append(
                {
                    "type": "LETTER",
                    "value": (
                        f"LETTER_{category}"
                    ),
                    "character": character,
                }
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Marks
        # ----------------------------------------------------

        if category.startswith("M"):

            tokens.append(
                {
                    "type": "MARK",
                    "value": (
                        f"MARK_{category}"
                    ),
                    "character": character,
                }
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Symbols
        # ----------------------------------------------------

        if category.startswith("S"):

            tokens.append(
                {
                    "type": "SYMBOL",
                    "value": (
                        f"SYMBOL_U+"
                        f"{ord(character):04X}"
                    ),
                    "character": character,
                }
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Punctuation
        # ----------------------------------------------------

        if category.startswith("P"):

            tokens.append(
                {
                    "type": "PUNCTUATION",
                    "value": (
                        f"PUNCT_U+"
                        f"{ord(character):04X}"
                    ),
                    "character": character,
                }
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Everything else
        # ----------------------------------------------------

        tokens.append(
            {
                "type": "OTHER",
                "value": (
                    f"OTHER_U+"
                    f"{ord(character):04X}"
                ),
                "character": character,
            }
        )

        i += 1

    return tokens


# ============================================================
# STRUCTURAL PATTERN
# ============================================================

def structural_pattern(text):
    tokens = tokenize(
        text
    )

    parts = []

    for token in tokens:

        token_type = token[
            "type"
        ]

        if token_type == "NUMBER":
            parts.append(
                "<NUM>"
            )

        elif token_type == "LETTER":
            parts.append(
                "<LETTER>"
            )

        elif token_type == "MARK":
            parts.append(
                "<MARK>"
            )

        elif token_type == "SYMBOL":
            parts.append(
                token["value"]
            )

        elif token_type == "PUNCTUATION":
            parts.append(
                token["value"]
            )

        else:
            parts.append(
                token["value"]
            )

    return " ".join(
        parts
    )


# ============================================================
# OCR TRANSFORMATION FINGERPRINT
# ============================================================

def _transformation_tokens(text):
    """Tokenize OCR transformation pairs without swallowing a leading +/-.

    A leading + or - can itself be the OCR-corrupted symbol (for example
    ± -> +).  For learning, it must therefore remain a separate token.
    Numeric magnitudes are still abstracted as NUM.
    """
    text = normalize_text(text)
    if not text:
        return []

    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue

        number_match = re.match(r"\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?", text[i:])
        if number_match:
            tokens.append({"type": "NUMBER", "value": "<NUM>"})
            i += number_match.end()
            continue

        category = unicodedata.category(ch)
        if category.startswith("L"):
            tokens.append({
                "type": "LETTER",
                "value": f"LETTER_{category}",
                "character": ch,
            })
        elif category.startswith("M"):
            tokens.append({
                "type": "MARK",
                "value": f"MARK_{category}",
                "character": ch,
            })
        elif category.startswith("S"):
            tokens.append({
                "type": "SYMBOL",
                "value": f"SYMBOL_U+{ord(ch):04X}",
                "character": ch,
            })
        elif category.startswith("P"):
            tokens.append({
                "type": "PUNCTUATION",
                "value": f"PUNCT_U+{ord(ch):04X}",
                "character": ch,
            })
        else:
            tokens.append({
                "type": "OTHER",
                "value": f"OTHER_U+{ord(ch):04X}",
                "character": ch,
            })
        i += 1

    return tokens


def transformation_fingerprint(original, source):
    original_tokens = _transformation_tokens(original)
    source_tokens = _transformation_tokens(source)

    if len(original_tokens) != len(source_tokens):
        return None

    result = []
    for a, b in zip(original_tokens, source_tokens):
        at = a.get("type", "")
        bt = b.get("type", "")

        if at == "NUMBER" and bt == "NUMBER":
            result.append("NUMBER->NUMBER")
            continue

        if a.get("value") == b.get("value"):
            result.append("SAME:" + a.get("value", ""))
            continue

        result.append(a.get("value", "") + "->" + b.get("value", ""))

    return " | ".join(result)


# ============================================================
# CONTEXT NORMALIZATION
# ============================================================

def context_pattern(text):
    """
    Context representation.

    Numbers are abstracted.
    Unicode character classes remain represented.
    """

    tokens = tokenize(
        text
    )

    parts = []

    for token in tokens:

        token_type = token[
            "type"
        ]

        if token_type == "NUMBER":
            parts.append(
                "<NUM>"
            )

        elif token_type == "LETTER":
            parts.append(
                "<LETTER>"
            )

        elif token_type == "MARK":
            parts.append(
                "<MARK>"
            )

        elif token_type == "SYMBOL":
            parts.append(
                token["value"]
            )

        elif token_type == "PUNCTUATION":
            parts.append(
                token["value"]
            )

        else:
            parts.append(
                token["value"]
            )

    return " ".join(
        parts
    )


# ============================================================
# PAGE LOOKUP
# ============================================================

def get_page(
    final_data,
    document_name,
    page_number,
):
    """
    Locate a page in either the current per-PDF JSON structure
    or the older combined structure.

    Current per-PDF structure:
        {
            "document": "...",
            "pages": [...]
        }

    Legacy combined structure:
        {
            "documents": [
                {
                    "document": "...",
                    "pages": [...]
                }
            ]
        }
    """

    # --------------------------------------------------------
    # CURRENT PER-PDF JSON
    # --------------------------------------------------------

    if isinstance(final_data, dict):

        current_document = str(
            final_data.get(
                "document",
                final_data.get(
                    "name",
                    "",
                ),
            )
        )

        pages = final_data.get(
            "pages",
            [],
        )

        if (
            isinstance(pages, list)
            and (
                not document_name
                or current_document == str(document_name)
            )
        ):

            try:
                wanted_page = int(
                    page_number
                )
            except (
                TypeError,
                ValueError,
            ):
                wanted_page = None

            for page in pages:

                if not isinstance(
                    page,
                    dict,
                ):
                    continue

                try:
                    current_page = int(
                        page.get(
                            "page",
                            0,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if (
                    wanted_page is not None
                    and current_page == wanted_page
                ):
                    return page

    # --------------------------------------------------------
    # LEGACY COMBINED JSON
    # --------------------------------------------------------

    if isinstance(final_data, dict):

        documents = final_data.get(
            "documents",
            [],
        )

        if isinstance(
            documents,
            list,
        ):

            for document in documents:

                if not isinstance(
                    document,
                    dict,
                ):
                    continue

                current_name = str(
                    document.get(
                        "document",
                        document.get(
                            "name",
                            "",
                        ),
                    )
                )

                if current_name != str(
                    document_name
                ):
                    continue

                for page in document.get(
                    "pages",
                    [],
                ):

                    try:
                        current_page = int(
                            page.get(
                                "page",
                                0,
                            )
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    try:
                        wanted_page = int(
                            page_number
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        return None

                    if current_page == wanted_page:
                        return page

    return None


# ============================================================
# ROBUST OBJECT TEXT EXTRACTION
# ============================================================

def extract_object_text(
    obj
):
    if not isinstance(
        obj,
        dict,
    ):
        return ""

    for field in (
        "text",
        "original_text",
        "content",
        "value",
        "ocr_text",
        "verified_text",
        "final_verified_text",
    ):

        value = obj.get(
            field
        )

        if value is not None:
            value = str(
                value
            )

            if value.strip():
                return value

    return ""


# ============================================================
# ROBUST PARAGRAPH BLOCK SEARCH
# ============================================================

def find_paragraph_block(
    final_data,
    document_name,
    page_number,
    paragraph_block,
):
    page = get_page(
        final_data,
        document_name,
        page_number,
    )

    if page is None:
        return None

    try:
        target_id = int(
            paragraph_block
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    # Stage 06 stores paragraph blocks canonically as:
    # page -> paragraph -> text_blocks.
    # Prefer this exact structure before checking legacy layouts.
    paragraph = page.get("paragraph", {})
    if isinstance(paragraph, dict):
        text_blocks = paragraph.get("text_blocks", [])
        if isinstance(text_blocks, list):
            if 0 <= target_id - 1 < len(text_blocks):
                block = text_blocks[target_id - 1]
                if isinstance(block, dict):
                    return block

    possible_containers = [
        page.get(
            "paragraph_blocks",
            [],
        ),
        page.get(
            "blocks",
            [],
        ),
        page.get(
            "text_blocks",
            [],
        ),
        page.get(
            "paragraphs",
            [],
        ),
        page.get(
            "objects",
            [],
        ),
    ]

    for container in possible_containers:

        if not isinstance(
            container,
            list,
        ):
            continue

        for block in container:

            if not isinstance(
                block,
                dict,
            ):
                continue

            for field in (
                "paragraph_block",
                "paragraph_id",
                "block",
                "block_id",
                "id",
                "object_id",
            ):

                value = block.get(
                    field
                )

                try:
                    if int(
                        value
                    ) == target_id:
                        return block
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    return None


# ============================================================
# ROBUST TABLE CELL SEARCH
# ============================================================

def find_table_cell(
    final_data,
    document_name,
    page_number,
    table_number,
    row_number,
    column_number,
):
    """Resolve a table cell from the canonical Stage 06 structure.

    Prefer explicit table/row/column metadata when it is present, then fall
    back to the documented 1-based list coordinates. This keeps the lookup
    generic while tolerating older JSON variants whose list position and
    stored coordinate metadata are not perfectly aligned.
    """
    page = get_page(
        final_data,
        document_name,
        page_number,
    )

    if page is None:
        return None

    try:
        requested_table = int(table_number)
        requested_row = int(row_number)
        requested_column = int(column_number)
    except (TypeError, ValueError):
        return None

    tables = page.get("tables", [])
    if not isinstance(tables, list):
        return None

    # First use explicit table metadata when available. Otherwise retain the
    # canonical 1-based table numbering used by the review records.
    table_candidates = []
    for table_index, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        metadata_values = []
        for field in ("table", "table_id", "index", "table_index"):
            value = table.get(field)
            try:
                metadata_values.append(int(value))
            except (TypeError, ValueError):
                pass
        if requested_table in metadata_values:
            table_candidates.insert(0, (table_index, table))
        else:
            table_candidates.append((table_index, table))

    if not table_candidates:
        return None

    # Prefer the canonical 1-based position. If that table does not contain
    # the requested cell, search the remaining tables by explicit cell
    # coordinates before failing.
    preferred_index = requested_table - 1
    ordered = sorted(
        table_candidates,
        key=lambda pair: (0 if pair[0] == preferred_index else 1, pair[0])
    )

    for _, table in ordered:
        rows = table.get("rows", [])
        if not isinstance(rows, list):
            continue

        # Explicit row/column metadata is the strongest coordinate match.
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells", [])
            if not isinstance(cells, list):
                continue
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                try:
                    cell_row = int(cell.get("row"))
                    cell_column = int(cell.get("column"))
                except (TypeError, ValueError):
                    continue
                if cell_row == requested_row and cell_column == requested_column:
                    return cell

        # Canonical fallback: 1-based row and column positions.
        row_index = requested_row - 1
        column_index = requested_column - 1
        if 0 <= row_index < len(rows):
            cells = rows[row_index].get("cells", [])
            if isinstance(cells, list) and 0 <= column_index < len(cells):
                cell = cells[column_index]
                if isinstance(cell, dict):
                    return cell

    return None


# ============================================================
# TARGET FINDER
# ============================================================

def find_target(
    final_data,
    item,
):
    """
    Locate the canonical target using mutually exclusive object
    types.  Table coordinates always mean TABLE CELL.  A
    paragraph_block means TEXT/PARAGRAPH BLOCK only.
    """
    document_name = str(
        item.get(
            "document",
            "",
        )
    )

    try:
        page_number = int(
            item.get(
                "page",
                0,
            )
        )
    except (TypeError, ValueError):
        page_number = 0

    table_number = item.get("table")
    row_number = item.get("row")
    column_number = item.get("column")

    # TABLE CELL HAS PRIORITY. A table review item is never
    # interpreted as a paragraph block, even if a legacy record
    # happens to contain a paragraph_block field.
    if (
        table_number not in (None, "", 0, "0")
        and row_number not in (None, "", 0, "0")
        and column_number not in (None, "", 0, "0")
    ):
        target = find_table_cell(
            final_data,
            document_name,
            page_number,
            table_number,
            row_number,
            column_number,
        )

        if target is not None:
            return target, "table_cell"

        return None, "table_cell"

    # PARAGRAPH/TEXT BLOCK. This means page -> paragraph ->
    # text_blocks in the Stage 06 canonical structure.
    paragraph_id = item.get("paragraph_block")

    if paragraph_id not in (None, "", 0, "0"):
        target = find_paragraph_block(
            final_data,
            document_name,
            page_number,
            paragraph_id,
        )

        if target is not None:
            return target, "paragraph_block"

        return None, "paragraph_block"

    return None, "unknown"


# ============================================================
# OBJECT TYPE
# ============================================================

def object_type(
    item
):
    """
    Object classification is mutually exclusive:

    - table_cell: table/row/column coordinates
    - paragraph_block: text block only

    Table coordinates take precedence so a table cell can never
    accidentally be classified as a paragraph block.
    """
    if (
        item.get("table") not in (None, "", 0, "0")
        and item.get("row") not in (None, "", 0, "0")
        and item.get("column") not in (None, "", 0, "0")
    ):
        return "table_cell"

    if item.get("paragraph_block") not in (None, "", 0, "0"):
        return "paragraph_block"

    return "unknown"


# ============================================================
# DECISION KEY
# ============================================================

def decision_key(
    item
):
    """
    Stable location key. Paragraph/text blocks and table cells
    intentionally have different namespaces.
    """
    document = str(
        item.get(
            "document",
            "",
        )
    )

    try:
        page = int(
            item.get(
                "page",
                0,
            )
        )
    except (TypeError, ValueError):
        page = 0

    # Table namespace always wins when complete table coordinates
    # are present.
    if (
        item.get("table") not in (None, "", 0, "0")
        and item.get("row") not in (None, "", 0, "0")
        and item.get("column") not in (None, "", 0, "0")
    ):
        try:
            table = int(item.get("table"))
            row = int(item.get("row"))
            column = int(item.get("column"))
        except (TypeError, ValueError):
            table = row = column = 0

        return (
            document,
            page,
            "table",
            table,
            row,
            column,
        )

    if item.get("paragraph_block") not in (None, "", 0, "0"):
        try:
            paragraph = int(item.get("paragraph_block"))
        except (TypeError, ValueError):
            paragraph = 0

        return (
            document,
            page,
            "paragraph",
            paragraph,
        )

    return (
        document,
        page,
        "unknown",
    )


# ============================================================
# CONTEXT FINGERPRINT
# ============================================================

def build_context_fingerprint(
    final_data,
    item,
):
    """
    Creates contextual information around the reviewed object.

    This prevents a learned transformation from becoming
    a global character replacement.
    """

    page = get_page(
        final_data,
        item.get(
            "document",
            "",
        ),
        int(
            item.get(
                "page",
                0,
            )
        ),
    )

    if page is None:
        return ""

    current_type = object_type(
        item
    )

    parts = [
        "OBJECT_TYPE="
        + current_type
    ]

    # --------------------------------------------------------
    # Table context
    # --------------------------------------------------------

    if current_type == "table_cell":

        try:
            table_number = int(
                item.get(
                    "table",
                    0,
                )
            )

            row_number = int(
                item.get(
                    "row",
                    0,
                )
            )

            column_number = int(
                item.get(
                    "column",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            table_number = (
                row_number
            ) = (
                column_number
            ) = 0

        tables = page.get(
            "tables",
            [],
        )

        table_index = (
            table_number - 1
        )

        row_index = (
            row_number - 1
        )

        column_index = (
            column_number - 1
        )

        if (
            0 <= table_index < len(
                tables
            )
        ):

            rows = tables[
                table_index
            ].get(
                "rows",
                [],
            )

            # Header
            if rows:

                header_cells = rows[
                    0
                ].get(
                    "cells",
                    [],
                )

                if (
                    0 <= column_index
                    < len(
                        header_cells
                    )
                ):

                    header_text = (
                        extract_object_text(
                            header_cells[
                                column_index
                            ]
                        )
                    )

                    if header_text:
                        parts.append(
                            "HEADER="
                            + context_pattern(
                                header_text
                            )
                        )

            # Row label
            if (
                0 <= row_index
                < len(rows)
            ):

                row_cells = rows[
                    row_index
                ].get(
                    "cells",
                    [],
                )

                if row_cells:

                    label = (
                        extract_object_text(
                            row_cells[0]
                        )
                    )

                    if label:
                        parts.append(
                            "ROW="
                            + context_pattern(
                                label
                            )
                        )

    # --------------------------------------------------------
    # Paragraph context
    # --------------------------------------------------------

    elif (
        current_type
        == "paragraph_block"
    ):

        paragraph_id = item.get(
            "paragraph_block"
        )

        try:
            target_id = int(
                paragraph_id
            )
        except (
            TypeError,
            ValueError,
        ):
            target_id = 0

        paragraph = page.get("paragraph", {})
        nested_text_blocks = (
            paragraph.get("text_blocks", [])
            if isinstance(paragraph, dict)
            else []
        )

        containers = [
            nested_text_blocks,
            page.get(
                "paragraph_blocks",
                [],
            ),
            page.get(
                "blocks",
                [],
            ),
            page.get(
                "text_blocks",
                [],
            ),
            page.get(
                "paragraphs",
                [],
            ),
        ]

        all_blocks = []

        for container in containers:

            if not isinstance(
                container,
                list,
            ):
                continue

            for block in container:

                if not isinstance(
                    block,
                    dict,
                ):
                    continue

                block_id = None

                for field in (
                    "paragraph_block",
                    "paragraph_id",
                    "block",
                    "block_id",
                    "id",
                    "object_id",
                ):

                    try:
                        block_id = int(
                            block.get(
                                field
                            )
                        )

                        break

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                if block_id is None:
                    continue

                text = extract_object_text(
                    block
                )

                all_blocks.append(
                    (
                        block_id,
                        text,
                    )
                )

        # Remove duplicates
        unique_blocks = {}

        for block_id, text in all_blocks:
            unique_blocks[
                block_id
            ] = text

        ordered = sorted(
            unique_blocks.items(),
            key=lambda x: x[0],
        )

        current_index = None

        for index, (
            block_id,
            text,
        ) in enumerate(
            ordered
        ):

            if block_id == target_id:
                current_index = index
                break

        if current_index is not None:

            if current_index > 0:

                previous_text = (
                    ordered[
                        current_index - 1
                    ][1]
                )

                if previous_text:
                    parts.append(
                        "PREV="
                        + context_pattern(
                            previous_text
                        )
                    )

            if (
                current_index + 1
                < len(ordered)
            ):

                next_text = (
                    ordered[
                        current_index + 1
                    ][1]
                )

                if next_text:
                    parts.append(
                        "NEXT="
                        + context_pattern(
                            next_text
                        )
                    )

    return "|".join(
        parts
    )


# ============================================================
# COMPLETE LEARNING FINGERPRINT
# ============================================================

def agreement_special_signature(text):
    """Return a generic signature for non-ASCII special characters.

    Used only for HIGH_RISK_AGREEMENT_REVIEW learning where OCR and
    source already agree. This never rewrites OCR text and never maps
    one character to another.
    """
    chars = []
    for ch in normalize_text(text):
        if ord(ch) <= 127:
            continue
        category = unicodedata.category(ch)
        if category.startswith(("S", "P", "M")):
            chars.append(f"U+{ord(ch):04X}")
    return "|".join(chars)


def build_learning_fingerprint(
    final_data,
    item,
):
    original = str(
        item.get(
            "original_text",
            "",
        )
    )

    source = str(
        item.get(
            "source_text",
            "",
        )
    )

    transformation = (
        transformation_fingerprint(
            original,
            source,
        )
    )

    generalized_transformation = _generalized_transformation(
        original,
        source,
    )

    if transformation is None:
        return None

    return {
        "object_type":
            object_type(item),

        "original_structure":
            structural_pattern(
                original
            ),

        "source_structure":
            structural_pattern(
                source
            ),

        "transformation":
            transformation,

        "generalized_transformation":
            generalized_transformation,

        "context":
            build_context_fingerprint(
                final_data,
                item,
            ),
    }


def fingerprint_key(
    fingerprint
):
    return json.dumps(
        fingerprint,
        ensure_ascii=False,
        sort_keys=True,
    )


def learning_action(decision):
    decision_type = decision.get("decision", "")
    if decision_type == "USER_CONFIRMED_ORIGINAL":
        return "USE_CURRENT_ORIGINAL"
    if decision_type == "USER_CONFIRMED_SOURCE":
        return "USE_CURRENT_SOURCE"
    return None


def _generalized_transformation(original, source):
    """Generalize OCR transformations without hard-coded document text."""
    ot = _transformation_tokens(original)
    st = _transformation_tokens(source)

    if len(ot) != len(st):
        return None

    def atom(token):
        t = token.get("type", "")
        ch = token.get("character", "")
        # For generalized OCR learning, letters and numbers are treated
        # as one generic alphanumeric class. This is deliberately generic:
        # it does not name or hard-code O/0, I/1, etc.
        # The actual OCR text is never changed; this is only a learning
        # fingerprint used to decide whether two transformation cases have
        # the same structural shape.
        if t in {"NUMBER", "LETTER"}:
            return "ALNUM"
        if t == "SYMBOL":
            return "SYMBOL:" + token.get("value", "")
        if t == "PUNCTUATION":
            return "PUNCT:" + token.get("value", "")
        return t

    result = []
    for a, b in zip(ot, st):
        a_atom = atom(a)
        b_atom = atom(b)
        if {a_atom, b_atom} <= {"AMBIG_ALNUM", "NUM"}:
            result.append("AMBIG_ALNUM->AMBIG_ALNUM")
        elif a_atom == b_atom:
            result.append("SAME:" + a_atom)
        else:
            result.append(a_atom + "->" + b_atom)

    return " | ".join(result)



def core_transformation_fingerprint(fingerprint):
    if not isinstance(fingerprint, dict):
        return None
    # Deliberately omit literal source text and exact source structure.
    # The learned rule is based on the generalized transformation plus
    # object type, not on a hard-coded occurrence.
    return {
        "object_type": fingerprint.get("object_type", ""),
        "generalized_transformation": fingerprint.get(
            "generalized_transformation",
            fingerprint.get("transformation", ""),
        ),
    }


# ============================================================
# MEMORY
# ============================================================

def empty_memory():
    return {
        "version": 7,
        "exact_patterns": {},
        "transformation_patterns": {},
        "structural_patterns": {},
        "context_patterns": {},
        "agreement_patterns": {},
        "symbol_confusion_patterns": {},
        "audit": [],
    }


def load_memory():
    """Load the clean Stage 09 memory without replacing its schema."""
    if not LEARNING_MEMORY_FILE.exists():
        return empty_memory()

    try:
        memory = load_json(LEARNING_MEMORY_FILE)
    except Exception:
        return empty_memory()

    if not isinstance(memory, dict):
        return empty_memory()

    memory.setdefault("version", 6)
    memory.setdefault("exact_patterns", {})
    memory.setdefault("transformation_patterns", {})
    memory.setdefault("structural_patterns", {})
    memory.setdefault("context_patterns", {})
    memory.setdefault("cross_context_generalized_patterns", {})
    memory.setdefault("cross_context_structural_patterns", {})
    memory.setdefault("local_transformation_patterns", {})
    memory.setdefault("agreement_patterns", {})
    memory.setdefault("symbol_confusion_patterns", {})
    memory.setdefault("audit", [])
    return memory


def save_memory(memory):
    save_json(LEARNING_MEMORY_FILE, memory)


# ============================================================
# BUILD MEMORY FROM HUMAN DECISIONS
# ============================================================

def build_memory_from_decisions(
    final_data,
    decisions,
    memory,
):
    """
    Rebuilds learning from trusted HUMAN decisions.

    Automatic decisions are NEVER allowed to teach the
    learning system.

    This is critical for preventing an OCR error from
    becoming learned truth.
    """

    for decision in decisions:

        if decision.get(
            "decision"
        ) not in TRUSTED_HUMAN_DECISIONS:
            continue

        original = str(
            decision.get(
                "original_text",
                "",
            )
        )

        source = str(
            decision.get(
                "source_text",
                "",
            )
        )

        verified = str(
            decision.get(
                "verified_text",
                "",
            )
        )

        if not verified:
            continue

        # ----------------------------------------------------
        # Exact pattern
        # ----------------------------------------------------

        ekey = json.dumps(
            [
                normalize_text(
                    original
                ),
                normalize_text(
                    source
                ),
            ],
            ensure_ascii=False,
        )

        memory[
            "exact_patterns"
        ][ekey] = {
            "verified_text":
                verified,
            "human_confirmed":
                True,
        }

        # ----------------------------------------------------
        # AGREEMENT PATTERN
        # ----------------------------------------------------
        # If OCR and source agree and a human confirmed the value,
        # remember the non-ASCII special-character signature. This
        # specifically addresses HIGH_RISK_AGREEMENT_REVIEW false
        # positives without creating a character substitution rule.
        if (
            normalize_text(original)
            and normalize_text(original) == normalize_text(source)
        ):
            signature = agreement_special_signature(original)
            if signature:
                akey = json.dumps(
                    [object_type(decision), signature],
                    ensure_ascii=False,
                )
                memory["agreement_patterns"][akey] = {
                    "action": "USE_CURRENT_ORIGINAL",
                    "repeat_count": int(
                        memory["agreement_patterns"]
                        .get(akey, {})
                        .get("repeat_count", 0) or 0
                    ) + 1,
                    "human_confirmed": True,
                }

        # ----------------------------------------------------
        # Transformation pattern
        # ----------------------------------------------------

        pseudo_item = dict(
            decision
        )

        fingerprint = (
            build_learning_fingerprint(
                final_data,
                pseudo_item,
            )
        )

        if fingerprint is None:
            continue

        fkey = fingerprint_key(
            fingerprint
        )

        existing = memory[
            "transformation_patterns"
        ].get(
            fkey,
            {},
        )

        repeat_count = int(
            existing.get(
                "repeat_count",
                0,
            )
            or 0
        )

        action = learning_action(decision)

        if action is not None:
            core_key = fingerprint_key(
                core_transformation_fingerprint(fingerprint)
            )
            memory["transformation_patterns"][core_key] = {
                "action": action,
                "repeat_count": repeat_count + 1,
                "human_confirmed": True,
            }

        # ----------------------------------------------------
        # Structural pattern
        # ----------------------------------------------------

        skey = (
            structural_pattern(
                original
            )
            + " || "
            + structural_pattern(
                source
            )
        )

        structural_existing = (
            memory[
                "structural_patterns"
            ].get(
                skey,
                {},
            )
        )

        structural_repeat = int(
            structural_existing.get(
                "repeat_count",
                0,
            )
            or 0
        )

        memory[
            "structural_patterns"
        ][skey] = {
            "verified_text":
                verified,
            "repeat_count":
                structural_repeat + 1,
            "human_confirmed":
                True,
        }

    return memory


# ============================================================
# FIND LEARNED PATTERN
# ============================================================

def _clean_learning_object_type(item):
    """Exactly match CLEAN Stage 09 object_type()."""
    explicit = str(item.get("object_type", "")).strip()
    if explicit:
        return explicit

    try:
        table = int(item.get("table", 0) or 0)
    except (TypeError, ValueError):
        table = 0

    if table > 0:
        return "TABLE_CELL"
    return "PARAGRAPH"


def _clean_learning_context(item):
    """Exactly match CLEAN Stage 09 learning_context()."""
    typ = _clean_learning_object_type(item)
    if typ == "TABLE_CELL":
        try:
            table = int(item.get("table", 0) or 0)
        except (TypeError, ValueError):
            table = 0
        try:
            column = int(item.get("column", 0) or 0)
        except (TypeError, ValueError):
            column = 0
        return f"{typ}|table={table}|column={column}"
    return typ


def _clean_char_class(ch):
    category = unicodedata.category(ch)
    if category.startswith("L"):
        return "LETTER"
    if category.startswith("N"):
        return "NUMBER"
    if category.startswith("P"):
        return "PUNCTUATION"
    if category.startswith("S"):
        return "SYMBOL"
    if category.startswith("M"):
        return "MARK"
    if category.startswith("Z"):
        return "SPACE"
    if category.startswith("C"):
        return "CONTROL"
    return category


def _clean_structural_pattern(text):
    """Exactly match CLEAN Stage 09 structural_pattern()."""
    text = normalize_text(text)
    parts = []
    for ch in text:
        cls = _clean_char_class(ch)
        if cls == "LETTER":
            parts.append("LETTER")
        elif cls == "NUMBER":
            parts.append("NUMBER")
        elif cls == "PUNCTUATION":
            parts.append("PUNCTUATION")
        elif cls == "SYMBOL":
            parts.append("SYMBOL")
        elif cls == "MARK":
            parts.append("MARK")
        elif cls == "SPACE":
            parts.append("SPACE")
        else:
            parts.append(cls)
    return " ".join(parts)


def _clean_generalized_shape(text):
    """Exactly match CLEAN Stage 09 generalized_shape()."""
    text = normalize_text(text)
    if not text:
        return ""
    groups = []
    current = None
    count = 0

    def flush():
        nonlocal current, count
        if current is not None:
            groups.append(f"{current}{count}")
        current = None
        count = 0

    for ch in text:
        cls = _clean_char_class(ch)
        if cls == "LETTER":
            base = "L"
        elif cls == "NUMBER":
            base = "N"
        elif cls == "SPACE":
            base = "W"
        else:
            flush()
            if cls == "SYMBOL":
                groups.append("S:" + unicodedata.name(ch, "UNKNOWN"))
            elif cls == "PUNCTUATION":
                groups.append("P:" + unicodedata.name(ch, "UNKNOWN"))
            elif cls == "MARK":
                groups.append("M:" + unicodedata.name(ch, "UNKNOWN"))
            else:
                groups.append(cls)
            continue
        if base == current:
            count += 1
        else:
            flush()
            current = base
            count = 1
    flush()
    return "|".join(groups)


def _clean_generalized_transformation(original, source):
    """Exactly match CLEAN Stage 09 generalized_transformation()."""
    a = normalize_text(original)
    b = normalize_text(source)
    if not a or not b:
        return None

    # Stage 09's transformation fingerprint is based on Unicode
    # character classes and sequence alignment, not literal OCR maps.
    aa = [c for c in a if not c.isspace()]
    bb = [c for c in b if not c.isspace()]
    sm = SequenceMatcher(a=aa, b=bb, autojunk=False)
    pieces = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pieces.append(f"SAME:{_clean_char_class(aa[i1])}:{i2-i1}")
        elif tag == "replace":
            left = "".join(_clean_char_class(c) for c in aa[i1:i2])
            right = "".join(_clean_char_class(c) for c in bb[j1:j2])
            pieces.append(f"REPLACE:{left}->{right}")
        elif tag == "delete":
            left = "".join(_clean_char_class(c) for c in aa[i1:i2])
            pieces.append(f"DELETE:{left}")
        elif tag == "insert":
            right = "".join(_clean_char_class(c) for c in bb[j1:j2])
            pieces.append(f"INSERT:{right}")

    return " | ".join(pieces)



def _local_transformation_pattern_keys(original: str, source: str, typ: str):
    """
    Build the exact local-transformation keys used by Stage 09 FIX25.

    This must mirror Stage 09's local_transformation_patterns() and
    local_pattern_key() exactly.  The learned pattern stores Unicode names,
    Unicode classes, and a small structural neighborhood around each changed
    character.  It never contains a predefined OCR correction dictionary.
    """
    original = normalize_text(original)
    source = normalize_text(source)

    if not original or not source or len(original) != len(source):
        return []

    out = []
    for i, (a, b) in enumerate(zip(original, source)):
        if a == b:
            continue

        left = original[max(0, i - 2):i]
        right = original[i + 1:i + 3]

        position_class = (
            "START" if i == 0
            else "END" if i == len(original) - 1
            else "MIDDLE"
        )

        item = {
            "original_name": unicodedata.name(a, "UNKNOWN"),
            "source_name": unicodedata.name(b, "UNKNOWN"),
            "original_class": _clean_char_class(a),
            "source_class": _clean_char_class(b),
            "left_shape": _clean_generalized_shape(left),
            "right_shape": _clean_generalized_shape(right),
            "position_class": position_class,
        }

        out.append(
            _memory_key([
                typ,
                item["original_name"],
                item["source_name"],
                item["original_class"],
                item["source_class"],
                item["left_shape"],
                item["right_shape"],
                item["position_class"],
            ])
        )

    return out

def _memory_key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def find_learned_pattern(final_data, item, memory):
    """
    Consume ONLY the CLEAN Stage 09 schema.

    Important:
      * no hard-coded OCR correction dictionary
      * no individual glyph substitution rules
      * exact reuse may reuse a previously human-verified value
      * generalized/structural reuse only acts when Stage 09 has at least
        two consistent trusted human examples
      * paragraph and table-cell learning remain separate
    """
    original = normalize_text(item.get("original_text", ""))
    source = normalize_text(item.get("source_text", ""))
    if not original or not source:
        return None

    typ = _clean_learning_object_type(item)
    context = _clean_learning_context(item)

    # 1. EXACT CLEAN-STAGE-09 PATTERN
    exact_key = _memory_key([original, source])
    exact = memory.get("exact_patterns", {}).get(exact_key)
    if isinstance(exact, dict) and exact.get("human_confirmed"):
        value = normalize_text(exact.get("verified_text", ""))
        if value:
            return {
                "type": "EXACT_CASE_REUSED",
                "verified_text": value,
                "repeat_count": int(exact.get("repeat_count", 0) or 0),
                "reason": "Exact OCR/source evidence was previously human-confirmed by the learning memory.",
            }

    # 2. GENERALIZED TRANSFORMATION — same schema as CLEAN Stage 09
    gt = _clean_generalized_transformation(original, source)
    if gt is not None:
        gkey = _memory_key((typ, context, gt))
        entry = memory.get("generalized_patterns", {}).get(gkey)
        if isinstance(entry, dict):
            count = int(entry.get("repeat_count", 0) or 0)
            action = entry.get("action")
            if count >= 2 and action == "USE_CURRENT_ORIGINAL":
                return {
                    "type": "LEARNED_GENERALIZED_TRANSFORMATION",
                    "verified_text": original,
                    "repeat_count": count,
                    "reason": "The same generalized OCR transformation was repeatedly human-confirmed as the original value.",
                }
            if count >= 2 and action == "USE_CURRENT_SOURCE":
                return {
                    "type": "LEARNED_GENERALIZED_TRANSFORMATION",
                    "verified_text": source,
                    "repeat_count": count,
                    "reason": "The same generalized OCR transformation was repeatedly human-confirmed as the source value.",
                }

    # 3. CROSS-CONTEXT GENERALIZED PATTERN
    # This is intentionally checked before context-specific structural
    # matching. Stage 09 FIX24 creates these only from >=3 trusted human
    # examples spanning >=2 documents with unanimous human action.
    if gt is not None:
        cgkey = _memory_key((typ, gt))
        entry = memory.get("cross_context_generalized_patterns", {}).get(cgkey)
        if isinstance(entry, dict):
            count = int(entry.get("repeat_count", 0) or 0)
            docs = int(entry.get("document_count", 0) or 0)
            action = entry.get("action")
            if count >= 3 and docs >= 2 and action == "USE_CURRENT_ORIGINAL":
                return {
                    "type": "LEARNED_CROSS_CONTEXT_TRANSFORMATION",
                    "verified_text": original,
                    "repeat_count": count,
                    "reason": "The same generalized OCR transformation was repeatedly human-confirmed across multiple documents as the original value.",
                }
            if count >= 3 and docs >= 2 and action == "USE_CURRENT_SOURCE":
                return {
                    "type": "LEARNED_CROSS_CONTEXT_TRANSFORMATION",
                    "verified_text": source,
                    "repeat_count": count,
                    "reason": "The same generalized OCR transformation was repeatedly human-confirmed across multiple documents as the source value.",
                }

    # 4. FIX25 LOCAL TRANSFORMATION PATTERN
    # A local Unicode-level behavior can generalize across otherwise
    # different strings. Stage 09 only stores this when it has >=3 trusted
    # human examples from >=2 documents with unanimous action.
    local_patterns = memory.get("local_transformation_patterns", {})
    local_matches = []
    for lk in _local_transformation_pattern_keys(original, source, typ):
        entry = local_patterns.get(lk)
        if isinstance(entry, dict) and entry.get("human_confirmed"):
            count = int(entry.get("repeat_count", 0) or 0)
            docs = int(entry.get("document_count", 0) or 0)
            if count >= 3 and docs >= 2:
                local_matches.append(entry)

    if local_matches:
        actions = {e.get("action") for e in local_matches}
        if len(actions) == 1:
            action = next(iter(actions))
            count = min(int(e.get("repeat_count", 0) or 0) for e in local_matches)
            if action == "USE_CURRENT_ORIGINAL":
                return {
                    "type": "LEARNED_LOCAL_TRANSFORMATION",
                    "verified_text": original,
                    "repeat_count": count,
                    "reason": "A local Unicode-level OCR transformation was repeatedly human-confirmed across multiple documents as the original value.",
                }
            if action == "USE_CURRENT_SOURCE":
                return {
                    "type": "LEARNED_LOCAL_TRANSFORMATION",
                    "verified_text": source,
                    "repeat_count": count,
                    "reason": "A local Unicode-level OCR transformation was repeatedly human-confirmed across multiple documents as the source value.",
                }

    # 5. STRUCTURAL PATTERN — same schema as CLEAN Stage 09
    original_shape = _clean_generalized_shape(original)
    source_shape = _clean_generalized_shape(source)
    skey = _memory_key((typ, context, original_shape, source_shape))
    structural = memory.get("structural_patterns", {}).get(skey)
    if isinstance(structural, dict):
        count = int(structural.get("repeat_count", 0) or 0)
        action = structural.get("action")
        if count >= 2 and action == "USE_CURRENT_ORIGINAL":
            return {
                "type": "LEARNED_STRUCTURAL_CONTEXT",
                "verified_text": original,
                "repeat_count": count,
                "reason": "The same generalized structure and context were repeatedly human-confirmed as the original value.",
            }
        if count >= 2 and action == "USE_CURRENT_SOURCE":
            return {
                "type": "LEARNED_STRUCTURAL_CONTEXT",
                "verified_text": source,
                "repeat_count": count,
                "reason": "The same generalized structure and context were repeatedly human-confirmed as the source value.",
            }

    # 5. CROSS-CONTEXT STRUCTURAL PATTERN
    skey2 = _memory_key((typ, original_shape, source_shape))
    structural2 = memory.get("cross_context_structural_patterns", {}).get(skey2)
    if isinstance(structural2, dict):
        count = int(structural2.get("repeat_count", 0) or 0)
        docs = int(structural2.get("document_count", 0) or 0)
        action = structural2.get("action")
        if count >= 3 and docs >= 2 and action == "USE_CURRENT_ORIGINAL":
            return {
                "type": "LEARNED_CROSS_CONTEXT_STRUCTURAL",
                "verified_text": original,
                "repeat_count": count,
                "reason": "The same generalized OCR structure was repeatedly human-confirmed across multiple documents as the original value.",
            }
        if count >= 3 and docs >= 2 and action == "USE_CURRENT_SOURCE":
            return {
                "type": "LEARNED_CROSS_CONTEXT_STRUCTURAL",
                "verified_text": source,
                "repeat_count": count,
                "reason": "The same generalized OCR structure was repeatedly human-confirmed across multiple documents as the source value.",
            }

    return None


# ============================================================
# APPLY FINAL VALUE
# ============================================================

def apply_final_value(
    target,
    verified_text,
    decision,
    original,
    source,
):
    if target is None:
        return

    target[
        "final_verified_text"
    ] = verified_text

    target[
        "final_verification_status"
    ] = decision

    target[
        "manual_review_original_ocr"
    ] = original

    target[
        "manual_review_source_ocr"
    ] = source


# ============================================================
# DECISION RECORD
# ============================================================

def make_decision_record(
    item,
    verified_text,
    decision,
    previous_status="",
    learning_type=None,
    learning_fingerprint=None,
):
    record = dict(
        item
    )

    record[
        "verified_text"
    ] = verified_text

    record[
        "decision"
    ] = decision

    record[
        "previous_status"
    ] = previous_status

    record[
        "timestamp_utc"
    ] = now_utc()

    if learning_type:

        record[
            "learning"
        ] = {
            "type":
                learning_type,
            "fingerprint":
                learning_fingerprint,
        }

    return record


# ============================================================
# DISPLAY
# ============================================================

def display_review_item(
    index,
    total,
    item,
):
    print()
    print("=" * 80)

    print(
        f"REVIEW ITEM "
        f"{index} OF {total}"
    )

    print("=" * 80)

    print(
        f"Document : "
        f"{item.get('document', '')}"
    )

    print(
        f"Page     : "
        f"{item.get('page', '')}"
    )

    has_table_coordinates = (
        item.get("table") not in (None, "", 0, "0")
        and item.get("row") not in (None, "", 0, "0")
        and item.get("column") not in (None, "", 0, "0")
    )

    if has_table_coordinates:

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

    original = str(
        item.get(
            "original_text",
            "",
        )
    )

    source = str(
        item.get(
            "source_text",
            "",
        )
    )

    print()
    print(
        "ORIGINAL OCR:"
    )

    print(
        original
    )

    print()
    print(
        "SOURCE / REFERENCE:"
    )

    print(
        source
    )

    verification = item.get(
        "verification",
        {},
    )

    print()
    print(
        "VERIFICATION DECISION:"
    )

    print(
        verification.get(
            "decision",
            "",
        )
    )

    print()
    print(
        "REASON:"
    )

    print(
        verification.get(
            "reason",
            "",
        )
    )

    confidence = verification.get(
        "confidence"
    )

    if confidence is not None:

        print()
        print(
            f"Confidence: "
            f"{confidence}"
        )

    special_chars = (
        detect_special_characters(
            original
        )
    )

    if special_chars:

        print()
        print(
            "DYNAMICALLY DETECTED "
            "SPECIAL CHARACTERS:"
        )

        for entry in special_chars:

            print(
                f"  {entry['character']} "
                f"{entry['unicode']} "
                f"{entry['name']}"
            )

    crop = item.get(
        "review_crop",
        "",
    )

    if crop:

        print()
        print(
            "SOURCE REVIEW CROP:"
        )

        print(
            crop
        )

    print()
    print("-" * 80)

    print(
        "[O] Keep ORIGINAL OCR"
    )

    print(
        "[S] Use SOURCE text"
    )

    print(
        "[C] Enter CORRECT text"
    )

    print(
        "[Q] Quit and save progress"
    )

    print("-" * 80)


# ============================================================
# ACTIVE DOCUMENTS / STALE REVIEW CLEANUP
# ============================================================

def get_active_document_names():
    """Return document stems for PDFs currently present in input/."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    return {
        path.stem
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    }


def get_active_review_files(active_document_names):
    """Return only review JSON files belonging to currently active PDFs.

    IMPORTANT: this function is deliberately NON-DESTRUCTIVE. Generated
    review JSON files are evidence/audit artifacts and must never be deleted
    merely because their PDF is not currently active. Stale files are simply
    ignored by the active UI. This prevents Stage 08 from destroying the
    Stage 06 review queue and also preserves prior review history.
    """
    active_document_names = {
        str(name).strip()
        for name in (active_document_names or set())
        if str(name).strip()
    }

    review_files = []

    if not REVIEW_DIR.exists():
        return review_files

    for review_file in sorted(REVIEW_DIR.glob("*.json")):
        active = review_file.stem in active_document_names

        if not active:
            try:
                review_data = load_json(review_file)
                review_document = (
                    str(review_data.get("document", "")).strip()
                    if isinstance(review_data, dict) else ""
                )
                active = review_document in active_document_names
            except Exception:
                active = False

        if active:
            review_files.append(review_file)

    return review_files


# ============================================================


# ============================================================
# VISUAL REVIEW HELPERS
# ============================================================

def _visual_page_image(document_name, page_number):
    """Resolve the exact rendered source page used by the OCR pipeline.

    No page +/- guessing is used. The path is derived only from the
    pipeline's standard output/pages/<document>/page_NNNN.png convention.
    """
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        return None

    candidates = [
        BASE_DIR / "output" / "pages" / str(document_name) / f"page_{page:04d}.png",
        BASE_DIR / "output" / "pages" / str(document_name) / f"page_{page:03d}.png",
        BASE_DIR / "output" / "pages" / str(document_name) / f"page_{page}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _visual_points_to_box(points):
    if not isinstance(points, (list, tuple)) or not points:
        return None
    if len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
        return [float(points[0]), float(points[1]), float(points[2]), float(points[3])]
    pts=[]
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try: pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError): pass
        elif isinstance(p, dict):
            try: pts.append((float(p.get("x")), float(p.get("y"))))
            except (TypeError, ValueError): pass
    if not pts:
        return None
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _visual_bbox(item):
    bbox = item.get("bbox")
    if isinstance(bbox, dict):
        keys=("x1","y1","x2","y2")
        if all(k in bbox for k in keys):
            try: return [float(bbox[k]) for k in keys]
            except (TypeError, ValueError): pass
        keys=("left","top","right","bottom")
        if all(k in bbox for k in keys):
            try: return [float(bbox[k]) for k in keys]
            except (TypeError, ValueError): pass
        for k in ("points","polygon","box"):
            if k in bbox:
                out=_visual_points_to_box(bbox[k])
                if out is not None: return out
    return _visual_points_to_box(bbox)


def _visual_any_bbox(item, final_data=None):
    """Return the best available stored OCR bbox for APPROXIMATE display.

    This is intentionally broader than the exact-resolution path. Exact
    verification still uses _visual_resolve_item_bbox(); this helper only
    answers: "does the data contain any numeric OCR region that we can draw
    in red so the reviewer can see what the OCR system attempted to review?"
    """
    candidates = []
    if isinstance(item, dict):
        candidates.extend([
            item.get("bbox"),
            item.get("source_bbox"),
            item.get("ocr_bbox"),
            item.get("stored_bbox"),
            item.get("canonical_bbox"),
        ])
        verification = item.get("verification")
        if isinstance(verification, dict):
            candidates.extend([
                verification.get("bbox"),
                verification.get("source_bbox"),
                verification.get("ocr_bbox"),
            ])
    if isinstance(final_data, dict):
        try:
            target, _ = find_target(final_data, item)
        except Exception:
            target = None
        if isinstance(target, dict):
            candidates.extend([
                target.get("bbox"),
                target.get("source_bbox"),
                target.get("ocr_bbox"),
                target.get("stored_bbox"),
                target.get("canonical_bbox"),
            ])

    for candidate in candidates:
        box = _visual_bbox({"bbox": candidate})
        if box is None:
            box = _visual_points_to_box(candidate)
        if box is None:
            continue
        try:
            vals = [float(v) for v in box]
            if all(v == v and abs(v) != float("inf") for v in vals):
                return vals
        except (TypeError, ValueError):
            continue
    return None


def _visual_bbox_is_suspicious(item, bbox, image_size):
    """Detect an implausible paragraph bbox before presenting it as exact."""
    if bbox is None or image_size is None or "paragraph_block" not in item:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        page_w, page_h = [float(v) for v in image_size]
    except (TypeError, ValueError):
        return False
    width = max(0.0, x2-x1)
    height = max(0.0, y2-y1)
    if width <= 0 or height <= 0 or page_w <= 0 or page_h <= 0:
        return True
    text = normalize_text(item.get("original_text", ""))
    near_full_height = height >= page_h * 0.40
    extreme_aspect = height / max(width, 1.0) >= 12.0
    short_text = len(text) <= 160
    return bool(near_full_height and extreme_aspect and short_text)


def _visual_resolve_item_bbox(final_data, item, image_size=None):
    candidates = []
    direct = _visual_bbox(item)
    if direct is not None:
        candidates.append(direct)
    try:
        target, _ = find_target(final_data, item)
    except Exception:
        target = None
    if isinstance(target, dict):
        target_bbox = _visual_bbox(target)
        if target_bbox is not None:
            candidates.append(target_bbox)
    for candidate in candidates:
        if not _visual_bbox_is_suspicious(item, candidate, image_size):
            return candidate
    return None


def _visual_scale_box(box, image_size):
    if box is None:
        return None
    w,h=image_size
    x1,y1,x2,y2=box
    # Normalized coordinates are accepted only when every coordinate is in [0,1].
    if all(0.0 <= v <= 1.0 for v in box):
        x1,x2=x1*w,x2*w
        y1,y2=y1*h,y2*h
    x1=max(0,min(w,x1)); x2=max(0,min(w,x2))
    y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
    if x2 < x1: x1,x2=x2,x1
    if y2 < y1: y1,y2=y2,y1
    return [int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))]


def _visual_mark(image, item):
    img=image.convert("RGB").copy()
    box=_visual_scale_box(_visual_bbox(item), img.size)
    if box is None:
        return img, None
    draw=ImageDraw.Draw(img)
    draw.rectangle(box, outline="red", width=max(3, min(img.size)//250))
    return img, box


def _visual_crop(image, box, margin=30):
    """Return the exact review crop, rotated for readability when needed.

    The page image remains untouched.  If the selected OCR region is a
    tall/narrow region (for example vertically oriented header/footer text),
    the review crop is rotated 90 degrees so the reviewer can read it
    horizontally.  No OCR text or coordinates are changed by this display
    transformation.
    """
    if box is None:
        return None
    w,h=image.size
    x1=max(0,box[0]-margin); y1=max(0,box[1]-margin)
    x2=min(w,box[2]+margin); y2=min(h,box[3]+margin)
    if x2<=x1 or y2<=y1: return None

    out=image.crop((x1,y1,x2,y2))

    # Rotate only clearly vertical review regions.  Normal paragraph/table
    # crops are left unchanged.
    if out.height > out.width * 1.25:
        out=out.rotate(270, expand=True)

    # Enlarge narrow OCR regions for human inspection. This affects only
    # the review display; source pixels and coordinates remain unchanged.
    if out.width < 900 or out.height < 180:
        scale = min(4.0, max(1.5, 900.0 / max(1, out.width)))
        out = out.resize(
            (int(out.width * scale), int(out.height * scale)),
            Image.Resampling.LANCZOS
        )

    return out


def _visual_crop_was_rotated(box, image_size, margin=30):
    """Return whether the displayed review crop will be rotated."""
    if box is None:
        return False
    w,h=image_size
    x1=max(0,box[0]-margin); y1=max(0,box[1]-margin)
    x2=min(w,box[2]+margin); y2=min(h,box[3]+margin)
    return (x2 > x1 and y2 > y1 and (y2-y1) > (x2-x1)*1.25)


def _visual_record_review_outcome(document_name, item, outcome,
                                  verified_text=None, source="STAGE08"):
    """Persist an audit-only Stage 08 review outcome.

    Review outcomes are separate from trusted human-learning decisions.
    AUTO_* outcomes are never treated as human evidence by Stage 09.
    USER_SKIPPED is persisted for audit/counting but does not modify OCR.
    """
    outcomes = []
    if DECISIONS_FILE.exists():
        try:
            data = load_json(DECISIONS_FILE)
            if isinstance(data, dict) and isinstance(data.get("review_outcomes"), list):
                outcomes = data.get("review_outcomes", [])
        except Exception:
            outcomes = []

    original = str(item.get("original_text", ""))
    source_text = str(item.get("source_text", ""))
    record = {
        "timestamp_utc": now_utc(),
        "document": item.get("document", document_name),
        "page": item.get("page", ""),
        "object_type": object_type(item),
        "table": item.get("table"),
        "row": item.get("row"),
        "column": item.get("column"),
        "paragraph_block": item.get("paragraph_block"),
        "bbox": item.get("bbox"),
        "original_text": original,
        "source_text": source_text,
        "outcome": outcome,
        "verified_text": verified_text,
        "source": source,
    }

    # One outcome per exact review item + outcome type. If the same item is
    # revisited later, the newest record replaces the older identical outcome.
    def outcome_key(x):
        return (
            str(x.get("document", "")).strip(),
            str(x.get("page", "")).strip(),
            str(x.get("table", "")).strip(),
            str(x.get("row", "")).strip(),
            str(x.get("column", "")).strip(),
            str(x.get("paragraph_block", "")).strip(),
            str(x.get("bbox", "")).strip(),
            str(x.get("original_text", "")).strip(),
            str(x.get("source_text", "")).strip(),
            str(x.get("outcome", "")).strip(),
        )

    key = outcome_key(record)
    outcomes = [x for x in outcomes if not (isinstance(x, dict) and outcome_key(x) == key)]
    outcomes.append(record)

    data = {}
    if DECISIONS_FILE.exists():
        try:
            loaded = load_json(DECISIONS_FILE)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    decisions = data.get("decisions", [])
    if not isinstance(decisions, list):
        decisions = []
    save_json(
        DECISIONS_FILE,
        {
            "total_items": len(decisions),
            "review_outcomes": outcomes,
            "decisions": decisions,
        },
    )
    return record


def _persist_decision_data(existing_decisions):
    """Save decisions without discarding the Stage 08 review-outcome ledger."""
    data = {}
    if DECISIONS_FILE.exists():
        try:
            loaded = load_json(DECISIONS_FILE)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    data["total_items"] = len(existing_decisions)
    data["decisions"] = existing_decisions
    data.setdefault("review_outcomes", [])
    save_json(DECISIONS_FILE, data)


def _append_memory_audit(memory, record):
    """Append a Stage 08 audit event without corrupting Stage 09 audit schema.

    Stage 09 stores `audit` as a summary dictionary, whereas older Stage 08
    versions expected it to be a list. Keep the Stage 09 summary intact and
    store incremental UI events in a dedicated list.
    """
    if not isinstance(memory, dict):
        return
    audit = memory.get("audit")
    if isinstance(audit, list):
        audit.append(record)
        return
    if isinstance(audit, dict):
        events = audit.setdefault("stage08_events", [])
        if isinstance(events, list):
            events.append(record)
        return
    memory["audit"] = {"stage08_events": [record]}


def _stable_review_location_key(record):
    """Stable identity for a reviewable object, independent of mutable OCR text/bbox."""
    if not isinstance(record, dict):
        return None
    document = str(record.get("document", "")).strip()
    page = _canonical_evidence_page(record.get("page", ""))
    obj = object_type(record)
    if obj == "table_cell":
        table = str(record.get("table", "")).strip()
        row = str(record.get("row", "")).strip()
        column = str(record.get("column", "")).strip()
        if document and page and table and row and column:
            return ("TABLE", document, page, table, row, column)
    if obj == "paragraph_block":
        block = str(record.get("paragraph_block", "")).strip()
        if document and page and block:
            return ("PARAGRAPH", document, page, block)
    return None


def _visual_record_human_decision(item, final_data, review_file, verified_file, document_name,
                                  existing_decisions, existing_lookup,
                                  same_content_lookup, memory,
                                  verified_text, decision):
    """Persist a human decision only after the canonical target is updated.

    This is deliberately transactional: if the current review item cannot be
    resolved to a canonical object, no decision, human-evidence record,
    learning-memory entry, or UI human-action count may be created.
    """
    target, target_type = find_target(final_data, item)
    if target is None:
        return None

    original = str(item.get("original_text", ""))
    source = str(item.get("source_text", ""))

    apply_final_value(
        target,
        verified_text,
        decision,
        original,
        source,
    )

    # Verify the mutation before writing any audit/learning/decision data.
    if (
        str(target.get("final_verified_text", "")) != str(verified_text)
        or str(target.get("final_verification_status", "")) != str(decision)
    ):
        return None
    source = str(item.get("source_text", ""))
    fingerprint = build_learning_fingerprint(final_data, item)
    decision_record = make_decision_record(
        item, verified_text, decision,
        previous_status=item.get("verification", {}).get("decision", ""),
        learning_type=("HUMAN_CONFIRMED_PATTERN" if fingerprint else None),
        learning_fingerprint=fingerprint,
    )
    existing_decisions.append(decision_record)
    existing_lookup[decision_key(item)] = decision_record

    human_evidence_record = {
        "evidence_type": "HUMAN_CONFIRMED_EVIDENCE",
        "document": item.get("document", document_name),
        "page": item.get("page", ""),
        "object_type": object_type(item),
        "bbox": item.get("bbox"),
        "paragraph_block": item.get("paragraph_block"),
        "table": item.get("table"),
        "row": item.get("row"),
        "column": item.get("column"),
        "original_text": original,
        "source_text": source,
        "verified_text": verified_text,
        "human_decision": decision,
        "timestamp_utc": now_utc(),
    }

    evidence_key=(
        str(human_evidence_record.get("document", "")).strip(),
        _canonical_evidence_page(human_evidence_record.get("page", "")),
        _canonical_evidence_text(original),
        _canonical_evidence_text(source),
    )
    replaced=False
    for i, existing_record in enumerate(existing_decisions):
        if not isinstance(existing_record, dict): continue
        if existing_record.get("evidence_type") != "HUMAN_CONFIRMED_EVIDENCE": continue
        existing_key=(
            str(existing_record.get("document", "")).strip(),
            _canonical_evidence_page(existing_record.get("page", "")),
            _canonical_evidence_text(existing_record.get("original_text", "")),
            _canonical_evidence_text(existing_record.get("source_text", "")),
        )
        if existing_key == evidence_key:
            existing_decisions[i]=human_evidence_record
            replaced=True
            break
    if not replaced:
        existing_decisions.append(human_evidence_record)

    same_content_lookup[_exact_evidence_key(
        item.get("document", document_name), item.get("page", ""), original, source
    )]=human_evidence_record

    _append_memory_audit(memory, {
        "timestamp_utc": now_utc(),
        "document": item.get("document", document_name),
        "page": item.get("page", ""),
        "object_type": object_type(item),
        "original_text": original,
        "source_text": source,
        "verified_text": verified_text,
        "decision": decision,
        "fingerprint": fingerprint,
    })

    # IMPORTANT: the active review queue is rebuilt from VERIFIED_DIR on every
    # Streamlit rerun.  Saving only to FINAL_DIR makes the correction appear to
    # succeed for the current process, but the next rerun reloads the old
    # unresolved JSON and shows the same item again.  Persist the mutation to
    # the exact canonical verified JSON that _visual_main() loaded.
    canonical_verified_file = Path(verified_file) if verified_file is not None else (
        VERIFIED_DIR / f"{review_file.stem}.json"
    )
    save_json(canonical_verified_file, final_data)

    # Keep the Stage 08 final-output mirror as well, if configured.
    final_output_file=FINAL_DIR / f"{review_file.stem}.json"
    if final_output_file.resolve() != canonical_verified_file.resolve():
        save_json(final_output_file, final_data)

    save_memory(memory)
    _persist_decision_data(existing_decisions)
    return final_data


def _visual_load_decision_state():
    existing_decisions=[]
    if DECISIONS_FILE.exists():
        try:
            data=load_json(DECISIONS_FILE)
            existing_decisions=data.get("decisions", []) if isinstance(data,dict) else []
            if not isinstance(existing_decisions,list): existing_decisions=[]
        except Exception:
            existing_decisions=[]

    # Compatibility: if an older Stage 08 ledger still exists under
    # verified_data, merge records that are not already in the authoritative
    # learning ledger. Never let the legacy mirror replace cumulative history.
    if LEGACY_DECISIONS_FILE.exists():
        try:
            legacy = load_json(LEGACY_DECISIONS_FILE)
            legacy_decisions = legacy.get("decisions", []) if isinstance(legacy, dict) else []
            if isinstance(legacy_decisions, list):
                seen = {json.dumps(x, ensure_ascii=False, sort_keys=True, default=str) for x in existing_decisions if isinstance(x, dict)}
                for rec in legacy_decisions:
                    if not isinstance(rec, dict):
                        continue
                    sig = json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str)
                    if sig not in seen:
                        existing_decisions.append(rec)
                        seen.add(sig)
        except Exception:
            pass

    existing_lookup={}
    same_content_lookup={}
    for d in existing_decisions:
        if not isinstance(d,dict): continue

        # Only decisions that actually resolve an item may be reused by
        # location. Historical REVIEW records are evidence that an item
        # still required verification; they must never make the item appear
        # resolved, and must never be copied back into final_verified.json.
        kind=_decision_kind(d)
        verified=_decision_verified(d)
        is_review_status=(
            "REVIEW" in kind
            or kind in {
                "SOURCE_DISAGREEMENT",
                "NUMERIC_DISAGREEMENT",
                "SOURCE_EMPTY",
                "NO_SOURCE_IMAGE",
            }
        )
        # Only genuine human evidence may resolve a current review item.
        # A historical automatic/session record may contain a verified value,
        # but it is NOT permission to skip source-image human review.
        reusable=(
            not is_review_status
            and _is_human_evidence_decision(d)
            and bool(verified or _reuse_exact_evidence_value(d, _decision_original(d), _decision_source(d)))
        )

        if reusable:
            existing_lookup[decision_key(d)]=d
            stable_key = _stable_review_location_key(d)
            if stable_key is not None:
                existing_lookup[("STABLE_LOCATION",) + stable_key]=d

        if _is_human_evidence_decision(d):
            o=_decision_original(d); s=_decision_source(d)
            if o and s and _reuse_exact_evidence_value(d,o,s) is not None:
                same_content_lookup[_exact_evidence_key(_decision_document(d),_decision_page(d),o,s)]=d
    return existing_decisions, existing_lookup, same_content_lookup


def _visual_process_auto_item(item, final_data, document_name, review_file,
                              existing_decisions, existing_lookup,
                              same_content_lookup, memory):
    """Run the original Stage 08 automatic reuse sequence for ONE item.

    Returns (handled, final_data, kind). If handled=False the item must be
    shown to a human. The sequence is intentionally kept the same as the
    original main(): location decision -> exact evidence -> visual evidence
    -> legacy fallback -> global exact -> learned pattern.
    """
    item=dict(item)
    item.setdefault("document", document_name)
    if not item.get("document"): item["document"]=document_name
    key=decision_key(item)
    original=str(item.get("original_text", ""))
    source=str(item.get("source_text", ""))

    # A current REVIEW status does NOT override a prior trusted human decision.
    # The current JSON can legitimately still carry REVIEW because an earlier
    # Stage 08 version failed to persist the human result there. Check stable
    # historical human evidence first.
    stable_key = _stable_review_location_key(item)
    previous = existing_lookup.get(("STABLE_LOCATION",) + stable_key) if stable_key is not None else None
    if previous is None:
        previous = existing_lookup.get(key)

    if previous is not None:
        previous_kind=_human_verification_kind(previous)
        previous_value=_decision_verified(previous)
        if not previous_value:
            if previous_kind == "USER_CONFIRMED_ORIGINAL": previous_value=_decision_original(previous)
            elif previous_kind == "USER_CONFIRMED_SOURCE": previous_value=_decision_source(previous)

        # A historical REVIEW/DISAGREEMENT record is evidence that the item
        # needed review, not a resolution. It must never satisfy the current
        # human-review queue merely because a record exists at the same key.
        # Existing location records are reusable only when they contain
        # actual human evidence. Automatic/session reuse records must never
        # make a current review candidate disappear.
        reusable = (
            bool(previous_value)
            and _is_human_evidence_decision(previous)
            and "REVIEW" not in previous_kind
            and "DISAGREEMENT" not in previous_kind
        )

        if reusable:
            # High-risk current items must not be silently closed by a historical
            # decision. If the canonical cell has not already been explicitly
            # verified in its current final status, leave it in the human queue.
            target,_=find_target(final_data,item)
            current_verification = str((target or {}).get("verification_status", "") or "").strip().upper()
            current_final = str((target or {}).get("final_verification_status", "") or "").strip().upper()
            is_high_risk = "HIGH_RISK" in current_verification or "HIGH_RISK" in str(item.get("verification_status", "") or "").upper()
            explicit_current_human = current_final.startswith(("USER_CONFIRMED", "USER_ENTERED", "HUMAN_CONFIRMED", "MANUAL_CONFIRMED"))
            if is_high_risk and not explicit_current_human:
                return False, final_data, "HIGH_RISK_REQUIRES_CURRENT_HUMAN_REVIEW"
            # Never report an automatic reuse as handled unless the exact
            # canonical object exists, is updated, and is persisted.
            if target is None:
                return False, final_data, "HUMAN_REVIEW_TARGET_NOT_FOUND"
            decision = _decision_kind(previous)
            apply_final_value(
                target, previous_value, decision,
                _decision_original(previous) or original,
                _decision_source(previous) or source,
            )
            if (
                str(target.get("final_verified_text", "")) != str(previous_value)
                or str(target.get("final_verification_status", "")) != str(decision)
            ):
                return False, final_data, "HUMAN_REVIEW_TARGET_UPDATE_FAILED"
            canonical_file = VERIFIED_DIR / f"{review_file.stem}.json"
            save_json(canonical_file, final_data)
            mirror_file = FINAL_DIR / f"{review_file.stem}.json"
            if mirror_file.resolve() != canonical_file.resolve():
                save_json(mirror_file, final_data)
            _visual_record_review_outcome(document_name,item,"AUTO_EXISTING_DECISION",previous_value,"AUTO")
            return True, final_data, "EXISTING_DECISION"

        # SAFETY GUARD:
    # High-risk items must not be automatically resolved by exact-evidence
    # reuse or learned-pattern reuse. Require explicit human confirmation
    # on the current canonical target before any automatic reuse can proceed.
    target, _ = find_target(final_data, item)

    current_verification = str(
        (target or {}).get("verification_status", "") or ""
    ).strip().upper()

    current_final = str(
        (target or {}).get("final_verification_status", "") or ""
    ).strip().upper()

    item_verification = str(
        item.get("verification_status", "") or ""
    ).strip().upper()

    is_high_risk = (
        "HIGH_RISK" in current_verification
        or "HIGH_RISK" in item_verification
    )

    explicit_current_human = current_final.startswith(
        (
            "USER_CONFIRMED",
            "USER_ENTERED",
            "HUMAN_CONFIRMED",
            "MANUAL_CONFIRMED",
        )
    )

    if is_high_risk and (
        target is None or not explicit_current_human
    ):
        return False, final_data, "HIGH_RISK_REQUIRES_CURRENT_HUMAN_REVIEW"

    content_key=_exact_evidence_key(item.get("document",document_name), item.get("page",""), original, source)
    same=same_content_lookup.get(content_key)
    if same is None:
        same=_find_exact_human_evidence(existing_decisions,item.get("document",document_name),item.get("page",""),original,source)
        if same is not None: same_content_lookup[content_key]=same
    if same is None:
        same=_find_human_evidence_by_original_and_bbox(existing_decisions,item)
        if same is not None: same_content_lookup[content_key]=same
    if same is None:
        same=_find_legacy_human_evidence_by_unique_original(existing_decisions,item)
        if same is not None: same_content_lookup[content_key]=same
    if same is None:
        same=_find_global_exact_human_evidence(existing_decisions,original,source)
        if same is not None: same_content_lookup[content_key]=same
    if same is None:
        same=_find_unambiguous_human_confirmation_by_exact_text(
            existing_decisions, original
        )

    if same is not None:
        learned_value=_reuse_exact_evidence_value(same,original,source)
        if learned_value is not None:
            learned_type="EXACT_EVIDENCE_REUSED"
            target,_=find_target(final_data,item)
            if target is None:
                return False, final_data, "HUMAN_REVIEW_TARGET_NOT_FOUND"
            apply_final_value(target,learned_value,learned_type,original,source)
            if (
                str(target.get("final_verified_text", "")) != str(learned_value)
                or str(target.get("final_verification_status", "")) != str(learned_type)
            ):
                return False, final_data, "HUMAN_REVIEW_TARGET_UPDATE_FAILED"
            fingerprint=build_learning_fingerprint(final_data,item)
            rec=make_decision_record(item,learned_value,learned_type,learning_type=learned_type,learning_fingerprint=fingerprint)
            existing_decisions.append(rec); existing_lookup[key]=rec
            _append_memory_audit(memory, {"timestamp_utc":now_utc(),"document":item.get("document",document_name),"page":item.get("page",""),"object_type":object_type(item),"decision":learned_type,"original_text":original,"source_text":source,"verified_text":learned_value,"reused_from_decision":_human_verification_kind(same),"fingerprint":fingerprint})
            canonical_file = VERIFIED_DIR / f"{review_file.stem}.json"
            save_json(canonical_file, final_data)
            mirror_file = FINAL_DIR / f"{review_file.stem}.json"
            if mirror_file.resolve() != canonical_file.resolve():
                save_json(mirror_file, final_data)
            save_memory(memory)
            _persist_decision_data(existing_decisions)
            _visual_record_review_outcome(document_name,item,"AUTO_EXACT_EVIDENCE_REUSED",learned_value,"AUTO")
            return True, final_data, learned_type

    learned=find_learned_pattern(final_data,item,memory)
    if learned is not None:
        learned_value=learned["verified_text"]; learned_type=learned["type"]
        target,_=find_target(final_data,item)
        if target is None:
            return False, final_data, "HUMAN_REVIEW_TARGET_NOT_FOUND"
        apply_final_value(target,learned_value,learned_type,original,source)
        if (
            str(target.get("final_verified_text", "")) != str(learned_value)
            or str(target.get("final_verification_status", "")) != str(learned_type)
        ):
            return False, final_data, "HUMAN_REVIEW_TARGET_UPDATE_FAILED"
        fingerprint=build_learning_fingerprint(final_data,item)
        rec=make_decision_record(item,learned_value,learned_type,learning_type=learned_type,learning_fingerprint=fingerprint)
        existing_decisions.append(rec); existing_lookup[key]=rec
        _append_memory_audit(memory, {"timestamp_utc":now_utc(),"document":item.get("document",document_name),"page":item.get("page",""),"object_type":object_type(item),"decision":learned_type,"original_text":original,"source_text":source,"verified_text":learned_value,"fingerprint":fingerprint})
        canonical_file = VERIFIED_DIR / f"{review_file.stem}.json"
        save_json(canonical_file, final_data)
        mirror_file = FINAL_DIR / f"{review_file.stem}.json"
        if mirror_file.resolve() != canonical_file.resolve():
            save_json(mirror_file, final_data)
        save_memory(memory)
        _persist_decision_data(existing_decisions)
        _visual_record_review_outcome(document_name,item,"AUTO_LEARNED_PATTERN",learned_value,"AUTO")
        return True, final_data, learned_type

    return False, final_data, "HUMAN_REVIEW"


def _visual_active_documents():
    names=sorted(get_active_document_names())
    return names


def _visual_review_file_for(document_name):
    for p in get_active_review_files({document_name}):
        if p.stem == document_name:
            return p
    files=get_active_review_files({document_name})
    return files[0] if files else None


def _visual_status_is_resolved(status):
    """Return True only for statuses that explicitly mean verification is complete.

    IMPORTANT: statuses such as SOURCE_DISAGREEMENT and NUMERIC_DISAGREEMENT
    are *unresolved* even though they do not contain the word REVIEW.  The
    previous implementation treated every non-REVIEW status as resolved and
    therefore silently removed older review items from the UI.
    """
    s = str(status or "").strip().upper()
    if not s:
        return False

    resolved_exact = {
        "SOURCE_VERIFIED",
        "SOURCE_VERIFIED_CLOSE",
        "USER_CONFIRMED_ORIGINAL",
        "USER_CONFIRMED_SOURCE",
        "USER_ENTERED_CORRECTION",
        "EXACT_EVIDENCE_REUSED",
        "EXACT_CASE_REUSED",
        "CONTEXT_PATTERN_REUSED",
        "OCR_TRANSFORMATION_PATTERN_REUSED",
        "STRUCTURAL_PATTERN_REUSED",
        "LEARNED_GENERALIZED_TRANSFORMATION",
        "LEARNED_STRUCTURAL_CONTEXT",
        "AGREEMENT_PATTERN_REUSED",
        "LEARNED_CROSS_CONTEXT_TRANSFORMATION",
        "LEARNED_CROSS_CONTEXT_STRUCTURAL",
        "SESSION_LEARNED_SYMBOL",
    }
    if s in resolved_exact:
        return True
    if s.startswith("SOURCE_VERIFIED"):
        return True
    if s.startswith("USER_CONFIRMED") or s.startswith("USER_ENTERED"):
        return True
    if s.startswith("LEARNED_") or s.endswith("_REUSED"):
        return True
    return False


def _visual_item_requires_current_review(final_data, item):
    """Return True only when the canonical current target is still unresolved.

    Historical review records are retained, but their current canonical target
    is authoritative. Any explicit disagreement/review/unknown verification
    state remains visible. Nothing is silently dropped merely because the
    status string does not contain the word REVIEW.
    """
    target, _ = find_target(final_data, item)
    if target is None:
        # Never silently drop an evidence item that cannot be mapped to the
        # canonical JSON. Keep it visible for human investigation.
        return True

    final_status = str(target.get("final_verification_status", "") or "").strip().upper()
    verification_status = str(target.get("verification_status", "") or "").strip().upper()

        # High-risk OCR items remain in the review queue unless the
    # current canonical target has an explicit human-confirmation status.
    is_high_risk = (
        "HIGH_RISK" in final_status
        or "HIGH_RISK" in verification_status
    )
    explicit_current_human = final_status.startswith(
        (
            "USER_CONFIRMED",
            "USER_ENTERED",
            "HUMAN_CONFIRMED",
            "MANUAL_CONFIRMED",
        )
    )
    if is_high_risk and not explicit_current_human:
        return True

    # A human/explicitly verified final status closes the item.
    if final_status and _visual_status_is_resolved(final_status):
        return False

    # If the final status is absent, a resolved base verification status may
    # still be enough to close a legacy item.
    if not final_status and verification_status and _visual_status_is_resolved(verification_status):
        return False

    # Disagreement, REVIEW, empty-final-status, and unknown statuses remain
    # actionable. This is deliberately conservative to prevent lost reviews.
    return True


def _visual_collect_canonical_unresolved(final_data, document_name):
    """Discover unresolved canonical objects that are missing from the old review JSON.

    Stage 08 historically depended on review_required/*.json. That file is an
    audit snapshot, not the authoritative current queue. This generic walk
    supplements the historical queue with any canonical object whose current
    verification status explicitly indicates disagreement/review/uncertainty.
    It does not invent rows, columns, or text; all coordinates and text come
    directly from the canonical final JSON.
    """
    found = {}

    def walk(node, ctx):
        if isinstance(node, dict):
            local = dict(ctx)
            for key in ("page", "table", "row", "column", "paragraph_block"):
                if key in node and node.get(key) not in (None, ""):
                    local[key] = node.get(key)

            final_status = str(node.get("final_verification_status", "") or "").strip().upper()
            verification_status = str(node.get("verification_status", "") or "").strip().upper()
            status = final_status or verification_status

            # Only explicit verification states are discovered here. Blank
            # statuses are handled through the historical review queue so we
            # do not turn every ordinary OCR object into a review candidate.
            if status and not _visual_status_is_resolved(status):
                has_identity = (
                    local.get("table") not in (None, "", 0, "0")
                    and local.get("row") not in (None, "", 0, "0")
                    and local.get("column") not in (None, "", 0, "0")
                ) or local.get("paragraph_block") not in (None, "", 0, "0")

                if has_identity:
                    item = dict(node)
                    item["document"] = document_name
                    for key, value in local.items():
                        item.setdefault(key, value)
                    if not item.get("original_text"):
                        item["original_text"] = item.get("text", "")
                    if not item.get("source_text"):
                        item["source_text"] = item.get("source_ocr", "")
                    if isinstance(item.get("verification"), dict):
                        v = item["verification"]
                        item.setdefault("original_text", v.get("verified_text", ""))
                        item.setdefault("source_text", v.get("source_text", ""))
                    try:
                        key = decision_key(item)
                    except Exception:
                        key = None
                    if key is not None:
                        found[key] = item

            for value in node.values():
                walk(value, local)

        elif isinstance(node, list):
            for value in node:
                walk(value, ctx)

    walk(final_data, {"document": document_name})
    return found


def _visual_skip_key(item):
    """Return a stable, serialization-safe identity for one unresolved review item.

    The normal decision_key() is authoritative for table/paragraph locations.
    The fallback keeps Skip functional even for a legacy review item whose
    canonical location fields are incomplete.  This key is audit identity only;
    it never changes OCR content or creates learning rules.
    """
    if not isinstance(item, dict):
        return None

    try:
        key = decision_key(item)
    except Exception:
        key = None

    if key is not None and not (
        isinstance(key, tuple)
        and len(key) >= 3
        and str(key[2]) == "unknown"
    ):
        return "DECISION|" + json.dumps(
            key, ensure_ascii=False, default=str, separators=(",", ":")
        )

    fallback = {
        "document": str(item.get("document", "")).strip(),
        "page": _canonical_evidence_page(item.get("page", "")),
        "object_type": object_type(item),
        "table": str(item.get("table", "")).strip(),
        "row": str(item.get("row", "")).strip(),
        "column": str(item.get("column", "")).strip(),
        "paragraph_block": str(item.get("paragraph_block", "")).strip(),
        "original_text": _canonical_evidence_text(item.get("original_text", "")),
        "source_text": _canonical_evidence_text(item.get("source_text", "")),
        "bbox": _bbox_tuple(item.get("bbox")),
    }
    return "FALLBACK|" + json.dumps(
        fallback, ensure_ascii=False, default=str, sort_keys=True,
        separators=(",", ":")
    )


def _visual_remove_from_session_queue(queue_key_name, item):
    """Remove one resolved/skipped item from the current Streamlit queue."""
    current_queue = st.session_state.get(queue_key_name, [])
    item_key = _visual_skip_key(item)
    st.session_state[queue_key_name] = [
        queued
        for queued in current_queue
        if _visual_skip_key(queued) != item_key
    ]


def _visual_main():
    st.set_page_config(page_title="OCR Human Review", layout="wide")
    st.title("OCR Human Review")
    st.caption("Visual source-image verification using the existing Stage 08 decision, evidence, audit, and learning-memory workflow.")

    active=_visual_active_documents()
    if not active:
        st.warning("No active input PDFs were found. Generated/stale review files are ignored.")
        return

    with st.sidebar:
        st.header("Document")
        selected=st.selectbox("Active PDF", active)
        if st.button("Reload / Rescan"):
            st.rerun()
        st.divider()
        st.write("Learning memory is cumulative and is read from the existing Stage 09 memory file.")

    review_file=_visual_review_file_for(selected)
    if review_file is None:
        st.success("No review file exists for this active PDF. Nothing needs human review.")
        return

    review_data=load_json(review_file)
    if not isinstance(review_data,dict):
        st.error("Review JSON is not a valid object.")
        return
    historical_items=review_data.get("items",[])
    if not isinstance(historical_items,list): historical_items=[]
    document_name=str(review_data.get("document",review_file.stem)).strip() or review_file.stem
    verified_file=VERIFIED_DIR / f"{review_file.stem}.json"
    if not verified_file.exists():
        alt=VERIFIED_DIR / f"{document_name}.json"
        if alt.exists(): verified_file=alt
    if not verified_file.exists():
        st.error(f"Verified JSON not found: {verified_file}")
        return
    final_data=load_json(verified_file)
    if isinstance(final_data,dict) and not final_data.get("document"):
        final_data["document"]=document_name

    existing_decisions, existing_lookup, same_content_lookup=_visual_load_decision_state()
    memory=load_memory()

    # The review JSON is cumulative audit evidence, not the complete current
    # queue. Build the current queue as a UNION of:
    #   (1) historical review records that are still unresolved now, and
    #   (2) unresolved canonical objects discovered directly in final_verified.
    # This prevents older disagreements from disappearing and also catches a
    # review item that was never present in the historical snapshot.
    items_by_key = {}
    for raw in historical_items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item.setdefault("document", document_name)
        if _visual_item_requires_current_review(final_data, item):
            items_by_key[decision_key(item)] = item

    canonical_unresolved = _visual_collect_canonical_unresolved(final_data, document_name)
    for key, item in canonical_unresolved.items():
        items_by_key.setdefault(key, item)

    items = list(items_by_key.values())
    # Keep the complete unresolved candidate set separately from the active
    # queue. Skipped items are removed from the active queue, but they must
    # still be counted and offered for later re-review.
    all_unresolved_candidates = [
        dict(candidate)
        for candidate in items
        if isinstance(candidate, dict)
    ]

    # Write a separate live-queue snapshot. Do NOT overwrite review_required/
    # historical JSON: those files are retained as audit evidence. This file
    # reflects only candidates unresolved in the canonical verified JSON now.
    try:
        current_queue_dir = REVIEW_DIR / "current_queue"
        current_queue_dir.mkdir(parents=True, exist_ok=True)
        current_queue_file = current_queue_dir / f"{review_file.stem}.json"
        save_json(
            current_queue_file,
            {
                "document": document_name,
                "source_review_file": str(review_file),
                "queue_type": "CURRENT_UNRESOLVED",
                "total_review_items": len(all_unresolved_candidates),
                "items": all_unresolved_candidates,
            },
        )
    except Exception as exc:
        # Queue snapshot failure must not prevent human review.
        st.warning(f"Could not write current-queue snapshot: {exc}")

    # ------------------------------------------------------------
    # PERSISTENT SKIP STATE
    # ------------------------------------------------------------
    # Skip is an AUDIT-ONLY state. It never verifies the OCR item, changes
    # final_verified.json, or teaches Stage 09.
    #
    # The previous implementation reconstructed skip identity through
    # decision_key() in several different places. Legacy records with slightly
    # different coordinate serialization could therefore fail to match the
    # current queue.  A dedicated stable skip_key is now persisted with every
    # skip and is used for both loading and filtering.
    skipped_key_name = f"ocr_visual_skipped_v2::{document_name}"
    released_key_name = f"ocr_visual_released_skips_v2::{document_name}"
    queue_key_name = f"ocr_visual_queue_v2::{document_name}"
    queue_total_key_name = f"ocr_visual_queue_total_v2::{document_name}"
    queue_position_key_name = f"ocr_visual_queue_positions_v2::{document_name}"
    session_skips_key_name = f"ocr_visual_session_skips::{document_name}"
    skip_just_happened_key_name = f"ocr_visual_skip_just_happened::{document_name}"

    # Preserve each review candidate's ORIGINAL position even after the active
    # queue removes resolved/skipped items. This keeps the UI numbering stable:
    # item 1 of 8 -> item 2 of 8, rather than restarting at item 1 of 7.
    if queue_total_key_name not in st.session_state:
        st.session_state[queue_total_key_name] = len(all_unresolved_candidates)
    if queue_position_key_name not in st.session_state:
        positions = {}
        for original_position, raw_candidate in enumerate(all_unresolved_candidates, 1):
            if isinstance(raw_candidate, dict):
                positions[_visual_skip_key(raw_candidate)] = original_position
        st.session_state[queue_position_key_name] = positions

    if session_skips_key_name not in st.session_state:
        st.session_state[session_skips_key_name] = set()
    if skip_just_happened_key_name not in st.session_state:
        st.session_state[skip_just_happened_key_name] = False

    if released_key_name not in st.session_state:
        st.session_state[released_key_name] = set()
    released_skips = st.session_state[released_key_name]

    persisted_skips = set()

    try:
        # Dedicated skip ledger.
        if SKIPS_FILE.exists():
            skip_data = load_json(SKIPS_FILE)
            records = (
                skip_data.get("skipped_items", [])
                if isinstance(skip_data, dict)
                else []
            )

            if isinstance(records, list):
                for skipped in records:
                    if not isinstance(skipped, dict):
                        continue
                    if str(skipped.get("document", "")).strip() != document_name:
                        continue

                    stored_key = skipped.get("skip_key")
                    if stored_key:
                        persisted_skips.add(str(stored_key))
                    else:
                        # Backward compatibility with older skip records.
                        key = _visual_skip_key(skipped)
                        if key:
                            persisted_skips.add(key)

        # Backward compatibility with USER_SKIPPED records in the main audit
        # ledger.
        if DECISIONS_FILE.exists():
            ledger = load_json(DECISIONS_FILE)

            if isinstance(ledger, dict):
                outcomes = ledger.get("review_outcomes", [])
                if isinstance(outcomes, list):
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue
                        if (
                            str(outcome.get("outcome", "")).strip().upper()
                            != "USER_SKIPPED"
                        ):
                            continue
                        if (
                            str(outcome.get("document", "")).strip()
                            != document_name
                        ):
                            continue

                        key = _visual_skip_key(outcome)
                        if key:
                            persisted_skips.add(key)

                skipped_items = ledger.get("skipped_items", [])
                if isinstance(skipped_items, list):
                    for skipped in skipped_items:
                        if not isinstance(skipped, dict):
                            continue
                        if (
                            str(skipped.get("document", "")).strip()
                            != document_name
                        ):
                            continue

                        key = _visual_skip_key(skipped)
                        if key:
                            persisted_skips.add(key)

    except Exception:
        # A damaged skip ledger must never break Stage 08.
        persisted_skips = set()

    # Session skips are authoritative immediately after the button click.
    # IMPORTANT:
    # A USER_SKIPPED record is itself an unresolved human-review state.
    # It must NEVER be intersected with the current active queue.
    #
    # The old implementation did:
    #
    #     skipped_keys &= candidate_skip_keys
    #
    # That is logically wrong for the LAST unresolved item because Skip
    # intentionally removes that item from the active queue. On the next
    # rerun the active queue can therefore be empty, making the skipped count
    # incorrectly become zero.
    #
    # The persistent skip ledger is the authoritative source for the skipped
    # count. A skipped item remains unresolved until the reviewer explicitly
    # reviews it again and confirms/corrects it.
    session_skips = set(st.session_state[session_skips_key_name])

    skipped_keys = (persisted_skips | session_skips) - set(released_skips)

    # DO NOT intersect skipped_keys with all_unresolved_candidates.
    # Skipped items are deliberately absent from the active queue.
    st.session_state[skipped_key_name] = skipped_keys

    # Build the active queue once per document/session state.  This is the
    # important part for the actual Skip button: after a click, the current
    # item is removed from the in-memory queue BEFORE rerun, so Streamlit cannot
    # redisplay the same item because of widget/session-state ordering.
    #
    # Human verification always causes a rerun with a fresh queue, so resolved
    # items are removed from the canonical unresolved set normally.
    if queue_key_name not in st.session_state:
        active_queue = []
        for raw in items:
            if not isinstance(raw, dict):
                continue

            raw_item = dict(raw)
            raw_item.setdefault("document", document_name)

            skip_key = _visual_skip_key(raw_item)
            if skip_key in skipped_keys:
                continue

            active_queue.append(raw_item)

        st.session_state[queue_key_name] = active_queue

    else:
        # Reconcile the session queue against the CURRENT canonical unresolved
        # set on every rerun. The old implementation only removed skipped items,
        # so a queue item that had just been human-confirmed could remain in
        # session state, while newly unresolved items could be omitted.
        #
        # Keep historical review_required/*.json untouched as audit snapshots.
        # The canonical verified JSON determines whether a candidate is still
        # unresolved; only candidates in `items` are eligible for the live queue.
        current_by_key = {}
        for raw in items:
            if not isinstance(raw, dict):
                continue
            current_item = dict(raw)
            current_item.setdefault("document", document_name)
            current_by_key[decision_key(current_item)] = current_item

        synchronized_queue = []
        seen_queue_keys = set()

        # Preserve existing queue order and original position for candidates
        # that remain unresolved.
        for raw in st.session_state[queue_key_name]:
            if not isinstance(raw, dict):
                continue
            old_item = dict(raw)
            old_item.setdefault("document", document_name)
            key = decision_key(old_item)
            if key not in current_by_key or key in seen_queue_keys:
                continue
            if _visual_skip_key(current_by_key[key]) in skipped_keys:
                continue
            synchronized_queue.append(current_by_key[key])
            seen_queue_keys.add(key)

        # Add candidates newly found by the canonical scan, without duplicates.
        for key, current_item in current_by_key.items():
            if key in seen_queue_keys:
                continue
            if _visual_skip_key(current_item) in skipped_keys:
                continue
            synchronized_queue.append(current_item)
            seen_queue_keys.add(key)

        st.session_state[queue_key_name] = synchronized_queue

    items = st.session_state[queue_key_name]
    skipped_item_count = len(skipped_keys)

    action_key_name = f"ocr_visual_actions::{document_name}"
    if action_key_name not in st.session_state:
        st.session_state[action_key_name] = {
            "human": 0,
            "skipped": 0,
        }
    session_actions = st.session_state[action_key_name]

    # Run the same automatic processing chain from the original Stage 08,
    # stopping only at the first genuinely unresolved item.
    #
    # IMPORTANT: immediately after Skip, do NOT run the automatic chain again.
    # The user explicitly asked to move to the next review item, so the next
    # unresolved candidate must be shown directly. Otherwise learned/exact
    # reuse can consume the rest of the queue in the same rerun and make the
    # screen jump straight to zero candidates.
    pending=None
    auto_count=0
    done_count=0
    skipped_count=len(skipped_keys)
    skip_just_happened = bool(st.session_state.get(skip_just_happened_key_name, False))

    if skip_just_happened:
        # Consume this one-shot flag. The next user action can again use the
        # normal automatic reuse chain.
        st.session_state[skip_just_happened_key_name] = False
        if items:
            first_item = dict(items[0])
            positions = st.session_state.get(queue_position_key_name, {})
            first_position = positions.get(_visual_skip_key(first_item), 1)
            pending=(first_position, st.session_state[queue_total_key_name], first_item)
    else:
        for position, raw in enumerate(items,1):
            if not isinstance(raw,dict):
                continue

            item=dict(raw)
            item.setdefault("document",document_name)
            key=decision_key(item)

            handled, final_data, kind=_visual_process_auto_item(
                item,final_data,document_name,review_file,
                existing_decisions,existing_lookup,same_content_lookup,memory
            )
            if handled:
                done_count += 1
                auto_count += 1
                continue

            positions = st.session_state.get(queue_position_key_name, {})
            current_position = positions.get(_visual_skip_key(item), position)
            pending=(current_position, st.session_state[queue_total_key_name], item)
            break

    total=len(items)

    # Persistent human-confirmation count for the selected document.
    # Session counters are intentionally separate because Streamlit session
    # state resets on restart and must not be presented as persistent history.
    def _visual_saved_human_decision_key(record):
        if not isinstance(record, dict):
            return None
        if str(record.get("document", "")).strip() != str(document_name).strip():
            return None
        if not _is_human_evidence_decision(record):
            return None
        location_fields = (
            "page", "object_type", "table", "row", "column",
            "paragraph_block", "bbox", "original_text", "source_text",
        )
        values = tuple(
            json.dumps(record.get(field), ensure_ascii=False, sort_keys=True, default=str)
            for field in location_fields
        )
        return (str(document_name).strip(),) + values

    saved_human_keys = {
        key for record in existing_decisions
        if (key := _visual_saved_human_decision_key(record)) is not None
    }
    saved_human_count = len(saved_human_keys)

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Unresolved review candidates",total)
    c2.metric("Human decisions saved",saved_human_count)
    c3.metric("Skipped unresolved",skipped_count)
    c4.metric("Auto-resolved this run",done_count)
    c5.metric("Current item",1 if pending else 0)

    st.caption(
        f"Human actions this session: {int(session_actions.get('human', 0))}. "
        "This session-only count resets when Streamlit restarts; "
        "'Human decisions saved' is read from the persistent decision ledger."
    )

    if pending is None:
        save_json(FINAL_DIR / f"{review_file.stem}.json",final_data)

        if skipped_count > 0:
            st.info(
                f"There are {skipped_count} unresolved item(s) skipped for now. "
                "They were NOT marked verified and were NOT added to learning memory."
            )

            if st.button("Review skipped items again", use_container_width=True):
                # Release the currently skipped items for this Streamlit session.
                # Their USER_SKIPPED audit records remain persisted, so a later
                # application restart will still remember that they were skipped.
                released_skips.update(skipped_keys)
                st.session_state[released_key_name] = released_skips
                st.session_state[session_skips_key_name].difference_update(skipped_keys)
                st.session_state[skipped_key_name] = set()
                st.session_state[skip_just_happened_key_name] = False

                # Rebuild the current queue DIRECTLY from the persisted skip
                # records. A skipped item is already known to be unresolved;
                # it must not depend on the current candidate-discovery/status
                # logic to become reviewable again.
                rebuilt_queue = []

                try:
                    if SKIPS_FILE.exists():
                        loaded_skips = load_json(SKIPS_FILE)
                        skip_records = (
                            loaded_skips.get("skipped_items", [])
                            if isinstance(loaded_skips, dict)
                            else []
                        )

                        if isinstance(skip_records, list):
                            for raw in skip_records:
                                if not isinstance(raw, dict):
                                    continue
                                if str(raw.get("document", "")).strip() != document_name:
                                    continue

                                candidate = dict(raw)
                                candidate.setdefault("document", document_name)

                                # Convert the audit record back into the normal
                                # review-item shape where possible.
                                candidate.setdefault(
                                    "original_text",
                                    candidate.get("original_text", "")
                                )
                                candidate.setdefault(
                                    "source_text",
                                    candidate.get("source_text", "")
                                )

                                candidate_key = _visual_skip_key(candidate)
                                if candidate_key in released_skips:
                                    rebuilt_queue.append(candidate)
                except Exception as exc:
                    st.error(f"Could not rebuild skipped review items: {exc}")

                # De-duplicate by stable Skip identity.
                unique_queue = {}
                for candidate in rebuilt_queue:
                    candidate_key = _visual_skip_key(candidate)
                    if candidate_key:
                        unique_queue[candidate_key] = candidate

                # If no persisted record could be reconstructed, fall back to
                # the historical/canonical sources using the released keys.
                if not unique_queue:
                    release_candidates = list(historical_items)
                    release_candidates.extend(
                        canonical_unresolved.values()
                        if isinstance(canonical_unresolved, dict)
                        else []
                    )
                    for raw in release_candidates:
                        if not isinstance(raw, dict):
                            continue
                        candidate = dict(raw)
                        candidate.setdefault("document", document_name)
                        candidate_key = _visual_skip_key(candidate)
                        if candidate_key in released_skips:
                            unique_queue[candidate_key] = candidate

                st.session_state[queue_key_name] = list(unique_queue.values())
                st.rerun()
        else:
            st.success("All non-skipped review candidates for this active PDF are resolved.")

        st.write(f"Final verified JSON: `{FINAL_DIR / f'{review_file.stem}.json'}`")
        return

    position,total,item=pending
    st.subheader(f"Human review — item {position} of {total}")
    original=str(item.get("original_text",""))
    source=str(item.get("source_text",""))
    page=item.get("page","")

    img_path=_visual_page_image(document_name,page)
    resolved_bbox=None
    display_bbox=None
    box=None
    crop_rotated=False
    bbox_is_approximate=False

    if img_path is None:
        st.warning(f"Source page image could not be resolved for page {page}. Text can still be reviewed, but do not treat a missing source image as visual verification.")
        marked=None; crop=None
    else:
        try:
            source_img=Image.open(img_path).convert("RGB")
            visual_item=dict(item)
            resolved_bbox=_visual_resolve_item_bbox(
                final_data, item, source_img.size
            )
            if resolved_bbox is not None:
                display_bbox=resolved_bbox
                visual_item["bbox"]=resolved_bbox
            else:
                # Even when the generic geometry check rejects a bbox as
                # unsuitable for an EXACT verification crop, show the raw
                # stored OCR bbox when it is numerically inside the page.
                # This lets the reviewer see what region the OCR system was
                # attempting to review, without falsely calling it exact.
                raw_bbox=_visual_any_bbox(item, final_data)
                raw_scaled=_visual_scale_box(raw_bbox, source_img.size)
                if raw_scaled is not None:
                    x1,y1,x2,y2=raw_scaled
                    if x2>x1 and y2>y1:
                        display_bbox=raw_bbox
                        bbox_is_approximate=True
                        visual_item["bbox"]=raw_bbox
            marked,box=_visual_mark(source_img,visual_item)
            crop=_visual_crop(source_img,box) if resolved_bbox is not None else None
            crop_rotated=_visual_crop_was_rotated(box,source_img.size) if resolved_bbox is not None else False
        except Exception as exc:
            st.error(f"Could not open source image: {exc}")
            marked=None; crop=None

    left,right=st.columns([2,1])
    with left:
        if resolved_bbox is not None:
            st.markdown("**SOURCE PAGE — RED BOX IS THE EXACT REGION TO VERIFY**")
        elif display_bbox is not None:
            st.markdown("**SOURCE PAGE — RED BOX SHOWS THE STORED OCR REGION (APPROXIMATE)**")
            st.warning("The red box shows where the OCR system stored this item. Its geometry failed the exact-region sanity check, so it must not be treated as an exact verification crop.")
        else:
            st.markdown("**SOURCE PAGE — NO STORED OCR REGION AVAILABLE**")
        if marked is not None:
            st.image(marked,use_container_width=True)
        else:
            st.info("Source image unavailable")
    with right:
        if resolved_bbox is not None:
            st.markdown("**REVIEW THIS REGION**")
            st.caption("The image below is the source pixels from the exact canonical bounding box. Only its display orientation/size may be changed for readability.")
            if crop is not None:
                st.image(crop,use_container_width=True)
                if crop_rotated:
                    st.caption("The review region was automatically rotated horizontally for readability. The source page and OCR coordinates were not changed.")
            else:
                st.info("Review crop unavailable")
        else:
            st.markdown("**REVIEW REGION UNAVAILABLE**")
            st.warning(
                "The stored OCR bounding box is geometrically inconsistent "
                "with this review item, so it is not being presented as an "
                "exact source region. Review the full source page instead."
            )

        st.markdown("**TEXT YOU ARE VERIFYING**")
        st.code(original or "<empty>")
        st.write(f"Page: {page}")
        st.write(f"Object: {object_type(item)}")
        for k in ("table","row","column","paragraph_block"):
            if item.get(k) is not None: st.write(f"{k}: {item.get(k)}")
        if resolved_bbox is not None:
            st.caption("Exact canonical source-image bounding box")
            st.code(str([round(v, 2) for v in resolved_bbox]))
        elif item.get("bbox") is not None:
            st.error(
                "The stored canonical bounding box failed the generic "
                "geometry sanity check. No exact crop is being shown."
            )
        else:
            st.error(
                "No canonical bounding box could be resolved. Do NOT "
                "verify this item from text alone."
            )

    st.divider()
    a,b=st.columns(2)
    with a:
        st.markdown("**Primary OCR**")
        st.code(original or "<empty>")
    with b:
        st.markdown("**Independent/source OCR**")
        st.code(source or "<empty>")

    verification=item.get("verification",{})
    if isinstance(verification,dict):
        st.info(f"Reason: {verification.get('reason','Not supplied')} | Confidence: {verification.get('confidence','N/A')}")

    # Skip is deliberately placed BEFORE the verification editor. Skipping
    # is an audit-only action and uses a dedicated ledger so it cannot be
    # blocked by the trusted-decision/learning-memory schema.
    skip_col, _ = st.columns([1, 3])
    with skip_col:
        if st.button("⏭ Skip unresolved", use_container_width=True, key=f"skip_top_{position}_{page}"):
            item_key = _visual_skip_key(item)
            if not item_key:
                st.error(
                    "This review item has no stable identity, so it cannot "
                    "be skipped safely."
                )
                st.stop()

            try:
                records = []

                if SKIPS_FILE.exists():
                    loaded = load_json(SKIPS_FILE)
                    if (
                        isinstance(loaded, dict)
                        and isinstance(loaded.get("skipped_items"), list)
                    ):
                        records = [
                            x for x in loaded["skipped_items"]
                            if isinstance(x, dict)
                        ]

                existing_skip_keys = set()
                for record in records:
                    stored_key = record.get("skip_key")
                    if stored_key:
                        existing_skip_keys.add(str(stored_key))
                    else:
                        legacy_key = _visual_skip_key(record)
                        if legacy_key:
                            existing_skip_keys.add(legacy_key)

                if item_key not in existing_skip_keys:
                    records.append({
                        "skip_key": item_key,
                        "document": item.get("document", document_name),
                        "page": item.get("page", ""),
                        "object_type": object_type(item),
                        "table": item.get("table"),
                        "row": item.get("row"),
                        "column": item.get("column"),
                        "paragraph_block": item.get("paragraph_block"),
                        "bbox": item.get("bbox"),
                        "original_text": item.get("original_text", ""),
                        "source_text": item.get("source_text", ""),
                        "timestamp_utc": now_utc(),
                    })

                save_json(
                    SKIPS_FILE,
                    {
                        "version": 2,
                        "total_items": len(records),
                        "skipped_items": records,
                    },
                )

                # Preserve the main audit trail when possible, but its failure
                # must never prevent the dedicated skip from taking effect.
                try:
                    _visual_record_review_outcome(
                        document_name,
                        item,
                        "USER_SKIPPED",
                        None,
                        "HUMAN",
                    )
                except Exception:
                    pass

                # Remove the item from the CURRENT queue immediately.
                # This guarantees that the next rerun cannot redisplay it.
                current_queue = st.session_state.get(queue_key_name, [])
                st.session_state[queue_key_name] = [
                    queued
                    for queued in current_queue
                    if _visual_skip_key(queued) != item_key
                ]

                # A previously released skip must become skipped again if the
                # reviewer skips this item a second time.  Otherwise the item
                # remains in released_skips and the next rerun calculates:
                #
                #     skipped_keys = (...) - released_skips
                #
                # which incorrectly removes this newly skipped item and makes
                # "Skipped unresolved" fall back to 0.
                released_skips.discard(item_key)
                st.session_state[released_key_name] = released_skips

                skipped_keys.add(item_key)
                st.session_state[session_skips_key_name].add(item_key)
                st.session_state[skipped_key_name] = skipped_keys

                # Tell the next rerun to show the next unresolved candidate
                # directly instead of auto-consuming the queue.
                st.session_state[skip_just_happened_key_name] = True

                session_actions["skipped"] += 1
                st.session_state[action_key_name] = session_actions

                st.rerun()

            except Exception as exc:
                st.error(f"Could not persist skip: {exc}")

    st.markdown("**Verified text**")
    correction=st.text_area("Edit only if the source image confirms a correction", value=original, key=f"correction_{position}_{page}", height=120)

    if resolved_bbox is None:
        st.error(
            "Do not confirm this item from the OCR text alone. The exact "
            "source region could not be established automatically."
        )
    else:
        st.warning(
            "Choose the value only after checking the highlighted source "
            "image. The original OCR and source OCR remain preserved in "
            "the decision/audit records."
        )

    b1,b2,b3=st.columns(3)
    with b1:
        if st.button("✓ Confirm primary OCR",use_container_width=True):
            saved=_visual_record_human_decision(item,final_data,review_file,verified_file,document_name,existing_decisions,existing_lookup,same_content_lookup,memory,original,"USER_CONFIRMED_ORIGINAL")
            if saved is None:
                st.error("Could not resolve this review item to its canonical target. Nothing was recorded as a human action.")
            else:
                _visual_record_review_outcome(document_name,item,"USER_CONFIRMED_ORIGINAL",original,"HUMAN")
                session_actions["human"] += 1
                st.session_state[action_key_name] = session_actions
                _visual_remove_from_session_queue(queue_key_name, item)
                st.rerun()
    with b2:
        if st.button("✓ Use source OCR",use_container_width=True):
            saved=_visual_record_human_decision(item,final_data,review_file,verified_file,document_name,existing_decisions,existing_lookup,same_content_lookup,memory,source,"USER_CONFIRMED_SOURCE")
            if saved is None:
                st.error("Could not resolve this review item to its canonical target. Nothing was recorded as a human action.")
            else:
                _visual_record_review_outcome(document_name,item,"USER_CONFIRMED_SOURCE",source,"HUMAN")
                session_actions["human"] += 1
                st.session_state[action_key_name] = session_actions
                _visual_remove_from_session_queue(queue_key_name, item)
                st.rerun()
    with b3:
        if st.button("✓ Save correction",use_container_width=True):
            if not correction.strip():
                st.error("Correction cannot be empty.")
            else:
                saved=_visual_record_human_decision(item,final_data,review_file,verified_file,document_name,existing_decisions,existing_lookup,same_content_lookup,memory,correction.strip(),"USER_ENTERED_CORRECTION")
                if saved is None:
                    st.error("Could not resolve this review item to its canonical target. Nothing was recorded as a human action.")
                else:
                    _visual_record_review_outcome(document_name,item,"USER_ENTERED_CORRECTION",correction.strip(),"HUMAN")
                    session_actions["human"] += 1
                    st.session_state[action_key_name] = session_actions
                    _visual_remove_from_session_queue(queue_key_name, item)
                    st.rerun()
    st.divider()
    st.caption(f"Source page file: {img_path if img_path else 'NOT FOUND'}")
    st.caption("Stage 09 remains responsible for rebuilding generalized learning. Stage 08 records human evidence; automatic decisions are never treated as human teaching evidence.")


# ============================================================
# ORIGINAL TERMINAL MAIN — PRESERVED FOR FALLBACK / AUDIT
# ============================================================
# This is the original Stage 08 main workflow, renamed only so the
# visual interface can be the default entry point. No original backend
# functions or decision logic are deleted.

# MAIN
# ============================================================

def legacy_terminal_main():

    print("=" * 80)
    print("MANUAL OCR REVIEW")
    print("PER-PDF HUMAN REVIEW")
    print("DYNAMIC PATTERN LEARNING")
    print()
    print("Human evidence persistence: ENABLED")
    print("Automatic reuse is never treated as human ground truth.")
    print("=" * 80)

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    if not REVIEW_DIR.exists():

        raise FileNotFoundError(
            f"Review directory not found:\n"
            f"{REVIEW_DIR}"
        )

    active_document_names = get_active_document_names()

    print()
    print(
        "Active input PDFs: "
        f"{len(active_document_names)}"
    )
    for name in sorted(active_document_names):
        print(f"  {name}.pdf")

    review_files = get_active_review_files(
        active_document_names
    )

    if not review_files:

        print()
        print(
            "No per-PDF review files found."
        )

        print(
            f"Directory: {REVIEW_DIR}"
        )

        print("=" * 80)

        return

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DECISIONS
    #
    # This file is intentionally cumulative across PDFs.
    # It is the human-evidence history consumed by Stage 09.
    # --------------------------------------------------------

    existing_decisions = []

    if DECISIONS_FILE.exists():

        try:

            decision_data = load_json(
                DECISIONS_FILE
            )

            existing_decisions = (
                decision_data.get(
                    "decisions",
                    [],
                )
            )

            if not isinstance(
                existing_decisions,
                list,
            ):
                existing_decisions = []

        except Exception:

            existing_decisions = []

    existing_lookup = {}

    for decision in existing_decisions:

        if not isinstance(
            decision,
            dict,
        ):
            continue

        existing_lookup[
            decision_key(
                decision
            )
        ] = decision

    # --------------------------------------------------------
    # SAME-CONTENT HUMAN EVIDENCE LOOKUP
    # --------------------------------------------------------

    same_content_lookup = {}

    for decision in existing_decisions:

        if not _is_human_evidence_decision(
            decision
        ):
            continue

        d_original = _decision_original(
            decision
        )

        d_source = _decision_source(
            decision
        )

        if not d_original or not d_source:
            continue

        if _reuse_exact_evidence_value(
            decision,
            d_original,
            d_source,
        ) is None:
            continue

        content_key = _exact_evidence_key(
            _decision_document(decision),
            _decision_page(decision),
            d_original,
            d_source,
        )

        same_content_lookup[
            content_key
        ] = decision

    # --------------------------------------------------------
    # MEMORY
    #
    # Stage 09 is authoritative for learning-memory construction.
    # Stage 08 only consumes the existing cumulative memory.
    # --------------------------------------------------------

    memory = load_memory()

    print()
    print(
        f"Per-PDF review files: "
        f"{len(review_files)}"
    )

    print(
        f"Existing human/audit decisions: "
        f"{len(existing_decisions)}"
    )

    # --------------------------------------------------------
    # PROCESS EACH PDF INDEPENDENTLY
    # --------------------------------------------------------

    grand_total = 0
    grand_reviewed = 0
    grand_auto_skipped = 0

    for review_file in review_files:

        print()
        print("=" * 80)
        print(
            f"PROCESSING PDF: "
            f"{review_file.stem}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # LOAD REVIEW JSON
        # ----------------------------------------------------

        review_data = load_json(
            review_file
        )

        if not isinstance(
            review_data,
            dict,
        ):
            print(
                "WARNING: Review JSON is not an object."
            )
            continue

        review_items = review_data.get(
            "items",
            [],
        )

        if not isinstance(
            review_items,
            list,
        ):
            review_items = []

        # ----------------------------------------------------
        # LOAD THIS PDF'S VERIFIED JSON
        # ----------------------------------------------------

        document_name = str(
            review_data.get(
                "document",
                review_file.stem,
            )
        ).strip()

        verified_file = (
            VERIFIED_DIR
            / f"{review_file.stem}.json"
        )

        if not verified_file.exists():

            # Some review files may carry an explicit document
            # name different from the filename.
            alternate_file = (
                VERIFIED_DIR
                / f"{document_name}.json"
            )

            if alternate_file.exists():
                verified_file = alternate_file

        if not verified_file.exists():

            print()
            print(
                "WARNING: Per-PDF verified JSON not found:"
            )

            print(
                verified_file
            )

            print(
                "Skipping this PDF."
            )

            continue

        final_data = load_json(
            verified_file
        )

        # ----------------------------------------------------
        # ENSURE DOCUMENT NAME IS AVAILABLE
        # ----------------------------------------------------

        if (
            isinstance(final_data, dict)
            and not final_data.get("document")
        ):
            final_data["document"] = document_name

        # ----------------------------------------------------
        # NO REVIEW
        # ----------------------------------------------------

        if not review_items:

            final_output_file = (
                FINAL_DIR
                / f"{review_file.stem}.json"
            )

            save_json(
                final_output_file,
                final_data,
            )

            print()
            print(
                "No items require manual review."
            )

            print(
                f"Final verified data: "
                f"{final_output_file}"
            )

            continue

        total = len(
            review_items
        )

        manually_reviewed = 0
        auto_skipped = 0

        grand_total += total

        # ====================================================
        # REVIEW LOOP
        # ====================================================

        for position, item in enumerate(
            review_items,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            # ------------------------------------------------
            # Ensure the item has the current PDF identity.
            # This prevents a per-PDF review file from accidentally
            # entering another document's namespace.
            # ------------------------------------------------

            item = dict(
                item
            )

            item.setdefault(
                "document",
                document_name,
            )

            if not item.get(
                "document"
            ):
                item["document"] = document_name

            key = decision_key(
                item
            )

            original = str(
                item.get(
                    "original_text",
                    "",
                )
            )

            source = str(
                item.get(
                    "source_text",
                    "",
                )
            )

            # ------------------------------------------------
            # EXISTING LOCATION DECISION
            # ------------------------------------------------

            if key in existing_lookup:

                previous = existing_lookup[
                    key
                ]

                previous_value = (
                    _decision_verified(
                        previous
                    )
                )

                # USER_CONFIRMED_ORIGINAL / SOURCE may not store
                # a separate verified_text in older records.
                if not previous_value:

                    previous_kind = (
                        _human_verification_kind(
                            previous
                        )
                    )

                    if (
                        previous_kind
                        == "USER_CONFIRMED_ORIGINAL"
                    ):
                        previous_value = (
                            _decision_original(
                                previous
                            )
                        )

                    elif (
                        previous_kind
                        == "USER_CONFIRMED_SOURCE"
                    ):
                        previous_value = (
                            _decision_source(
                                previous
                            )
                        )

                target, target_type = (
                    find_target(
                        final_data,
                        item,
                    )
                )

                if (
                    target is not None
                    and previous_value
                ):

                    apply_final_value(
                        target,
                        previous_value,
                        _decision_kind(
                            previous
                        ),
                        _decision_original(
                            previous
                        ) or original,
                        _decision_source(
                            previous
                        ) or source,
                    )

                print()
                print(
                    f"Item {position}/{total} "
                    f"already has a decision."
                )

                continue

            # ------------------------------------------------
            # SAME OBSERVED OCR/SOURCE EVIDENCE
            # ------------------------------------------------

            content_key = _exact_evidence_key(
                item.get(
                    "document",
                    document_name,
                ),
                item.get(
                    "page",
                    "",
                ),
                original,
                source,
            )

            same_content_decision = (
                same_content_lookup.get(
                    content_key
                )
            )

            if same_content_decision is None:

                same_content_decision = (
                    _find_exact_human_evidence(
                        existing_decisions,
                        item.get(
                            "document",
                            document_name,
                        ),
                        item.get(
                            "page",
                            "",
                        ),
                        original,
                        source,
                    )
                )

                if same_content_decision is not None:

                    same_content_lookup[
                        content_key
                    ] = same_content_decision

            # ------------------------------------------------
            # SAME VISUAL EVIDENCE
            # ------------------------------------------------

            if same_content_decision is None:

                same_content_decision = (
                    _find_human_evidence_by_original_and_bbox(
                        existing_decisions,
                        item,
                    )
                )

            if same_content_decision is not None:

                same_content_lookup[
                    content_key
                ] = same_content_decision

            # ------------------------------------------------
            # LEGACY PARAGRAPH -> TABLE FALLBACK
            # ------------------------------------------------

            if same_content_decision is None:

                same_content_decision = (
                    _find_legacy_human_evidence_by_unique_original(
                        existing_decisions,
                        item,
                    )
                )

            if same_content_decision is not None:

                same_content_lookup[
                    content_key
                ] = same_content_decision

            # ------------------------------------------------
            # GLOBAL EXACT EVIDENCE
            #
            # This is allowed across documents only when the
            # COMPLETE OCR/source pair is identical.
            # It is NOT generalized learning.
            # ------------------------------------------------

            if same_content_decision is None:

                same_content_decision = (
                    _find_global_exact_human_evidence(
                        existing_decisions,
                        original,
                        source,
                    )
                )

                if same_content_decision is not None:

                    same_content_lookup[
                        content_key
                    ] = same_content_decision

            # ------------------------------------------------
            # REUSE IDENTICAL HUMAN EVIDENCE
            # ------------------------------------------------

            if same_content_decision is not None:

                reused_decision = (
                    _human_verification_kind(
                        same_content_decision
                    )
                )

                learned_value = (
                    _reuse_exact_evidence_value(
                        same_content_decision,
                        original,
                        source,
                    )
                )

                if learned_value is not None:

                    learned_type = (
                        "EXACT_EVIDENCE_REUSED"
                    )

                    print()
                    print(
                        f"Item {position}/{total}"
                    )

                    print(
                        "DUPLICATE EVIDENCE "
                        "AUTOMATICALLY SKIPPED"
                    )

                    print(
                        f"Original OCR: {original}"
                    )

                    print(
                        f"Source OCR: {source}"
                    )

                    print(
                        f"Previous human decision: "
                        f"{reused_decision}"
                    )

                    print(
                        f"Final verified value: "
                        f"{learned_value}"
                    )

                    target, target_type = (
                        find_target(
                            final_data,
                            item,
                        )
                    )

                    if target is not None:

                        apply_final_value(
                            target,
                            learned_value,
                            learned_type,
                            original,
                            source,
                        )

                    fingerprint = (
                        build_learning_fingerprint(
                            final_data,
                            item,
                        )
                    )

                    decision_record = (
                        make_decision_record(
                            item,
                            learned_value,
                            learned_type,
                            learning_type=
                                learned_type,
                            learning_fingerprint=
                                fingerprint,
                        )
                    )

                    existing_decisions.append(
                        decision_record
                    )

                    existing_lookup[
                        key
                    ] = decision_record

                    memory[
                        "audit"
                    ].append(
                        {
                            "timestamp_utc":
                                now_utc(),
                            "document":
                                item.get(
                                    "document",
                                    document_name,
                                ),
                            "page":
                                item.get(
                                    "page",
                                    "",
                                ),
                            "object_type":
                                object_type(
                                    item
                                ),
                            "decision":
                                learned_type,
                            "original_text":
                                original,
                            "source_text":
                                source,
                            "verified_text":
                                learned_value,
                            "reused_from_decision":
                                reused_decision,
                            "fingerprint":
                                fingerprint,
                        }
                    )

                    auto_skipped += 1
                    grand_auto_skipped += 1

                    final_output_file = (
                        FINAL_DIR
                        / f"{review_file.stem}.json"
                    )

                    save_json(
                        final_output_file,
                        final_data,
                    )

                    save_memory(
                        memory
                    )

                    _persist_decision_data(existing_decisions)

                    continue

            # ------------------------------------------------
            # LEARNED PATTERN
            #
            # Stage 08 only consumes patterns already present in
            # cumulative memory. Stage 09 is responsible for
            # building/generalizing that memory.
            # ------------------------------------------------

            learned = find_learned_pattern(
                final_data,
                item,
                memory,
            )

            if learned is not None:

                learned_value = learned[
                    "verified_text"
                ]

                learned_type = learned[
                    "type"
                ]

                print()
                print(
                    f"Item {position}/{total}"
                )

                print(
                    "AUTOMATICALLY SKIPPED"
                )

                print(
                    f"Original OCR: {original}"
                )

                print(
                    f"Source OCR: {source}"
                )

                print(
                    f"Learned pattern: "
                    f"{learned_type}"
                )

                print(
                    f"Reason: "
                    f"{learned['reason']}"
                )

                print(
                    f"Final verified value: "
                    f"{learned_value}"
                )

                target, target_type = (
                    find_target(
                        final_data,
                        item,
                    )
                )

                if target is not None:

                    apply_final_value(
                        target,
                        learned_value,
                        learned_type,
                        original,
                        source,
                    )

                fingerprint = (
                    build_learning_fingerprint(
                        final_data,
                        item,
                    )
                )

                decision_record = (
                    make_decision_record(
                        item,
                        learned_value,
                        learned_type,
                        learning_type=
                            learned_type,
                        learning_fingerprint=
                            fingerprint,
                    )
                )

                existing_decisions.append(
                    decision_record
                )

                existing_lookup[
                    key
                ] = decision_record

                memory[
                    "audit"
                ].append(
                    {
                        "timestamp_utc":
                            now_utc(),
                        "document":
                            item.get(
                                "document",
                                document_name,
                            ),
                        "page":
                            item.get(
                                "page",
                                "",
                            ),
                        "object_type":
                            object_type(
                                item
                            ),
                        "decision":
                            learned_type,
                        "original_text":
                            original,
                        "source_text":
                            source,
                        "verified_text":
                            learned_value,
                        "fingerprint":
                            fingerprint,
                    }
                )

                auto_skipped += 1
                grand_auto_skipped += 1

                final_output_file = (
                    FINAL_DIR
                    / f"{review_file.stem}.json"
                )

                save_json(
                    final_output_file,
                    final_data,
                )

                save_memory(
                    memory
                )

                _persist_decision_data(existing_decisions)

                continue

            # =================================================
            # HUMAN REVIEW
            # =================================================

            display_review_item(
                position,
                total,
                item,
            )

            while True:

                choice = normalize_choice(
                    input(
                        "Choose O / S / C / Q: "
                    )
                )

                if choice == "o":

                    verified_text = (
                        original
                    )

                    decision = (
                        "USER_CONFIRMED_ORIGINAL"
                    )

                    break

                if choice == "s":

                    verified_text = (
                        source
                    )

                    decision = (
                        "USER_CONFIRMED_SOURCE"
                    )

                    break

                if choice == "c":

                    print()

                    verified_text = input(
                        "Enter the correct value exactly: "
                    ).strip()

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

                    final_output_file = (
                        FINAL_DIR
                        / f"{review_file.stem}.json"
                    )

                    save_json(
                        final_output_file,
                        final_data,
                    )

                    save_memory(
                        memory
                    )

                    _persist_decision_data(existing_decisions)

                    print()
                    print(
                        "Progress saved."
                    )

                    print(
                        f"Current PDF final data: "
                        f"{final_output_file}"
                    )

                    return

                print(
                    "Invalid choice. "
                    "Enter O, S, C, or Q."
                )

            # ------------------------------------------------
            # TARGET
            # ------------------------------------------------

            target, target_type = (
                find_target(
                    final_data,
                    item,
                )
            )

            if target is None:

                print()
                print(
                    "WARNING: Target object "
                    "could not be located in final data."
                )

                print(
                    "The decision will still "
                    "be recorded."
                )

            else:

                apply_final_value(
                    target,
                    verified_text,
                    decision,
                    original,
                    source,
                )

            # ------------------------------------------------
            # LEARNING FINGERPRINT
            # ------------------------------------------------

            fingerprint = (
                build_learning_fingerprint(
                    final_data,
                    item,
                )
            )

            decision_record = (
                make_decision_record(
                    item,
                    verified_text,
                    decision,
                    previous_status=
                        item.get(
                            "verification",
                            {},
                        ).get(
                            "decision",
                            "",
                        ),
                    learning_type=
                        (
                            "HUMAN_CONFIRMED_PATTERN"
                            if fingerprint
                            else None
                        ),
                    learning_fingerprint=
                        fingerprint,
                )
            )

            existing_decisions.append(
                decision_record
            )

            existing_lookup[
                key
            ] = decision_record

            # ------------------------------------------------
            # IMMUTABLE HUMAN EVIDENCE
            # ------------------------------------------------

            human_evidence_record = {
                "evidence_type":
                    "HUMAN_CONFIRMED_EVIDENCE",

                "document":
                    item.get(
                        "document",
                        document_name,
                    ),

                "page":
                    item.get(
                        "page",
                        "",
                    ),

                "object_type":
                    object_type(
                        item
                    ),

                "bbox":
                    item.get(
                        "bbox"
                    ),

                "paragraph_block":
                    item.get(
                        "paragraph_block"
                    ),

                "table":
                    item.get(
                        "table"
                    ),

                "row":
                    item.get(
                        "row"
                    ),

                "column":
                    item.get(
                        "column"
                    ),

                "original_text":
                    original,

                "source_text":
                    source,

                "verified_text":
                    verified_text,

                "human_decision":
                    decision,

                "timestamp_utc":
                    now_utc(),
            }

            # ------------------------------------------------
            # REPLACE SAME HUMAN EVIDENCE ONLY
            # ------------------------------------------------

            evidence_key = (
                str(
                    human_evidence_record.get(
                        "document",
                        "",
                    )
                ).strip(),

                _canonical_evidence_page(
                    human_evidence_record.get(
                        "page",
                        "",
                    )
                ),

                _canonical_evidence_text(
                    original
                ),

                _canonical_evidence_text(
                    source
                ),
            )

            replaced_human_evidence = False

            for evidence_index, existing_record in enumerate(
                existing_decisions
            ):

                if not isinstance(
                    existing_record,
                    dict,
                ):
                    continue

                if existing_record.get(
                    "evidence_type"
                ) != "HUMAN_CONFIRMED_EVIDENCE":
                    continue

                existing_evidence_key = (
                    str(
                        existing_record.get(
                            "document",
                            "",
                        )
                    ).strip(),

                    _canonical_evidence_page(
                        existing_record.get(
                            "page",
                            "",
                        )
                    ),

                    _canonical_evidence_text(
                        existing_record.get(
                            "original_text",
                            "",
                        )
                    ),

                    _canonical_evidence_text(
                        existing_record.get(
                            "source_text",
                            "",
                        )
                    ),
                )

                if (
                    existing_evidence_key
                    == evidence_key
                ):

                    existing_decisions[
                        evidence_index
                    ] = (
                        human_evidence_record
                    )

                    replaced_human_evidence = True

                    break

            if not replaced_human_evidence:

                existing_decisions.append(
                    human_evidence_record
                )

            # ------------------------------------------------
            # SAME-CONTENT LOOKUP
            # ------------------------------------------------

            same_content_lookup[
                _exact_evidence_key(
                    item.get(
                        "document",
                        document_name,
                    ),
                    item.get(
                        "page",
                        "",
                    ),
                    original,
                    source,
                )
            ] = human_evidence_record

            # ------------------------------------------------
            # IMPORTANT:
            # Stage 08 records human evidence.
            # Stage 09 is responsible for rebuilding the
            # cumulative generalized learning memory.
            #
            # We intentionally DO NOT create generalized
            # learning rules here.
            # ------------------------------------------------

            memory[
                "audit"
            ].append(
                {
                    "timestamp_utc":
                        now_utc(),

                    "document":
                        item.get(
                            "document",
                            document_name,
                        ),

                    "page":
                        item.get(
                            "page",
                            "",
                        ),

                    "object_type":
                        object_type(
                            item
                        ),

                    "original_text":
                        original,

                    "source_text":
                        source,

                    "verified_text":
                        verified_text,

                    "decision":
                        decision,

                    "fingerprint":
                        fingerprint,
                }
            )

            # ------------------------------------------------
            # SAVE AFTER EVERY HUMAN DECISION
            # ------------------------------------------------

            final_output_file = (
                FINAL_DIR
                / f"{review_file.stem}.json"
            )

            save_json(
                final_output_file,
                final_data,
            )

            save_memory(
                memory
            )

            _persist_decision_data(existing_decisions)

            manually_reviewed += 1
            grand_reviewed += 1

            print()
            print(
                f"Recorded: {decision}"
            )

            print(
                f"Final verified value: "
                f"{verified_text}"
            )

            print()
            print(
                "Original OCR remains preserved."
            )

            print(
                "Source/reference OCR remains preserved."
            )

            print(
                f"Per-PDF final data saved: "
                f"{final_output_file}"
            )

        # ====================================================
        # END OF CURRENT PDF
        # ====================================================

        final_output_file = (
            FINAL_DIR
            / f"{review_file.stem}.json"
        )

        save_json(
            final_output_file,
            final_data,
        )

        print()
        print("-" * 80)

        print(
            f"PDF COMPLETE: "
            f"{review_file.stem}"
        )

        print(
            f"Review candidates     : "
            f"{total}"
        )

        print(
            f"Human reviewed        : "
            f"{manually_reviewed}"
        )

        print(
            f"Automatically skipped : "
            f"{auto_skipped}"
        )

        print(
            f"Final verified data   : "
            f"{final_output_file}"
        )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_memory(
        memory
    )

    _persist_decision_data(existing_decisions)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)

    print(
        "PER-PDF HUMAN REVIEW COMPLETE"
    )

    print("=" * 80)

    print()
    print(
        f"PDFs processed         : "
        f"{len(review_files)}"
    )

    print(
        f"Review candidates      : "
        f"{grand_total}"
    )

    print(
        f"Human reviewed         : "
        f"{grand_reviewed}"
    )

    print(
        f"Automatically skipped  : "
        f"{grand_auto_skipped}"
    )

    print(
        f"Decision records       : "
        f"{len(existing_decisions)}"
    )

    print()
    print(
        "Cumulative human decisions:"
    )

    print(
        DECISIONS_FILE
    )

    print()
    print(
        "Cumulative learning memory:"
    )

    print(
        LEARNING_MEMORY_FILE
    )

    print()
    print(
        "Per-PDF final verified JSON:"
    )

    print(
        FINAL_DIR
    )

    print()
    print(
        "Original OCR was preserved."
    )

    print(
        "Source OCR was preserved."
    )

    print(
        "Final verified text is separate."
    )

    print("=" * 80)




if __name__ == "__main__":
    _visual_main()
