import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

from PIL import Image, ImageOps, ImageEnhance
import pytesseract


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "final_data"
    / "all_documents.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "verified_data"
)

PAGES_DIR = (
    BASE_DIR
    / "output"
    / "pages"
)

REVIEW_CROP_DIR = (
    OUTPUT_DIR
    / "review_crops"
)

VERIFIED_FILE = (
    OUTPUT_DIR
    / "all_documents_verified.json"
)

REVIEW_FILE = (
    OUTPUT_DIR
    / "review_required.json"
)

LEARNING_MEMORY_FILE = (
    OUTPUT_DIR
    / "ocr_learning_memory.json"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REVIEW_CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

CLOSE_SIMILARITY = 0.85

CELL_PADDING = 8

UPSCALE = 3

REVIEW_DECISIONS = {
    "HIGH_RISK_SYMBOL_REVIEW",
    "HIGH_RISK_AGREEMENT_REVIEW",
    "NUMERIC_DISAGREEMENT",
    "SOURCE_DISAGREEMENT",
    "SOURCE_EMPTY",
    "NO_SOURCE_IMAGE",
}


# ============================================================
# HIGH-RISK TECHNICAL CHARACTERS
# ============================================================

HIGH_RISK_CHARACTERS = {
    # Proper Unicode
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
    "∂",
    "∫",
    "∇",
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
    "∈",
    "∉",
    "⊂",
    "⊃",
    "∩",
    "∪",

    # Common mojibake seen in the project
    "Î©",
    "Ï‰",
    "Â±",
    "âˆ“",
    "â‰¥",
    "â‰¤",
    "Î”",
    "âˆ†",
    "Î¼",
    "Âµ",
    "Â°",
    "â†’",
    "â†”",
    "â‰ ",
    "â‰ˆ",
    "â‰…",
    "Ã—",
    "Ã·",
    "âˆž",
    "âˆ‘",
    "âˆš",
    "âˆ‚",
    "âˆ«",
    "âˆ",
    "Î»",
    "Î¸",
    "Ï†",
    "Ïƒ",
    "Ï„",
    "Î±",
    "Î²",
    "Î³",
    "Î´",
    "Îµ",
    "Î·",
    "Îº",
    "Ï",
    "Ï‡",
    "Ïˆ",
    "Î¾",
    "Î¶",
    "âˆˆ",
    "âˆ‰",
    "âŠ‚",
    "âŠƒ",
    "âˆ©",
    "âˆª",
}


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def normalize_text(text):
    """
    Basic whitespace normalization.
    Meaningful internal spaces are preserved.
    """

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
    """
    Repair common UTF-8/Windows mojibake patterns.

    This is deliberately limited to known OCR/document
    encoding problems used by this project.
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

        "â‰ ": "≠",
        "â‰ˆ": "≈",
        "â‰…": "≅",

        "Ã—": "×",
        "Ã·": "÷",

        "âˆž": "∞",
        "âˆ‘": "∑",
        "âˆš": "√",
        "âˆ‚": "∂",
        "âˆ«": "∫",

        "Î»": "λ",
        "Î¸": "θ",
        "Ï†": "φ",
        "Ïƒ": "σ",
        "Ï„": "τ",
        "Î±": "α",
        "Î²": "β",
        "Î³": "γ",
        "Î´": "δ",
        "Îµ": "ε",
        "Î·": "η",
        "Îº": "κ",
        "Ï‡": "χ",
        "Ïˆ": "ψ",
        "Î¾": "ξ",
        "Î¶": "ζ",

        "âˆˆ": "∈",
        "âˆ‰": "∉",
        "âŠ‚": "⊂",
        "âŠƒ": "⊃",
        "âˆ©": "∩",
        "âˆª": "∪",

        "â€“": "–",
        "â€”": "—",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€ ": "”",

        "Â°C": "°C",
        "Âµm": "µm",
        "Î¼m": "μm",

        "℃": "°C",
        "°℃": "°C",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


def normalize_for_comparison(text):
    """
    Normalize harmless representation differences while
    preserving semantic differences.

    Examples:

        Ω != Q
        ± != +
        ≥ != >
        ≤ != <

    But:

        μm == um
        °℃ == °C
        different dash styles == -
    """

    text = normalize_text(
        text
    )

    text = repair_mojibake(
        text
    )

    # --------------------------------------------------------
    # Unit / formatting normalization
    # --------------------------------------------------------

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

    text = text.replace(
        "Ohm",
        "Ω"
    )

    text = text.replace(
        "ohm",
        "Ω"
    )

    # --------------------------------------------------------
    # Dash normalization
    # --------------------------------------------------------

    dash_variants = [
        "–",
        "—",
        "−",
        "-",
        "‒",
        "﹘",
        "﹣",
        "－",
    ]

    for dash in dash_variants:

        text = text.replace(
            dash,
            "-"
        )

    # --------------------------------------------------------
    # Whitespace around symbols / punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"\s*°\s*C\b",
        " °C",
        text
    )

    text = re.sub(
        r"\s*%\b",
        "%",
        text
    )

    return text.strip().casefold()


# ============================================================
# TECHNICAL SYMBOL DETECTION
# ============================================================

def contains_technical_symbol(text):
    """
    Detect special/technical characters other than ordinary
    letters, digits, whitespace, and normal separators.
    """

    if text is None:
        return False

    text = str(text)

    ordinary_separators = {
        "-",
        "_",
        "/",
        ":",
        ".",
        ",",
    }

    for character in text:

        if character.isspace():
            continue

        if character.isalnum():
            continue

        if character in ordinary_separators:
            continue

        return True

    return False


def contains_high_risk_character(text):
    if text is None:
        return False

    return any(
        character in HIGH_RISK_CHARACTERS
        for character in str(text)
    )


def extract_high_risk_characters(text):
    found = []

    if text is None:
        return found

    for character in str(text):

        if character in HIGH_RISK_CHARACTERS:

            if character not in found:
                found.append(
                    character
                )

    return found


def extract_technical_symbols(text):
    found = []

    if text is None:
        return found

    ordinary_separators = {
        "-",
        "_",
        "/",
        ":",
        ".",
        ",",
    }

    for character in str(text):

        if character.isspace():
            continue

        if character.isalnum():
            continue

        if character in ordinary_separators:
            continue

        if character not in found:
            found.append(
                character
            )

    return found


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def is_identifier_like(text):
    """
    Detect values that look like IDs / codes.

    Examples:

        C-01
        PRB-O0I1
        MX-47A-19
        SN_00481-X
        IMP-07A
    """

    if text is None:
        return False

    text = str(text).strip()

    if not text:
        return False

    has_letter = bool(
        re.search(
            r"[A-Za-z]",
            text
        )
    )

    has_digit = bool(
        re.search(
            r"\d",
            text
        )
    )

    has_separator = bool(
        re.search(
            r"[_:/.\-]",
            text
        )
    )

    has_letter_digit_transition = bool(
        re.search(
            r"[A-Za-z]\d|\d[A-Za-z]",
            text
        )
    )

    return (
        has_letter
        and has_digit
        and (
            has_separator
            or has_letter_digit_transition
        )
    )


# ============================================================
# NUMERIC HELPERS
# ============================================================

def numeric_content(text):
    """
    Extract standalone numeric values.

    Embedded digits inside identifiers are excluded.
    """

    if text is None:
        return []

    text = str(text)

    pattern = r"""
        (?<![A-Za-z_])
        [+-]?
        (?:
            \d+(?:\.\d*)?
            |
            \.\d+
        )
        (?![A-Za-z_])
    """

    return re.findall(
        pattern,
        text,
        flags=re.VERBOSE
    )


def numeric_values_equal(
    a,
    b
):
    a_numbers = numeric_content(
        a
    )

    b_numbers = numeric_content(
        b
    )

    if len(a_numbers) != len(b_numbers):
        return False

    try:

        for x, y in zip(
            a_numbers,
            b_numbers
        ):

            if abs(
                float(x) - float(y)
            ) > 1e-12:
                return False

        return True

    except ValueError:

        return False


def non_numeric_skeleton(text):
    """
    Replace standalone numeric content with #.
    """

    if text is None:
        return ""

    text = str(text)

    pattern = r"""
        (?<![A-Za-z_])
        [+-]?
        (?:
            \d+(?:\.\d*)?
            |
            \.\d+
        )
        (?![A-Za-z_])
    """

    return re.sub(
        pattern,
        "#",
        text,
        flags=re.VERBOSE
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def resolve_image_path(
    document,
    page
):
    candidates = [

        PAGES_DIR
        / str(document)
        / f"page_{int(page):04d}.png",

        PAGES_DIR
        / str(document)
        / f"page_{int(page):04d}.jpg",

        PAGES_DIR
        / str(document)
        / f"page_{int(page):04d}.jpeg",
    ]

    for path in candidates:

        if path.exists():
            return path

    return None


def crop_with_padding(
    image,
    bbox,
    padding=CELL_PADDING
):
    if not bbox:
        return None

    if len(bbox) < 4:
        return None

    try:

        x1, y1, x2, y2 = map(
            int,
            bbox[:4]
        )

    except (
        TypeError,
        ValueError
    ):

        return None

    width, height = image.size

    x1 = max(
        0,
        x1 - padding
    )

    y1 = max(
        0,
        y1 - padding
    )

    x2 = min(
        width,
        x2 + padding
    )

    y2 = min(
        height,
        y2 + padding
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return image.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# SOURCE OCR PREPROCESSING
# ============================================================

def prepare_ocr_image(
    crop,
    mode
):
    image = crop.resize(
        (
            crop.width * UPSCALE,
            crop.height * UPSCALE
        ),
        Image.Resampling.LANCZOS
    )

    if mode == "original":
        return image

    if mode == "grayscale":

        return ImageOps.grayscale(
            image
        )

    if mode == "contrast":

        image = ImageOps.grayscale(
            image
        )

        enhancer = ImageEnhance.Contrast(
            image
        )

        return enhancer.enhance(
            2.0
        )

    if mode == "sharp":

        image = ImageOps.grayscale(
            image
        )

        enhancer = ImageEnhance.Sharpness(
            image
        )

        return enhancer.enhance(
            2.0
        )

    return image


# ============================================================
# SOURCE OCR
# ============================================================

def source_ocr_variants(crop):
    """
    Run several OCR passes.

    They are witnesses, not ground truth.
    """

    results = {}

    configurations = [
        (
            "original",
            "--psm 7"
        ),
        (
            "grayscale",
            "--psm 7"
        ),
        (
            "contrast",
            "--psm 7"
        ),
        (
            "sharp",
            "--psm 7"
        ),
    ]

    for mode, config in configurations:

        prepared = prepare_ocr_image(
            crop,
            mode
        )

        # ----------------------------------------------------
        # OCR TEXT
        # ----------------------------------------------------

        try:

            text = pytesseract.image_to_string(
                prepared,
                config=config
            )

        except Exception as exc:

            results[mode] = {
                "text": "",
                "confidence": 0.0,
                "error": str(exc),
            }

            continue

        text = normalize_text(
            text
        )

        # ----------------------------------------------------
        # OCR CONFIDENCE
        # ----------------------------------------------------

        confidence = 0.0

        try:

            data = pytesseract.image_to_data(
                prepared,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            values = []

            for value in data.get(
                "conf",
                []
            ):

                try:

                    number = float(
                        value
                    )

                    if number >= 0:
                        values.append(
                            number
                        )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

            if values:

                confidence = (
                    sum(values)
                    / len(values)
                    / 100.0
                )

        except Exception:

            confidence = 0.0

        results[mode] = {
            "text": text,
            "confidence": round(
                confidence,
                4
            ),
        }

    return results


# ============================================================
# CONSENSUS
# ============================================================

def consensus_text(variants):
    texts = []

    for result in variants.values():

        text = normalize_text(
            result.get(
                "text",
                ""
            )
        )

        if text:
            texts.append(
                text
            )

    if not texts:
        return ""

    counts = Counter(
        texts
    )

    return counts.most_common(1)[0][0]


def consensus_confidence(variants):
    values = []

    for result in variants.values():

        try:

            value = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            if value >= 0:
                values.append(
                    value
                )

        except (
            TypeError,
            ValueError
        ):

            pass

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


# ============================================================
# LEARNING MEMORY
# ============================================================

def load_learning_memory():
    """
    Load reusable whole-cell patterns produced by Step 09.
    Symbol-level patterns are kept separate and are not used
    by find_learned_match().
    """

    if not LEARNING_MEMORY_FILE.exists():
        return []

    try:
        with open(
            LEARNING_MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        patterns = data.get(
            "reusable_text_patterns",
            []
        )

        if isinstance(patterns, list):
            return patterns

    except Exception:
        pass

    return []

def find_learned_match(
    original_text,
    source_text,
    learning_patterns
):
    """
    Find an exact repeated human-verified OCR pattern.

    Both OCR witnesses must match the learned pattern
    after safe normalization.

    The pattern must also have been verified at least
    twice by a human through Step 09.
    """

    original = normalize_for_comparison(
        original_text
    )

    source = normalize_for_comparison(
        source_text
    )

    if not original or not source:
        return None

    for pattern in learning_patterns:

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

        learned_original = normalize_for_comparison(
            pattern.get(
                "original_ocr",
                ""
            )
        )

        learned_source = normalize_for_comparison(
            pattern.get(
                "source_ocr",
                ""
            )
        )

        learned_verified = normalize_text(
            pattern.get(
                "verified_text",
                ""
            )
        )

        if not learned_verified:
            continue

        if (
            original == learned_original
            and
            source == learned_source
        ):

            return {
                "verified_text":
                    learned_verified,

                "repeat_count":
                    repeat_count,

                "learned_original":
                    learned_original,

                "learned_source":
                    learned_source,
            }

    return None


# ============================================================
# CELL VERIFICATION
# ============================================================

def verify_cell(
    original_text,
    source_text,
    source_confidence
):
    """
    Conservative source-image verification.

    Rules:

    1. Empty source OCR -> review.
    2. Exact harmless normalized agreement -> verified.
    3. Technical/special-symbol disagreement -> review.
    4. Identifier disagreement -> review.
    5. Standalone numeric disagreement -> review.
    6. Same numbers + same structure -> close verification.
    7. General similarity -> close verification.
    8. Everything else -> review.
    """

    original = normalize_text(
        original_text
    )

    source = normalize_text(
        source_text
    )

    original_normalized = normalize_for_comparison(
        original
    )

    source_normalized = normalize_for_comparison(
        source
    )

    # --------------------------------------------------------
    # 1. SOURCE OCR EMPTY
    # --------------------------------------------------------

    if not source_normalized:

        return {
            "decision":
                "SOURCE_EMPTY",

            "verified_text":
                original,

            "reason":
                (
                    "Source-image OCR returned no usable text."
                ),

            "confidence":
                source_confidence,
        }

    # --------------------------------------------------------
    # 2. EXACT AGREEMENT
    # --------------------------------------------------------

    if (
        original_normalized
        ==
        source_normalized
    ):

        # Exact agreement between two OCR passes is normally safe
        # for ordinary text. However, OCR systems can sometimes
        # make the SAME mistake on a technical symbol.
        #
        # We therefore flag only clearly suspicious substitutions
        # instead of sending every identifier or unit to review.

        suspicious_ocr_characters = {
            "J",
            "j",
            "V",
            "v",
            "d",
            "D",
            "c",
            "C",
            "E",
            "€",
            "o",
            "O",
        }

        suspicious_agreement = any(
            character in suspicious_ocr_characters
            for character in original
        )

        if suspicious_agreement and (
            contains_technical_symbol(original)
            or
            contains_high_risk_character(original)
            or
            "NEW-" in original
        ):

            return {
                "decision":
                    "HIGH_RISK_AGREEMENT_REVIEW",

                "verified_text":
                    original,

                "reason":
                    (
                        "Original OCR and source-image OCR "
                        "agree, but the result contains a "
                        "character commonly associated with "
                        "technical-symbol OCR substitutions. "
                        "The original image must be visually "
                        "reviewed because identical OCR output "
                        "does not prove that the symbol was "
                        "recognized correctly."
                    ),

                "confidence":
                    source_confidence,

                "technical_symbols":
                    extract_technical_symbols(
                        original
                    ),

                "high_risk_characters":
                    extract_high_risk_characters(
                        original
                    ),
            }

        return {
            "decision":
                "SOURCE_VERIFIED",

            "verified_text":
                original,

            "reason":
                (
                    "Original extraction and source-image "
                    "OCR agree after harmless normalization."
                ),

            "confidence":
                source_confidence,
        }

    # 3. TECHNICAL / SPECIAL SYMBOL DISAGREEMENT
    # --------------------------------------------------------

    original_has_symbol = (
        contains_high_risk_character(
            original
        )
        or
        contains_technical_symbol(
            original
        )
    )

    source_has_symbol = (
        contains_high_risk_character(
            source
        )
        or
        contains_technical_symbol(
            source
        )
    )

    if (
        original_has_symbol
        or source_has_symbol
    ):

        return {
            "decision":
                "HIGH_RISK_SYMBOL_REVIEW",

            "verified_text":
                original,

            "reason":
                (
                    "Original and source-image OCR disagree "
                    "on a technical or special character. "
                    "The original image must be visually reviewed; "
                    "no OCR candidate is automatically treated "
                    "as ground truth."
                ),

            "confidence":
                source_confidence,

            "technical_symbols_original":
                extract_technical_symbols(
                    original
                ),

            "technical_symbols_source":
                extract_technical_symbols(
                    source
                ),

            "high_risk_characters_original":
                extract_high_risk_characters(
                    original
                ),

            "high_risk_characters_source":
                extract_high_risk_characters(
                    source
                ),
        }

    # --------------------------------------------------------
    # 4. IDENTIFIER DISAGREEMENT
    # --------------------------------------------------------

    if (
        is_identifier_like(
            original
        )
        or
        is_identifier_like(
            source
        )
    ):

        similarity = SequenceMatcher(
            None,
            original_normalized,
            source_normalized
        ).ratio()

        return {
            "decision":
                "SOURCE_DISAGREEMENT",

            "verified_text":
                original,

            "reason":
                (
                    "The cell appears to contain an "
                    "alphanumeric identifier. The extracted "
                    "and source-image OCR differ, so the "
                    "original image must be reviewed instead "
                    "of interpreting embedded digits as "
                    "measurement values."
                ),

            "confidence":
                source_confidence,

            "similarity":
                round(
                    similarity,
                    4
                ),
        }

    # --------------------------------------------------------
    # 5. STANDALONE NUMERIC DISAGREEMENT
    # --------------------------------------------------------

    original_numbers = numeric_content(
        original
    )

    source_numbers = numeric_content(
        source
    )

    if (
        original_numbers
        or source_numbers
    ):

        if not numeric_values_equal(
            original,
            source
        ):

            return {
                "decision":
                    "NUMERIC_DISAGREEMENT",

                "verified_text":
                    original,

                "reason":
                    (
                        "Standalone numeric content differs "
                        "between the extracted OCR and the "
                        "source-image OCR. No automatic numeric "
                        "correction is allowed."
                    ),

                "confidence":
                    source_confidence,

                "original_numbers":
                    original_numbers,

                "source_numbers":
                    source_numbers,
            }

    # --------------------------------------------------------
    # 6. SAME NUMBERS + SAME STRUCTURE
    # --------------------------------------------------------

    if (
        numeric_values_equal(
            original,
            source
        )
        and
        (
            non_numeric_skeleton(
                original_normalized
            )
            ==
            non_numeric_skeleton(
                source_normalized
            )
        )
    ):

        return {
            "decision":
                "SOURCE_VERIFIED_CLOSE",

            "verified_text":
                original,

            "reason":
                (
                    "Numeric content and non-numeric structure "
                    "agree despite a minor representation difference."
                ),

            "confidence":
                source_confidence,
        }

    # --------------------------------------------------------
    # 7. GENERAL STRING SIMILARITY
    # --------------------------------------------------------

    similarity = SequenceMatcher(
        None,
        original_normalized,
        source_normalized
    ).ratio()

    if similarity >= CLOSE_SIMILARITY:

        return {
            "decision":
                "SOURCE_VERIFIED_CLOSE",

            "verified_text":
                original,

            "reason":
                (
                    "Source OCR is sufficiently similar to the "
                    "original extraction and no technical-symbol "
                    "disagreement was detected."
                ),

            "confidence":
                source_confidence,

            "similarity":
                round(
                    similarity,
                    4
                ),
        }

    # --------------------------------------------------------
    # 8. GENERAL DISAGREEMENT
    # --------------------------------------------------------

    return {
        "decision":
            "SOURCE_DISAGREEMENT",

        "verified_text":
            original,

        "reason":
            (
                "Source-image OCR differs materially from "
                "the original extraction."
            ),

        "confidence":
            source_confidence,

        "similarity":
            round(
                similarity,
                4
            ),
    }

# ============================================================
# PARAGRAPH BLOCK VERIFICATION
# ============================================================

def verify_paragraph_block(
    original_text,
    source_text,
    source_confidence
):
    """
    Conservative verification for paragraph text blocks.

    Paragraph blocks are treated more strictly than normal text:
    technical/special-character disagreements always require review,
    and suspicious technical-character agreement also requires review.
    """

    original = normalize_text(
        original_text
    )

    source = normalize_text(
        source_text
    )

    original_normalized = normalize_for_comparison(
        original
    )

    source_normalized = normalize_for_comparison(
        source
    )

    # --------------------------------------------------------
    # 1. EMPTY SOURCE OCR
    # --------------------------------------------------------

    if not source_normalized:

        return {
            "decision": "SOURCE_EMPTY",
            "verified_text": original,
            "reason": (
                "Source-image OCR returned no usable text "
                "for this paragraph block."
            ),
            "confidence": source_confidence,
        }

    # --------------------------------------------------------
    # 2. EXACT AGREEMENT
    # --------------------------------------------------------

    if original_normalized == source_normalized:

        suspicious_ocr_characters = {
            "J", "j",
            "V", "v",
            "d", "D",
            "c", "C",
            "E",
            "€",
            "o", "O",
        }

        suspicious_agreement = any(
            character in suspicious_ocr_characters
            for character in original
        )

        if suspicious_agreement and (
            contains_technical_symbol(original)
            or contains_high_risk_character(original)
        ):

            return {
                "decision":
                    "HIGH_RISK_AGREEMENT_REVIEW",

                "verified_text":
                    original,

                "reason": (
                    "Original OCR and source-image OCR agree, "
                    "but the paragraph contains a character "
                    "commonly associated with OCR substitutions. "
                    "The original image must be visually reviewed."
                ),

                "confidence":
                    source_confidence,

                "technical_symbols":
                    extract_technical_symbols(original),

                "high_risk_characters":
                    extract_high_risk_characters(original),
            }

        return {
            "decision":
                "SOURCE_VERIFIED",

            "verified_text":
                original,

            "reason": (
                "Original paragraph OCR and source-image "
                "OCR agree after harmless normalization."
            ),

            "confidence":
                source_confidence,
        }

    # --------------------------------------------------------
    # 3. TECHNICAL / SPECIAL SYMBOL DISAGREEMENT
    # --------------------------------------------------------

    if (
        contains_high_risk_character(original)
        or contains_technical_symbol(original)
        or contains_high_risk_character(source)
        or contains_technical_symbol(source)
    ):

        return {
            "decision":
                "HIGH_RISK_SYMBOL_REVIEW",

            "verified_text":
                original,

            "reason": (
                "Paragraph OCR and source-image OCR disagree "
                "on a technical or special character. "
                "The original image must be visually reviewed."
            ),

            "confidence":
                source_confidence,

            "technical_symbols_original":
                extract_technical_symbols(original),

            "technical_symbols_source":
                extract_technical_symbols(source),

            "high_risk_characters_original":
                extract_high_risk_characters(original),

            "high_risk_characters_source":
                extract_high_risk_characters(source),
        }

    # --------------------------------------------------------
    # 4. GENERAL PARAGRAPH DISAGREEMENT
    # --------------------------------------------------------

    similarity = SequenceMatcher(
        None,
        original_normalized,
        source_normalized
    ).ratio()

    if similarity >= CLOSE_SIMILARITY:

        return {
            "decision":
                "SOURCE_VERIFIED_CLOSE",

            "verified_text":
                original,

            "reason": (
                "Source-image OCR is sufficiently similar "
                "to the paragraph extraction."
            ),

            "confidence":
                source_confidence,

            "similarity":
                round(
                    similarity,
                    4
                ),
        }

    return {
        "decision":
            "SOURCE_DISAGREEMENT",

        "verified_text":
            original,

        "reason": (
            "Source-image OCR differs materially "
            "from the paragraph extraction."
        ),

        "confidence":
            source_confidence,

        "similarity":
            round(
                similarity,
                4
            ),
    }

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SOURCE IMAGE ACCURACY VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD INPUT DATA
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(
            f
        )

    verified_data = json.loads(
        json.dumps(
            data
        )
    )

    # --------------------------------------------------------
    # LOAD LEARNING MEMORY
    # --------------------------------------------------------

    learning_patterns = load_learning_memory()

    print()
    print(
        f"Learned OCR patterns : "
        f"{len(learning_patterns)}"
    )

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    review_items = []

    total_cells = 0

    automatic_verified = 0

    needs_review = 0

    learned_verified = 0

    decisions = {}

    # --------------------------------------------------------
    # PROCESS DOCUMENTS
    # --------------------------------------------------------

    for document in verified_data.get(
        "documents",
        []
    ):

        document_name = document.get(
            "document",
            document.get(
                "name",
                "unknown_document"
            )
        )

        print()
        print(
            f"Document: {document_name}"
        )

        for page in document.get(
            "pages",
            []
        ):

            page_number = page.get(
                "page"
            )

            image_path = resolve_image_path(
                document_name,
                page_number
            )

            print(
                f"  Page {page_number}"
            )

            if image_path:

                print(
                    f"    Source image: "
                    f"{image_path}"
                )

            else:

                print(
                    "    Source image: "
                    "NOT FOUND"
                )


                        # ------------------------------------------------
            # PARAGRAPH BLOCKS
            # ------------------------------------------------

            paragraph = page.get(
                "paragraph",
                {}
            )

            paragraph_blocks = paragraph.get(
                "text_blocks",
                []
            )

            for paragraph_index, block in enumerate(
                paragraph_blocks,
                start=1
            ):

                original_text = normalize_text(
                    block.get(
                        "text",
                        ""
                    )
                )

                bbox = block.get(
                    "bbox"
                )

                crop = None

                # --------------------------------------------
                # LOAD / CROP SOURCE IMAGE
                # --------------------------------------------

                if (
                    image_path
                    and image_path.exists()
                    and bbox
                ):

                    try:

                        page_image = Image.open(
                            image_path
                        ).convert(
                            "RGB"
                        )

                        crop = crop_with_padding(
                            page_image,
                            bbox
                        )

                    except Exception:

                        crop = None

                # --------------------------------------------
                # SOURCE IMAGE UNAVAILABLE
                # --------------------------------------------

                if crop is None:

                    result = {
                        "decision":
                            "NO_SOURCE_IMAGE",

                        "verified_text":
                            original_text,

                        "reason":
                            (
                                "Original page image or "
                                "paragraph bounding box "
                                "is unavailable."
                            ),

                        "confidence":
                            0.0,
                    }

                    needs_review += 1

                    decision_name = result[
                        "decision"
                    ]

                    decisions[
                        decision_name
                    ] = (
                        decisions.get(
                            decision_name,
                            0
                        )
                        + 1
                    )

                    review_items.append({

                        "document":
                            document_name,

                        "page":
                            page_number,

                        "paragraph_block":
                            paragraph_index,

                        "original_text":
                            original_text,

                        "source_text":
                            "",

                        "verification":
                            result,

                        "review_crop":
                            "",
                    })

                    block[
                        "verification_status"
                    ] = result[
                        "decision"
                    ]

                    block[
                        "verified_text"
                    ] = result[
                        "verified_text"
                    ]

                    block[
                        "final_verified_text"
                    ] = result[
                        "verified_text"
                    ]

                    block[
                        "final_verification_status"
                    ] = result[
                        "decision"
                    ]

                    continue

                # --------------------------------------------
                # SOURCE OCR
                # --------------------------------------------

                variants = source_ocr_variants(
                    crop
                )

                source_text = consensus_text(
                    variants
                )

                source_confidence = (
                    consensus_confidence(
                        variants
                    )
                )

                # --------------------------------------------
                # VERIFY PARAGRAPH
                # --------------------------------------------

                result = verify_paragraph_block(
                    original_text,
                    source_text,
                    source_confidence
                )

                decision = result[
                    "decision"
                ]

                decisions[
                    decision
                ] = (
                    decisions.get(
                        decision,
                        0
                    )
                    + 1
                )

                # --------------------------------------------
                # WRITE VERIFICATION METADATA
                # --------------------------------------------

                block[
                    "verification_status"
                ] = decision

                block[
                    "verified_text"
                ] = result[
                    "verified_text"
                ]

                block[
                    "final_verified_text"
                ] = result[
                    "verified_text"
                ]

                block[
                    "final_verification_status"
                ] = decision

                block[
                    "source_ocr"
                ] = source_text

                block[
                    "source_ocr_confidence"
                ] = round(
                    source_confidence,
                    4
                )

                block[
                    "verification_reason"
                ] = result.get(
                    "reason",
                    ""
                )

                block[
                    "source_ocr_variants"
                ] = variants

                # --------------------------------------------
                # REVIEW OR AUTO-ACCEPT
                # --------------------------------------------

                if decision in REVIEW_DECISIONS:

                    needs_review += 1

                    crop_name = (
                        f"{document_name}"
                        f"_p{int(page_number):03d}"
                        f"_para{int(paragraph_index):03d}"
                        f".png"
                    )

                    crop_path = (
                        REVIEW_CROP_DIR
                        / crop_name
                    )

                    try:

                        crop.save(
                            crop_path
                        )

                        block[
                            "review_crop"
                        ] = str(
                            crop_path
                        )

                    except Exception:

                        block[
                            "review_crop"
                        ] = ""

                    review_items.append({

                        "document":
                            document_name,

                        "page":
                            page_number,

                        "paragraph_block":
                            paragraph_index,

                        "original_text":
                            original_text,

                        "source_text":
                            source_text,

                        "source_ocr_variants":
                            variants,

                        "verification":
                            result,

                        "review_crop":
                            block.get(
                                "review_crop",
                                ""
                            ),
                    })

                else:

                    automatic_verified += 1

            # ------------------------------------------------
            # TABLES
            # ------------------------------------------------

            for table_index, table in enumerate(
                page.get(
                    "tables",
                    []
                ),
                start=1
            ):

                for row_index, row in enumerate(
                    table.get(
                        "rows",
                        []
                    ),
                    start=1
                ):

                    for column_index, cell in enumerate(
                        row.get(
                            "cells",
                            []
                        ),
                        start=1
                    ):

                        total_cells += 1

                        original_text = normalize_text(
                            cell.get(
                                "text",
                                ""
                            )
                        )

                        bbox = cell.get(
                            "bbox"
                        )

                        crop = None

                        # ------------------------------------
                        # LOAD / CROP IMAGE
                        # ------------------------------------

                        if (
                            image_path
                            and image_path.exists()
                            and bbox
                        ):

                            try:

                                page_image = Image.open(
                                    image_path
                                ).convert(
                                    "RGB"
                                )

                                crop = crop_with_padding(
                                    page_image,
                                    bbox
                                )

                            except Exception:

                                crop = None

                        # ------------------------------------
                        # SOURCE IMAGE UNAVAILABLE
                        # ------------------------------------

                        if crop is None:

                            result = {

                                "decision":
                                    "NO_SOURCE_IMAGE",

                                "verified_text":
                                    original_text,

                                "reason":
                                    (
                                        "Original page image "
                                        "or cell bounding box "
                                        "is unavailable."
                                    ),

                                "confidence":
                                    0.0,
                            }

                            needs_review += 1

                            decision_name = result[
                                "decision"
                            ]

                            decisions[
                                decision_name
                            ] = (
                                decisions.get(
                                    decision_name,
                                    0
                                )
                                + 1
                            )

                            review_items.append({

                                "document":
                                    document_name,

                                "page":
                                    page_number,

                                "table":
                                    table_index,

                                "row":
                                    row_index,

                                "column":
                                    column_index,

                                "original_text":
                                    original_text,

                                "source_text":
                                    "",

                                "verification":
                                    result,

                                "review_crop":
                                    "",
                            })

                            cell[
                                "verification_status"
                            ] = result[
                                "decision"
                            ]

                            cell[
                                "verified_text"
                            ] = result[
                                "verified_text"
                            ]

                            continue

                        # ------------------------------------
                        # SOURCE OCR
                        # ------------------------------------

                        variants = source_ocr_variants(
                            crop
                        )

                        source_text = consensus_text(
                            variants
                        )

                        source_confidence = (
                            consensus_confidence(
                                variants
                            )
                        )

                        # ------------------------------------
                        # STANDARD VERIFICATION
                        # ------------------------------------

                        result = verify_cell(
                            original_text,
                            source_text,
                            source_confidence
                        )

                        # ------------------------------------
                        # LEARNED HUMAN-VERIFIED PATTERN
                        # ------------------------------------
                        #
                        # IMPORTANT:
                        #
                        # Learning is checked AFTER the normal
                        # verification logic has produced a
                        # candidate, but a learned exact match
                        # can safely replace a review decision.
                        #
                        # The learned pattern must match BOTH
                        # original OCR and source OCR after
                        # normalization, and must have been
                        # verified at least twice by a human.
                        # ------------------------------------

                        learned_match = find_learned_match(
                            original_text,
                            source_text,
                            learning_patterns
                        )

                        if learned_match:

                            result = {

                                "decision":
                                    "LEARNED_PATTERN_VERIFIED",

                                "verified_text":
                                    learned_match[
                                        "verified_text"
                                    ],

                                "reason":
                                    (
                                        "The same original/source "
                                        "OCR disagreement was "
                                        "previously verified by a "
                                        "human and confirmed at "
                                        f"least "
                                        f"{learned_match['repeat_count']} "
                                        "times."
                                    ),

                                "confidence":
                                    1.0,

                                "learned_repeat_count":
                                    learned_match[
                                        "repeat_count"
                                    ],
                            }

                            learned_verified += 1

                        # ------------------------------------
                        # DECISION COUNT
                        # ------------------------------------

                        decision = result[
                            "decision"
                        ]

                        decisions[
                            decision
                        ] = (
                            decisions.get(
                                decision,
                                0
                            )
                            + 1
                        )

                        # ------------------------------------
                        # WRITE VERIFICATION METADATA
                        # ------------------------------------

                        cell[
                            "verification_status"
                        ] = decision

                        cell[
                            "verified_text"
                        ] = result[
                            "verified_text"
                        ]

                        cell[
                            		    "final_verified_text"
                        		] = result[
                        		         "verified_text"
                        		]

                        cell[
                        		         "final_verification_status"
                        		] = decision

			                   
                        cell[
                            "source_ocr"
                        ] = source_text

                        cell[
                            "source_ocr_confidence"
                        ] = round(
                            source_confidence,
                            4
                        )

                        cell[
                            "verification_reason"
                        ] = result.get(
                            "reason",
                            ""
                        )

                        cell[
                            "source_ocr_variants"
                        ] = variants

                        if (
                            decision
                            ==
                            "LEARNED_PATTERN_VERIFIED"
                        ):

                            cell[
                                "learned_pattern"
                            ] = True

                            cell[
                                "learned_repeat_count"
                            ] = result.get(
                                "learned_repeat_count",
                                0
                            )

                        else:

                            cell[
                                "learned_pattern"
                            ] = False

                        # ------------------------------------
                        # REVIEW OR AUTO-ACCEPT
                        # ------------------------------------

                        if decision in REVIEW_DECISIONS:

                            needs_review += 1

                            crop_name = (
                                f"{document_name}"
                                f"_p{int(page_number):03d}"
                                f"_t{table_index:02d}"
                                f"_r{row_index:02d}"
                                f"_c{column_index:02d}"
                                f".png"
                            )

                            crop_path = (
                                REVIEW_CROP_DIR
                                / crop_name
                            )

                            try:

                                crop.save(
                                    crop_path
                                )

                                cell[
                                    "review_crop"
                                ] = str(
                                    crop_path
                                )

                            except Exception:

                                cell[
                                    "review_crop"
                                ] = ""

                            review_items.append({

                                "document":
                                    document_name,

                                "page":
                                    page_number,

                                "table":
                                    table_index,

                                "row":
                                    row_index,

                                "column":
                                    column_index,

                                "original_text":
                                    original_text,

                                "source_text":
                                    source_text,

                                "source_ocr_variants":
                                    variants,

                                "verification":
                                    result,

                                "review_crop":
                                    cell.get(
                                        "review_crop",
                                        ""
                                    ),
                            })

                        else:

                            automatic_verified += 1

    # ========================================================
    # SAVE VERIFIED DATA
    # ========================================================

    with open(
        VERIFIED_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            verified_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE REVIEW DATA
    # ========================================================

    review_output = {

        "description": (
            "Cells requiring visual review because "
            "source-image OCR disagreed with extracted OCR, "
            "a technical/special character is involved, "
            "or the source image was unavailable. "
            "Repeated human-verified OCR patterns may be "
            "automatically accepted through learning memory."
        ),

        "total_review_items":
            len(review_items),

        "items":
            review_items,
    }

    with open(
        REVIEW_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            review_output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Total cells            : "
        f"{total_cells}"
    )

    print(
        f"Automatically verified : "
        f"{automatic_verified}"
    )

    print(
        f"Learned pattern verified: "
        f"{learned_verified}"
    )

    print(
        f"Needs review           : "
        f"{needs_review}"
    )

    print()
    print("=" * 70)
    print("DECISIONS")
    print("=" * 70)

    print()

    for name in sorted(
        decisions
    ):

        print(
            f"{name:<32}: "
            f"{decisions[name]}"
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print()

    print(
        "Verified data:"
    )

    print(
        VERIFIED_FILE
    )

    print()

    print(
        "Review required:"
    )

    print(
        REVIEW_FILE
    )

    print()

    print(
        "Review crops:"
    )

    print(
        REVIEW_CROP_DIR
    )

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()