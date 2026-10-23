import json
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

from PIL import Image, ImageOps, ImageEnhance
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\LIBRARY.LIBRARY-5.000\AppData\Local\Tesseract-OCR\tesseract.exe"

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Resolve the real project root when Stage 06 is running from
# a job-local copied source directory.
PROJECT_ROOT = BASE_DIR

if (
    BASE_DIR.parent.name == "jobs"
    and BASE_DIR.parent.parent.name == "output"
):
    PROJECT_ROOT = BASE_DIR.parents[2]

# Stage 06 processes one PDF JSON independently.
# The learning memory remains cumulative across all PDFs.

INPUT_DIR = (
    BASE_DIR
    / "output"
    / "final_data"
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

REVIEW_REQUIRED_DIR = (
    OUTPUT_DIR
    / "review_required"
)

REVIEW_CROP_ROOT_DIR = (
    OUTPUT_DIR
    / "review_crops"
)

# Stage 09 stores the cumulative learning memory in output/learning.
# Stage 06 must read that same shared memory; it must not create a
# separate per-verification copy under output/verified_data.
LEARNING_MEMORY_FILE = (
    PROJECT_ROOT
    / "output"
    / "learning"
    / "ocr_learning_memory.json"
)

# These are assigned for the individual PDF currently being processed.
INPUT_FILE = None
REVIEW_CROP_DIR = None
VERIFIED_FILE = None
REVIEW_FILE = None

# PERFORMANCE CACHES
# Reuse the page image and identical source-image crops within a run.
PAGE_IMAGE_CACHE = {}
SOURCE_OCR_CACHE = {}


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

CLOSE_SIMILARITY = 0.85
BBOX_OVERLAP_THRESHOLD = 0.90

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


def contains_mojibake_artifact(text):
    """
    Return True only for recognized encoding-corruption sequences.
    Valid Unicode technical symbols (such as °, Ω, μ, and ℃) are
    not mojibake merely because they are non-ASCII characters.
    """
    if text is None:
        return False

    raw_text = str(text)
    mojibake_markers = (
        "Î©", "Ï‰", "Â±", "âˆ“", "â‰¥", "â‰¤",
        "Î”", "âˆ†", "Î¼", "Âµ", "Â°", "â†’",
        "â†”", "â‰ ", "â‰ˆ", "â‰…", "Ã—", "Ã·",
        "âˆž", "âˆ‘", "âˆš", "âˆ‚", "âˆ«", "Î»",
        "Î¸", "Ï†", "Ïƒ", "Ï„", "Î±", "Î²", "Î³",
        "Î´", "Îµ", "Î·", "Îº", "Ï‡", "Ïˆ", "Î¾",
        "Î¶", "âˆˆ", "âˆ‰", "âŠ‚", "âŠƒ", "âˆ©",
        "âˆª", "â€“", "â€”", "â€˜", "â€™", "â€œ",
        "â€ ", "Âµm", "Î¼m",
    )
    return any(marker in raw_text for marker in mojibake_markers)


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

    # OCR commonly inserts spaces around a decimal point (e.g. "24 .4"
    # or "24. 4"). Normalize only a decimal point surrounded by digits;
    # do not rewrite other punctuation or change numeric digits.
    text = re.sub(
        r"(?<=\d)\s*\.\s*(?=\d)",
        ".",
        text,
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


def _normalize_decimal_spacing(text):
    """Ignore OCR-inserted whitespace immediately around decimal points."""
    if text is None:
        return ""

    return re.sub(
        r"(?<=\d)\s*\.\s*(?=\d)",
        ".",
        str(text)
    )


def numeric_values_equal(
    a,
    b
):
    # Compare numeric formatting, not OCR-added spaces around a decimal point.
    a_numbers = numeric_content(
        _normalize_decimal_spacing(a)
    )

    b_numbers = numeric_content(
        _normalize_decimal_spacing(b)
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

    text = _normalize_decimal_spacing(text)

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

def source_ocr_variants(crop, modes=None):
    """Run one or more independent Tesseract preprocessing passes."""
    results = {}
    configurations = [
        ("original", "--psm 7"),
        ("grayscale", "--psm 7"),
        ("contrast", "--psm 7"),
        ("sharp", "--psm 7"),
    ]
    if modes is not None:
        requested = set(modes)
        configurations = [x for x in configurations if x[0] in requested]
    for mode, config in configurations:
        prepared = prepare_ocr_image(crop, mode)
        try:
            data = pytesseract.image_to_data(
                prepared, config=config, output_type=pytesseract.Output.DICT
            )
            words, values = [], []
            for text_value, conf_value in zip(data.get("text", []), data.get("conf", [])):
                text_value = normalize_text(text_value)
                if text_value:
                    words.append(text_value)
                try:
                    number = float(conf_value)
                    if number >= 0:
                        values.append(number)
                except (TypeError, ValueError):
                    pass
            text = normalize_text(" ".join(words))
            confidence = sum(values) / len(values) / 100.0 if values else 0.0
            results[mode] = {"text": text, "confidence": round(confidence, 4)}
        except Exception as exc:
            results[mode] = {"text": "", "confidence": 0.0, "error": str(exc)}
    return results


def _needs_full_source_ocr(original_text, fast_variants):
    """Escalate to all four OCR witnesses only when evidence is uncertain."""
    fast_text = consensus_text(fast_variants)
    if not fast_text:
        return True
    original = normalize_text(original_text)
    source = normalize_text(fast_text)
    if normalize_for_comparison(original) != normalize_for_comparison(source):
        return True
    if contains_high_risk_character(original) or contains_high_risk_character(source):
        return True
    return False


def get_source_ocr_evidence(crop, original_text):
    """Tier 1: original crop; Tier 2: three extra passes only if needed."""
    fast_variants = source_ocr_variants(crop, modes=["original"])
    if _needs_full_source_ocr(original_text, fast_variants):
        variants = source_ocr_variants(crop)
    else:
        variants = fast_variants
    return variants, consensus_text(variants), consensus_confidence(variants)


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

def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pattern_action_result(action, original_text, source_text, verified_text=""):
    """Resolve a learned action using the CURRENT document evidence."""
    if action == "USE_CURRENT_ORIGINAL":
        return normalize_text(original_text)
    if action == "USE_CURRENT_SOURCE":
        return normalize_text(source_text)
    if verified_text:
        return normalize_text(verified_text)
    return ""


def load_learning_memory():
    """
    Load reusable human-verified learning rules from Stage 09's
    cumulative ``output/learning/ocr_learning_memory.json``.

    Stage 09 memory v7 contains several rule families.  Stage 06
    consumes only rules that have already passed Stage 09's repetition
    and unanimity requirements:

      * exact_patterns                  : repeat_count >= 2
      * generalized_patterns            : repeat_count >= 2
      * structural_patterns             : repeat_count >= 2
      * context_patterns                : repeat_count >= 2
      * cross-context generalized       : repeat_count >= 3, >= 2 docs
      * cross-context structural        : repeat_count >= 3, >= 2 docs
      * local transformations           : repeat_count >= 3, >= 2 docs

    The original human evidence is never modified.  This function only
    converts reusable memory entries into an internal matcher format.
    """
    if not LEARNING_MEMORY_FILE.exists():
        return []

    try:
        with open(LEARNING_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    reusable = []

    def add_rule(rule):
        if not isinstance(rule, dict):
            return
        if not rule.get("human_confirmed", False):
            return
        repeat_count = _safe_int(rule.get("repeat_count", 0))
        if repeat_count < 2:
            return
        rule = dict(rule)
        rule["repeat_count"] = repeat_count
        rule["memory_version"] = data.get("version", 0)
        reusable.append(rule)

    # --------------------------------------------------------
    # EXACT PATTERNS
    # --------------------------------------------------------
    exact_patterns = data.get("exact_patterns", {})
    if isinstance(exact_patterns, dict):
        for encoded_key, record in exact_patterns.items():
            if not isinstance(record, dict):
                continue
            repeat_count = _safe_int(record.get("repeat_count", 0))
            if repeat_count < 2 or not record.get("human_confirmed", False):
                continue
            try:
                key_data = json.loads(encoded_key)
                if not isinstance(key_data, list) or len(key_data) != 2:
                    continue
                object_type = str(key_data[0])
                evidence = json.loads(key_data[1])
                if not isinstance(evidence, list) or len(evidence) != 2:
                    continue
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            verified_text = normalize_text(record.get("verified_text", ""))
            if not verified_text:
                continue
            reusable.append({
                "family": "exact",
                "object_type": object_type,
                "original_ocr": str(evidence[0]),
                "source_ocr": str(evidence[1]),
                "verified_text": verified_text,
                "repeat_count": repeat_count,
                "human_confirmed": True,
                "memory_version": data.get("version", 0),
            })

    # --------------------------------------------------------
    # STRUCTURED STAGE 09 RULE FAMILIES
    # --------------------------------------------------------
    family_specs = [
        ("generalized_patterns", "generalized", 2),
        ("structural_patterns", "structural", 2),
        ("context_patterns", "context", 2),
        ("cross_context_generalized_patterns", "cross_generalized", 3),
        ("cross_context_structural_patterns", "cross_structural", 3),
        ("local_transformation_patterns", "local", 3),
        ("character_edit_transformation_patterns", "character_edit", 2),
    ]

    for field_name, family, min_repeat in family_specs:
        patterns = data.get(field_name, {})
        if not isinstance(patterns, dict):
            continue
        for encoded_key, record in patterns.items():
            if not isinstance(record, dict):
                continue
            repeat_count = _safe_int(record.get("repeat_count", 0))
            document_count = _safe_int(record.get("document_count", 0))
            if repeat_count < min_repeat or not record.get("human_confirmed", False):
                continue
            if family in {"cross_generalized", "cross_structural", "local"} and document_count < 2:
                continue
            try:
                key_data = json.loads(encoded_key)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            reusable.append({
                "family": family,
                "key": key_data,
                "action": str(record.get("action", "")),
                "repeat_count": repeat_count,
                "document_count": document_count,
                "human_confirmed": True,
                "memory_version": data.get("version", 0),
            })

    return reusable


def _current_object_type(object_type):
    return "TABLE_CELL" if str(object_type).upper() == "TABLE_CELL" else "PARAGRAPH"


def _current_learning_context(object_type, table_index=0, column_index=0):
    typ = _current_object_type(object_type)
    if typ == "TABLE_CELL":
        return f"{typ}|table={_safe_int(table_index)}|column={_safe_int(column_index)}"
    return typ


def _generalized_shape(text):
    """Mirror Stage 09's compact generalized shape."""
    text = normalize_text(text)
    if not text:
        return ""
    result = []
    current = None
    count = 0

    def flush():
        if current is not None and count:
            result.append(f"{current}{count}")

    for ch in text:
        cat = ch.isalpha()
        if cat:
            cls = "L"
        elif ch.isdigit():
            cls = "N"
        elif ch.isspace():
            cls = "W"
        else:
            # Keep punctuation/symbol names in the same spirit as Stage 09.
            import unicodedata
            try:
                name = unicodedata.name(ch, "UNKNOWN")
            except Exception:
                name = "UNKNOWN"
            category = unicodedata.category(ch)
            cls = ("P:" if category.startswith("P") else "S:") + name
        if cls == current:
            count += 1
        else:
            flush()
            current = cls
            count = 1
    flush()
    return "|".join(result)


def _stage09_char_class(ch):
    import unicodedata
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


def _stage09_structural_pattern(text):
    import unicodedata
    text = normalize_text(text)
    out = []
    for ch in text:
        cls = _stage09_char_class(ch)
        if cls == "LETTER":
            out.append("L")
        elif cls == "NUMBER":
            out.append("N")
        elif cls == "SYMBOL":
            out.append("S:" + unicodedata.name(ch, "UNKNOWN"))
        elif cls == "PUNCTUATION":
            out.append("P:" + unicodedata.name(ch, "UNKNOWN"))
        elif cls == "MARK":
            out.append("M:" + unicodedata.name(ch, "UNKNOWN"))
        elif cls == "SPACE":
            out.append("SPACE")
        else:
            out.append(cls)
    return "|".join(out)


def _stage09_generalized_transformation(original, source):
    import unicodedata
    original = normalize_text(original)
    source = normalize_text(source)
    if not original or not source:
        return None
    differences = []
    if len(original) == len(source):
        for a, b in zip(original, source):
            if a != b:
                differences.append((_stage09_char_class(a), _stage09_char_class(b)))
    return {
        "same_length": len(original) == len(source),
        "length_delta": len(source) - len(original),
        "original_generalized_shape": _generalized_shape(original),
        "source_generalized_shape": _generalized_shape(source),
        "difference_count": len(differences),
        "difference_classes": differences,
    }


def _local_signature_matches(original, source, key_data):
    """Match one Stage 09 local transformation without literal OCR mappings."""
    if not isinstance(key_data, list) or len(key_data) != 8:
        return False
    _, original_name, source_name, original_class, source_class, left_shape, right_shape, position_class = key_data
    if len(original) != len(source) or not original:
        return False
    import unicodedata
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(original, source)) if a != b]
    if len(differences) != 1:
        return False
    i, a, b = differences[0]
    if unicodedata.name(a, "UNKNOWN") != original_name:
        return False
    if unicodedata.name(b, "UNKNOWN") != source_name:
        return False
    if _stage09_char_class(a) != original_class or _stage09_char_class(b) != source_class:
        return False
    left = _generalized_shape(original[max(0, i-2):i])
    right = _generalized_shape(original[i+1:i+3])
    position = "START" if i == 0 else "END" if i == len(original)-1 else "MIDDLE"
    return left == left_shape and right == right_shape and position == position_class


def find_learned_match(
    original_text,
    source_text,
    learning_patterns,
    object_type="PARAGRAPH",
    table_index=0,
    column_index=0,
):
    """
    Apply Stage 09's reusable rule families conservatively.

    Exact evidence is strongest.  Context/structural/generalized rules are
    only used with their stored context.  Cross-context and local rules are
    used only when Stage 09 marked them reusable (including its document-count
    requirements).  Learned rules never invent a literal correction: they
    choose the current original/source witness according to the learned action.
    """
    original = normalize_for_comparison(original_text)
    source = normalize_for_comparison(source_text)
    if not original or not source:
        return None

    typ = _current_object_type(object_type)
    context = _current_learning_context(typ, table_index, column_index)

    # Prefer the most specific rule family.
    family_order = [
        "exact", "generalized", "structural", "context",
        "cross_generalized", "cross_structural", "character_edit", "local"
    ]

    for family in family_order:
        for pattern in learning_patterns:
            if pattern.get("family") != family:
                continue
            if pattern.get("object_type", typ) not in {typ, ""}:
                continue
            action = pattern.get("action", "")
            verified_text = pattern.get("verified_text", "")
            key_data = pattern.get("key")

            if family == "exact":
                if (
                    normalize_for_comparison(pattern.get("original_ocr", "")) == original
                    and normalize_for_comparison(pattern.get("source_ocr", "")) == source
                ):
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text, verified_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
                    }

            elif family == "generalized":
                if not isinstance(key_data, list) or len(key_data) != 3:
                    continue
                _, stored_context, stored_gt_json = key_data
                if str(stored_context) != context:
                    continue
                try:
                    stored_gt = json.loads(stored_gt_json)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                current_gt = _stage09_generalized_transformation(original_text, source_text)
                if current_gt == stored_gt:
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
                    }

            elif family == "structural":
                if not isinstance(key_data, list) or len(key_data) != 4:
                    continue
                _, stored_context, orig_shape, src_shape = key_data
                if str(stored_context) != context:
                    continue
                if (
                    _generalized_shape(original_text) == str(orig_shape)
                    and _generalized_shape(source_text) == str(src_shape)
                ):
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
                    }

            elif family == "context":
                if not isinstance(key_data, list) or len(key_data) != 3:
                    continue
                _, stored_context, orig_structural = key_data
                if (
                    str(stored_context) == context
                    and _stage09_structural_pattern(original_text) == str(orig_structural)
                ):
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
                    }

            elif family == "cross_generalized":
                if not isinstance(key_data, list) or len(key_data) != 2:
                    continue
                stored_typ, stored_gt_json = key_data
                if str(stored_typ) != typ:
                    continue
                try:
                    stored_gt = json.loads(stored_gt_json)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if _stage09_generalized_transformation(original_text, source_text) == stored_gt:
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
                    }

            elif family == "cross_structural":
                if not isinstance(key_data, list) or len(key_data) != 3:
                    continue
                stored_typ, orig_shape, src_shape = key_data
                if (
                    str(stored_typ) == typ
                    and _generalized_shape(original_text) == str(orig_shape)
                    and _generalized_shape(source_text) == str(src_shape)
                ):
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
                    }

            elif family == "character_edit":
                if not isinstance(key_data, list) or len(key_data) != 8:
                    continue
                (
                    stored_typ,
                    witness_name,
                    verified_name,
                    witness_class,
                    verified_class,
                    left_shape,
                    right_shape,
                    position_class,
                ) = key_data
                # Character-edit rules represent repeated Unicode-level OCR
                # confusions.  They are intentionally reusable across
                # paragraphs and table cells once promoted by Stage 09.
                # Other learning families remain object-type restricted.
                if str(stored_typ) not in {"PARAGRAPH", "TABLE_CELL", typ}:
                    continue
                if not original_text or not source_text:
                    continue

                # Stage 09 learns from the OCR witness that the human did
                # NOT confirm. Therefore matching must be performed against
                # source_text when the source was the unconfirmed witness,
                # while the returned value is reconstructed from the current
                # original_text at the same character position.
                #
                # Use the same harmless text normalization already used by
                # Stage 06 for verification before comparing the character
                # edit. This prevents whitespace/encoding representation
                # differences from blocking an otherwise valid learned
                # character-edit rule. No literal OCR value is stored or
                # matched here.
                original_match_text = repair_mojibake(
                    normalize_text(original_text)
                )
                source_match_text = repair_mojibake(
                    normalize_text(source_text)
                )

                if (
                    not original_match_text
                    or not source_match_text
                    or len(original_match_text) != len(source_match_text)
                ):
                    continue

                differences = []
                for i, ch in enumerate(source_match_text):
                    if unicodedata.name(ch, "UNKNOWN") != str(witness_name):
                        continue
                    if _stage09_char_class(ch) != str(witness_class):
                        continue
                    # Character-edit rules use the learned witness
                    # character itself, not a literal/local numeric context.
                    # Older records may contain local shapes, so preserve
                    # those checks for such records; new ANY fields are
                    # intentionally context-independent.
                    if (
                        str(left_shape) not in {"ANY", ""}
                        or str(right_shape) not in {"ANY", ""}
                        or str(position_class) not in {"ANY", ""}
                    ):
                        left = _generalized_shape(source_match_text[max(0, i-2):i])
                        right = _generalized_shape(source_match_text[i+1:i+3])
                        position = (
                            "START" if i == 0 else
                            "END" if i == len(source_text)-1 else
                            "MIDDLE"
                        )
                        if not (
                            left == str(left_shape)
                            and right == str(right_shape)
                            and position == str(position_class)
                        ):
                            continue
                    differences.append((i, ch))

                if len(differences) != 1:
                    continue


                position = differences[0][0]

                # Reject invalid positions before indexing the original text.
                if (
                    position < 0
                    or position >= len(original_match_text)
                ):
                    continue

                # Resolve the learned target Unicode character safely.
                try:
                    target_char = unicodedata.lookup(str(verified_name))
                except (KeyError, TypeError):
                    continue

                if _stage09_char_class(target_char) != str(verified_class):
                    continue

                # The original character must be either the learned OCR
                # witness or the verified target. Never alter an unrelated
                # character merely because it occupies the same position.
                original_char = original_match_text[position]
                witness_char = differences[0][1]

                if original_char not in {witness_char, target_char}:
                    continue

                # No correction is necessary if the original is already right.
                if original_char == target_char:
                    continue

                verified_value = (
                    original_match_text[:position]
                    + target_char
                    + original_match_text[position + 1:]
                )

                # A learned correction must produce an actual text change.
                if verified_value == original_match_text:
                    continue

                return {
                    "verified_text": verified_value,
                    "repeat_count": pattern.get("repeat_count", 0),
                    "learned_family": family,
                }


            elif family == "local":
                if _local_signature_matches(original_text, source_text, key_data):
                    return {
                        "verified_text": _pattern_action_result(action, original_text, source_text),
                        "repeat_count": pattern.get("repeat_count", 0),
                        "learned_family": family,
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

        # Apply the shared technical-character policy directly.
        # Do not infer risk from arbitrary Latin letters or document-specific
        # prefixes; those exceptions create inconsistent, hardcoded behavior.
        if (
            contains_mojibake_artifact(original)
            or contains_mojibake_artifact(source)
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

        # Use the same shared technical-character policy as table cells.
        # Avoid a second, slightly different list of suspicious letters.
        if (
            contains_mojibake_artifact(original)
            or contains_mojibake_artifact(source)
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
    # 4. NUMERIC DISAGREEMENT
    # --------------------------------------------------------

    # Keep numeric conflicts out of the general similarity path. A paragraph
    # can be textually very similar while containing a materially different
    # measurement (for example, 24.4 versus 24.9). This mirrors the conservative
    # table-cell policy: flag the conflict, preserve the extracted value, and
    # never automatically substitute the source OCR value.
    original_numbers = numeric_content(original)
    source_numbers = numeric_content(source)

    if (original_numbers or source_numbers) and not numeric_values_equal(
        original,
        source,
    ):
        return {
            "decision": "NUMERIC_DISAGREEMENT",
            "verified_text": original,
            "reason": (
                "Numeric content differs between the paragraph extraction "
                "and source-image OCR. No automatic numeric correction is allowed."
            ),
            "confidence": source_confidence,
            "original_numbers": original_numbers,
            "source_numbers": source_numbers,
        }

    # --------------------------------------------------------
    # 5. GENERAL PARAGRAPH DISAGREEMENT
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

def _process_current_input():

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
        f"Reusable learned OCR patterns : "
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
    # PROCESS THIS SINGLE DOCUMENT
    #
    # Stage 04 now writes one document object directly to each
    # JSON file, rather than wrapping it in {"documents": [...]}.
    # --------------------------------------------------------

    document = verified_data

    document_name = document.get(
        "document",
        document.get(
            "name",
            INPUT_FILE.stem
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

                        image_key = str(image_path)
                        page_image = PAGE_IMAGE_CACHE.get(image_key)
                        if page_image is None:
                            page_image = Image.open(
                                image_path
                            ).convert("RGB")
                            PAGE_IMAGE_CACHE[image_key] = page_image

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

                crop_key = (
                    str(image_path),
                    tuple(int(v) for v in bbox[:4])
                    if bbox and len(bbox) >= 4 else None,
                )
                evidence_key = (
                    crop_key,
                    normalize_for_comparison(original_text),
                )
                cached_evidence = SOURCE_OCR_CACHE.get(evidence_key)
                if cached_evidence is None:
                    cached_evidence = get_source_ocr_evidence(
                        crop,
                        original_text,
                    )
                    SOURCE_OCR_CACHE[evidence_key] = cached_evidence

                variants, source_text, source_confidence = cached_evidence

                # --------------------------------------------
                # VERIFY PARAGRAPH
                # --------------------------------------------

                result = verify_paragraph_block(
                    original_text,
                    source_text,
                    source_confidence
                )

                learned_match = find_learned_match(
                    original_text,
                    source_text,
                    learning_patterns,
                    object_type="PARAGRAPH",
                )

                if learned_match and (
                    # Matching original/source OCR is stronger evidence than
                    # a reusable learned transformation. Never let a learned
                    # rule rewrite text when both OCR witnesses agree.
                    normalize_for_comparison(original_text)
                    != normalize_for_comparison(source_text)
                ) and (
                    result.get("decision") in REVIEW_DECISIONS
                    or learned_match.get("learned_family") == "character_edit"
                ):
                    result = {
                        "decision": "LEARNED_PATTERN_VERIFIED",
                        "verified_text": learned_match["verified_text"],
                        "reason": (
                            "A reusable Stage 09 human-verified learning rule "
                            "matched the current OCR evidence."
                        ),
                        "confidence": 1.0,
                        "learned_repeat_count": learned_match["repeat_count"],
                        "learned_family": learned_match["learned_family"],
                    }
                    learned_verified += 1

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

                        # ------------------------------------
                        # EMPTY STRUCTURAL / PLACEHOLDER CELL
                        # ------------------------------------
                        #
                        # Table reconstruction can legitimately create
                        # an empty cell for a structural position, for
                        # example when a header is merged across columns.
                        # Such a cell has neither OCR text nor a source
                        # bounding box, so there is no visual evidence
                        # for Stage 06 to verify or crop.
                        #
                        # Do NOT turn this into NO_SOURCE_IMAGE and do
                        # NOT place it in the manual-review queue. This
                        # rule is deliberately generic: it depends only
                        # on the cell's own evidence, never on a page,
                        # table, row, column, or document name.
                        #
                        # A cell with an empty text value BUT a valid bbox
                        # is NOT treated as structural; its source image
                        # can still contain text, so normal verification
                        # continues below.
                        # ------------------------------------

                        if (
                            not original_text
                            and not bbox
                        ):

                            result = {

                                "decision":
                                    "EMPTY_STRUCTURAL_CELL",

                                "verified_text":
                                    "",

                                "reason":
                                    (
                                        "The reconstructed table cell is empty "
                                        "and has no source bounding box; it is "
                                        "treated as a structural or placeholder "
                                        "cell rather than as a missing source image."
                                    ),

                                "confidence":
                                    1.0,
                            }

                            decisions[
                                "EMPTY_STRUCTURAL_CELL"
                            ] = (
                                decisions.get(
                                    "EMPTY_STRUCTURAL_CELL",
                                    0
                                )
                                + 1
                            )

                            cell[
                                "verification_status"
                            ] = (
                                "EMPTY_STRUCTURAL_CELL"
                            )

                            cell[
                                "verified_text"
                            ] = ""

                            cell[
                                "final_verified_text"
                            ] = ""

                            cell[
                                "final_verification_status"
                            ] = (
                                "EMPTY_STRUCTURAL_CELL"
                            )

                            cell[
                                "review_crop"
                            ] = ""

                            # It is not a review item and therefore does
                            # not increment needs_review. Continue to the
                            # next reconstructed cell.
                            continue

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

                                image_key = str(image_path)
                                page_image = PAGE_IMAGE_CACHE.get(image_key)
                                if page_image is None:
                                    page_image = Image.open(
                                        image_path
                                    ).convert("RGB")
                                    PAGE_IMAGE_CACHE[image_key] = page_image

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

                        crop_key = (
                            str(image_path),
                            tuple(int(v) for v in bbox[:4])
                            if bbox and len(bbox) >= 4 else None,
                        )
                        evidence_key = (
                            crop_key,
                            normalize_for_comparison(original_text),
                        )
                        cached_evidence = SOURCE_OCR_CACHE.get(evidence_key)
                        if cached_evidence is None:
                            cached_evidence = get_source_ocr_evidence(
                                crop,
                                original_text,
                            )
                            SOURCE_OCR_CACHE[evidence_key] = cached_evidence

                        variants, source_text, source_confidence = cached_evidence

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
                            learning_patterns,
                            object_type="TABLE_CELL",
                            table_index=table_index,
                            column_index=column_index,
                        )

                        if learned_match and (
                    # Matching original/source OCR is stronger evidence than
                    # a reusable learned transformation. Never let a learned
                    # rule rewrite text when both OCR witnesses agree.
                    normalize_for_comparison(original_text)
                    != normalize_for_comparison(source_text)
                ) and (
                    result.get("decision") in REVIEW_DECISIONS
                    or learned_match.get("learned_family") == "character_edit"
                ):

                            result = {

                                "decision":
                                    "LEARNED_PATTERN_VERIFIED",

                                "verified_text":
                                    learned_match[
                                        "verified_text"
                                    ],

                                "reason":
                                    (
                                        "A reusable Stage 09 human-verified "
                                        "learning rule matched the current OCR "
                                        "evidence and selected the learned action."
                                    ),

                                "confidence":
                                    1.0,

                                "learned_repeat_count":
                                    learned_match[
                                        "repeat_count"
                                    ],

                                "learned_family":
                                    learned_match.get(
                                        "learned_family",
                                        "unknown"
                                    ),
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
    # REVIEW QUEUE DEDUPLICATION
    # ========================================================

    def _review_object_type(item):
        if "table" in item:
            return "table_cell"
        if "paragraph_block" in item:
            return "paragraph_block"
        return "unknown"

    def _bbox_tuple(item):
        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            return None
        try:
            return tuple(float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            return None

    def _bbox_overlap_ratio(a, b):
        box_a = _bbox_tuple(a)
        box_b = _bbox_tuple(b)
        if not box_a or not box_b:
            return 0.0
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smallest = min(area_a, area_b)
        return intersection / smallest if smallest > 0 else 0.0

    def _review_text_key(item):
        return (
            str(item.get("document", "")),
            str(item.get("page", "")),
            normalize_text(item.get("original_text", "")).casefold(),
            normalize_text(item.get("source_text", "")).casefold(),
        )

    def _deduplicate_review_items(items):
        exact_seen = {}
        exact_removed = 0
        deduped = []

        for item in items:
            key = _review_text_key(item)
            if key not in exact_seen:
                exact_seen[key] = len(deduped)
                deduped.append(item)
                continue

            existing_index = exact_seen[key]
            existing = deduped[existing_index]
            # Prefer a table-cell review over a paragraph review for the
            # same exact evidence because table cells are the canonical
            # structured representation of tabular content.
            if (
                _review_object_type(existing) != "table_cell"
                and _review_object_type(item) == "table_cell"
            ):
                deduped[existing_index] = item
            exact_removed += 1

        table_items = [
            item for item in deduped
            if _review_object_type(item) == "table_cell"
        ]

        final_items = []
        overlap_removed = 0
        for item in deduped:
            if _review_object_type(item) != "paragraph_block":
                final_items.append(item)
                continue

            duplicate_region = False
            for table_item in table_items:
                if (
                    str(item.get("document", ""))
                    == str(table_item.get("document", ""))
                    and str(item.get("page", ""))
                    == str(table_item.get("page", ""))
                    and _bbox_overlap_ratio(item, table_item) >= BBOX_OVERLAP_THRESHOLD
                ):
                    duplicate_region = True
                    break

            if duplicate_region:
                overlap_removed += 1
            else:
                final_items.append(item)

        return final_items, exact_removed, overlap_removed

    review_items, exact_removed, overlap_removed = _deduplicate_review_items(
        review_items
    )

    # ========================================================
    # REMOVE STALE REVIEWS ALREADY RESOLVED BY LEARNING
    # ========================================================
    #
    # A learned rule can update the verification metadata for an
    # evidence item while an older review entry for the same
    # original/source pair still exists in the accumulated queue.
    # Such an item must not be sent to Stage 08 again.
    #
    # Match on document + page + original text + source text.
    # This deliberately removes only evidence that is explicitly
    # marked LEARNED_PATTERN_VERIFIED in the current verified data.
    # Legitimate unresolved reviews remain untouched.
    # ========================================================

    learned_evidence = set()

    def _collect_learned_evidence(node, current_document=""):

        if isinstance(node, dict):

            document_name_for_node = str(
                node.get(
                    "document",
                    current_document
                )
            ).strip()

            original_value = normalize_text(
                node.get(
                    "original_text",
                    node.get(
                        "text",
                        ""
                    )
                )
            )

            source_value = normalize_text(
                node.get(
                    "source_ocr",
                    ""
                )
            )

            final_status = str(
                node.get(
                    "final_verification_status",
                    node.get(
                        "verification_status",
                        ""
                    )
                )
            ).strip()

            if (
                final_status == "LEARNED_PATTERN_VERIFIED"
                and original_value
                and source_value
            ):
                learned_evidence.add(
                    (
                        document_name_for_node,
                        str(
                            node.get(
                                "page",
                                ""
                            )
                        ),
                        original_value.casefold(),
                        source_value.casefold(),
                    )
                )

            for value in node.values():
                _collect_learned_evidence(
                    value,
                    document_name_for_node
                )

        elif isinstance(node, list):

            for value in node:
                _collect_learned_evidence(
                    value,
                    current_document
                )

    _collect_learned_evidence(
        verified_data
    )

    learned_reviews_removed = 0
    cleaned_review_items = []

    for item in review_items:

        item_key = (
            str(
                item.get(
                    "document",
                    ""
                )
            ).strip(),
            str(
                item.get(
                    "page",
                    ""
                )
            ),
            normalize_text(
                item.get(
                    "original_text",
                    ""
                )
            ).casefold(),
            normalize_text(
                item.get(
                    "source_text",
                    ""
                )
            ).casefold(),
        )

        if item_key in learned_evidence:
            learned_reviews_removed += 1
            continue

        cleaned_review_items.append(
            item
        )

    review_items = cleaned_review_items
    needs_review = len(review_items)

    print()
    print(
        "Learned-pattern stale review items removed : "
        + str(learned_reviews_removed)
    )
    print("REVIEW QUEUE DEDUPLICATION")
    print("Exact duplicate review items removed : " + str(exact_removed))
    print("Overlapping paragraph/table reviews removed : " + str(overlap_removed))
    print("Final review items : " + str(len(review_items)))

    # Recalculate decision counts from the final queue so the summary
    # reflects the actual review file rather than the pre-dedup queue.
    final_review_decisions = Counter()
    for item in review_items:
        decision_name = item.get("verification", {}).get("decision", "")
        if decision_name:
            final_review_decisions[decision_name] += 1

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

def main():
    """
    Process every individual PDF JSON independently.

    Each PDF gets its own verified JSON and review JSON.
    The OCR learning memory remains one cumulative file shared by
    all documents.
    """

    global INPUT_FILE
    global REVIEW_CROP_DIR
    global VERIFIED_FILE
    global REVIEW_FILE

    print("=" * 70)
    print("SOURCE IMAGE ACCURACY VERIFICATION")
    print("PER-PDF VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT DIRECTORY
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Final data directory not found:\n"
            f"{INPUT_DIR}"
        )

    json_files = sorted(
        [
            path
            for path in INPUT_DIR.glob("*.json")
            if path.name != "all_documents.json"
        ]
    )

    print()
    print(
        f"Individual document JSON files: "
        f"{len(json_files)}"
    )

    for path in json_files:
        print(
            f"  {path.name}"
        )

    if not json_files:

        raise FileNotFoundError(
            "No individual PDF JSON files were found in "
            f"{INPUT_DIR}"
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REVIEW_REQUIRED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REVIEW_CROP_ROOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # REMOVE OLD COMBINED STAGE-06 OUTPUTS
    # --------------------------------------------------------

    old_outputs = [
        OUTPUT_DIR / "all_documents_verified.json",
        OUTPUT_DIR / "review_required.json",
    ]

    for old_file in old_outputs:

        if old_file.exists():

            print()
            print(
                f"Removing old combined output: "
                f"{old_file}"
            )

            old_file.unlink()

    # --------------------------------------------------------
    # PROCESS EACH PDF JSON
    # --------------------------------------------------------

    processed = 0

    for input_json in json_files:

        document_name = input_json.stem

        INPUT_FILE = input_json

        VERIFIED_FILE = (
            OUTPUT_DIR
            / f"{document_name}.json"
        )

        REVIEW_FILE = (
            REVIEW_REQUIRED_DIR
            / f"{document_name}.json"
        )

        REVIEW_CROP_DIR = (
            REVIEW_CROP_ROOT_DIR
            / document_name
        )

        REVIEW_CROP_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print("=" * 70)
        print(
            f"PROCESSING: "
            f"{document_name}"
        )
        print("=" * 70)

        _process_current_input()

        processed += 1

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-PDF VERIFICATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Documents processed: "
        f"{processed}"
    )

    print()
    print(
        "Verified JSON directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Review JSON directory:"
    )

    print(
        REVIEW_REQUIRED_DIR
    )

    print()
    print(
        "Review crops directory:"
    )

    print(
        REVIEW_CROP_ROOT_DIR
    )

    print()
    print(
        "Cumulative learning memory:"
    )

    print(
        LEARNING_MEMORY_FILE
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
