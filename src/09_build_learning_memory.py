"""
Stage 09 — Learning Memory Builder (CLEAN)

Purpose
-------
Build reusable OCR-learning evidence from TRUSTED HUMAN decisions.

Design rules
------------
1. No hard-coded OCR error dictionary.
2. No hard-coded list of "learnable" characters.
3. Unicode properties are used dynamically.
4. Exact evidence remains available.
5. Generalized patterns are learned only from human decisions.
6. Generalized patterns require repeated, consistent human evidence.
7. Object type is preserved: paragraph blocks and table cells are never merged.
8. Conflicting human evidence is never made reusable.
9. This stage does NOT change OCR text. It only creates learning memory.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LEARNING_DIR = BASE_DIR / "output" / "learning"
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

# Persistent files: never remove these when clearing disposable OCR outputs.
DECISIONS_FILE = LEARNING_DIR / "manual_review_decisions.json"
MEMORY_FILE = LEARNING_DIR / "ocr_learning_memory.json"

# Minimum independent human confirmations before a generalized rule
# becomes eligible for automatic reuse.
MIN_REPEATS = 2

TRUSTED_HUMAN_DECISIONS = {
    "USER_CONFIRMED_ORIGINAL",
    "USER_CONFIRMED_SOURCE",
    "USER_ENTERED_CORRECTION",
    "USER_CORRECTION",
}


# ============================================================
# BASIC IO
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_review_outcomes(path: Path):
    """Load Stage 08 review outcomes separately from trusted learning decisions."""
    if not path.exists():
        return []
    try:
        data = load_json(path)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    outcomes = data.get("review_outcomes", [])
    if not isinstance(outcomes, list):
        outcomes = []

    # Backward compatibility: older Stage 08 files had only trusted decision
    # records. These are valid human outcomes, but we do not manufacture
    # skipped/automatic outcomes that were never recorded.
    if not outcomes:
        for d in data.get("decisions", []):
            if not isinstance(d, dict):
                continue
            if d.get("decision") not in TRUSTED_HUMAN_DECISIONS:
                continue
            outcomes.append({
                "document": d.get("document", ""),
                "page": d.get("page", ""),
                "object_type": d.get("object_type", ""),
                "original_text": d.get("original_text", ""),
                "source_text": d.get("source_text", ""),
                "verified_text": d.get("verified_text", ""),
                "outcome": d.get("decision", ""),
                "source": "LEGACY_DECISION_BACKFILL",
            })
    return outcomes


def review_outcome_summary(outcomes):
    """Summarize review activity without treating it as learning evidence."""
    counts = defaultdict(int)
    unique = set()
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        outcome = str(item.get("outcome", "")).strip()
        if not outcome:
            continue
        counts[outcome] += 1
        unique.add((
            str(item.get("document", "")).strip(),
            str(item.get("page", "")).strip(),
            str(item.get("table", "")).strip(),
            str(item.get("row", "")).strip(),
            str(item.get("column", "")).strip(),
            str(item.get("paragraph_block", "")).strip(),
            str(item.get("bbox", "")).strip(),
            str(item.get("original_text", "")).strip(),
            str(item.get("source_text", "")).strip(),
            outcome,
        ))
    human = sum(v for k,v in counts.items() if k in TRUSTED_HUMAN_DECISIONS)
    skipped = counts.get("USER_SKIPPED", 0)
    automatic = sum(v for k,v in counts.items() if k.startswith("AUTO_"))
    return {
        "total_review_outcomes": len(unique),
        "human_review_outcomes": human,
        "skipped_review_outcomes": skipped,
        "automatic_review_outcomes": automatic,
        "by_outcome": dict(sorted(counts.items())),
    }


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# SAFE TEXT NORMALIZATION
# ============================================================

def normalize_text(value) -> str:
    """
    Normalization only.

    IMPORTANT:
    This function does not contain OCR corrections.
    It does not say that O means 0, ± means +, Ω means O, etc.

    Unicode NFC normalization and whitespace normalization are
    representation normalization, not OCR learning.
    """
    text = "" if value is None else str(value)

    text = unicodedata.normalize("NFC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# DYNAMIC UNICODE / CHARACTER FEATURES
# ============================================================

def char_class(ch: str) -> str:
    """
    Determine a character class dynamically from Unicode properties.

    No document-specific character list is used.
    """
    if not ch:
        return "EMPTY"

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


def structural_pattern(text: str) -> str:
    """
    Convert text into a generic structural fingerprint.

    Literal letters/numbers are deliberately not retained.
    Symbols and punctuation retain their Unicode category and
    code point so that mathematical structure is not erased.

    Examples of the concept:

        T01      -> LETTER NUMBER NUMBER
        T02      -> LETTER NUMBER NUMBER

    But this function never says:
        T01 == T02
    as text. It only describes their structure.
    """
    text = normalize_text(text)

    result = []

    for ch in text:
        cls = char_class(ch)

        if cls == "LETTER":
            result.append("L")
        elif cls == "NUMBER":
            result.append("N")
        elif cls == "SYMBOL":
            result.append(
                "S:" + unicodedata.name(ch, "UNKNOWN")
            )
        elif cls == "PUNCTUATION":
            result.append(
                "P:" + unicodedata.name(ch, "UNKNOWN")
            )
        elif cls == "MARK":
            result.append(
                "M:" + unicodedata.name(ch, "UNKNOWN")
            )
        elif cls == "SPACE":
            result.append("SPACE")
        else:
            result.append(cls)

    return "|".join(result)


def generalized_shape(text: str) -> str:
    """
    More compact structure used for generalized matching.

    Consecutive letters become L{count}.
    Consecutive numbers become N{count}.
    Symbols/punctuation remain explicit.

    This is structural abstraction only. It is not a correction rule.
    """
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
        cls = char_class(ch)

        if cls == "LETTER":
            base = "L"
        elif cls == "NUMBER":
            base = "N"
        elif cls == "SPACE":
            base = "W"
        else:
            flush()

            if cls == "SYMBOL":
                groups.append(
                    "S:" + unicodedata.name(ch, "UNKNOWN")
                )
            elif cls == "PUNCTUATION":
                groups.append(
                    "P:" + unicodedata.name(ch, "UNKNOWN")
                )
            elif cls == "MARK":
                groups.append(
                    "M:" + unicodedata.name(ch, "UNKNOWN")
                )
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


# ============================================================
# OBJECT TYPE
# ============================================================

def object_type(decision: dict) -> str:
    """
    Preserve paragraph/table separation.

    Table cells remain TABLE_CELL.
    Paragraph blocks remain PARAGRAPH.
    """
    explicit = str(decision.get("object_type", "")).strip()

    if explicit:
        return explicit

    # Existing decision records normally contain table coordinates.
    # A positive table identifier is treated as a table cell.
    try:
        table = int(decision.get("table", 0) or 0)
    except (TypeError, ValueError):
        table = 0

    if table > 0:
        return "TABLE_CELL"

    return "PARAGRAPH"


# ============================================================
# CONTEXT
# ============================================================

def table_context(decision: dict) -> str:
    """
    Stable table context.

    Row number is intentionally excluded so that a learned pattern
    can apply to another row of the same column.

    Document identity is retained because formatting conventions can
    differ between document families.
    """
    try:
        table = int(decision.get("table", 0) or 0)
    except (TypeError, ValueError):
        table = 0

    try:
        column = int(decision.get("column", 0) or 0)
    except (TypeError, ValueError):
        column = 0

    if table <= 0:
        return ""

    return (
        f"table={table}|"
        f"column={column}"
    )


def learning_context(decision: dict) -> str:
    """
    Build a conservative context fingerprint.

    Table learning is column-aware.
    Paragraph learning is kept separate.
    """
    typ = object_type(decision)

    if typ == "TABLE_CELL":
        return (
            typ
            + "|"
            + table_context(decision)
        )

    return typ


# ============================================================
# TRANSFORMATION DESCRIPTION
# ============================================================

def transformation_signature(original: str, source: str):
    """
    Describe how the two OCR streams differ without embedding
    document-specific OCR correction mappings.

    The signature contains only:
      - lengths
      - structural shapes
      - character-class transitions
      - number of differing positions

    Literal O/0, I/1, Ω/O, ±/+ mappings are NOT hard-coded.
    """
    original = normalize_text(original)
    source = normalize_text(source)

    if not original or not source:
        return None

    original_classes = [char_class(c) for c in original]
    source_classes = [char_class(c) for c in source]

    differences = []

    if len(original) == len(source):
        for i, (a, b) in enumerate(
            zip(original, source)
        ):
            if a != b:
                differences.append(
                    (
                        char_class(a),
                        char_class(b),
                    )
                )

    return {
        "same_length": len(original) == len(source),
        "original_length": len(original),
        "source_length": len(source),
        "length_delta": len(source) - len(original),
        "original_shape": structural_pattern(original),
        "source_shape": structural_pattern(source),
        "original_generalized_shape": generalized_shape(original),
        "source_generalized_shape": generalized_shape(source),
        "difference_count": len(differences),
        "difference_classes": differences,
    }


def generalized_transformation(original: str, source: str):
    """
    Produce a transformation class that can be shared by different
    literal values.

    Literal characters are not used as a correction dictionary.
    """
    sig = transformation_signature(original, source)

    if sig is None:
        return None

    return {
        "same_length": sig["same_length"],
        "length_delta": sig["length_delta"],
        "original_generalized_shape": sig[
            "original_generalized_shape"
        ],
        "source_generalized_shape": sig[
            "source_generalized_shape"
        ],
        "difference_count": sig["difference_count"],
        "difference_classes": sig[
            "difference_classes"
        ],
    }


# ============================================================
# LOCAL / CHARACTER-LEVEL TRANSFORMATION
# ============================================================

def local_transformation_patterns(original: str, source: str):
    """
    Describe substitution-level OCR behavior using Unicode properties.

    This is deliberately learned from the human-reviewed evidence.  It does
    not contain a predefined OCR mapping such as Ω->O or ±->+.

    Only same-length substitutions are emitted here. Insertions/deletions are
    left to the broader structural learners because alignment can otherwise
    create unsafe local rules.
    """
    original = normalize_text(original)
    source = normalize_text(source)
    if not original or not source or len(original) != len(source):
        return []

    out = []
    for i, (a, b) in enumerate(zip(original, source)):
        if a == b:
            continue

        left = original[max(0, i-2):i]
        right = original[i+1:i+3]
        out.append({
            "original_name": unicodedata.name(a, "UNKNOWN"),
            "source_name": unicodedata.name(b, "UNKNOWN"),
            "original_class": char_class(a),
            "source_class": char_class(b),
            "left_shape": generalized_shape(left),
            "right_shape": generalized_shape(right),
            "position_class": (
                "START" if i == 0 else
                "END" if i == len(original)-1 else
                "MIDDLE"
            ),
        })
    return out


def local_pattern_key(typ: str, item: dict):
    return json.dumps(
        [
            typ,
            item["original_name"],
            item["source_name"],
            item["original_class"],
            item["source_class"],
            item["left_shape"],
            item["right_shape"],
            item["position_class"],
        ],
        ensure_ascii=False,
    )


def character_edit_patterns(witness: str, verified: str):
    """
    Describe the human-confirmed OCR witness as a Unicode-level edit.

    The witness is the OCR value that was *not* the human-confirmed value:
      - USER_CONFIRMED_ORIGINAL  -> source -> verified
      - USER_CONFIRMED_SOURCE    -> original -> verified
      - USER_ENTERED_CORRECTION  -> original -> verified

    This distinction is critical.  A USER_CONFIRMED_ORIGINAL decision means
    the source OCR is the erroneous witness, even though ``original_text``
    already contains the correct value.  Only one-character substitutions
    are promoted, so a numeric value or surrounding text is never learned as
    a literal replacement.
    """
    witness = normalize_text(witness)
    verified = normalize_text(verified)
    if not witness or not verified or len(witness) != len(verified):
        return []

    differences = []
    for i, (a, b) in enumerate(zip(witness, verified)):
        if a == b:
            continue
        differences.append((i, a, b))

    if len(differences) != 1:
        return []

    i, a, b = differences[0]
    # Do not bind the learned symbol edit to the literal numeric/text
    # neighborhood or to its exact position. The same OCR confusion can
    # occur in different numeric layouts. Safety is provided by the
    # repeated unanimous evidence across multiple documents required when
    # the pattern is promoted into memory.
    return [{
        "witness_name": unicodedata.name(a, "UNKNOWN"),
        "verified_name": unicodedata.name(b, "UNKNOWN"),
        "witness_class": char_class(a),
        "verified_class": char_class(b),
        "left_shape": "ANY",
        "right_shape": "ANY",
        "position_class": "ANY",
    }]


def character_edit_key(typ: str, item: dict):
    return json.dumps(
        [
            typ,
            item["witness_name"],
            item["verified_name"],
            item["witness_class"],
            item["verified_class"],
            item["left_shape"],
            item["right_shape"],
            item["position_class"],
        ],
        ensure_ascii=False,
    )


# ============================================================
# HUMAN ACTION
# ============================================================

def learning_action(decision: dict):
    decision_type = decision.get("decision", "")

    if decision_type == "USER_CONFIRMED_ORIGINAL":
        return "USE_CURRENT_ORIGINAL"

    if decision_type == "USER_CONFIRMED_SOURCE":
        return "USE_CURRENT_SOURCE"

    # A user-entered correction is evidence that the literal
    # verified value matters, but it is not converted into a
    # generalized replacement rule here.
    return None


# ============================================================
# EXACT EVIDENCE
# ============================================================

def exact_key(original: str, source: str):
    return json.dumps(
        [
            normalize_text(original),
            normalize_text(source),
        ],
        ensure_ascii=False,
    )


# ============================================================
# DECISION EVIDENCE
# ============================================================

def evidence_identity(decision: dict):
    """
    Identify the human-reviewed observation.

    This is used to ensure repeated evidence is not merely the
    same decision duplicated in the decision file.
    """
    return (
        str(decision.get("document", "")),
        str(decision.get("page", "")),
        str(decision.get("table", "")),
        str(decision.get("row", "")),
        str(decision.get("column", "")),
        str(decision.get("bbox", "")),
        normalize_text(
            decision.get("original_text", "")
        ),
        normalize_text(
            decision.get("source_text", "")
        ),
    )


# ============================================================
# BUILD MEMORY
# ============================================================

def empty_memory():
    return {
        "version": 7,
        "description": (
            "Learning memory generated only from trusted human "
            "review decisions. Exact evidence is retained. "
            "Generalized structural and transformation patterns "
            "are derived dynamically from Unicode properties and "
            "document context. No predefined OCR correction "
            "dictionary is used."
        ),
        "minimum_repeats_required": MIN_REPEATS,
        "total_manual_decisions": 0,
        "trusted_human_decisions": 0,
        "exact_patterns": {},
        "generalized_patterns": {},
        "structural_patterns": {},
        "context_patterns": {},
        "cross_context_generalized_patterns": {},
        "cross_context_structural_patterns": {},
        "local_transformation_patterns": {},
        "character_edit_transformation_patterns": {},
        "conflicts": [],
        "audit": [],
        "review_activity": {},
    }


def add_pattern(
    groups,
    key,
    decision,
):
    groups[key].append(decision)


def build_memory(decisions):
    memory = empty_memory()

    exact_groups = defaultdict(list)
    generalized_groups = defaultdict(list)
    structural_groups = defaultdict(list)
    context_groups = defaultdict(list)
    cross_generalized_groups = defaultdict(list)
    cross_structural_groups = defaultdict(list)
    local_groups = defaultdict(list)
    character_edit_groups = defaultdict(list)

    seen_evidence = set()

    for decision in decisions:
        if (
            decision.get("decision")
            not in TRUSTED_HUMAN_DECISIONS
        ):
            continue

        original = normalize_text(
            decision.get("original_text", "")
        )

        source = normalize_text(
            decision.get("source_text", "")
        )

        verified = normalize_text(
            decision.get("verified_text", "")
        )

        if not verified:
            continue

        memory["trusted_human_decisions"] += 1

        identity = evidence_identity(decision)

        if identity in seen_evidence:
            continue

        seen_evidence.add(identity)

        typ = object_type(decision)
        context = learning_context(decision)
        action = learning_action(decision)

        # --------------------------------------------------------
        # Exact evidence
        # --------------------------------------------------------

        if original or source:
            add_pattern(
                exact_groups,
                (
                    typ,
                    exact_key(original, source),
                ),
                decision,
            )

        # --------------------------------------------------------
        # Generalized transformation
        # --------------------------------------------------------

        if original and source:
            gt = generalized_transformation(
                original,
                source,
            )

            if gt is not None:
                key = (
                    typ,
                    context,
                    json.dumps(
                        gt,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )

                # Cross-context candidate: deliberately excludes document,
                # table-column, and paragraph/table context. It is still
                # restricted to the same object family and the exact same
                # generalized transformation. Reuse later requires repeated
                # evidence from at least two documents.
                cross_generalized_groups[
                    (
                        typ,
                        json.dumps(
                            gt,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                ].append(decision)

                add_pattern(
                    generalized_groups,
                    key,
                    decision,
                )

                structural_key = (
                    typ,
                    context,
                    gt[
                        "original_generalized_shape"
                    ],
                    gt[
                        "source_generalized_shape"
                    ],
                )

                cross_structural_groups[
                    (
                        typ,
                        gt[
                            "original_generalized_shape"
                        ],
                        gt[
                            "source_generalized_shape"
                        ],
                    )
                ].append(decision)

                add_pattern(
                    structural_groups,
                    structural_key,
                    decision,
                )

        # --------------------------------------------------------
        # Local character-level transformation candidates
        # --------------------------------------------------------

        if original and source:
            for local_item in local_transformation_patterns(original, source):
                local_groups[
                    local_pattern_key(typ, local_item)
                ].append(decision)

        # --------------------------------------------------------
        # Human-confirmed character-edit candidates
        # --------------------------------------------------------
        # The witness must be the OCR value that the human did NOT confirm.
        # For USER_CONFIRMED_ORIGINAL this is SOURCE; for
        # USER_CONFIRMED_SOURCE and explicit corrections this is ORIGINAL.

        decision_type = str(decision.get("decision", "")).strip().upper()
        if verified:
            if decision_type == "USER_CONFIRMED_ORIGINAL":
                character_edit_witness = source
            else:
                character_edit_witness = original

            if character_edit_witness:
                for edit_item in character_edit_patterns(
                    character_edit_witness,
                    verified,
                ):
                    character_edit_groups[
                        character_edit_key(typ, edit_item)
                    ].append(decision)

        # --------------------------------------------------------
        # Context + original structure
        # --------------------------------------------------------

        if original:
            context_key = (
                typ,
                context,
                structural_pattern(original),
            )

            add_pattern(
                context_groups,
                context_key,
                decision,
            )

    # ============================================================
    # EXACT PATTERNS
    # ============================================================

    for key, examples in exact_groups.items():
        verified_values = {
            normalize_text(
                x.get("verified_text", "")
            )
            for x in examples
            if normalize_text(
                x.get("verified_text", "")
            )
        }

        if len(verified_values) == 1:
            verified_value = next(
                iter(verified_values)
            )

            memory["exact_patterns"][
                json.dumps(
                    key,
                    ensure_ascii=False,
                )
            ] = {
                "verified_text": verified_value,
                "repeat_count": len(examples),
                "human_confirmed": True,
            }

        elif len(verified_values) > 1:
            memory["conflicts"].append(
                {
                    "type": "EXACT_CONFLICT",
                    "key": key,
                    "verified_values": sorted(
                        verified_values
                    ),
                    "examples": len(examples),
                }
            )

    # ============================================================
    # GENERALIZED PATTERNS
    # ============================================================

    for key, examples in generalized_groups.items():
        actions = [
            learning_action(x)
            for x in examples
        ]

        actions = [
            x for x in actions
            if x is not None
        ]

        if not actions:
            continue

        action_counts = defaultdict(int)

        for action in actions:
            action_counts[action] += 1

        best_action, best_count = max(
            action_counts.items(),
            key=lambda pair: pair[1],
        )

        total = len(actions)

        # Require repeated evidence AND unanimous action.
        if (
            total >= MIN_REPEATS
            and best_count == total
        ):
            memory["generalized_patterns"][
                json.dumps(
                    key,
                    ensure_ascii=False,
                )
            ] = {
                "action": best_action,
                "repeat_count": total,
                "human_confirmed": True,
            }

        elif len(action_counts) > 1:
            memory["conflicts"].append(
                {
                    "type":
                        "GENERALIZED_ACTION_CONFLICT",
                    "key": key,
                    "actions": dict(
                        action_counts
                    ),
                }
            )

    # ============================================================
    # CROSS-CONTEXT GENERALIZED PATTERNS
    # ============================================================
    # These are intentionally stricter than ordinary generalized patterns:
    #   * same object family only
    #   * same generalized transformation
    #   * at least 3 trusted examples
    #   * evidence from at least 2 different documents
    #   * unanimous human action
    # This lets genuinely repeated OCR behavior transfer to a new document
    # without creating a literal character replacement rule.

    for key, examples in cross_generalized_groups.items():
        actions = [learning_action(x) for x in examples]
        actions = [x for x in actions if x is not None]
        docs = {str(x.get("document", "")).strip() for x in examples}
        docs.discard("")

        if not actions:
            continue

        counts = defaultdict(int)
        for action in actions:
            counts[action] += 1

        best_action, best_count = max(
            counts.items(), key=lambda pair: pair[1]
        )

        if len(actions) >= 3 and len(docs) >= 2 and best_count == len(actions):
            memory["cross_context_generalized_patterns"][
                json.dumps(key, ensure_ascii=False)
            ] = {
                "action": best_action,
                "repeat_count": len(actions),
                "document_count": len(docs),
                "human_confirmed": True,
            }

    # ============================================================
    # CROSS-CONTEXT STRUCTURAL PATTERNS
    # ============================================================

    for key, examples in cross_structural_groups.items():
        actions = [learning_action(x) for x in examples]
        actions = [x for x in actions if x is not None]
        docs = {str(x.get("document", "")).strip() for x in examples}
        docs.discard("")

        if not actions:
            continue

        counts = defaultdict(int)
        for action in actions:
            counts[action] += 1

        best_action, best_count = max(
            counts.items(), key=lambda pair: pair[1]
        )

        if len(actions) >= 3 and len(docs) >= 2 and best_count == len(actions):
            memory["cross_context_structural_patterns"][
                json.dumps(key, ensure_ascii=False)
            ] = {
                "action": best_action,
                "repeat_count": len(actions),
                "document_count": len(docs),
                "human_confirmed": True,
            }

    # ============================================================
    # LOCAL / CHARACTER-LEVEL PATTERNS
    # ============================================================
    # A local pattern is reusable only when the same Unicode-level
    # substitution behavior has been confirmed repeatedly across documents.
    # This is much less dependent on the complete surrounding sentence/cell
    # than the old generalized matcher, while still requiring strong evidence.

    for key, examples in local_groups.items():
        actions = [learning_action(x) for x in examples]
        actions = [x for x in actions if x is not None]
        docs = {str(x.get("document", "")).strip() for x in examples}
        docs.discard("")

        if not actions:
            continue

        counts = defaultdict(int)
        for action in actions:
            counts[action] += 1

        best_action, best_count = max(
            counts.items(), key=lambda pair: pair[1]
        )

        # Stronger than the legacy MIN_REPEATS rule: at least 3 examples
        # spanning at least 2 independent documents and unanimous action.
        if len(actions) >= 3 and len(docs) >= 2 and best_count == len(actions):
            memory["local_transformation_patterns"][key] = {
                "action": best_action,
                "repeat_count": len(actions),
                "document_count": len(docs),
                "human_confirmed": True,
            }
        elif len(counts) > 1:
            memory["conflicts"].append({
                "type": "LOCAL_TRANSFORMATION_ACTION_CONFLICT",
                "key": key,
                "actions": dict(counts),
            })

    # ============================================================
    # CHARACTER-EDIT TRANSFORMATION PATTERNS
    # ============================================================
    # Human-entered corrections are allowed to teach a reusable Unicode
    # character edit, but only after repeated unanimous evidence across at
    # least two documents.  The surrounding numeric/text value is never
    # stored as the correction.

    for key, examples in character_edit_groups.items():
        docs = {str(x.get("document", "")).strip() for x in examples}
        docs.discard("")

        if len(examples) < 2 or len(docs) < 2:
            continue

        verified_actions = [
            normalize_text(x.get("verified_text", ""))
            for x in examples
        ]
        if not all(verified_actions):
            continue

        memory["character_edit_transformation_patterns"][key] = {
            "action": "APPLY_CHARACTER_EDIT",
            "repeat_count": len(examples),
            "document_count": len(docs),
            "human_confirmed": True,
        }

    # ============================================================
    # STRUCTURAL PATTERNS
    # ============================================================

    for key, examples in structural_groups.items():
        actions = [
            learning_action(x)
            for x in examples
        ]

        actions = [
            x for x in actions
            if x is not None
        ]

        if not actions:
            continue

        counts = defaultdict(int)

        for action in actions:
            counts[action] += 1

        best_action, best_count = max(
            counts.items(),
            key=lambda pair: pair[1],
        )

        total = len(actions)

        if (
            total >= MIN_REPEATS
            and best_count == total
        ):
            memory["structural_patterns"][
                json.dumps(
                    key,
                    ensure_ascii=False,
                )
            ] = {
                "action": best_action,
                "repeat_count": total,
                "human_confirmed": True,
            }

    # ============================================================
    # CONTEXT PATTERNS
    # ============================================================

    for key, examples in context_groups.items():
        actions = [
            learning_action(x)
            for x in examples
        ]

        actions = [
            x for x in actions
            if x is not None
        ]

        if not actions:
            continue

        counts = defaultdict(int)

        for action in actions:
            counts[action] += 1

        best_action, best_count = max(
            counts.items(),
            key=lambda pair: pair[1],
        )

        total = len(actions)

        if (
            total >= MIN_REPEATS
            and best_count == total
        ):
            memory["context_patterns"][
                json.dumps(
                    key,
                    ensure_ascii=False,
                )
            ] = {
                "action": best_action,
                "repeat_count": total,
                "human_confirmed": True,
            }

    # ============================================================
    # AUDIT
    # ============================================================

    memory["audit"] = {
        "unique_human_evidence": len(
            seen_evidence
        ),
        "exact_candidate_groups": len(
            exact_groups
        ),
        "generalized_candidate_groups": len(
            generalized_groups
        ),
        "structural_candidate_groups": len(
            structural_groups
        ),
        "context_candidate_groups": len(
            context_groups
        ),
        "reusable_exact_patterns": len(
            memory["exact_patterns"]
        ),
        "reusable_generalized_patterns": len(
            memory["generalized_patterns"]
        ),
        "reusable_structural_patterns": len(
            memory["structural_patterns"]
        ),
        "reusable_cross_context_generalized_patterns": len(
            memory["cross_context_generalized_patterns"]
        ),
        "reusable_cross_context_structural_patterns": len(
            memory["cross_context_structural_patterns"]
        ),
        "reusable_local_transformation_patterns": len(
            memory["local_transformation_patterns"]
        ),
        "reusable_character_edit_transformation_patterns": len(
            memory["character_edit_transformation_patterns"]
        ),
        "reusable_context_patterns": len(
            memory["context_patterns"]
        ),
        "conflict_count": len(
            memory["conflicts"]
        ),
    }

    return memory


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("OCR LEARNING MEMORY BUILDER — CLEAN")
    print("=" * 70)
    print()
    print(
        "No hard-coded OCR correction dictionary."
    )
    print(
        "Unicode character classes are detected dynamically."
    )
    print(
        f"Minimum repeated human evidence: {MIN_REPEATS}"
    )
    print()

    if not DECISIONS_FILE.exists():
        raise FileNotFoundError(
            "Manual decision file not found:\n"
            f"{DECISIONS_FILE}"
        )

    data = load_json(
        DECISIONS_FILE
    )

    decisions = data.get(
        "decisions",
        [],
    )
    if not isinstance(decisions, list):
        decisions = []

    review_outcomes = load_review_outcomes(DECISIONS_FILE)
    review_activity = review_outcome_summary(review_outcomes)

    print(
        f"Review outcomes recorded : "
        f"{review_activity['total_review_outcomes']}"
    )
    print(
        f"  Human review outcomes   : "
        f"{review_activity['human_review_outcomes']}"
    )
    print(
        f"  Skipped reviews         : "
        f"{review_activity['skipped_review_outcomes']}"
    )
    print(
        f"  Automatic resolutions   : "
        f"{review_activity['automatic_review_outcomes']}"
    )
    print(
        f"Trusted learning decisions: "
        f"{sum(1 for d in decisions if isinstance(d, dict) and d.get('decision') in TRUSTED_HUMAN_DECISIONS)}"
    )

    memory = build_memory(
        decisions
    )
    memory["review_activity"] = review_activity

    memory["total_manual_decisions"] = (
        len(decisions)
    )

    save_json(
        MEMORY_FILE,
        memory,
    )

    print()
    print("=" * 70)
    print("LEARNING MEMORY COMPLETE")
    print("=" * 70)
    print()
    print(
        f"Trusted human decisions       : "
        f"{memory['trusted_human_decisions']}"
    )
    print(
        f"Unique human evidence         : "
        f"{memory['audit']['unique_human_evidence']}"
    )
    print()
    print(
        f"Reusable exact patterns       : "
        f"{memory['audit']['reusable_exact_patterns']}"
    )
    print(
        f"Reusable generalized patterns : "
        f"{memory['audit']['reusable_generalized_patterns']}"
    )
    print(
        f"Reusable structural patterns  : "
        f"{memory['audit']['reusable_structural_patterns']}"
    )
    print(
        f"Reusable cross-context generalized: "
        f"{memory['audit']['reusable_cross_context_generalized_patterns']}"
    )
    print(
        f"Reusable cross-context structural : "
        f"{memory['audit']['reusable_cross_context_structural_patterns']}"
    )
    print(
        f"Reusable local transformations   : "
        f"{memory['audit']['reusable_local_transformation_patterns']}"
    )
    print(
        f"Reusable character-edit transformations: "
        f"{memory['audit']['reusable_character_edit_transformation_patterns']}"
    )
    print(
        f"Reusable context patterns     : "
        f"{memory['audit']['reusable_context_patterns']}"
    )
    print(
        f"Conflicting patterns blocked  : "
        f"{memory['audit']['conflict_count']}"
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
