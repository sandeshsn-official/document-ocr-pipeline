import re
import math
import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("output/final")
OUTPUT_DIR = Path("output/reconstructed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GENERIC CONFIGURATION
# ============================================================

ROW_TOLERANCE = 25

MIN_TABLE_ROWS = 3

# Number of OCR blocks needed for a row to be considered
# structurally strong enough to help discover a table.
MIN_STRUCTURED_BLOCKS = 2

# Normal distance between neighbouring structured rows.
MAX_ROW_GAP = 150

# Larger gap allowed when two sections have compatible
# horizontal geometry.
MAX_SECTION_GAP = 300

# Larger gap allowed when deciding whether two sections are
# part of the same physical table.
MAX_TABLE_CONTINUITY_GAP = 350

# X clustering tolerance.
COLUMN_CLUSTER_TOLERANCE = 100

# Maximum normal distance when assigning a block to a column.
COLUMN_ASSIGNMENT_TOLERANCE = 180

# Minimum number of structured rows required to establish
# a table section.
MIN_ANCHOR_ROWS = 2

# Minimum geometric score.
MIN_TABLE_SCORE = 0.40

# Minimum horizontal overlap required when joining nearby
# sections.
MIN_HORIZONTAL_OVERLAP = 0.30

# ------------------------------------------------------------
# FALLBACK TABLE EVIDENCE
# ------------------------------------------------------------
# These are generic thresholds. They are NOT document/page
# specific. Fallback reconstruction is deliberately conservative:
# paragraph geometry alone must show sustained table-like
# repetition before a table is manufactured.
FALLBACK_MIN_ROWS = 4
FALLBACK_MIN_DENSE_RATIO = 0.55
FALLBACK_MIN_DOMINANT_RATIO = 0.45
FALLBACK_MIN_STRUCTURE_SIMILARITY = 0.55
FALLBACK_MIN_REPEATED_RUN_RATIO = 0.50
FALLBACK_MIN_STABLE_ROWS = 4

# Two-column layouts are common in normal documents, so they need
# stronger repetition than 3+ column layouts.
TWO_COLUMN_MIN_DENSE_RATIO = 0.72
TWO_COLUMN_MIN_DOMINANT_RATIO = 0.60
TWO_COLUMN_MIN_STRUCTURE_SIMILARITY = 0.70
TWO_COLUMN_MIN_REPEATED_RUN_RATIO = 0.65

# Optional image evidence. OpenCV is used only when available and
# when the page image can be located. It is supporting evidence,
# never a hard-coded page/document rule.
IMAGE_LINE_MIN_LENGTH_RATIO = 0.25
IMAGE_TABLE_INTERSECTION_MIN = 12
IMAGE_TABLE_INTERSECTION_STRONG = 40


# ============================================================
# BBOX NORMALIZATION
# ============================================================

def normalize_bbox(bbox):

    if bbox is None:
        return None

    try:

        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float))
                for value in bbox
            )
        ):

            x1, y1, x2, y2 = bbox

            return [
                float(min(x1, x2)),
                float(min(y1, y2)),
                float(max(x1, x2)),
                float(max(y1, y2))
            ]

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

                xs = [
                    point[0]
                    for point in points
                ]

                ys = [
                    point[1]
                    for point in points
                ]

                return [
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys)
                ]

    except Exception:
        return None

    return None


# ============================================================
# BBOX CENTER
# ============================================================

def bbox_center(bbox):

    bbox = normalize_bbox(bbox)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


# ============================================================
# ROW GEOMETRY
# ============================================================

def row_center_y(row):

    if not row:
        return 0.0

    return (
        sum(
            block["y"]
            for block in row
        )
        / len(row)
    )


def row_min_y(row):

    if not row:
        return 0.0

    return min(
        block["bbox"][1]
        for block in row
    )


def row_max_y(row):

    if not row:
        return 0.0

    return max(
        block["bbox"][3]
        for block in row
    )


def row_min_x(row):

    if not row:
        return 0.0

    return min(
        block["bbox"][0]
        for block in row
    )


def row_max_x(row):

    if not row:
        return 0.0

    return max(
        block["bbox"][2]
        for block in row
    )


# ============================================================
# REGION GEOMETRY
# ============================================================

def region_min_y(region):

    if not region:
        return 0.0

    valid_rows = [
        row
        for row in region
        if row
    ]

    if not valid_rows:
        return 0.0

    return min(
        row_min_y(row)
        for row in valid_rows
    )


def region_max_y(region):

    if not region:
        return 0.0

    valid_rows = [
        row
        for row in region
        if row
    ]

    if not valid_rows:
        return 0.0

    return max(
        row_max_y(row)
        for row in valid_rows
    )


def region_min_x(region):

    if not region:
        return 0.0

    valid_rows = [
        row
        for row in region
        if row
    ]

    if not valid_rows:
        return 0.0

    return min(
        row_min_x(row)
        for row in valid_rows
    )


def region_max_x(region):

    if not region:
        return 0.0

    valid_rows = [
        row
        for row in region
        if row
    ]

    if not valid_rows:
        return 0.0

    return max(
        row_max_x(row)
        for row in valid_rows
    )


# ============================================================
# HORIZONTAL OVERLAP
# ============================================================

def horizontal_overlap_ratio(
    region_a,
    region_b
):

    if not region_a or not region_b:
        return 0.0

    a1 = region_min_x(region_a)
    a2 = region_max_x(region_a)

    b1 = region_min_x(region_b)
    b2 = region_max_x(region_b)

    intersection = max(
        0.0,
        min(a2, b2)
        - max(a1, b1)
    )

    width_a = max(
        1.0,
        a2 - a1
    )

    width_b = max(
        1.0,
        b2 - b1
    )

    return intersection / min(
        width_a,
        width_b
    )


# ============================================================
# PARAGRAPH OCR BLOCKS
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

    text_blocks = paragraph.get(
        "text_blocks",
        []
    )

    if not isinstance(
        text_blocks,
        list
    ):
        return []

    output = []

    for block in text_blocks:

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

        if not text or bbox is None:
            continue

        center = bbox_center(
            bbox
        )

        if center is None:
            continue

        output.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": block.get(
                    "confidence"
                ),
                "x": center[0],
                "y": center[1]
            }
        )

    return output


# ============================================================
# VISUAL ROW GROUPING
# ============================================================

def group_blocks_into_visual_rows(
    blocks
):
    """
    Group OCR blocks into visual rows using their vertical intervals rather
    than allowing a running row mean to absorb nearby blocks.

    The grouping is deliberately conservative about transitive drift:
    blocks are compared with the current row's median geometry and vertical
    span.  A block belongs to a row when its vertical footprint overlaps the
    row or its center is close to the robust row center.  Large vertical
    gaps create a new row even when preceding rows contain unusually tall
    OCR boxes.

    No document-specific coordinates, row counts, or table dimensions are
    used.
    """
    if not blocks:
        return []

    valid = [
        block for block in blocks
        if isinstance(block, dict)
        and block.get("bbox")
        and len(block.get("bbox")) == 4
    ]
    if not valid:
        return []

    valid = sorted(
        valid,
        key=lambda block: (
            float(block["bbox"][1]),
            float(block["bbox"][0])
        )
    )

    heights = sorted(
        max(1.0, float(b["bbox"][3]) - float(b["bbox"][1]))
        for b in valid
    )
    median_height = heights[len(heights) // 2]

    rows = []

    def row_stats(row):
        centers = sorted(float(b["y"]) for b in row)
        hvals = sorted(
            max(1.0, float(b["bbox"][3]) - float(b["bbox"][1]))
            for b in row
        )
        median_center = centers[len(centers) // 2]
        median_h = hvals[len(hvals) // 2]
        top = min(float(b["bbox"][1]) for b in row)
        bottom = max(float(b["bbox"][3]) for b in row)
        return median_center, median_h, top, bottom

    for block in valid:
        bx1 = float(block["bbox"][1])
        bx2 = float(block["bbox"][3])
        bc = float(block["y"])
        bh = max(1.0, bx2 - bx1)

        candidates = []
        for idx, row in enumerate(rows):
            rc, rh, rt, rb = row_stats(row)
            overlap = max(0.0, min(bx2, rb) - max(bx1, rt))
            overlap_ratio = overlap / max(1.0, min(bh, rb - rt if rb > rt else bh))
            center_distance = abs(bc - rc)

            # Use a robust tolerance, capped so a tall/merged OCR block cannot
            # pull a neighbouring line into the same row.
            tolerance = min(
                max(8.0, ROW_TOLERANCE, 0.55 * median_height, 0.55 * rh, 0.55 * bh),
                max(ROW_TOLERANCE, 1.25 * median_height)
            )

            # A meaningful vertical intersection is strong evidence.  Without
            # intersection, require a genuinely close center.
            if overlap_ratio >= 0.20 or center_distance <= tolerance:
                candidates.append((center_distance, -overlap_ratio, idx))

        if candidates:
            _, _, best_idx = min(candidates)
            row = rows[best_idx]
            rc, rh, rt, rb = row_stats(row)

            # Final guard against chain-merging adjacent lines: when the new
            # block is separated from every existing block in the row by a
            # substantial vertical gap, do not attach it merely because the
            # aggregate row span is tall.
            nearest_block_gap = min(
                max(
                    0.0,
                    max(float(block["bbox"][1]) - float(other["bbox"][3]),
                        float(other["bbox"][1]) - float(block["bbox"][3]))
                )
                for other in row
            )
            local_gap_limit = max(10.0, 0.75 * median_height)

            if nearest_block_gap <= local_gap_limit or any(
                max(0.0, min(bx2, float(other["bbox"][3])) -
                    max(bx1, float(other["bbox"][1]))) > 0
                for other in row
            ):
                row.append(block)
            else:
                rows.append([block])
        else:
            rows.append([block])

    for row in rows:
        row.sort(key=lambda block: float(block["bbox"][0]))

    rows.sort(key=lambda row: row_center_y(row))
    return rows


# ============================================================
# ROW X POSITIONS
# ============================================================

def row_x_positions(row):

    return sorted(
        block["x"]
        for block in row
    )


# ============================================================
# ROW STRUCTURE SIMILARITY
# ============================================================

def row_structure_similarity(
    row_a,
    row_b
):

    if not row_a or not row_b:
        return 0.0

    xs_a = row_x_positions(
        row_a
    )

    xs_b = row_x_positions(
        row_b
    )

    if (
        len(xs_a) < 2
        or len(xs_b) < 2
    ):
        return 0.0

    count_difference = abs(
        len(xs_a)
        - len(xs_b)
    )

    count_score = max(
        0.0,
        1.0
        - (
            count_difference
            / max(
                len(xs_a),
                len(xs_b)
            )
        )
    )

    min_a = min(xs_a)
    max_a = max(xs_a)

    min_b = min(xs_b)
    max_b = max(xs_b)

    range_a = max(
        1.0,
        max_a - min_a
    )

    range_b = max(
        1.0,
        max_b - min_b
    )

    normalized_a = [
        (x - min_a)
        / range_a
        for x in xs_a
    ]

    normalized_b = [
        (x - min_b)
        / range_b
        for x in xs_b
    ]

    count = min(
        len(normalized_a),
        len(normalized_b)
    )

    differences = [
        abs(
            normalized_a[i]
            - normalized_b[i]
        )
        for i in range(count)
    ]

    if not differences:
        return count_score

    position_score = (
        1.0
        - (
            sum(differences)
            / len(differences)
        )
    )

    position_score = max(
        0.0,
        min(
            1.0,
            position_score
        )
    )

    return (
        0.45 * count_score
        + 0.55 * position_score
    )


# ============================================================
# STRUCTURED ROWS
# ============================================================

def get_structured_row_indices(
    rows
):

    return [
        index
        for index, row in enumerate(rows)
        if len(row)
        >= MIN_STRUCTURED_BLOCKS
    ]


# ============================================================
# BUILD ANCHOR GROUPS
# ============================================================

def build_anchor_groups(
    rows
):

    indices = get_structured_row_indices(
        rows
    )

    if len(indices) < MIN_ANCHOR_ROWS:
        return []

    groups = []

    current = [
        indices[0]
    ]

    for index in indices[1:]:

        previous_index = current[-1]

        previous_row = rows[
            previous_index
        ]

        current_row = rows[
            index
        ]

        gap = (
            row_center_y(
                current_row
            )
            - row_center_y(
                previous_row
            )
        )

        similarity = (
            row_structure_similarity(
                previous_row,
                current_row
            )
        )

        same_group = (
            (
                gap <= MAX_ROW_GAP
                and similarity >= 0.40
            )
            or
            (
                gap <= MAX_SECTION_GAP
                and similarity >= 0.65
            )
        )

        if same_group:

            current.append(
                index
            )

        else:

            if len(current) >= MIN_ANCHOR_ROWS:

                groups.append(
                    current
                )

            current = [
                index
            ]

    if len(current) >= MIN_ANCHOR_ROWS:

        groups.append(
            current
        )

    return groups


# ============================================================
# COLUMN CLUSTERING
# ============================================================

def cluster_column_centers(
    rows
):

    x_positions = []

    for row in rows:

        if len(row) < MIN_STRUCTURED_BLOCKS:
            continue

        for block in row:

            text_value = str(block.get("text", "")).strip()
            compact = "".join(ch for ch in text_value if not ch.isspace())
            if compact and all(not ch.isalnum() for ch in compact):
                continue
            bbox = normalize_bbox(block.get("bbox"))
            if bbox is not None and (bbox[2] - bbox[0]) < 0.20 * max(1.0, COLUMN_CLUSTER_TOLERANCE) and len(compact) <= 2:
                continue

            x_positions.append(
                block["x"]
            )

    if not x_positions:
        return []

    x_positions.sort()

    clusters = []

    for x in x_positions:

        if not clusters:

            clusters.append(
                [x]
            )

            continue

        current_center = (
            sum(
                clusters[-1]
            )
            / len(
                clusters[-1]
            )
        )

        if abs(
            x - current_center
        ) <= COLUMN_CLUSTER_TOLERANCE:

            clusters[-1].append(
                x
            )

        else:

            clusters.append(
                [x]
            )

    centers = [
        sum(cluster)
        / len(cluster)
        for cluster in clusters
    ]

    return sorted(
        centers
    )


# ============================================================
# REFINE COLUMNS
# ============================================================

def refine_column_centers(
    centers
):

    if not centers:
        return []

    centers = sorted(
        centers
    )

    output = [
        centers[0]
    ]

    for center in centers[1:]:

        if abs(
            center - output[-1]
        ) <= 40:

            output[-1] = (
                output[-1]
                + center
            ) / 2.0

        else:

            output.append(
                center
            )

    return output


# ============================================================
# TABLE SCORE
# ============================================================

def table_core_score(
    rows
):

    if len(rows) < MIN_TABLE_ROWS:
        return 0.0

    structured = [
        row
        for row in rows
        if len(row)
        >= MIN_STRUCTURED_BLOCKS
    ]

    if len(structured) < MIN_ANCHOR_ROWS:
        return 0.0

    centers = cluster_column_centers(
        structured
    )

    centers = refine_column_centers(
        centers
    )

    if len(centers) < 2:
        return 0.0

    structured_ratio = (
        len(structured)
        / len(rows)
    )

    total = 0
    matched = 0

    for row in rows:

        for block in row:

            total += 1

            distance = min(
                abs(
                    block["x"]
                    - center
                )
                for center in centers
            )

            if (
                distance
                <= COLUMN_ASSIGNMENT_TOLERANCE
            ):

                matched += 1

    column_fit = (
        matched / total
        if total
        else 0.0
    )

    similarities = []

    for i in range(
        len(structured) - 1
    ):

        similarities.append(
            row_structure_similarity(
                structured[i],
                structured[i + 1]
            )
        )

    repeated_structure = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0.0
    )

    return (
        0.30 * structured_ratio
        + 0.40 * column_fit
        + 0.30 * repeated_structure
    )


# ============================================================
# FIND TABLE CORES
# ============================================================

def find_table_cores(
    rows
):

    groups = build_anchor_groups(
        rows
    )

    candidates = []

    for group in groups:

        start = min(group)
        end = max(group)

        core = rows[
            start:end + 1
        ]

        score = table_core_score(
            core
        )

        if score >= MIN_TABLE_SCORE:

            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "score": score
                }
            )

    return candidates


# ============================================================
# BUILD REGION FROM CANDIDATE
# ============================================================

def candidate_to_region(
    rows,
    candidate
):

    start = candidate["start"]
    end = candidate["end"]

    if start < 0:
        return []

    if end >= len(rows):
        return []

    return rows[
        start:end + 1
    ]


# ============================================================
# GENERIC TABLE-CONTINUITY TEST
# ============================================================

def regions_belong_to_same_table(
    region_a,
    region_b
):
    """
    Decide whether two nearby regions are likely parts of the
    same physical table.

    This is intentionally based on geometry, not text.

    This handles cases where:
        - the main table has many columns
        - subtotal rows have fewer populated columns
        - total rows have only a few populated columns
        - continuation rows have one populated column
    """

    if not region_a or not region_b:
        return False

    # --------------------------------------------------------
    # Vertical distance.
    # --------------------------------------------------------

    gap = (
        region_min_y(region_b)
        - region_max_y(region_a)
    )

    if gap < 0:

        gap = (
            region_min_y(region_a)
            - region_max_y(region_b)
        )

    if gap > MAX_TABLE_CONTINUITY_GAP:
        return False

    # --------------------------------------------------------
    # Horizontal footprint.
    # --------------------------------------------------------

    overlap = (
        horizontal_overlap_ratio(
            region_a,
            region_b
        )
    )

    if overlap < MIN_HORIZONTAL_OVERLAP:
        return False

    # --------------------------------------------------------
    # Width compatibility.
    # --------------------------------------------------------

    width_a = max(
        1.0,
        region_max_x(region_a)
        - region_min_x(region_a)
    )

    width_b = max(
        1.0,
        region_max_x(region_b)
        - region_min_x(region_b)
    )

    width_ratio = (
        min(width_a, width_b)
        / max(width_a, width_b)
    )

    # A summary section can be narrower than the main table,
    # so width ratio is deliberately NOT required to be high.
    #
    # Strong horizontal overlap is sufficient.
    if (
        overlap >= 0.55
        and gap <= MAX_TABLE_CONTINUITY_GAP
    ):

        return True

    if (
        overlap >= 0.30
        and width_ratio >= 0.45
        and gap <= MAX_SECTION_GAP
    ):

        return True

    return False


# ============================================================
# MERGE RELATED TABLE SECTIONS
# ============================================================

def merge_related_table_regions(
    rows,
    regions
):
    """
    Merge nearby table sections that belong to one physical
    table.

    IMPORTANT:

    This is why a main table + subtotal + total section is
    represented as ONE table.

    No text is inspected.
    """

    if not regions:
        return []

    # --------------------------------------------------------
    # Convert regions to index ranges.
    # --------------------------------------------------------

    ranges = []

    for region in regions:

        if not region:
            continue

        first_row = rows.index(
            region[0]
        )

        last_row = rows.index(
            region[-1]
        )

        ranges.append(
            {
                "start": first_row,
                "end": last_row
            }
        )

    if not ranges:
        return []

    ranges.sort(
        key=lambda item:
        item["start"]
    )

    changed = True

    while changed:

        changed = False

        merged_ranges = []

        i = 0

        while i < len(ranges):

            current = ranges[i]

            if (
                i + 1
                >= len(ranges)
            ):

                merged_ranges.append(
                    current
                )

                break

            next_range = ranges[
                i + 1
            ]

            current_region = rows[
                current["start"]:
                current["end"] + 1
            ]

            next_region = rows[
                next_range["start"]:
                next_range["end"] + 1
            ]

            if regions_belong_to_same_table(
                current_region,
                next_region
            ):

                merged_ranges.append(
                    {
                        "start": min(
                            current["start"],
                            next_range["start"]
                        ),
                        "end": max(
                            current["end"],
                            next_range["end"]
                        )
                    }
                )

                changed = True
                i += 2

            else:

                merged_ranges.append(
                    current
                )

                i += 1

        ranges = merged_ranges

    # --------------------------------------------------------
    # Convert merged ranges back to row regions.
    #
    # This is critical because ALL rows between the two
    # sections are now included.
    # --------------------------------------------------------

    output = []

    for item in ranges:

        region = rows[
            item["start"]:
            item["end"] + 1
        ]

        if len(region) >= MIN_TABLE_ROWS:

            output.append(
                region
            )

    return output


# ============================================================
# FIND TABLE REGIONS
# ============================================================

def find_table_regions(
    rows
):

    if not rows:
        return []

    candidates = find_table_cores(
        rows
    )

    if not candidates:
        return []

    regions = []

    for candidate in candidates:

        region = candidate_to_region(
            rows,
            candidate
        )

        if region:

            regions.append(
                region
            )

    # --------------------------------------------------------
    # Merge table sections belonging to the same physical
    # table.
    # --------------------------------------------------------

    regions = merge_related_table_regions(
        rows,
        regions
    )

    return regions


# ============================================================
# TABLE COLUMN CENTERS
# ============================================================

def get_table_column_centers(
    region
):

    structured = [
        row
        for row in region
        if len(row)
        >= MIN_STRUCTURED_BLOCKS
    ]

    if not structured:
        return []

    # OCR fragmentation or an unresolved coordinate mismatch can otherwise
    # manufacture an implausibly large number of columns. The table model
    # must be supported by repeated row geometry, not by every tiny fragment.
    if len(structured) >= 3:
        row_x_counts = [
            len({
                round(float(block["x"]) / max(1.0, COLUMN_CLUSTER_TOLERANCE), 2)
                for block in row
            })
            for row in structured
        ]

        if row_x_counts:
            median_row_x_count = sorted(row_x_counts)[
                len(row_x_counts) // 2
            ]

            if median_row_x_count > 30:
                return []

    # Start with the global x-cluster model.
    global_centers = cluster_column_centers(structured)
    global_centers = refine_column_centers(global_centers)

    if not global_centers:
        return []

    # A wide OCR text block can move its center substantially from row to row
    # (for example, a long Designation/Description cell).  Global clustering
    # can then manufacture a phantom column inside that real text column.
    # Prefer a high-information row as the geometric anchor when it provides
    # a coherent set of distinct column positions.  This is document-agnostic:
    # no header text, column count, or page coordinates are assumed.
    max_blocks = max(len(row) for row in structured)
    anchor_rows = [
        row for row in structured
        if len(row) == max_blocks
    ]

    def anchor_quality(row):
        xs = sorted(float(block["x"]) for block in row)
        if len(xs) < 3:
            return (-1.0, -1.0)

        # Penalize unusually tight adjacent anchors, which are more likely
        # to be OCR fragments than independent table columns.
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        positive = [g for g in gaps if g > 0]
        if not positive:
            return (-1.0, -1.0)
        median_gap = sorted(positive)[len(positive) // 2]
        tight = sum(1 for g in positive if g < max(40.0, 0.35 * median_gap))

        # Reward anchors that agree with the global model, while allowing one
        # global phantom cluster to be absent from the anchor row.
        matches = 0
        used = set()
        for x in xs:
            choices = sorted(
                (abs(x - c), i)
                for i, c in enumerate(global_centers)
                if i not in used
            )
            if choices and choices[0][0] <= COLUMN_ASSIGNMENT_TOLERANCE:
                used.add(choices[0][1])
                matches += 1

        return (
            matches - 0.75 * tight,
            -abs(len(global_centers) - len(xs))
        )

    best_anchor = max(anchor_rows, key=anchor_quality) if anchor_rows else None

    if best_anchor is not None and len(best_anchor) >= 3:
        anchor_centers = sorted(
            float(block["x"]) for block in best_anchor
        )

        # Use the anchor only when its positions are sufficiently distinct.
        gaps = [b - a for a, b in zip(anchor_centers, anchor_centers[1:])]
        if gaps and min(gaps) >= max(40.0, 0.20 * sorted(gaps)[len(gaps) // 2]):
            return anchor_centers

    return global_centers


# ============================================================
# COLUMN ASSIGNMENT
# ============================================================

def nearest_column_index(
    x,
    column_centers
):

    if not column_centers:
        return None

    distances = [
        abs(
            x - center
        )
        for center in column_centers
    ]

    return min(
        range(
            len(distances)
        ),
        key=lambda index:
        distances[index]
    )


def assign_row_to_columns(
    row,
    column_centers
):
    """
    Assign OCR blocks to columns using their horizontal footprint first,
    with center distance as a secondary signal.

    This avoids losing text when an OCR block's center falls closer to a
    neighboring column even though most of its actual bounding box lies
    inside the intended column.

    No document-specific dimensions or text rules are used.
    """

    assignments = {
        index: []
        for index in range(
            len(column_centers)
        )
    }

    if not row or not column_centers:
        return assignments

    centers = sorted(
        float(center)
        for center in column_centers
    )

    # Convert column centers into non-overlapping horizontal Voronoi slots.
    boundaries = []

    for index, center in enumerate(centers):
        if index == 0:
            left = float("-inf")
        else:
            left = (
                centers[index - 1]
                + center
            ) / 2.0

        if index == len(centers) - 1:
            right = float("inf")
        else:
            right = (
                center
                + centers[index + 1]
            ) / 2.0

        boundaries.append(
            (left, right)
        )

    for block in row:
        bbox = block.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1 = float(bbox[0])
        x2 = float(bbox[2])

        if x2 <= x1:
            continue

        block_width = x2 - x1
        block_center = (
            x1 + x2
        ) / 2.0

        scores = []

        for index, (left, right) in enumerate(
            boundaries
        ):
            overlap_left = max(
                x1,
                left
            )
            overlap_right = min(
                x2,
                right
            )

            overlap_width = max(
                0.0,
                overlap_right - overlap_left
            )

            overlap_ratio = (
                overlap_width / block_width
            )

            center_distance = abs(
                block_center
                - centers[index]
            )

            # Primary signal: fraction of OCR bbox inside slot.
            # Secondary signal: center proximity.
            scores.append(
                (
                    overlap_ratio,
                    -center_distance,
                    index
                )
            )

        best_overlap, _, best_index = max(
            scores,
            key=lambda item: (
                item[0],
                item[1]
            )
        )

        # If the block crosses a column boundary, use center proximity
        # only when no column owns a meaningful part of the block.
        if best_overlap <= 0.0:
            best_index = min(
                range(len(centers)),
                key=lambda index:
                abs(
                    block_center
                    - centers[index]
                )
            )

        assignments[
            best_index
        ].append(
            block
        )

    return assignments


# ============================================================
# CELL TEXT
# ============================================================

def combine_cell_text(
    blocks
):

    if not blocks:
        return ""

    blocks = sorted(
        blocks,
        key=lambda block: (
            block["y"],
            block["x"]
        )
    )

    parts = []

    for block in blocks:

        text = str(
            block.get(
                "text",
                ""
            )
        ).strip()

        if text:

            parts.append(
                text
            )

    return "\n".join(
        parts
    )


# ============================================================
# CELL BBOX
# ============================================================

def combine_cell_bbox(
    blocks
):

    if not blocks:
        return None

    bboxes = [
        block["bbox"]
        for block in blocks
        if block.get(
            "bbox"
        )
    ]

    if not bboxes:
        return None

    return [
        min(
            bbox[0]
            for bbox in bboxes
        ),
        min(
            bbox[1]
            for bbox in bboxes
        ),
        max(
            bbox[2]
            for bbox in bboxes
        ),
        max(
            bbox[3]
            for bbox in bboxes
        )
    ]


# ============================================================
# CELL CONFIDENCE
# ============================================================

def combine_cell_confidence(
    blocks
):

    values = []

    for block in blocks:

        confidence = block.get(
            "confidence"
        )

        if isinstance(
            confidence,
            (int, float)
        ):

            values.append(
                float(confidence)
            )

    if not values:
        return 0.0

    return min(
        values
    )


# ============================================================
# BUILD FALLBACK TABLE
# ============================================================

def build_fallback_table(
    region
):

    if not region:
        return None

    column_centers = (
        get_table_column_centers(
            region
        )
    )

    if len(column_centers) < 2:
        return None

    cells = []

    # --------------------------------------------------------
    # EVERY row in the region is retained.
    # --------------------------------------------------------

    for row_number, row in enumerate(
        region,
        start=1
    ):

        assignments = (
            assign_row_to_columns(
                row,
                column_centers
            )
        )

        # ----------------------------------------------------
        # Every detected column is created.
        #
        # Missing OCR stays empty.
        # ----------------------------------------------------

        for column_index in range(
            len(column_centers)
        ):

            blocks = assignments[
                column_index
            ]

            if blocks:

                text = (
                    combine_cell_text(
                        blocks
                    )
                )

                bbox = (
                    combine_cell_bbox(
                        blocks
                    )
                )

                confidence = (
                    combine_cell_confidence(
                        blocks
                    )
                )

                status = (
                    "FALLBACK_OCR"
                )

            else:

                text = ""
                bbox = None
                confidence = 0.0

                status = (
                    "MISSING_OCR"
                )

            cells.append(
                {
                    "row": row_number,
                    "column": (
                        column_index + 1
                    ),
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                    "status": status
                }
            )

    # --------------------------------------------------------
    # Row objects.
    # --------------------------------------------------------

    output_rows = []

    for row_number in range(
        1,
        len(region) + 1
    ):

        row_cells = [
            cell
            for cell in cells
            if cell["row"]
            == row_number
        ]

        output_rows.append(
            {
                "row": row_number,
                "cells": row_cells
            }
        )

    return {
        "row_count": len(
            output_rows
        ),
        "cell_count": len(
            cells
        ),
        "rows": output_rows,
        "status": "FALLBACK_OCR",
        "source": "paragraph_text_blocks",
        "column_centers": column_centers
    }


# ============================================================
# PADDLE TABLE
# ============================================================

def reconstruct_paddle_table(
    table
):

    cells = table.get(
        "cells",
        []
    )

    if not isinstance(
        cells,
        list
    ):
        return None

    if not cells:
        return None

    prepared = []

    for cell in cells:

        bbox = normalize_bbox(
            cell.get(
                "bbox"
            )
        )

        if bbox is None:
            continue

        center = bbox_center(
            bbox
        )

        if center is None:
            continue

        prepared.append(
            {
                "cell": cell,
                "x": center[0],
                "y": center[1]
            }
        )

    if not prepared:
        return None

    prepared.sort(
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    rows = []

    for item in prepared:

        placed = False

        for row in rows:

            average_y = (
                sum(
                    entry["y"]
                    for entry in row
                )
                / len(row)
            )

            if abs(
                item["y"]
                - average_y
            ) <= ROW_TOLERANCE:

                row.append(
                    item
                )

                placed = True
                break

        if not placed:

            rows.append(
                [item]
            )

    rows.sort(
        key=lambda row:
        sum(
            item["y"]
            for item in row
        )
        / len(row)
    )

    output_rows = []

    for row_number, row in enumerate(
        rows,
        start=1
    ):

        row.sort(
            key=lambda item:
            item["x"]
        )

        output_cells = []

        for column_number, item in enumerate(
            row,
            start=1
        ):

            cell = item["cell"]

            text = cell.get(
                "final_text"
            )

            if text is None:

                text = cell.get(
                    "text",
                    ""
                )

            output_cells.append(
                {
                    "column": column_number,
                    "text": text,
                    "paddle_text": cell.get(
                        "paddle_text",
                        ""
                    ),
                    "tesseract_text": cell.get(
                        "tesseract_text",
                        ""
                    ),
                    "paddle_confidence": cell.get(
                        "paddle_confidence",
                        cell.get(
                            "confidence",
                            0.0
                        )
                    ),
                    "similarity": cell.get(
                        "similarity",
                        0.0
                    ),
                    "verification_status": cell.get(
                        "status",
                        ""
                    ),
                    "bbox": cell.get(
                        "bbox"
                    )
                }
            )

        output_rows.append(
            {
                "row": row_number,
                "cells": output_cells
            }
        )

    return {
        "row_count": len(
            output_rows
        ),
        "cell_count": sum(
            len(
                row["cells"]
            )
            for row in output_rows
        ),
        "rows": output_rows,
        "status": table.get(
            "status",
            ""
        ),
        "source": "paddle"
    }


# ============================================================
# TABLE BBOX
# ============================================================

def get_table_bbox(
    table
):

    bboxes = []

    for row in table.get(
        "rows",
        []
    ):

        for cell in row.get(
            "cells",
            []
        ):

            bbox = normalize_bbox(
                cell.get(
                    "bbox"
                )
            )

            if bbox:

                bboxes.append(
                    bbox
                )

    if not bboxes:
        return None

    return [
        min(
            bbox[0]
            for bbox in bboxes
        ),
        min(
            bbox[1]
            for bbox in bboxes
        ),
        max(
            bbox[2]
            for bbox in bboxes
        ),
        max(
            bbox[3]
            for bbox in bboxes
        )
    ]


# ============================================================
# BBOX OVERLAP
# ============================================================

def bbox_overlap_ratio(
    bbox_a,
    bbox_b
):

    if not bbox_a or not bbox_b:
        return 0.0

    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    if (
        ix2 <= ix1
        or iy2 <= iy1
    ):
        return 0.0

    intersection = (
        ix2 - ix1
    ) * (
        iy2 - iy1
    )

    area_a = max(
        1.0,
        (
            ax2 - ax1
        ) * (
            ay2 - ay1
        )
    )

    return (
        intersection
        / area_a
    )


# ============================================================
# DUPLICATE FALLBACK FILTER
# ============================================================

def filter_duplicate_fallbacks(
    fallback_tables,
    paddle_tables
):

    if not fallback_tables:
        return []

    if not paddle_tables:
        return fallback_tables

    paddle_bboxes = [
        get_table_bbox(
            table
        )
        for table in paddle_tables
    ]

    output = []

    for fallback in fallback_tables:

        fallback_bbox = (
            get_table_bbox(
                fallback
            )
        )

        duplicate = False

        for paddle_bbox in paddle_bboxes:

            if paddle_bbox is None:
                continue

            overlap = (
                bbox_overlap_ratio(
                    fallback_bbox,
                    paddle_bbox
                )
            )

            if overlap >= 0.90:

                duplicate = True
                break

        if not duplicate:

            output.append(
                fallback
            )

    return output


# ============================================================
# FALLBACK TABLE EVIDENCE
# ============================================================

def _longest_true_run(values):

    best = 0
    current = 0

    for value in values:

        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def _fallback_region_statistics(region):
    """
    Calculate generic structural evidence for a possible table.

    IMPORTANT:
    OCR may fragment one physical table cell into several OCR blocks.
    Therefore the number of OCR blocks per row is NOT treated as the
    table's column count.  Column geometry is inferred independently
    from the x-centres of structured OCR blocks.
    """

    if not region:
        return None

    row_count = len(region)

    block_counts = [
        len(row)
        for row in region
    ]

    dense_flags = [
        count >= MIN_STRUCTURED_BLOCKS
        for count in block_counts
    ]

    dense_count = sum(dense_flags)
    dense_ratio = (
        dense_count / row_count
        if row_count
        else 0.0
    )

    repeated_run = _longest_true_run(
        dense_flags
    )

    repeated_run_ratio = (
        repeated_run / row_count
        if row_count
        else 0.0
    )

    structured_rows = [
        row
        for row in region
        if len(row) >= MIN_STRUCTURED_BLOCKS
    ]

    # --------------------------------------------------------
    # Infer physical columns from x-centres.
    # --------------------------------------------------------
    column_centers = get_table_column_centers(
        region
    )

    inferred_column_count = len(
        column_centers
    )

    # --------------------------------------------------------
    # Measure how consistently rows occupy the same columns.
    # --------------------------------------------------------
    occupancy_ratios = []
    column_presence = [
        0
        for _ in column_centers
    ]

    for row in structured_rows:

        if not column_centers:
            continue

        assigned = set()

        for block in row:

            center = bbox_center(
                block.get("bbox")
            )

            if center is None:
                continue

            nearest = nearest_column_index(
                center[0],
                column_centers
            )

            if nearest is None:
                continue

            distance = abs(
                center[0]
                - column_centers[nearest]
            )

            if (
                distance
                <= COLUMN_ASSIGNMENT_TOLERANCE
            ):
                assigned.add(nearest)

        if assigned:
            occupancy_ratios.append(
                len(assigned)
                / max(
                    1,
                    inferred_column_count
                )
            )

            for index in assigned:
                column_presence[index] += 1

    mean_column_occupancy = (
        sum(occupancy_ratios)
        / len(occupancy_ratios)
        if occupancy_ratios
        else 0.0
    )

    stable_column_ratio = (
        sum(
            1
            for count in column_presence
            if count >= max(
                2,
                int(
                    0.50
                    * max(
                        1,
                        len(structured_rows)
                    )
                )
            )
        )
        / inferred_column_count
        if inferred_column_count
        else 0.0
    )

    # --------------------------------------------------------
    # Repeated row geometry.
    # --------------------------------------------------------
    similarities = []

    for index in range(
        len(structured_rows) - 1
    ):

        similarities.append(
            row_structure_similarity(
                structured_rows[index],
                structured_rows[index + 1]
            )
        )

    structure_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0.0
    )

    # --------------------------------------------------------
    # Generic prose-flow evidence.
    #
    # A normal two-column paper can accidentally look like a borderless
    # table after OCR because several text lines may be grouped into the
    # same visual row.  In that situation the inferred columns can be very
    # stable even though there is no table.  Measure the geometry that is
    # characteristic of continuous prose: OCR blocks fill their column
    # slots almost completely, rows contain repeated line fragments, and
    # individual grouped rows span multiple OCR line heights.
    #
    # This is only negative evidence.  It does not inspect document names,
    # page numbers, known text, or a fixed table shape.
    # --------------------------------------------------------
    prose_full_slot_values = []
    prose_long_block_values = []
    prose_row_fragmentation = []
    prose_internal_span_values = []

    if column_centers and len(column_centers) >= 2:
        centers = sorted(float(c) for c in column_centers)
        boundaries = []
        for index, center in enumerate(centers):
            left = (
                float("-inf")
                if index == 0
                else (centers[index - 1] + center) / 2.0
            )
            right = (
                float("inf")
                if index == len(centers) - 1
                else (center + centers[index + 1]) / 2.0
            )
            boundaries.append((left, right))

        all_heights = [
            max(1.0, float(block["bbox"][3]) - float(block["bbox"][1]))
            for row in region
            for block in row
            if isinstance(block, dict)
            and block.get("bbox")
            and len(block.get("bbox")) == 4
        ]
        median_block_height = (
            sorted(all_heights)[len(all_heights) // 2]
            if all_heights
            else 1.0
        )

        for row in region:
            if not row:
                continue

            prose_row_fragmentation.append(
                len(row) / max(1.0, float(len(centers)))
            )

            y_values = []
            for block in row:
                bbox = block.get("bbox") if isinstance(block, dict) else None
                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = map(float, bbox)
                width_value = max(1.0, x2 - x1)
                center_x = (x1 + x2) / 2.0
                nearest = min(
                    range(len(centers)),
                    key=lambda i: abs(center_x - centers[i])
                )
                slot_left, slot_right = boundaries[nearest]

                finite_left = (
                    slot_left
                    if slot_left != float("-inf")
                    else region_min_x(region)
                )
                finite_right = (
                    slot_right
                    if slot_right != float("inf")
                    else region_max_x(region)
                )
                slot_width = max(1.0, finite_right - finite_left)

                overlap = max(
                    0.0,
                    min(x2, finite_right) - max(x1, finite_left)
                )
                prose_full_slot_values.append(
                    overlap / width_value
                )
                prose_long_block_values.append(
                    width_value / slot_width
                )
                y_values.append((y1 + y2) / 2.0)

            if y_values:
                prose_internal_span_values.append(
                    (max(y_values) - min(y_values))
                    / max(1.0, median_block_height)
                )

    full_slot_ratio = (
        sum(value >= 0.80 for value in prose_full_slot_values)
        / len(prose_full_slot_values)
        if prose_full_slot_values
        else 0.0
    )
    long_block_ratio = (
        sum(value >= 0.65 for value in prose_long_block_values)
        / len(prose_long_block_values)
        if prose_long_block_values
        else 0.0
    )
    high_fragmentation_ratio = (
        sum(value >= 2.0 for value in prose_row_fragmentation)
        / len(prose_row_fragmentation)
        if prose_row_fragmentation
        else 0.0
    )
    multi_line_row_ratio = (
        sum(value >= 1.75 for value in prose_internal_span_values)
        / len(prose_internal_span_values)
        if prose_internal_span_values
        else 0.0
    )

    prose_flow_score = (
        0.35 * full_slot_ratio
        + 0.25 * long_block_ratio
        + 0.20 * high_fragmentation_ratio
        + 0.20 * multi_line_row_ratio
    )

    # A second, deliberately conservative prose detector.  Journal prose
    # often produces long OCR lines that occupy most of their inferred
    # column slot.  A real borderless table normally has compact cell text
    # distributed across several columns instead.  This detector is only
    # negative evidence and is never sufficient by itself to create a table.
    continuous_prose_layout = (
        (
            full_slot_ratio >= 0.55
            and long_block_ratio >= 0.60
            and (
                high_fragmentation_ratio >= 0.20
                or multi_line_row_ratio >= 0.20
            )
        )
        or (
            long_block_ratio >= 0.75
            and high_fragmentation_ratio >= 0.50
            and multi_line_row_ratio >= 0.50
        )
    )

    # --------------------------------------------------------
    # Column-wise prose-content evidence.
    #
    # A journal page can create a false 3-column candidate when OCR lines
    # from two page columns are aligned by Y.  Geometry alone is not enough
    # in that situation.  Real table cells are usually compact and vary in
    # length/type across columns; continuous prose produces sentence-like
    # blocks in the same column repeatedly.
    #
    # Measure this generically per inferred column.  We deliberately require
    # the prose characteristic to be present across most columns, so a table
    # containing one long description column is not rejected.
    # --------------------------------------------------------
    prose_sentence_column_flags = []

    if column_centers:
        column_blocks = [[] for _ in column_centers]

        for row in region:
            for block in row:
                if not isinstance(block, dict):
                    continue
                bbox = block.get("bbox")
                text = str(block.get("text", "")).strip()
                if not bbox or len(bbox) != 4 or not text:
                    continue

                center = bbox_center(bbox)
                if center is None:
                    continue

                nearest = nearest_column_index(
                    float(center[0]),
                    column_centers
                )
                if nearest is None:
                    continue

                column_blocks[nearest].append(text)

        for texts in column_blocks:
            if len(texts) < 2:
                prose_sentence_column_flags.append(False)
                continue

            sentence_like = 0
            for text in texts:
                words = [w for w in re.split(r"\s+", text) if w]
                alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
                if len(text) >= 40 and len(alpha_words) >= 7:
                    sentence_like += 1

            prose_sentence_column_flags.append(
                sentence_like / len(texts) >= 0.60
            )

    prose_sentence_column_ratio = (
        sum(prose_sentence_column_flags)
        / len(prose_sentence_column_flags)
        if prose_sentence_column_flags
        else 0.0
    )

    # Region-wide sentence density catches the remaining failure mode where
    # OCR creates a synthetic third column from ordinary two-column prose.
    # A real table may contain one long-description column, so this signal is
    # combined with the column-wise test rather than used alone.
    prose_sentence_blocks = 0
    prose_content_blocks = 0

    for row in region:
        for block in row:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            words = [w for w in re.split(r"\s+", text) if w]
            alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
            prose_content_blocks += 1
            if len(text) >= 40 and len(alpha_words) >= 7:
                prose_sentence_blocks += 1

    prose_sentence_block_ratio = (
        prose_sentence_blocks / prose_content_blocks
        if prose_content_blocks
        else 0.0
    )

    # This flag is intentionally conservative: it fires only when most
    # inferred columns repeatedly contain sentence-sized OCR blocks.  It is
    # therefore a rejection signal for multi-column prose, not a table rule.
    columnar_prose_layout = (
        len(column_centers) >= 2
        and len(region) >= 3
        and prose_sentence_column_ratio >= 0.67
    )

    sentence_dense_prose_columns = sum(
        1 for flag in prose_sentence_column_flags
        if flag
    )

    sentence_dense_prose_layout = (
        len(column_centers) >= 3
        and len(region) >= 3
        and prose_sentence_block_ratio >= 0.55
        and sentence_dense_prose_columns >= 2
    )

    # Synthetic columns created by a journal page layout are often extremely
    # unevenly spaced: several OCR lines from one physical page column form
    # one or more nearby x-centres, while the next real page column is far
    # away.  Real table columns can be unequal, so this becomes a rejection
    # signal only when combined with sentence-dense prose.
    column_spacing_imbalance = False
    if len(column_centers) >= 3:
        gaps = [
            float(column_centers[i + 1]) - float(column_centers[i])
            for i in range(len(column_centers) - 1)
        ]
        positive_gaps = [gap for gap in gaps if gap > 0]
        if positive_gaps:
            sorted_gaps = sorted(positive_gaps)
            median_gap = sorted_gaps[len(sorted_gaps) // 2]
            minimum_gap = min(sorted_gaps)
            column_spacing_imbalance = (
                median_gap > 0
                and minimum_gap / median_gap < 0.35
            )

    uneven_column_prose_layout = (
        len(column_centers) >= 3
        and prose_sentence_block_ratio >= 0.50
        and column_spacing_imbalance
    )

    # --------------------------------------------------------
    # Original table-core score.
    # --------------------------------------------------------
    core_score = table_core_score(
        region
    )

    return {
        "row_count": row_count,
        "dense_ratio": dense_ratio,
        "max_blocks": max(block_counts) if block_counts else 0,
        "average_blocks": (
            sum(block_counts)
            / len(block_counts)
            if block_counts
            else 0.0
        ),
        "inferred_column_count": inferred_column_count,
        "mean_column_occupancy": mean_column_occupancy,
        "stable_column_ratio": stable_column_ratio,
        "repeated_run_ratio": repeated_run_ratio,
        "structure_similarity": structure_similarity,
        "core_score": core_score,
        # Retained for backwards-compatible evidence output.
        "dominant_count": max(
            set(block_counts),
            key=block_counts.count
        ) if block_counts else 0,
        "dominant_ratio": (
            max(
                block_counts.count(value)
                for value in set(block_counts)
            ) / row_count
            if block_counts and row_count
            else 0.0
        ),
        "stable_rows": max(
            (
                block_counts.count(value)
                for value in set(block_counts)
            ),
            default=0
        ),
        "prose_full_slot_ratio": full_slot_ratio,
        "prose_long_block_ratio": long_block_ratio,
        "prose_high_fragmentation_ratio": high_fragmentation_ratio,
        "prose_multi_line_row_ratio": multi_line_row_ratio,
        "prose_flow_score": prose_flow_score,
        "continuous_prose_layout": continuous_prose_layout,
        "prose_sentence_column_ratio": prose_sentence_column_ratio,
        "columnar_prose_layout": columnar_prose_layout,
        "prose_sentence_block_ratio": prose_sentence_block_ratio,
        "sentence_dense_prose_columns": sentence_dense_prose_columns,
        "sentence_dense_prose_layout": sentence_dense_prose_layout,
        "column_spacing_imbalance": column_spacing_imbalance,
        "uneven_column_prose_layout": uneven_column_prose_layout
    }


def _image_table_evidence(
    image_path,
    region
):
    """
    Return generic image-based line evidence for a region.

    This is deliberately optional. If OpenCV or the image is
    unavailable, it returns neutral evidence rather than failing.
    """

    result = {
        "available": False,
        "horizontal_lines": 0,
        "vertical_lines": 0,
        "intersections": 0,
        "line_pixels": 0
    }

    if image_path is None:
        return result

    try:
        import cv2
    except Exception:
        return result

    try:
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or not region:
            return result

        x1 = max(
            0,
            int(region_min_x(region))
        )
        y1 = max(
            0,
            int(region_min_y(region))
        )
        x2 = min(
            image.shape[1],
            int(region_max_x(region))
        )
        y2 = min(
            image.shape[0],
            int(region_max_y(region))
        )

        if x2 <= x1 or y2 <= y1:
            return result

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return result

        _, binary = cv2.threshold(
            crop,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )

        height, width = binary.shape[:2]

        horizontal_kernel = max(
            10,
            int(width * IMAGE_LINE_MIN_LENGTH_RATIO)
        )

        vertical_kernel = max(
            10,
            int(height * IMAGE_LINE_MIN_LENGTH_RATIO)
        )

        horizontal = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (horizontal_kernel, 1)
            )
        )

        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, vertical_kernel)
            )
        )

        horizontal_pixels = int(
            cv2.countNonZero(horizontal)
        )

        vertical_pixels = int(
            cv2.countNonZero(vertical)
        )

        intersections_mask = cv2.bitwise_and(
            horizontal,
            vertical
        )

        intersections = int(
            cv2.countNonZero(
                intersections_mask
            )
        )

        horizontal_lines = 0
        vertical_lines = 0

        contours, _ = cv2.findContours(
            horizontal,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= horizontal_kernel:
                horizontal_lines += 1

        contours, _ = cv2.findContours(
            vertical,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= vertical_kernel:
                vertical_lines += 1

        result = {
            "available": True,
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "intersections": intersections,
            "line_pixels": (
                horizontal_pixels
                + vertical_pixels
            )
        }

    except Exception:
        return result

    return result


# ============================================================
# SOURCE-IMAGE GRID REGION DETECTION
# ============================================================

def _image_grid_regions(image_path):
    """Detect independent grid-backed table regions directly from the page image.

    This is intentionally independent of OCR row/column inference.  It is used
    to prevent normal journal/page columns from being mistaken for tables and to
    keep physically separate side-by-side tables separate.

    The detector looks for repeated horizontal and vertical line structure at
    several conservative intensity levels.  It does not use document names,
    page numbers, expected table dimensions, or known text.
    """
    if image_path is None:
        return []

    try:
        import cv2
    except Exception:
        return []

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []

        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return []

        # Several thresholds make the detector robust to light/gray table
        # borders while retaining ordinary dark text suppression through
        # long line morphology.
        threshold_values = [160, 180, 200, 220, 235]
        raw_regions = []

        h_kernel_len = max(12, int(width * 0.02))
        v_kernel_len = max(12, int(height * 0.02))

        for threshold_value in threshold_values:
            _, binary = cv2.threshold(
                image, threshold_value, 255, cv2.THRESH_BINARY_INV
            )

            horizontal = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (h_kernel_len, 1)
                )
            )
            vertical = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (1, v_kernel_len)
                )
            )

            combined = cv2.bitwise_or(horizontal, vertical)
            # Join crossings and small breaks in otherwise continuous grid
            # lines without joining unrelated page objects.
            joined = cv2.morphologyEx(
                combined,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (7, 7)
                )
            )

            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                joined, 8
            )

            for index in range(1, count):
                x, y, w, h, area = [int(v) for v in stats[index]]

                if w < max(60, int(width * 0.10)):
                    continue
                if h < max(60, int(height * 0.08)):
                    continue

                # Ignore page-wide rules and very thin artifacts.
                if w > int(width * 0.90) and h < int(height * 0.15):
                    continue
                if h > int(height * 0.90) and w < int(width * 0.15):
                    continue

                crop_h = horizontal[y:y+h, x:x+w]
                crop_v = vertical[y:y+h, x:x+w]

                def _line_count(mask, horizontal_line=True):
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    total = 0
                    for contour in contours:
                        cx, cy, cw, ch = cv2.boundingRect(contour)
                        length = cw if horizontal_line else ch
                        if length >= (h_kernel_len if horizontal_line else v_kernel_len):
                            total += 1
                    return total

                horizontal_count = _line_count(crop_h, True)
                vertical_count = _line_count(crop_v, False)
                intersections = int(
                    cv2.countNonZero(cv2.bitwise_and(crop_h, crop_v))
                )

                # Use two generic evidence levels.  Large grids require
                # repeated lines and many crossings; small UI/report tables
                # can legitimately have only two rows or a single data row,
                # so they need a smaller but still two-directional grid.
                # A one-box rectangle is not accepted as a table candidate.
                strong_grid = (
                    horizontal_count >= 6
                    and vertical_count >= 4
                    and intersections >= 20
                )
                small_grid = (
                    horizontal_count >= 2
                    and vertical_count >= 2
                    and intersections >= 4
                )
                if not (strong_grid or small_grid):
                    continue

                raw_regions.append({
                    "bbox": [x, y, x + w, y + h],
                    "horizontal_lines": horizontal_count,
                    "vertical_lines": vertical_count,
                    "intersections": intersections,
                    "threshold": threshold_value,
                })

        if not raw_regions:
            return []

        # Merge detections of the same physical grid across thresholds.
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aa = max(1, (ax2 - ax1) * (ay2 - ay1))
            bb = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(aa + bb - inter)

        merged = []
        for item in sorted(
            raw_regions,
            key=lambda r: (r["intersections"], r["horizontal_lines"] + r["vertical_lines"]),
            reverse=True,
        ):
            attached = False
            for current in merged:
                if iou(item["bbox"], current["bbox"]) >= 0.30:
                    x1 = min(item["bbox"][0], current["bbox"][0])
                    y1 = min(item["bbox"][1], current["bbox"][1])
                    x2 = max(item["bbox"][2], current["bbox"][2])
                    y2 = max(item["bbox"][3], current["bbox"][3])
                    current["bbox"] = [x1, y1, x2, y2]
                    current["horizontal_lines"] = max(current["horizontal_lines"], item["horizontal_lines"])
                    current["vertical_lines"] = max(current["vertical_lines"], item["vertical_lines"])
                    current["intersections"] = max(current["intersections"], item["intersections"])
                    attached = True
                    break
            if not attached:
                merged.append(dict(item))

        # Keep spatially distinct grids distinct.
        merged.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return merged

    except Exception:
        return []


def _small_grid_regions(image_path):
    """
    Detect small bordered tables/boxes from paired physical line segments.

    This supplements the large-grid detector.  It is intentionally generic:
    candidate rectangles are created only when multiple horizontal and
    vertical source-image rules intersect over the same region.  OCR content
    is still required by the caller before a candidate becomes a table.
    """
    if image_path is None:
        return []
    try:
        import cv2
    except Exception:
        return []
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return []

        proposals = []
        for threshold_value in (180, 200, 220, 235):
            _, binary = cv2.threshold(
                image, threshold_value, 255, cv2.THRESH_BINARY_INV
            )
            hk = max(6, int(width * 0.01))
            vk = max(6, int(height * 0.01))
            horizontal = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
            )
            vertical = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
            )

            hs = []
            for contour in cv2.findContours(
                horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                x, y, w, h = cv2.boundingRect(contour)
                if w < max(50, int(width * 0.15)):
                    continue
                if h > max(8, int(width * 0.01)):
                    continue
                hs.append((float(x), float(y), float(x + w), float(y + h)))

            vs = []
            for contour in cv2.findContours(
                vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                x, y, w, h = cv2.boundingRect(contour)
                if h < max(20, int(height * 0.02)):
                    continue
                if w > max(8, int(width * 0.01)):
                    continue
                vs.append((float(x), float(y), float(x + w), float(y + h)))

            if len(hs) < 2 or len(vs) < 2:
                continue

            # Cluster near-identical horizontal rules produced by multiple
            # thresholds.
            hs.sort(key=lambda r: (r[1], r[0]))
            hclusters = []
            for rule in hs:
                cy = (rule[1] + rule[3]) / 2.0
                attached = False
                for cluster in hclusters:
                    if abs(cy - cluster["y"]) <= max(4.0, 0.006 * height):
                        cluster["items"].append(rule)
                        cluster["y"] = sum((r[1] + r[3]) / 2.0 for r in cluster["items"]) / len(cluster["items"])
                        cluster["x1"] = min(cluster["x1"], rule[0])
                        cluster["x2"] = max(cluster["x2"], rule[2])
                        attached = True
                        break
                if not attached:
                    hclusters.append({
                        "y": cy,
                        "x1": rule[0],
                        "x2": rule[2],
                        "items": [rule],
                    })

            hclusters.sort(key=lambda c: c["y"])

            # Adjacent horizontal rules plus at least two vertical rules form
            # a physical rectangle.  Repeated adjacent rectangles are merged
            # below into the complete table region.
            for a, b in zip(hclusters, hclusters[1:]):
                y1, y2 = a["y"], b["y"]
                gap = y2 - y1
                if gap < 8 or gap > 0.40 * height:
                    continue
                common_x1 = max(a["x1"], b["x1"])
                common_x2 = min(a["x2"], b["x2"])
                if common_x2 - common_x1 < max(50.0, 0.15 * width):
                    continue

                usable_verticals = []
                for v in vs:
                    vx = (v[0] + v[2]) / 2.0
                    vy1, vy2 = v[1], v[3]
                    if vx < common_x1 or vx > common_x2:
                        continue
                    overlap = max(0.0, min(vy2, y2) - max(vy1, y1))
                    if overlap >= 0.35 * gap:
                        usable_verticals.append(v)

                if len(usable_verticals) < 2:
                    continue

                x1 = min(common_x1, min(v[0] for v in usable_verticals))
                x2 = max(common_x2, max(v[2] for v in usable_verticals))
                proposals.append({
                    "bbox": [x1, y1, x2, y2],
                    "horizontal_lines": 2,
                    "vertical_lines": len(usable_verticals),
                    "intersections": max(4, 2 * len(usable_verticals)),
                    "threshold": threshold_value,
                })

        if not proposals:
            return []

        # Merge row-by-row proposals belonging to the same physical table.
        merged = []
        for item in sorted(proposals, key=lambda r: (r["bbox"][1], r["bbox"][0])):
            attached = False
            for current in merged:
                a = current["bbox"]
                b = item["bbox"]
                x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                min_width = max(1.0, min(a[2] - a[0], b[2] - b[0]))
                y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
                if x_overlap / min_width >= 0.50 and y_gap <= max(10.0, 0.025 * height):
                    current["bbox"] = [
                        min(a[0], b[0]), min(a[1], b[1]),
                        max(a[2], b[2]), max(a[3], b[3]),
                    ]
                    current["horizontal_lines"] += 1
                    current["vertical_lines"] = max(current["vertical_lines"], item["vertical_lines"])
                    current["intersections"] += item["intersections"]
                    attached = True
                    break
            if not attached:
                merged.append(dict(item))

        return merged
    except Exception:
        return []


def _blocks_inside_image_region(blocks, bbox):
    """Return OCR blocks whose centers fall inside an image-derived grid."""
    if not blocks or not bbox:
        return []

    x1, y1, x2, y2 = map(float, bbox)
    selected = []
    for block in blocks:
        bb = block.get("bbox") if isinstance(block, dict) else None
        if not bb or len(bb) != 4:
            continue
        cx = (float(bb[0]) + float(bb[2])) / 2.0
        cy = (float(bb[1]) + float(bb[3])) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            selected.append(block)
    return selected



# ============================================================
# SOURCE-IMAGE HORIZONTAL-RULE TABLE DETECTION
# ============================================================

def _image_horizontal_rule_regions(image_path, blocks):
    """
    Detect table candidates that use horizontal rules but no vertical grid.

    The source image is used to establish the physical horizontal extent of
    the ruled structure.  A pair of unrelated rules elsewhere on a page is
    not sufficient: rules must have a persistent, overlapping x-range and
    the OCR blocks used for reconstruction must occupy that same horizontal
    footprint.  This is important for two-column papers, where unrelated
    horizontal separators can otherwise surround ordinary prose and create a
    false table candidate.

    No document name, page number, expected row count, or known text is used.
    """
    if image_path is None or not blocks:
        return []

    try:
        import cv2
    except Exception:
        return []

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []

        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return []

        kernel_len = max(24, int(width * 0.10))
        detected_rules = []

        for threshold_value in (160, 180, 200, 220, 235):
            _, binary = cv2.threshold(
                image, threshold_value, 255, cv2.THRESH_BINARY_INV
            )
            horizontal = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT, (kernel_len, 1)
                ),
            )

            contours, _ = cv2.findContours(
                horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < max(80, int(width * 0.35)):
                    continue
                if h > max(8, int(width * 0.01)):
                    continue

                detected_rules.append({
                    "y": float(y + h / 2.0),
                    "x1": float(x),
                    "x2": float(x + w),
                    "width": float(w),
                    "threshold": threshold_value,
                })

        if len(detected_rules) < 2:
            return []

        # ------------------------------------------------------------
        # Cluster repeated detections of the same physical rule.
        # Keep the horizontal interval, not only its y-position.  The
        # interval is essential for rejecting unrelated page separators.
        # ------------------------------------------------------------
        detected_rules.sort(key=lambda item: item["y"])
        cluster_tolerance = max(5.0, height * 0.006)
        clusters = []

        for rule in detected_rules:
            if not clusters or abs(rule["y"] - clusters[-1]["y"]) > cluster_tolerance:
                clusters.append({
                    "y": rule["y"],
                    "x1": rule["x1"],
                    "x2": rule["x2"],
                    "count": 1,
                    "samples": [rule],
                })
            else:
                current = clusters[-1]
                n = current["count"]
                current["y"] = (current["y"] * n + rule["y"]) / (n + 1)
                current["x1"] = min(current["x1"], rule["x1"])
                current["x2"] = max(current["x2"], rule["x2"])
                current["count"] += 1
                current["samples"].append(rule)

        rules = [c for c in clusters if c["count"] >= 1]
        if len(rules) < 2:
            return []

        valid_blocks = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            bbox = block.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                bx1, by1, bx2, by2 = map(float, bbox)
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
            except Exception:
                continue
            valid_blocks.append((block, cx, cy, bx1, bx2))

        if not valid_blocks:
            return []

        # ------------------------------------------------------------
        # Build rule groups using BOTH vertical continuity and horizontal
        # footprint continuity.  A single table can contain large vertical
        # whitespace gaps between its header, item rows, and summary rows, so
        # y-distance alone is not a safe grouping criterion.  Rules that share
        # a substantial x-footprint are first allowed to remain in the same
        # physical table candidate; the OCR/table evidence gate below decides
        # whether the complete span is actually tabular.
        # ------------------------------------------------------------
        max_rule_gap = max(120.0, height * 0.25)
        groups = []
        current = [rules[0]]

        for rule in rules[1:]:
            previous = current[-1]
            vertical_gap = rule["y"] - previous["y"]
            local_x1 = max(r["x1"] for r in current)
            local_x2 = min(r["x2"] for r in current)
            common_width_local = max(0.0, local_x2 - local_x1)
            candidate_x1 = max(local_x1, rule["x1"])
            candidate_x2 = min(local_x2, rule["x2"])
            candidate_common = max(0.0, candidate_x2 - candidate_x1)
            rule_width = max(1.0, rule["x2"] - rule["x1"])
            footprint_ratio = candidate_common / rule_width

            same_physical_footprint = (
                candidate_common >= max(80.0, width * 0.20)
                and footprint_ratio >= 0.45
            )

            if vertical_gap <= max_rule_gap or same_physical_footprint:
                current.append(rule)
            else:
                if len(current) >= 2:
                    groups.append(current)
                current = [rule]

        if len(current) >= 2:
            groups.append(current)

        candidates = []

        for group in groups:
            if len(group) < 2:
                continue

            top_rule = group[0]["y"]
            bottom_rule = group[-1]["y"]
            if bottom_rule <= top_rule:
                continue

            # --------------------------------------------------------
            # Find a persistent horizontal footprint shared by the rules.
            # We use the intersection of rule intervals first.  If OCR/table
            # scans contain small breaks, allow a small amount of relaxation,
            # but require substantial overlap among the actual rules.
            # --------------------------------------------------------
            common_x1 = max(rule["x1"] for rule in group)
            common_x2 = min(rule["x2"] for rule in group)
            common_width = max(0.0, common_x2 - common_x1)

            shorter_width = min(
                max(1.0, rule["x2"] - rule["x1"])
                for rule in group
            )
            overlap_ratio = common_width / max(1.0, shorter_width)

            # A real ruled table normally has a substantial common horizontal
            # span.  This rejects separate rules belonging to two page
            # columns or unrelated headings.
            if common_width <= 0 or overlap_ratio < 0.45:
                continue

            # Require the shared ruled footprint itself to be meaningful at
            # page scale.  This remains scale-independent and does not encode
            # any document-specific width.
            if common_width < max(80.0, width * 0.20):
                continue

            # Count how many rules actually cover the common interval.  This
            # protects against one long rule plus unrelated shorter rules.
            supporting_rules = 0
            for rule in group:
                rw = max(1.0, rule["x2"] - rule["x1"])
                overlap = max(
                    0.0,
                    min(rule["x2"], common_x2)
                    - max(rule["x1"], common_x1),
                )
                if overlap / rw >= 0.45:
                    supporting_rules += 1

            if supporting_rules < max(2, int(0.60 * len(group))):
                continue

            # Expand the common interval slightly to include OCR boxes that
            # sit just inside the drawn rule endpoints.
            padding = max(8.0, min(40.0, 0.02 * common_width))
            region_x1 = max(0.0, common_x1 - padding)
            region_x2 = min(float(width), common_x2 + padding)

            selected = [
                block for block, cx, cy, bx1, bx2 in valid_blocks
                if (
                    top_rule - 2.0 <= cy <= bottom_rule + 2.0
                    and region_x1 <= cx <= region_x2
                )
            ]
            if not selected:
                continue

            # OCR can legitimately extend a little beyond the physical rule
            # endpoints.  This happens when a horizontal-rule table has a
            # text column whose glyphs are not covered by the drawn separator
            # (for example a rightmost numeric/tax column).  Do not enlarge
            # the region by a fixed coordinate or a fixed column count.
            # Instead, look immediately outside each rule edge for a stable
            # OCR x-cluster that repeats across the same vertical table band.
            # Only clusters with repeated row support are admitted.
            def _expand_edge(edge, direction):
                band_blocks = []
                search_limit = max(30.0, min(0.12 * common_width, 180.0))
                for block, cx, cy, bx1, bx2 in valid_blocks:
                    if not (top_rule - 2.0 <= cy <= bottom_rule + 2.0):
                        continue
                    if direction < 0:
                        distance = edge - cx
                        if distance < 0 or distance > search_limit:
                            continue
                    else:
                        distance = cx - edge
                        if distance < 0 or distance > search_limit:
                            continue
                    band_blocks.append((block, cx, cy, bx1, bx2))

                if len(band_blocks) < 3:
                    return edge, []

                # Cluster centers using a tolerance derived from OCR box
                # widths.  This is deliberately adaptive to resolution.
                centers = sorted(band_blocks, key=lambda item: item[1])
                clusters = []
                for item in centers:
                    cx = item[1]
                    if not clusters:
                        clusters.append([item])
                        continue
                    previous_cx = sum(v[1] for v in clusters[-1]) / len(clusters[-1])
                    tol = max(10.0, min(0.04 * common_width, 45.0))
                    if abs(cx - previous_cx) <= tol:
                        clusters[-1].append(item)
                    else:
                        clusters.append([item])

                # The OCR rows are reconstructed only after edge expansion,
                # so this step must not depend on candidate_rows. Derive the
                # repetition threshold from the OCR y-bands already present
                # inside the physical rule span.
                row_quantum = max(8.0, 0.012 * height)
                all_row_keys = {
                    round(cy / row_quantum)
                    for _, _, cy, _, _ in valid_blocks
                    if top_rule - 2.0 <= cy <= bottom_rule + 2.0
                }
                min_support = max(3, int(max(1, len(all_row_keys)) * 0.18))
                best = None
                for cluster in clusters:
                    # Distinct OCR rows, rather than raw block count, are the
                    # useful repetition signal.
                    row_keys = set()
                    for _, _, cy, _, _ in cluster:
                        row_keys.add(round(cy / max(8.0, 0.012 * height)))
                    support = len(row_keys)
                    if support < min_support:
                        continue
                    if len(cluster) < support:
                        continue
                    score = support * 2.0 + len(cluster) * 0.25
                    if best is None or score > best[0]:
                        best = (score, cluster, support)

                if best is None:
                    return edge, []

                cluster = best[1]
                if direction < 0:
                    new_edge = min(v[3] for v in cluster) - padding
                else:
                    new_edge = max(v[4] for v in cluster) + padding
                return new_edge, [v[0] for v in cluster]

            expanded_left, left_extra = _expand_edge(region_x1, -1)
            expanded_right, right_extra = _expand_edge(region_x2, +1)

            if expanded_left < region_x1:
                region_x1 = max(0.0, expanded_left)
                selected.extend(left_extra)
            if expanded_right > region_x2:
                region_x2 = min(float(width), expanded_right)
                selected.extend(right_extra)

            # De-duplicate blocks while preserving the OCR objects themselves.
            seen_ids = set()
            selected_unique = []
            for block in selected:
                marker = id(block)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                selected_unique.append(block)
            selected = selected_unique

            # ------------------------------------------------------------
            # SECOND-PASS OCR FOOTPRINT CLOSURE
            # ------------------------------------------------------------
            # Some horizontal-rule tables draw separators that stop before a
            # rightmost/leftmost OCR column.  The first edge-expansion pass
            # above handles the common case.  This second pass is deliberately
            # conservative and uses only repeated OCR geometry within the same
            # ruled y-band.  It prevents a legitimate column from disappearing
            # merely because the first cluster did not meet its support
            # threshold, while still avoiding arbitrary page-wide expansion.
            core_rows = group_blocks_into_visual_rows(selected)
            core_row_count = len(core_rows)
            if core_row_count >= MIN_TABLE_ROWS:
                row_height_values = []
                for row in core_rows:
                    if not row:
                        continue
                    ys = [float(b["bbox"][1]) for b in row if b.get("bbox")]
                    ye = [float(b["bbox"][3]) for b in row if b.get("bbox")]
                    if ys and ye:
                        row_height_values.append(max(1.0, max(ye) - min(ys)))

                median_row_height = (
                    sorted(row_height_values)[len(row_height_values) // 2]
                    if row_height_values else max(12.0, 0.012 * height)
                )
                y_tolerance = max(8.0, min(24.0, 0.60 * median_row_height))
                edge_search = max(40.0, min(0.15 * common_width, 220.0))

                selected_ids = {id(block) for block in selected}

                def _repeated_edge_clusters(direction):
                    edge = region_x2 if direction > 0 else region_x1
                    candidates = []
                    for block, cx, cy, bx1, bx2 in valid_blocks:
                        if id(block) in selected_ids:
                            continue
                        if cy < top_rule - 2.0 or cy > bottom_rule + 2.0:
                            continue
                        distance = (cx - edge) if direction > 0 else (edge - cx)
                        if distance < 0.0 or distance > edge_search:
                            continue
                        candidates.append((block, cx, cy, bx1, bx2))

                    if len(candidates) < 3:
                        return []

                    candidates.sort(key=lambda item: item[1])
                    clusters = []
                    x_tolerance = max(12.0, min(0.035 * common_width, 55.0))
                    for item in candidates:
                        if not clusters:
                            clusters.append([item])
                            continue
                        prev = sum(v[1] for v in clusters[-1]) / len(clusters[-1])
                        if abs(item[1] - prev) <= x_tolerance:
                            clusters[-1].append(item)
                        else:
                            clusters.append([item])

                    accepted = []
                    min_support = max(3, int(0.25 * core_row_count))
                    for cluster in clusters:
                        # Count distinct visual rows represented by this x-cluster.
                        ys = sorted(v[2] for v in cluster)
                        row_groups = []
                        for y in ys:
                            if not row_groups or y - row_groups[-1][-1] > y_tolerance:
                                row_groups.append([y])
                            else:
                                row_groups[-1].append(y)
                        support = len(row_groups)
                        if support < min_support:
                            continue
                        if len(cluster) < support:
                            continue

                        # A stable column should not be spread across a large
                        # fraction of the search band.
                        x_values = [v[1] for v in cluster]
                        x_span = max(x_values) - min(x_values)
                        if x_span > max(30.0, min(0.05 * common_width, 75.0)):
                            continue

                        accepted.append(cluster)

                    return accepted

                for direction in (-1, 1):
                    clusters = _repeated_edge_clusters(direction)
                    if not clusters:
                        continue
                    best_cluster = max(
                        clusters,
                        key=lambda cluster: (
                            len({round(v[2] / max(1.0, y_tolerance)) for v in cluster}),
                            len(cluster),
                        ),
                    )
                    extras = [v[0] for v in best_cluster]
                    selected.extend(extras)
                    selected_ids.update(id(block) for block in extras)
                    if direction > 0:
                        region_x2 = min(
                            float(width),
                            max(region_x2, max(float(v[4]) for v in best_cluster) + padding),
                        )
                    else:
                        region_x1 = max(
                            0.0,
                            min(region_x1, min(float(v[3]) for v in best_cluster) - padding),
                        )

            # De-duplicate again after the closure pass.
            seen_ids = set()
            selected_unique = []
            for block in selected:
                marker = id(block)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                selected_unique.append(block)
            selected = selected_unique

            candidate_rows = group_blocks_into_visual_rows(selected)
            if len(candidate_rows) < MIN_TABLE_ROWS:
                continue

            stats = _fallback_region_statistics(candidate_rows)
            if stats is None:
                continue

            column_count = stats["inferred_column_count"]
            if column_count < 3:
                continue
            if stats["mean_column_occupancy"] < 0.40:
                continue
            if stats["stable_column_ratio"] < 0.40:
                continue
            if stats["repeated_run_ratio"] < 0.30:
                continue

            # The OCR footprint should agree with the final reconstructed
            # physical span.  Use the expanded region here rather than the
            # original rule intersection: a legitimate OCR column may sit
            # just outside the drawn separator endpoints.
            ocr_x1 = min(float(b["bbox"][0]) for b in selected)
            ocr_x2 = max(float(b["bbox"][2]) for b in selected)
            ocr_width = max(1.0, ocr_x2 - ocr_x1)
            final_region_width = max(1.0, region_x2 - region_x1)
            footprint_overlap = max(
                0.0,
                min(ocr_x2, region_x2) - max(ocr_x1, region_x1),
            ) / max(1.0, min(ocr_width, final_region_width))

            if footprint_overlap < 0.45:
                continue

            candidates.append({
                "bbox": [
                    region_x1,
                    max(0.0, top_rule - 2.0),
                    region_x2,
                    min(float(height), bottom_rule + 2.0),
                ],
                "rows": candidate_rows,
                "horizontal_rules": len(group),
                "supporting_rules": supporting_rules,
                "vertical_rules": 0,
                "rule_overlap_ratio": overlap_ratio,
                "rule_common_width": common_width,
            })

        # Merge overlapping detections produced by different threshold
        # levels or nearby rule groups.  Preserve distinct side-by-side
        # physical structures.
        merged = []
        for item in candidates:
            attached = False
            for current in merged:
                a = current["bbox"]
                b = item["bbox"]
                ix1 = max(a[0], b[0])
                iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2])
                iy2 = min(a[3], b[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
                area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                if inter / min(area_a, area_b) >= 0.50:
                    current["bbox"] = [
                        min(a[0], b[0]), min(a[1], b[1]),
                        max(a[2], b[2]), max(a[3], b[3]),
                    ]
                    current["horizontal_rules"] = max(
                        current["horizontal_rules"], item["horizontal_rules"]
                    )
                    current["supporting_rules"] = max(
                        current.get("supporting_rules", 0),
                        item.get("supporting_rules", 0),
                    )
                    current["rule_overlap_ratio"] = max(
                        current.get("rule_overlap_ratio", 0.0),
                        item.get("rule_overlap_ratio", 0.0),
                    )
                    current["rule_common_width"] = max(
                        current.get("rule_common_width", 0.0),
                        item.get("rule_common_width", 0.0),
                    )
                    if len(item["rows"]) > len(current["rows"]):
                        current["rows"] = item["rows"]
                    attached = True
                    break
            if not attached:
                merged.append(item)

        merged.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return merged

    except Exception:
        return []

def _fallback_region_is_table(
    region,
    image_path=None
):
    """
    Generic evidence gate for paragraph-based table reconstruction.

    Supports three physical table styles:
      1. fully bordered/grid-backed tables;
      2. horizontally ruled tables without vertical borders;
      3. completely borderless tables.

    Borderless recognition is adaptive rather than requiring a fixed row or
    column count.  Multi-column journal prose is rejected unless it shows
    substantially stronger repeated cell geometry than normal text flow.
    """
    stats = _fallback_region_statistics(region)
    if stats is None:
        return False, {"accepted": False, "reason": "NO_REGION"}

    row_count = stats["row_count"]
    column_count = stats["inferred_column_count"]
    dense_ratio = stats["dense_ratio"]
    column_occupancy = stats["mean_column_occupancy"]
    stable_column_ratio = stats["stable_column_ratio"]
    structure_similarity = stats["structure_similarity"]
    repeated_run_ratio = stats["repeated_run_ratio"]
    core_score = stats["core_score"]
    prose_flow_score = stats.get("prose_flow_score", 0.0)
    prose_full_slot_ratio = stats.get("prose_full_slot_ratio", 0.0)
    prose_high_fragmentation_ratio = stats.get("prose_high_fragmentation_ratio", 0.0)
    prose_multi_line_row_ratio = stats.get("prose_multi_line_row_ratio", 0.0)
    continuous_prose_layout = stats.get("continuous_prose_layout", False)
    prose_sentence_column_ratio = stats.get("prose_sentence_column_ratio", 0.0)
    columnar_prose_layout = stats.get("columnar_prose_layout", False)
    prose_sentence_block_ratio = stats.get("prose_sentence_block_ratio", 0.0)
    sentence_dense_prose_columns = stats.get("sentence_dense_prose_columns", 0)
    sentence_dense_prose_layout = stats.get("sentence_dense_prose_layout", False)
    column_spacing_imbalance = stats.get("column_spacing_imbalance", False)
    uneven_column_prose_layout = stats.get("uneven_column_prose_layout", False)

    image = _image_table_evidence(image_path, region)

    # Fully bordered/grid-backed table.
    image_grid_supported = (
        image["available"]
        and image.get("horizontal_lines", 0) >= 3
        and image.get("vertical_lines", 0) >= 2
        and image.get("intersections", 0) >= max(20, IMAGE_TABLE_INTERSECTION_MIN)
        and row_count >= 4
        and column_count >= 3
        and dense_ratio >= 0.40
        and column_occupancy >= 0.45
        and core_score >= 0.45
    )

    # Horizontal-rule table: vertical borders are intentionally NOT required.
    horizontal_rule_table = (
        image["available"]
        and image.get("horizontal_lines", 0) >= 2
        and image.get("vertical_lines", 0) <= 1
        and column_count >= 3
        and row_count >= 3
        and dense_ratio >= 0.45
        and column_occupancy >= 0.40
        and stable_column_ratio >= 0.40
        and repeated_run_ratio >= 0.30
        and structure_similarity >= 0.45
        and core_score >= 0.40
        and not continuous_prose_layout
    )

    # Adaptive borderless evidence. There is no hard-coded minimum of six
    # rows or three columns. Smaller tables must compensate with stronger
    # alignment; larger tables can accumulate evidence across more rows.
    alignment_score = (
        0.35 * stable_column_ratio
        + 0.25 * structure_similarity
        + 0.20 * repeated_run_ratio
        + 0.20 * column_occupancy
    )

    # 3+ columns: normal borderless table path.
    borderless_3plus = (
        column_count >= 3
        and row_count >= 3
        and dense_ratio >= 0.50
        and column_occupancy >= 0.55
        and stable_column_ratio >= 0.55
        and structure_similarity >= 0.65
        and repeated_run_ratio >= 0.50
        and alignment_score >= 0.58
        and core_score >= 0.55
        # Continuous prose can satisfy the positive geometry tests.  Do not
        # accept it unless the OCR blocks also fail the prose-flow test.
        and prose_flow_score < 0.72
        and not continuous_prose_layout
        and not columnar_prose_layout
        and not sentence_dense_prose_layout
        and not uneven_column_prose_layout
    )

    # Very small 3+ column tables: stronger geometry compensates for fewer rows.
    small_borderless = (
        column_count >= 3
        and 2 <= row_count < 3
        and dense_ratio >= 0.50
        and column_occupancy >= 0.70
        and stable_column_ratio >= 0.70
        and structure_similarity >= 0.70
        and core_score >= 0.60
        and prose_flow_score < 0.72
        and not continuous_prose_layout
        and not columnar_prose_layout
        and not sentence_dense_prose_layout
        and not uneven_column_prose_layout
    )

    # Two-column borderless tables are the highest-risk case because journal
    # papers commonly use exactly two page columns. Require very strong
    # repeated cell geometry and compact row structure.
    borderless_2col = (
        column_count == 2
        and row_count >= 3
        and dense_ratio >= 0.70
        and column_occupancy >= 0.85
        and stable_column_ratio >= 0.85
        and structure_similarity >= 0.78
        and repeated_run_ratio >= 0.70
        and core_score >= 0.65
        # Two-column prose is the most common false-positive layout.
        # Require stronger row geometry and reject continuous line-flow.
        and prose_flow_score < 0.60
        and not continuous_prose_layout
        and not columnar_prose_layout
        and not sentence_dense_prose_layout
        and not uneven_column_prose_layout
    )

    accepted = (
        image_grid_supported
        or horizontal_rule_table
        or borderless_3plus
        or small_borderless
        or borderless_2col
    )

    evidence = {
        **stats,
        "image": image,
        "image_grid_supported": image_grid_supported,
        "horizontal_rule_table": horizontal_rule_table,
        "borderless_3plus": borderless_3plus,
        "small_borderless": small_borderless,
        "borderless_2col": borderless_2col,
        "borderless_alignment_score": round(alignment_score, 4),
        "prose_flow_score": round(prose_flow_score, 4),
        "prose_full_slot_ratio": round(prose_full_slot_ratio, 4),
        "prose_long_block_ratio": round(stats.get("prose_long_block_ratio", 0.0), 4),
        "prose_high_fragmentation_ratio": round(prose_high_fragmentation_ratio, 4),
        "continuous_prose_layout": continuous_prose_layout,
        "prose_multi_line_row_ratio": round(prose_multi_line_row_ratio, 4),
        "prose_sentence_column_ratio": round(prose_sentence_column_ratio, 4),
        "columnar_prose_layout": columnar_prose_layout,
        "prose_sentence_block_ratio": round(prose_sentence_block_ratio, 4),
        "sentence_dense_prose_columns": sentence_dense_prose_columns,
        "sentence_dense_prose_layout": sentence_dense_prose_layout,
        "column_spacing_imbalance": column_spacing_imbalance,
        "uneven_column_prose_layout": uneven_column_prose_layout,
        "accepted": accepted,
    }

    if image_grid_supported:
        evidence["reason"] = "OCR_GEOMETRY_PLUS_IMAGE_GRID"
    elif horizontal_rule_table:
        evidence["reason"] = "OCR_GEOMETRY_PLUS_HORIZONTAL_RULES"
    elif small_borderless:
        evidence["reason"] = "SMALL_BORDERLESS_REPEATED_STRUCTURE"
    elif borderless_3plus:
        evidence["reason"] = "ADAPTIVE_BORDERLESS_MULTI_COLUMN_STRUCTURE"
    elif borderless_2col:
        evidence["reason"] = "HIGH_CONFIDENCE_TWO_COLUMN_TABLE_STRUCTURE"
    elif uneven_column_prose_layout:
        evidence["reason"] = "UNEVEN_COLUMN_PROSE_LAYOUT_REJECTED"
    elif sentence_dense_prose_layout:
        evidence["reason"] = "SENTENCE_DENSE_PROSE_LAYOUT_REJECTED"
    elif columnar_prose_layout:
        evidence["reason"] = "COLUMNAR_PROSE_LAYOUT_REJECTED"
    elif prose_flow_score >= 0.72:
        evidence["reason"] = "CONTINUOUS_PROSE_LAYOUT_REJECTED"
    else:
        evidence["reason"] = "INSUFFICIENT_TABLE_EVIDENCE"

    return accepted, evidence


def _row_table_likeness(row, column_centers):
    """Measure how strongly one OCR row occupies inferred table columns."""
    if not row or not column_centers:
        return 0.0

    assigned = set()
    for block in row:
        center = bbox_center(block.get("bbox"))
        if center is None:
            continue
        nearest = nearest_column_index(center[0], column_centers)
        if nearest is None:
            continue
        if abs(center[0] - column_centers[nearest]) <= COLUMN_ASSIGNMENT_TOLERANCE:
            assigned.add(nearest)

    return len(assigned) / max(1, len(column_centers))


def _trim_fallback_region_to_sustained_structure(region):
    """
    Trim a broad fallback candidate to its strongest sustained table-like
    subregion.

    This is document-independent. It uses only row geometry, inferred
    columns, row occupancy and continuity. It never uses page numbers,
    document names, expected row counts, or known answers.

    A broad candidate can contain surrounding prose because find_table_regions()
    intentionally errs toward recall. Before reconstruction, retain the
    strongest contiguous run of rows that repeatedly occupies the inferred
    columns.
    """
    if not region or len(region) < FALLBACK_MIN_ROWS:
        return region

    centers = get_table_column_centers(region)
    if len(centers) < 2:
        return region

    scores = [
        _row_table_likeness(row, centers)
        for row in region
    ]

    # A row is a strong table row when it occupies a meaningful fraction
    # of the inferred columns. Keep a lower threshold for OCR fragmentation.
    threshold = 0.35 if len(centers) >= 4 else 0.50
    flags = [score >= threshold for score in scores]

    # Find contiguous runs of structurally strong rows.
    runs = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(region) - 1))

    # Do not trim unless a sufficiently long sustained run exists.
    min_run = max(4, int(len(region) * 0.20))
    eligible = [
        (a, b) for a, b in runs
        if (b - a + 1) >= min_run
    ]

    if not eligible:
        return region

    # Score each run by length and mean column occupancy.
    best = None
    best_score = -1.0
    for a, b in eligible:
        run_scores = scores[a:b + 1]
        mean_score = sum(run_scores) / len(run_scores)
        run_length = b - a + 1
        score = mean_score * 0.70 + min(1.0, run_length / max(1, len(region))) * 0.30
        if score > best_score:
            best_score = score
            best = (a, b)

    if best is None:
        return region

    a, b = best

    # Preserve a genuinely table-like boundary row immediately adjacent
    # to the sustained run (headers can have fewer occupied columns).
    if a > 0 and scores[a - 1] >= 0.20:
        a -= 1
    if b + 1 < len(region) and scores[b + 1] >= 0.20:
        b += 1

    trimmed = region[a:b + 1]

    # Safety: never replace a candidate with a smaller region unless it
    # still contains enough rows and multiple inferred columns.
    if len(trimmed) >= MIN_TABLE_ROWS:
        return trimmed

    return region



def _global_horizontal_split(region, image_path=None):
    """
    Detect a persistent page-level vertical whitespace corridor from the
    distribution of OCR bounding boxes.  This is a fallback for side-by-side
    structures whose left and right rows are vertically offset enough that a
    row-by-row separator test cannot see both sides simultaneously.

    The split is inferred from observed geometry and robust gap statistics;
    it never uses document-specific coordinates or expected column counts.
    """
    if not region:
        return [region]

    blocks = [
        b for row in region for b in row
        if isinstance(b, dict) and b.get("bbox") and len(b.get("bbox")) == 4
    ]
    if len(blocks) < 8:
        return [region]

    # A global split is deliberately a high-recall mechanism, but page-layout
    # columns are common in papers and reports.  Before allowing a global
    # split to alter the candidate, require independent image evidence that
    # the broad region contains table-like ruling/intersections.  This keeps
    # ordinary multi-column prose from being promoted into tables.
    image_evidence = _image_table_evidence(image_path, region)
    if not image_evidence.get("available"):
        return [region]
    if image_evidence.get("intersections", 0) < IMAGE_TABLE_INTERSECTION_MIN:
        return [region]

    centers = sorted(
        [((float(b["bbox"][0]) + float(b["bbox"][2])) / 2.0, b) for b in blocks],
        key=lambda item: item[0]
    )

    # Estimate the natural spacing between repeated x centers by clustering
    # nearby centers.  A persistent component boundary should create a gap
    # materially larger than ordinary center spacing.
    xs = [x for x, _ in centers]
    gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
    if len(gaps) < 4:
        return [region]

    gaps_sorted = sorted(gaps)
    median_gap = gaps_sorted[len(gaps_sorted) // 2]
    q75 = gaps_sorted[int(0.75 * (len(gaps_sorted) - 1))]
    q90 = gaps_sorted[int(0.90 * (len(gaps_sorted) - 1))]

    candidates = []
    for i, gap in enumerate(gaps):
        if gap < max(1.8 * median_gap, q75):
            continue
        x = (xs[i] + xs[i + 1]) / 2.0
        left = [b for cx, b in centers if cx < x]
        right = [b for cx, b in centers if cx >= x]
        if len(left) < 3 or len(right) < 3:
            continue

        # Require both sides to occupy multiple vertical levels.  This avoids
        # mistaking a large gap inside a single sparse line for a component
        # boundary.
        left_ys = sorted((float(b["bbox"][1]) + float(b["bbox"][3])) / 2 for b in left)
        right_ys = sorted((float(b["bbox"][1]) + float(b["bbox"][3])) / 2 for b in right)
        y_span = max(
            max(left_ys[-1] - left_ys[0], 0.0),
            max(right_ys[-1] - right_ys[0], 0.0)
        )
        if y_span <= 0:
            continue

        strength = gap / max(1.0, median_gap)
        if gap >= q90:
            strength += 0.5
        candidates.append((strength, gap, x))

    if not candidates:
        return [region]

    candidates.sort(reverse=True)
    _, _, separator = candidates[0]

    left_region = []
    right_region = []
    for row in region:
        left_row = []
        right_row = []
        for block in row:
            bbox = block.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            if cx < separator:
                left_row.append(block)
            else:
                right_row.append(block)
        if left_row:
            left_region.append(sorted(left_row, key=lambda b: float(b["bbox"][0])))
        if right_row:
            right_region.append(sorted(right_row, key=lambda b: float(b["bbox"][0])))

    if len(left_region) < 4 or len(right_region) < 4:
        return [region]

    left_structured = sum(len(r) >= 2 for r in left_region)
    right_structured = sum(len(r) >= 2 for r in right_region)
    if left_structured < 2 or right_structured < 2:
        return [region]

    # Validate the proposed components against the source image as well.
    # This prevents a large whitespace gap in a normal two-column document
    # from becoming a table separator merely because the OCR geometry is
    # structured.
    left_image = _image_table_evidence(image_path, left_region)
    right_image = _image_table_evidence(image_path, right_region)
    def _has_grid_evidence(evidence):
        if not evidence.get("available"):
            return False
        return (
            evidence.get("horizontal_lines", 0) >= 3
            and evidence.get("vertical_lines", 0) >= 2
            and evidence.get("intersections", 0) >= max(20, IMAGE_TABLE_INTERSECTION_MIN)
        )

    # A global split is only allowed when both proposed components show
    # actual grid-like image evidence.  This is important because ordinary
    # paper layouts also contain persistent vertical whitespace corridors.
    if not (_has_grid_evidence(left_image) and _has_grid_evidence(right_image)):
        return [region]

    return [left_region, right_region]


def _horizontal_region_components(region, image_path=None):
    """
    Split a broad fallback region into physically separated horizontal
    components using persistent whitespace corridors in OCR geometry.

    The important distinction is between an ordinary inter-column gap and a
    document-level separation: an ordinary column gap is usually small
    relative to the OCR boxes around it, while a true side-by-side separation
    persists across many rows and is substantially wider than the local gaps.
    No document-specific coordinates or column counts are used.
    """
    if not region:
        return []

    rows = []
    for row in region:
        blocks = [
            b for b in row
            if isinstance(b, dict)
            and b.get("bbox")
            and len(b.get("bbox")) == 4
        ]
        if blocks:
            rows.append(sorted(blocks, key=lambda b: float(b["bbox"][0])))

    if len(rows) < 4:
        return [region]

    widths = []
    all_gaps = []
    row_intervals = []

    for row in rows:
        intervals = []
        for block in row:
            x1, _, x2, _ = map(float, block["bbox"])
            if x2 > x1:
                widths.append(x2 - x1)
                intervals.append((x1, x2))
        intervals.sort()
        row_intervals.append(intervals)
        for left, right in zip(intervals, intervals[1:]):
            gap = max(0.0, right[0] - left[1])
            if gap > 0:
                all_gaps.append(gap)

    if not widths or not all_gaps:
        return [region]

    widths_sorted = sorted(widths)
    median_width = widths_sorted[len(widths_sorted) // 2]
    gaps_sorted = sorted(all_gaps)
    median_gap = gaps_sorted[len(gaps_sorted) // 2]
    upper_gap = gaps_sorted[int(0.85 * (len(gaps_sorted) - 1))]

    # Candidate x positions are midpoints of every observed between-block gap.
    candidates = []
    for ri, intervals in enumerate(row_intervals):
        for left, right in zip(intervals, intervals[1:]):
            gap = max(0.0, right[0] - left[1])
            if gap <= 0:
                continue
            x = (left[1] + right[0]) / 2.0
            candidates.append((x, gap, ri))

    if len(candidates) < 4:
        return [region]

    # Cluster nearby gap positions. The scale is derived from OCR box size,
    # not from the current document's dimensions.
    tolerance = max(20.0, min(80.0, 0.40 * median_width))
    clusters = []
    for item in sorted(candidates, key=lambda z: z[0]):
        if not clusters:
            clusters.append([item])
            continue
        center = sum(v[0] for v in clusters[-1]) / len(clusters[-1])
        if abs(item[0] - center) <= tolerance:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    scored = []
    for cluster in clusters:
        x = sum(v[0] for v in cluster) / len(cluster)
        support_rows = {v[2] for v in cluster}
        support = len(support_rows) / len(rows)
        gaps = [v[1] for v in cluster]
        median_cluster_gap = sorted(gaps)[len(gaps) // 2]

        # Count rows that have meaningful OCR on both sides of the candidate.
        both = 0
        left_counts = []
        right_counts = []
        for intervals in row_intervals:
            left_count = sum(1 for x1, x2 in intervals if x2 <= x)
            right_count = sum(1 for x1, x2 in intervals if x1 >= x)
            left_counts.append(left_count)
            right_counts.append(right_count)
            if left_count >= 2 and right_count >= 2:
                both += 1

        both_ratio = both / len(rows)
        left_structured = sum(1 for n in left_counts if n >= 2) / len(rows)
        right_structured = sum(1 for n in right_counts if n >= 2) / len(rows)

        # A separator should be materially wider than ordinary local gaps.
        gap_ratio = median_cluster_gap / max(1.0, median_gap)
        width_ratio = median_cluster_gap / max(1.0, median_width)

        # Estimate how much of the page lies on each side. This rejects
        # separators that sit at the extreme edge of a candidate region.
        xs = [v for intervals in row_intervals for pair in intervals for v in pair]
        left_edge, right_edge = min(xs), max(xs)
        position = (x - left_edge) / max(1.0, right_edge - left_edge)
        balanced = 1.0 if 0.15 <= position <= 0.85 else 0.0

        # Persistence is the primary signal; gap width and two-sided
        # structure distinguish a real component boundary from normal columns.
        score = (
            0.30 * support
            + 0.25 * both_ratio
            + 0.15 * left_structured
            + 0.15 * right_structured
            + 0.10 * min(1.0, gap_ratio / 3.0)
            + 0.05 * min(1.0, width_ratio / 1.5)
        )

        # Require a genuinely broad corridor, or an exceptionally persistent
        # two-sided separator. This remains scale-independent.
        broad_gap = (
            median_cluster_gap >= max(
                2.5 * median_gap,
                0.75 * median_width
            )
            or median_cluster_gap >= upper_gap
        )
        persistent = support >= 0.50 and both_ratio >= 0.35

        if balanced and persistent and broad_gap and score >= 0.50:
            scored.append({
                "x": x,
                "score": score,
                "support": support,
                "both_ratio": both_ratio,
                "median_gap": median_cluster_gap,
            })

    if not scored:
        global_components = _global_horizontal_split(region, image_path=image_path)
        return global_components if len(global_components) >= 2 else [region]

    scored.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    min_sep = max(50.0, 0.90 * median_width)
    for candidate in scored:
        if all(abs(candidate["x"] - old["x"]) >= min_sep for old in selected):
            selected.append(candidate)

    if not selected:
        return [region]

    selected.sort(key=lambda c: c["x"])
    separators = [c["x"] for c in selected]

    components = []
    for ci in range(len(separators) + 1):
        left_boundary = separators[ci - 1] if ci else float("-inf")
        right_boundary = separators[ci] if ci < len(separators) else float("inf")
        subregion = []
        for original_row in region:
            chosen = []
            for block in original_row:
                bbox = block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                center = (float(bbox[0]) + float(bbox[2])) / 2.0
                if left_boundary <= center < right_boundary:
                    chosen.append(block)
            if chosen:
                subregion.append(sorted(chosen, key=lambda b: float(b["bbox"][0])))

        if subregion:
            components.append(subregion)

    # Only accept a split when every retained component has sustained
    # structured rows. Otherwise preserve the original candidate unchanged.
    valid = []
    for component in components:
        structured = sum(1 for row in component if len(row) >= 2)
        if len(component) >= 4 and structured >= max(2, int(0.35 * len(component))):
            valid.append(component)

    return valid if len(valid) >= 2 else [region]

def _table_bbox(table):
    """Return the overall bbox of a reconstructed table when available."""
    bboxes = []
    for row in table.get("rows", []):
        for cell in row.get("cells", []):
            bbox = normalize_bbox(cell.get("bbox"))
            if bbox is not None:
                bboxes.append(bbox)
    if not bboxes:
        return None
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _bbox_overlap_ratio(a, b):
    """Intersection over the smaller bbox area."""
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / min(aa, bb)


def _caption_is_figure(text):
    """Generic figure-caption detector; never tied to a specific document."""
    if not text:
        return False
    return bool(re.search(
        r"\b(?:figure|fig\.)\s*[A-Za-z0-9IVXivx]+\b",
        str(text),
        flags=re.IGNORECASE,
    ))


def _tesseract_fallback_blocks(image_path):
    """
    Optional page-image OCR fallback for pages where the primary OCR did not
    produce usable paragraph blocks. This is generic and is used only to
    recover spatial text evidence needed by table reconstruction and figure
    caption detection. It never replaces the primary OCR output.
    """
    if image_path is None:
        return []
    try:
        import pytesseract
    except Exception:
        return []
    try:
        data = pytesseract.image_to_data(
            str(image_path),
            output_type=pytesseract.Output.DICT,
            config="--psm 11"
        )
    except Exception:
        return []

    blocks = []
    texts = data.get("text", [])
    count = len(texts)
    for i in range(count):
        text = str(texts[i]).strip()
        if not text:
            continue
        try:
            x = float(data["left"][i])
            y = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        try:
            confidence = float(data.get("conf", [0] * count)[i])
        except Exception:
            confidence = 0.0
        blocks.append({
            "text": text,
            "bbox": [x, y, x + w, y + h],
            "confidence": confidence,
            "x": x + (w / 2.0),
            "y": y + (h / 2.0),
        })
    return blocks


def _align_ocr_blocks_to_image_coordinates(blocks, image_path):
    """
    Align OCR bounding boxes with source-image pixel coordinates when the
    OCR engine returned coordinates in a different raster scale.

    This is detected from robust bbox extents versus the actual image size.
    A few corrupted/outlier boxes cannot trigger the rescaling because
    percentiles are used instead of raw maxima.

    No document, page, text, or expected table dimension is used.
    """
    if not blocks or image_path is None:
        return list(blocks or [])

    try:
        import cv2
        import numpy as np

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            return list(blocks)

        image_h, image_w = image.shape[:2]

        x2_values = []
        y2_values = []

        for block in blocks:
            bb = normalize_bbox(block.get("bbox"))

            if bb is None:
                continue

            x2_values.append(float(bb[2]))
            y2_values.append(float(bb[3]))

        if len(x2_values) < 5 or len(y2_values) < 5:
            return list(blocks)

        p95_x = float(np.percentile(x2_values, 95))
        p95_y = float(np.percentile(y2_values, 95))

        # Detect scale independently on each axis. OCR/layout engines can
        # preserve one axis correctly while returning the other in a different
        # raster coordinate system. Requiring both axes to be oversized leaves
        # exactly that class of malformed geometry untreated.
        scale_x = 1.0
        scale_y = 1.0

        if p95_x > 1.35 * image_w:
            candidate_x = image_w / max(1.0, p95_x)
            if 0.10 <= candidate_x <= 1.10:
                scale_x = candidate_x

        if p95_y > 1.35 * image_h:
            candidate_y = image_h / max(1.0, p95_y)
            if 0.10 <= candidate_y <= 1.10:
                scale_y = candidate_y

        if scale_x == 1.0 and scale_y == 1.0:
            return list(blocks)

        aligned = []

        for block in blocks:
            copied = dict(block)
            bb = normalize_bbox(block.get("bbox"))

            if bb is not None:
                copied["bbox"] = [
                    bb[0] * scale_x,
                    bb[1] * scale_y,
                    bb[2] * scale_x,
                    bb[3] * scale_y,
                ]

                # Some paragraph helpers cache the left x-coordinate
                # separately. Keep it synchronized with the bbox.
                copied["x"] = copied["bbox"][0]

            aligned.append(copied)

        return aligned

    except Exception:
        return list(blocks)


def _effective_ocr_blocks(page_blocks, image_path):
    """Use primary OCR blocks, aligned to the current source-image coordinate space."""
    blocks = _align_ocr_blocks_to_image_coordinates(
        list(page_blocks or []),
        image_path
    )

    if len(blocks) >= 2:
        return blocks

    fallback = _tesseract_fallback_blocks(image_path)
    fallback = _align_ocr_blocks_to_image_coordinates(
        fallback,
        image_path
    )

    return fallback if len(fallback) > len(blocks) else blocks


def _ocr_region_blocks(image_path, bbox):
    """Run one OCR pass over a table region and return global-coordinate blocks.

    Cell-by-cell Tesseract calls are prohibitively expensive when a noisy grid
    temporarily produces many empty cells. A single region pass preserves the
    same source-image fallback principle while keeping runtime bounded by the
    number of detected visual regions rather than the number of cells.
    """
    if image_path is None or not bbox:
        return []
    try:
        import cv2
        import pytesseract

        image = cv2.imread(str(image_path))
        if image is None:
            return []

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return []

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if max(gray.shape[:2]) < 1200:
            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            scale = 1.5
        else:
            scale = 1.0

        def _run_region_ocr(psm):
            try:
                return pytesseract.image_to_data(
                    gray,
                    output_type=pytesseract.Output.DICT,
                    config=f"--psm {psm}"
                )
            except Exception:
                return {}

        def _convert_region_data(data):
            converted = []
            texts = data.get("text", [])
            count = len(texts)

            for i in range(count):
                text_value = str(texts[i]).strip()
                if not text_value:
                    continue

                try:
                    tx = float(data["left"][i]) / scale + x1
                    ty = float(data["top"][i]) / scale + y1
                    tw = float(data["width"][i]) / scale
                    th = float(data["height"][i]) / scale
                except Exception:
                    continue

                if tw <= 0 or th <= 0:
                    continue

                try:
                    conf = float(
                        data.get("conf", [0.0] * count)[i]
                    )
                except Exception:
                    conf = 0.0

                converted.append({
                    "text": text_value,
                    "bbox": [
                        tx,
                        ty,
                        tx + tw,
                        ty + th
                    ],
                    "confidence": conf,
                    "x": tx + tw / 2.0,
                    "y": ty + th / 2.0,
                    "source": "region_image_ocr"
                })

            return converted

        # First pass: preserve the existing table-oriented OCR behavior.
        primary_data = _run_region_ocr(6)
        output = _convert_region_data(primary_data)

        # Recovery pass: PSM 11 is substantially better for sparse cells,
        # isolated values, and OCR text separated by table rules.  Run it only
        # when the first pass produced little/no usable evidence, so ordinary
        # pages do not pay for two OCR passes unnecessarily.
        if len(output) < 2:
            recovery_data = _run_region_ocr(11)
            recovery = _convert_region_data(recovery_data)

            # Keep both passes only when they provide genuinely different
            # spatial text.  This prevents duplicate words from contaminating
            # cell assignment while still recovering text missed by PSM 6.
            for candidate in recovery:
                cbb = normalize_bbox(candidate.get("bbox"))
                duplicate = False

                if cbb is not None:
                    for existing in output:
                        ebb = normalize_bbox(existing.get("bbox"))
                        if ebb is None:
                            continue

                        overlap = _bbox_overlap_ratio(
                            cbb,
                            ebb
                        )

                        if overlap >= 0.60:
                            duplicate = True
                            break

                if not duplicate:
                    output.append(candidate)

        return output
    except Exception:
        return []


def _ocr_crop_text(image_path, bbox):
    """Optional cell-level OCR used only when no primary OCR text landed in a cell."""
    if image_path is None or not bbox:
        return "", 0.0
    try:
        import cv2
        import pytesseract
        image = cv2.imread(str(image_path))
        if image is None:
            return "", 0.0
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return "", 0.0
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Preserve the source content; only enlarge small cells for OCR.
        if max(gray.shape[:2]) < 500:
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"
        )
        parts = []
        scores = []
        for i, text in enumerate(data.get("text", [])):
            text = str(text).strip()
            if not text:
                continue
            parts.append(text)
            try:
                scores.append(float(data.get("conf", [0])[i]))
            except Exception:
                pass
        if not parts:
            return "", 0.0
        return " ".join(parts), (sum(scores) / len(scores) if scores else 0.0)
    except Exception:
        return "", 0.0


def _large_visual_regions(image_path):
    """
    Find page-level raster/visual containers that are much larger than a
    normal table region. This is deliberately conservative so a scanned page
    with ordinary text is not classified as an embedded figure.
    """
    if image_path is None:
        return []
    try:
        import cv2
    except Exception:
        return []
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []
        height, width = image.shape[:2]
        _, binary = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        regions = []
        for index in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[index]]
            if w < 0.75 * width or h < 0.40 * height:
                continue
            if area < 0.10 * width * height:
                continue
            regions.append({
                "bbox": [x, y, x + w, y + h],
                "kind": "large_visual_container",
            })
        return regions
    except Exception:
        return []


def _embedded_visual_regions(image_path, blocks):
    """
    Identify source-image regions that should not be reconstructed as document
    tables because they are embedded figures/screenshots.

    Evidence is generic:
      * a detected visual/grid region has a nearby Figure/Fig. caption; or
      * a very large page-level visual container is present.

    This protects table reconstruction from OCR text inside figures and
    screenshots without using page numbers, document names, or known answers.
    """
    if image_path is None:
        return []

    try:
        import cv2
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return []
        page_h, page_w = image.shape[:2]
    except Exception:
        return []

    valid_blocks = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        bbox = normalize_bbox(block.get("bbox"))
        if bbox is None:
            continue
        valid_blocks.append((block, bbox))

    # OCR engines may return a caption as one block or as many word blocks.
    # Build a line-level caption bbox around any Figure/Fig. token.
    figure_captions = []
    for block, bbox in valid_blocks:
        text = str(block.get("text", "")).strip()
        if not re.search(r"\b(?:figure|fig\.)\s*[A-Za-z0-9IVXivx]+\b|^\s*(?:figure|fig\.)", text, re.IGNORECASE):
            continue
        cy = (bbox[1] + bbox[3]) / 2.0
        line_members = []
        for _, obox in valid_blocks:
            oy = (obox[1] + obox[3]) / 2.0
            if abs(oy - cy) <= max(12.0, 1.5 * (bbox[3] - bbox[1])):
                line_members.append(obox)
        if line_members:
            figure_captions.append([
                min(x[0] for x in line_members),
                min(x[1] for x in line_members),
                max(x[2] for x in line_members),
                max(x[3] for x in line_members),
            ])
        else:
            figure_captions.append(bbox)

    regions = []
    grid_regions = _image_grid_regions(image_path)

    # Figure caption evidence: the caption normally sits immediately below
    # the visual. Require horizontal overlap and a modest vertical gap.
    max_gap = max(80.0, 0.16 * page_h)
    for region in grid_regions:
        rb = region.get("bbox")
        if not rb:
            continue
        rx1, ry1, rx2, ry2 = rb
        for cb in figure_captions:
            cx1, cy1, cx2, cy2 = cb
            overlap = max(0.0, min(rx2, cx2) - max(rx1, cx1))
            overlap_ratio = overlap / max(1.0, min(rx2 - rx1, cx2 - cx1))
            vertical_gap = cy1 - ry2
            if overlap_ratio >= 0.25 and -10 <= vertical_gap <= max_gap:
                regions.append({
                    "bbox": rb,
                    "kind": "figure_caption_region",
                })
                break

    # Large screenshots/collages are visual objects even when they have no
    # caption. Keep this rule conservative enough not to catch ordinary
    # scanned pages made of separated text blocks.
    regions.extend(_large_visual_regions(image_path))

    # Merge near-identical regions.
    merged = []
    for item in regions:
        bbox = item["bbox"]
        if any(_bbox_overlap_ratio(bbox, old["bbox"]) >= 0.70 for old in merged):
            continue
        merged.append(item)
    return merged


def _blocks_outside_regions(blocks, regions):
    """Keep OCR blocks whose centers are outside excluded visual regions."""
    if not regions:
        return list(blocks)
    output = []
    for block in blocks:
        bbox = normalize_bbox(block.get("bbox")) if isinstance(block, dict) else None
        if bbox is None:
            continue
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        inside = any(
            r["bbox"][0] <= cx <= r["bbox"][2]
            and r["bbox"][1] <= cy <= r["bbox"][3]
            for r in regions
        )
        if not inside:
            output.append(block)
    return output


def _grid_line_positions(image_path, bbox, small_grid=False):
    """Recover physical grid lines from the source image.

    Straight Hough lines are preferred when they provide a stable repeated
    structure (especially for screenshot row separators). Morphological lines
    remain the fallback for light borders that Hough does not see.
    """
    if image_path is None or not bbox:
        return [], []
    try:
        import cv2
        import numpy as np
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return [], []

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return [], []
        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]

        def cluster(values, tolerance):
            if not values:
                return []
            values = sorted(float(v) for v in values)
            groups = [[values[0]]]
            for value in values[1:]:
                if abs(value - groups[-1][-1]) <= tolerance:
                    groups[-1].append(value)
                else:
                    groups.append([value])
            return [sum(g) / len(g) for g in groups]

        # Morphological evidence.
        morph_y = []
        morph_x = []
        for threshold in ((180, 200, 220, 235) if small_grid else (180, 200, 220)):
            _, binary = cv2.threshold(crop, threshold, 255, cv2.THRESH_BINARY_INV)
            hk = max(8, int(w * 0.015))
            vk = max(8, int(h * 0.015))
            horizontal = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
            )
            vertical = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
            )
            for contour in cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                xx, yy, ww, hh = cv2.boundingRect(contour)
                if ww >= 0.30 * w:
                    morph_y.append(yy + hh / 2.0)
            for contour in cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                xx, yy, ww, hh = cv2.boundingRect(contour)
                if hh >= 0.45 * h:
                    morph_x.append(xx + ww / 2.0)

        morph_y = cluster(morph_y, max(3.0, 0.012 * min(w, h)))
        morph_x = cluster(morph_x, max(3.0, 0.012 * min(w, h)))

        # Straight-line evidence.
        edges = cv2.Canny(crop, 40, 120)
        min_h_len = max(20, int(w * 0.45))
        min_v_len = max(15, int(h * 0.45))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180.0,
            threshold=max(12, int(min(h, w) * 0.12)),
            minLineLength=min(min_h_len, min_v_len),
            maxLineGap=10
        )
        hough_y = []
        hough_x = []
        if lines is not None:
            # OpenCV may return HoughLinesP output as (N, 1, 4) or (N, 4),
            # depending on the runtime/build. Normalize both forms before
            # iterating. Never index with [:, 0] blindly because that can
            # yield scalar numpy values and silently break reconstruction.
            lines = np.asarray(lines)
            if lines.ndim == 3 and lines.shape[-1] == 4:
                line_segments = lines.reshape(-1, 4)
            elif lines.ndim == 2 and lines.shape[-1] == 4:
                line_segments = lines
            else:
                line_segments = np.empty((0, 4), dtype=np.int32)

            for line in line_segments:
                xa, ya, xb, yb = [int(v) for v in line]
                dx, dy = abs(xb - xa), abs(yb - ya)
                if dy <= 2 and dx >= min_h_len:
                    hough_y.append((ya + yb) / 2.0)
                elif dx <= 2 and dy >= min_v_len:
                    hough_x.append((xa + xb) / 2.0)

        hough_y = cluster(hough_y, max(3.0, 0.012 * min(w, h)))
        hough_x = cluster(hough_x, max(3.0, 0.012 * min(w, h)))

        # Hough is preferred for repeated table row separators.  With only
        # one interval it is safer to use morphology because text/crop edges
        # can produce two outer lines only.
        ys = hough_y if len(hough_y) >= 3 else morph_y
        xs = hough_x if len(hough_x) >= 2 else morph_x

        # Collapse near-duplicate horizontal rules. Screenshots and UI tables
        # can produce two very close detections for the same physical border
        # (for example an anti-aliased line plus its adjacent dark edge). A
        # duplicate must not become an extra table row. This is based only on
        # relative grid spacing, never on document-specific row counts.
        if len(ys) >= 3:
            ordered = sorted(float(v) for v in ys)
            gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 2.0]
            if gaps:
                median_gap = float(np.median(gaps))
                if median_gap > 0:
                    cleaned = [ordered[0]]
                    for value in ordered[1:]:
                        if value - cleaned[-1] < 0.30 * median_gap:
                            cleaned[-1] = (cleaned[-1] + value) / 2.0
                        else:
                            cleaned.append(value)
                    ys = cleaned

        if len(xs) > 80 or len(ys) > 200:
            return [], []
        if len(xs) < 2 or len(ys) < 2:
            return [], []

        return [x1 + x for x in xs], [y1 + y for y in ys]
    except Exception:
        return [], []


def _prune_empty_spurious_columns(xs, blocks):
    """Remove a narrow text-free sliver caused by a spurious vertical edge."""
    if len(xs) <= 3:
        return list(xs)
    xs = list(sorted(xs))
    if not blocks:
        return xs

    changed = True
    while changed and len(xs) > 2:
        changed = False
        for i in range(1, len(xs) - 1):
            left, mid, right = xs[i - 1], xs[i], xs[i + 1]
            mid_width = right - mid
            left_width = mid - left
            if mid_width <= 0 or left_width <= 0:
                continue

            has_text = False
            for block in blocks:
                bb = normalize_bbox(block.get("bbox"))
                if bb is None:
                    continue
                cx = (bb[0] + bb[2]) / 2.0
                if mid <= cx <= right:
                    has_text = True
                    break

            if has_text:
                continue

            # A narrow empty sliver between a populated left cell and a much
            # wider right cell is characteristic of a duplicated edge. Remove
            # the boundary on its right so the right cell absorbs the sliver.
            if mid_width <= 0.45 * left_width:
                del xs[i]
                changed = True
                break

    return xs


def _extend_open_bottom_grid(image_path, bbox, ys, xs=None):
    """Infer one missing bottom boundary when a visible final row is open."""
    if image_path is None or len(ys) < 4 or not bbox:
        return ys
    try:
        import cv2
        import pytesseract
        import numpy as np
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ys

        ys = sorted(float(v) for v in ys)
        gaps = [float(g) for g in np.diff(ys) if g > 5]
        if len(gaps) < 3:
            return ys
        median_gap = float(np.median(gaps))
        stable = sum(abs(g - median_gap) <= 0.25 * median_gap for g in gaps)
        if stable / len(gaps) < 0.65:
            return ys

        last = ys[-1]
        x1, _, x2, _ = [int(round(v)) for v in bbox]
        x1 = max(0, x1); x2 = min(image.shape[1], x2)
        strip_end = min(image.shape[0], int(round(last + 1.15 * median_gap)))
        if strip_end <= last + 5:
            return ys

        strip = image[int(round(last)):strip_end, x1:x2]
        if strip.size == 0:
            return ys

        # IMPORTANT: text below a table is not sufficient evidence for an
        # additional row.  Screenshots and scanned pages can have status bars,
        # footers, captions, or nearby prose immediately below the last table
        # rule.  Before extending an open-bottom grid, require the table's
        # vertical boundaries to continue into the proposed next-row band.
        # This is source-image geometry, not document-specific text.
        try:
            gray_strip = strip
            binary = cv2.threshold(
                gray_strip,
                235,
                255,
                cv2.THRESH_BINARY_INV
            )[1]

            strip_h, strip_w = binary.shape[:2]
            if strip_h < 8 or strip_w < 8:
                return ys

            # Test continuation at the already recovered vertical table
            # boundaries.  Looking at known boundaries is more reliable than
            # connected morphology here because a status bar/footer can form
            # one large dark horizontal component across the entire strip.
            boundary_positions = []
            if xs and len(xs) >= 2:
                for value in xs:
                    rel = int(round(float(value) - x1))
                    if 2 <= rel < strip_w - 2:
                        boundary_positions.append(rel)

            if not boundary_positions:
                # Generic fallback when column boundaries are unavailable.
                vertical_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (1, max(5, int(strip_h * 0.45)))
                )
                vertical = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_OPEN,
                    vertical_kernel
                )
                support = (vertical > 0).sum(axis=0) / float(strip_h)
                supported_x = [
                    index for index, value in enumerate(support)
                    if value >= 0.45
                ]
                clusters = []
                for index in supported_x:
                    if not clusters or index - clusters[-1][-1] > 3:
                        clusters.append([index])
                    else:
                        clusters[-1].append(index)
                boundary_positions = [
                    int(round(sum(cluster) / len(cluster)))
                    for cluster in clusters
                ]

            # Require the recovered vertical rules to continue through a
            # meaningful fraction of the proposed next-row band. A table that
            # already ends at the last detected rule should not be extended
            # merely because nearby page text creates dark pixels.
            continuation_h = max(3, int(round(0.50 * median_gap)))
            continuation_crop = image[
                int(round(last)):min(
                    image.shape[0],
                    int(round(last + continuation_h))
                ),
                x1:x2
            ]
            if continuation_crop.size == 0:
                return ys

            continued_boundaries = 0
            for rel_x in boundary_positions:
                lo = max(0, rel_x - 2)
                hi = min(continuation_crop.shape[1], rel_x + 3)
                if float(np.mean(continuation_crop[:, lo:hi] < 235)) >= 0.40:
                    continued_boundaries += 1

            if continued_boundaries < 2:
                return ys

            supported_boundaries = 0
            for rel_x in boundary_positions:
                lo = max(0, rel_x - 2)
                hi = min(strip_w, rel_x + 3)
                line_support = float(
                    np.mean(binary[:, lo:hi] > 0)
                )
                if line_support >= 0.45:
                    supported_boundaries += 1

            # A continuation must retain at least two physical boundaries.
            # This rejects nearby footer text while allowing a partially
            # clipped final row to be recovered from its remaining grid.
            if supported_boundaries < 2:
                return ys
        except Exception:
            return ys

        # Do not require a second OCR engine to read the final row.  In
        # screenshots, the final row can be partially clipped or too small
        # for Tesseract even though the source-image grid is still visible.
        # The continuing vertical rules are already strong structural
        # evidence. Confirm that the proposed band also contains non-background
        # content away from the rules, then recover the missing bottom boundary.
        proposed_end = min(
            image.shape[0] - 1,
            int(round(last + median_gap))
        )
        if proposed_end <= last + 4:
            return ys

        band = image[int(round(last)):proposed_end, x1:x2]
        if band.size == 0:
            return ys

        # Remove a narrow margin around the expected vertical rules so the
        # decision is based on cell content rather than the rules themselves.
        content_band = band.copy()
        try:
            content_binary = cv2.threshold(
                content_band,
                245,
                255,
                cv2.THRESH_BINARY_INV
            )[1]
            ink_ratio = float(np.mean(content_binary > 0))
        except Exception:
            ink_ratio = 0.0

        # A tiny amount of source-image content is sufficient because the
        # vertical-rule test above already proves that this is a continuation
        # of the same grid. Pure blank space must not create a row.
        if ink_ratio < 0.003:
            return ys

        new_boundary = last + median_gap
        if new_boundary < image.shape[0] - 2:
            return ys + [new_boundary]
    except Exception:
        pass
    return ys


def _repair_missing_horizontal_grid_lines(ys, blocks, bbox, image_path=None):
    """
    Repair missing horizontal rules in an otherwise regular physical grid.

    Some screenshots/tables have a very light or anti-aliased rule that the
    line detector misses. The remaining rules still reveal the normal row
    spacing. When one gap is approximately two normal row intervals, insert
    the missing boundary at the midpoint. This is geometry-based and does
    not depend on any document, page, or expected row count.
    """
    if not ys or len(ys) < 3 or not bbox:
        return sorted(float(v) for v in ys)

    values = sorted(float(v) for v in ys)
    gaps = [b - a for a, b in zip(values, values[1:]) if b - a > 4.0]
    if len(gaps) < 2:
        return values

    import statistics
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return values

    repaired = list(values)

    # OCR row centers are supporting evidence. We only use them to confirm
    # that both sides of a large physical gap contain text from the same
    # compact table region.
    centers = []
    for block in blocks or []:
        bb = normalize_bbox(block.get("bbox"))
        if bb is None:
            continue
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0
        if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
            centers.append(cy)

    for gap_index, (left, right) in enumerate(zip(values, values[1:])):
        gap = right - left

        # A terminal gap may belong to an open-bottom row or nearby page/UI
        # content; the dedicated open-bottom logic handles that case.
        if gap_index >= len(values) - 2:
            continue

        # A large early gap can be a legitimate multi-line header rather than
        # a missing rule. Use OCR content only as a generic discriminator: if
        # the gap contains multiple visual text rows and all of them are
        # header-like/alphabetic, do not synthesize a separator. If a later
        # row is data-like (for example numeric identifiers), a missing table
        # rule remains plausible.
        if blocks and 1.60 * 1.0 <= gap:
            try:
                visual_rows = group_blocks_into_visual_rows(blocks)
                in_gap = []
                for row in visual_rows:
                    center = row_center_y(row)
                    if left < center < right:
                        text = " ".join(
                            str(block.get("text", ""))
                            for block in row
                        ).strip()
                        in_gap.append(text)

                if len(in_gap) >= 2:
                    digit_ratios = []
                    for text in in_gap:
                        compact = re.sub(r"\s+", "", text)
                        if compact:
                            digit_ratios.append(
                                sum(ch.isdigit() for ch in compact)
                                / len(compact)
                            )
                    if digit_ratios and max(digit_ratios) < 0.20:
                        continue
            except Exception:
                pass

        # A missing single rule normally makes one gap close to two normal
        # row intervals. Do not manufacture boundaries for genuinely large
        # section gaps.
        if gap < 1.60 * median_gap or gap > 2.40 * median_gap:
            continue

        midpoint = (left + right) / 2.0

        left_text = any(left < cy < midpoint for cy in centers)
        right_text = any(midpoint < cy < right for cy in centers)

        if not (left_text and right_text) and image_path is not None:
            try:
                import cv2
                import numpy as np

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is not None:
                    x1 = max(0, int(round(bbox[0])))
                    x2 = min(image.shape[1], int(round(bbox[2])))
                    y_left = max(0, int(round(left)))
                    y_mid = min(image.shape[0], int(round(midpoint)))
                    y_right = min(image.shape[0], int(round(right)))

                    if x2 > x1 and y_mid > y_left and y_right > y_mid:
                        left_crop = image[y_left:y_mid, x1:x2]
                        right_crop = image[y_mid:y_right, x1:x2]

                        left_ink = (
                            float(np.mean(left_crop < 245))
                            if left_crop.size else 0.0
                        )
                        right_ink = (
                            float(np.mean(right_crop < 245))
                            if right_crop.size else 0.0
                        )

                        left_text = left_ink >= 0.003
                        right_text = right_ink >= 0.003

            except Exception:
                pass

        if not (left_text and right_text):
            continue

        if all(
            abs(existing - midpoint)
            > max(3.0, 0.12 * median_gap)
            for existing in repaired
        ):
            repaired.append(midpoint)

    return sorted(repaired)


def _prune_spurious_grid_lines(values):
    """Remove duplicate/phantom physical grid rules using relative geometry."""
    if not values:
        return []

    values = sorted(float(v) for v in values)
    if len(values) < 3:
        return values

    import numpy as np

    # First collapse near-coincident detections.  Anti-aliased borders often
    # appear as two rules only a few pixels apart.  The tolerance scales with
    # the detected span, so it works across page resolutions.
    span = max(1.0, values[-1] - values[0])
    gaps = np.diff(values)
    positive = gaps[gaps > 0.0]
    if len(positive) == 0:
        return [values[0]]

    # Use a robust local scale rather than a fixed pixel threshold. When a
    # grid contains a few very small duplicate-edge gaps alongside much
    # larger genuine cell intervals, estimate the duplicate scale from the
    # larger gaps instead of allowing those artifacts to dominate the median.
    median_gap = float(np.median(positive))
    max_gap = float(np.max(positive))
    large_scale_values = positive[positive >= 0.50 * max_gap]
    large_scale = (
        float(np.median(large_scale_values))
        if len(large_scale_values)
        else median_gap
    )
    duplicate_gap = min(
        0.04 * span,
        max(5.0, 0.15 * large_scale)
    )

    collapsed = [values[0]]
    for value in values[1:]:
        if value - collapsed[-1] <= duplicate_gap:
            collapsed[-1] = (collapsed[-1] + value) / 2.0
        else:
            collapsed.append(value)

    if len(collapsed) < 3:
        return collapsed

    # Detect a second, larger spacing scale.  In screenshots and UI tables,
    # OCR/image artifacts can create a spurious rule inside a genuine row.
    # That produces two short neighbouring gaps whose sum is approximately
    # one normal row interval.  The ordinary median-gap rule cannot see this
    # when the artifact occurs in several rows, because the short gaps become
    # the median.  Identify the larger repeated interval only when there is
    # enough evidence for a second scale; never assume a particular row count.
    collapsed = list(collapsed)
    if len(collapsed) >= 5:
        cgaps = np.diff(collapsed)
        positive_cgaps = cgaps[cgaps > 0.0]
        if len(positive_cgaps) >= 4:
            base_gap = float(np.median(positive_cgaps))
            large_candidates = positive_cgaps[positive_cgaps >= 1.55 * base_gap]
            if len(large_candidates) >= 2:
                dominant_large = float(np.median(large_candidates))
                cleaned = [collapsed[0]]
                i = 1
                while i < len(collapsed) - 1:
                    left_gap = float(collapsed[i] - cleaned[-1])
                    right_gap = float(collapsed[i + 1] - collapsed[i])
                    combined = left_gap + right_gap

                    # Remove only an internal boundary that divides one
                    # dominant row interval into two substantially smaller
                    # pieces.  This is intentionally relative and therefore
                    # adapts to different resolutions and table sizes.
                    if (
                        left_gap > 0
                        and right_gap > 0
                        and left_gap <= 0.72 * dominant_large
                        and right_gap <= 0.72 * dominant_large
                        and 0.72 * dominant_large <= combined <= 1.28 * dominant_large
                    ):
                        i += 1
                        continue

                    cleaned.append(collapsed[i])
                    i += 1

                cleaned.append(collapsed[-1])
                collapsed = cleaned

    # Then remove a remaining isolated phantom when a tiny interval sits next
    # to an otherwise normal interval. This catches cases where the first
    # pass did not collapse the pair because of a skewed global spacing.
    keep = list(collapsed)
    changed = True
    while changed and len(keep) >= 4:
        changed = False
        gaps = np.diff(keep)
        positive = gaps[gaps > 0.0]
        if len(positive) < 2:
            break
        scale = float(np.median(positive))

        for i in range(len(gaps) - 1):
            left_gap = float(gaps[i])
            right_gap = float(gaps[i + 1])

            if left_gap <= 0 or right_gap <= 0:
                continue

            small_gap = min(left_gap, right_gap)
            large_gap = max(left_gap, right_gap)

            if (
                small_gap < 0.12 * scale
                and 0.65 * scale <= large_gap <= 1.45 * scale
                and 0.75 * scale <= small_gap + large_gap <= 1.30 * scale
            ):
                remove_index = i + 1 if left_gap <= right_gap else i + 2
                keep.pop(remove_index)
                changed = True
                break

    return keep


def _derive_small_grid_column_boundaries(
    blocks,
    physical_xs,
    bbox
):
    """
    Recover internal columns for compact/screenshot tables when the source
    image exposes only outer vertical rules.

    OCR blocks are grouped into visual rows first. Horizontal positions are
    clustered across rows, and only positions recurring across multiple rows
    are treated as column evidence.

    This is generic geometry/OCR reasoning. It does not use document names,
    page numbers, known table dimensions, IP addresses, labels, or other
    document-specific values.
    """
    if not bbox or not physical_xs:
        return physical_xs

    import numpy as np

    left = float(min(physical_xs))
    right = float(max(physical_xs))

    if right <= left:
        return physical_xs

    rows = group_blocks_into_visual_rows(blocks or [])
    if not rows:
        return [left, right]

    row_positions = []

    for row in rows:
        row = sorted(row, key=lambda block: float(block["bbox"][0]))
        groups = []

        heights = [
            max(1.0, float(block["bbox"][3]) - float(block["bbox"][1]))
            for block in row
            if normalize_bbox(block.get("bbox")) is not None
        ]
        median_height = float(np.median(heights)) if heights else 20.0
        merge_gap = max(12.0, 0.90 * median_height)

        for block in row:
            bb = normalize_bbox(block.get("bbox"))
            if bb is None:
                continue
            if bb[2] < left or bb[0] > right:
                continue

            if not groups:
                groups.append([block])
                continue

            previous = groups[-1][-1]
            pb = normalize_bbox(previous.get("bbox"))
            if pb is not None and float(bb[0]) - float(pb[2]) <= merge_gap:
                groups[-1].append(block)
            else:
                groups.append([block])

        centers = []
        for group in groups:
            gbs = [normalize_bbox(block.get("bbox")) for block in group]
            gbs = [bb for bb in gbs if bb is not None]
            if not gbs:
                continue
            centers.append(
                (
                    min(bb[0] for bb in gbs)
                    + max(bb[2] for bb in gbs)
                ) / 2.0
            )

        centers = sorted(centers)

        if len(centers) >= 2:
            row_positions.append(centers)

    if not row_positions:
        return [left, right]

    all_centers = sorted(
        center
        for row in row_positions
        for center in row
    )

    if len(all_centers) < 2:
        return [left, right]

    width = max(1.0, right - left)

    cluster_tolerance = max(
        12.0,
        0.080 * width
    )

    clusters = [[all_centers[0]]]

    for center in all_centers[1:]:
        cluster_center = (
            sum(clusters[-1])
            / len(clusters[-1])
        )

        if abs(center - cluster_center) <= cluster_tolerance:
            clusters[-1].append(center)
        else:
            clusters.append([center])

    cluster_centers = [
        sum(cluster) / len(cluster)
        for cluster in clusters
    ]

    minimum_row_support = max(
        1,
        int(math.ceil(0.30 * len(row_positions)))
    )

    stable_centers = []

    for cluster_center in cluster_centers:
        support = 0

        for row in row_positions:
            if any(
                abs(x - cluster_center) <= cluster_tolerance
                for x in row
            ):
                support += 1

        if support >= minimum_row_support:
            stable_centers.append(cluster_center)

    stable_centers = sorted(stable_centers)

    deduplicated = []

    for center in stable_centers:
        if (
            not deduplicated
            or abs(center - deduplicated[-1])
            > max(10.0, 0.50 * cluster_tolerance)
        ):
            deduplicated.append(center)

    stable_centers = deduplicated

    if len(stable_centers) < 2:
        return [left, right]

    # If OCR produces a pathological number of repeated centers, do not turn
    # those fragments into a synthetic table grid.
    if len(stable_centers) > 20:
        return [left, right]

    boundaries = [left]

    for a, b in zip(
        stable_centers,
        stable_centers[1:]
    ):
        boundary = (a + b) / 2.0

        if left < boundary < right:
            boundaries.append(boundary)

    boundaries.append(right)

    output = []

    for value in sorted(boundaries):
        if (
            not output
            or value - output[-1] >= 5.0
        ):
            output.append(round(value, 2))

    return output if len(output) >= 3 else [left, right]




def _build_single_row_two_column_table(blocks, image_region, image_path):
    """Build a genuine bordered 1x2 table when the source grid has one row.

    This is intentionally restricted to exactly two OCR blocks plus source
    image border/separator evidence. It prevents ordinary single-line prose
    or button labels from becoming tables while allowing compact UI tables
    whose body contains no additional rows.
    """
    if not blocks or not image_region or image_path is None:
        return None

    rows = group_blocks_into_visual_rows(blocks)
    if len(rows) != 1 or len(rows[0]) != 2:
        return None

    row = sorted(rows[0], key=lambda b: float(b["bbox"][0]))
    left_block, right_block = row
    lb = normalize_bbox(left_block.get("bbox"))
    rb = normalize_bbox(right_block.get("bbox"))
    if lb is None or rb is None or rb[0] <= lb[2]:
        return None

    import cv2
    import numpy as np

    gap = float(rb[0] - lb[2])
    text_span = max(1.0, float(rb[2] - lb[0]))
    if gap < 0.08 * text_span:
        return None

    heights = [
        max(1.0, float(bb[3] - bb[1]))
        for bb in (lb, rb)
    ]
    pad = max(8.0, 0.65 * float(np.median(heights)))

    bbox = image_region.get("bbox")
    if not bbox:
        return None

    x1 = max(0.0, lb[0] - pad)
    x2 = min(float(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).shape[1]), rb[2] + pad)
    region_bbox = image_region.get("bbox")
    if not region_bbox:
        return None
    y1 = max(0.0, float(region_bbox[1]))
    y2 = min(float(region_bbox[3]), max(y1 + 1.0, float(region_bbox[3])))

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    ix1 = max(0, int(round(x1)))
    ix2 = min(image.shape[1], int(round(x2)))
    iy1 = max(0, int(round(y1)))
    iy2 = min(image.shape[0], int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    crop = image[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None

    # Strong horizontal border evidence near the top/bottom of the OCR row.
    edge_y = np.abs(np.diff(crop.astype(np.int16), axis=0)).mean(axis=1)
    if edge_y.size == 0:
        return None
    top_band = edge_y[:max(3, min(12, edge_y.size))]
    bottom_band = edge_y[-max(3, min(12, edge_y.size)):]
    horizontal_score = max(float(np.max(top_band)), float(np.max(bottom_band)))
    if horizontal_score < 8.0:
        return None

    # A vertical separator should lie in the broad gap between the two OCR
    # cells. Search around their midpoint for a sustained vertical edge.
    separator = (lb[2] + rb[0]) / 2.0
    sx = int(round(separator))
    search = max(3, int(round(0.12 * gap)))
    va = max(ix1 + 1, sx - search)
    vb = min(ix2 - 1, sx + search)
    if vb <= va:
        return None

    v_support = []
    for xx in range(va, vb + 1):
        column = image[iy1:iy2, max(0, xx - 1):min(image.shape[1], xx + 2)]
        v_support.append(float(np.mean(column < 210)))
    if not v_support or max(v_support) < 0.04:
        return None

    # Use the OCR text boxes to define the semantic cells. The image-derived
    # outer box is only provenance geometry; no document-specific dimensions
    # are assumed.
    boundary = (lb[2] + rb[0]) / 2.0
    cells = [
        {
            "row": 1,
            "column": 1,
            "text": combine_cell_text([left_block]),
            "bbox": [x1, y1, boundary, y2],
            "confidence": combine_cell_confidence([left_block]),
            "status": "FALLBACK_OCR",
        },
        {
            "row": 1,
            "column": 2,
            "text": combine_cell_text([right_block]),
            "bbox": [boundary, y1, x2, y2],
            "confidence": combine_cell_confidence([right_block]),
            "status": "FALLBACK_OCR",
        },
    ]

    return {
        "row_count": 1,
        "cell_count": 2,
        "rows": [{"row": 1, "cells": cells}],
        "status": "FALLBACK_OCR",
        "source": "source_image_grid",
        "column_centers": [
            (x1 + boundary) / 2.0,
            (boundary + x2) / 2.0,
        ],
        "bbox": [x1, y1, x2, y2],
    }


def _project_horizontal_grid_lines(image_path, bbox, small_grid=False):
    """Find physical horizontal rules by long-run pixel support."""
    if image_path is None or not bbox:
        return []
    try:
        import cv2, numpy as np
        image=cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
        if image is None:return []
        x1,y1,x2,y2=[int(round(v)) for v in bbox]
        x1=max(0,x1); y1=max(0,y1); x2=min(image.shape[1],x2); y2=min(image.shape[0],y2)
        if x2<=x1 or y2<=y1:return []
        crop=image[y1:y2,x1:x2]; h,w=crop.shape[:2]
        kernel=max(4,int(round(w*0.01)))
        required=0.30 if small_grid else 0.60
        candidates=[]
        for threshold in (180,200,220,235):
            _,binary=cv2.threshold(crop,threshold,255,cv2.THRESH_BINARY_INV)
            horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,1)))
            support=np.mean(horizontal>0,axis=1)
            active=np.where(support>=required)[0]
            groups=[]
            for yy in active:
                if not groups or yy>groups[-1][-1]+1:groups.append([])
                groups[-1].append(int(yy))
            for group in groups:
                candidates.append((float(sum(group)/len(group)+y1),float(max(support[group]))))
        if not candidates:return []
        candidates.sort(); clusters=[]; tol=3.0
        for y,strength in candidates:
            if not clusters or abs(y-clusters[-1][0])>tol:clusters.append([y,strength,1])
            else:
                n=clusters[-1][2]; clusters[-1][0]=(clusters[-1][0]*n+y)/(n+1); clusters[-1][1]=max(clusters[-1][1],strength); clusters[-1][2]=n+1
        values=sorted(c[0] for c in clusters if c[1]>=required)
        if len(values) >= 4:
            changed = True
            while changed and len(values) >= 4:
                changed = False
                gaps = np.diff(values)
                positive = gaps[gaps > 3.0]
                if len(positive) < 3:
                    break
                median_gap = float(np.median(positive))
                for i in range(len(gaps) - 1):
                    g1, g2 = float(gaps[i]), float(gaps[i + 1])
                    if (
                        g1 < 0.60 * median_gap
                        and g2 < 0.75 * median_gap
                        and 0.70 * median_gap <= g1 + g2 <= 1.30 * median_gap
                    ):
                        values.pop(i + 1 if g1 <= g2 else i + 2)
                        changed = True
                        break
        return values if len(values)>=2 else []
    except Exception:return []


def _expand_grid_bbox_by_vertical_edges(image_path, bbox):
    """Expand a partial bordered grid when both side rules continue vertically."""
    if image_path is None or not bbox:
        return list(bbox) if bbox else bbox
    try:
        import cv2, numpy as np
        image=cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
        if image is None:return list(bbox)
        x1,y1,x2,y2=map(float,bbox)
        ix1=max(0,int(round(x1))); ix2=min(image.shape[1],int(round(x2)))
        if ix2<=ix1:return list(bbox)
        search_x=max(4,int(round(0.02*max(20.0,x2-x1))))
        runs=[]
        for edge in (ix1,ix2):
            best=None
            for xx in range(max(0,edge-search_x),min(image.shape[1],edge+search_x+1)):
                dark=np.mean(image[:,xx:xx+1]<235,axis=1)
                active=np.where(dark>=0.75)[0]
                groups=[]
                for yy in active:
                    if not groups or yy>groups[-1][-1]+1:groups.append([])
                    groups[-1].append(int(yy))
                for group in groups:
                    if len(group)<max(20,int(0.25*(y2-y1))):continue
                    overlap=max(0,min(group[-1],int(y2))-max(group[0],int(y1)))
                    if overlap<0.50*(y2-y1):continue
                    score=(overlap,len(group),-abs(xx-edge))
                    if best is None or score>best[0]:best=(score,group[0],group[-1])
            if best:runs.append((best[1],best[2]))
        if len(runs)!=2:return list(bbox)
        top=max(r[0] for r in runs); bottom=min(r[1] for r in runs)
        out=list(bbox)
        if top<y1-2:out[1]=float(top)
        if bottom>y2+2:out[3]=float(bottom)
        return out
    except Exception:return list(bbox)


def _derive_small_grid_row_boundaries(blocks,bbox,physical_ys):
    """Derive compact-table row boundaries from repeated OCR visual rows."""
    if not blocks or not bbox:return physical_ys
    try:
        rows=group_blocks_into_visual_rows(blocks)
        centers=[]
        for row in rows:
            values=[]
            for block in row:
                bb=normalize_bbox(block.get("bbox"))
                if bb is not None:values.append((bb[1]+bb[3])/2.0)
            if values:centers.append(float(sum(values)/len(values)))
        centers=sorted(centers)
        if len(centers)<2:return physical_ys
        _,top,_,bottom=map(float,bbox)
        boundaries=[top]
        for a,b in zip(centers,centers[1:]):
            mid=(a+b)/2.0
            if top<mid<bottom:boundaries.append(mid)
        boundaries.append(bottom)
        return boundaries if len(boundaries)>=3 else physical_ys
    except Exception:return physical_ys

def _build_grid_table_from_image(blocks, image_region, image_path):
    """
    Reconstruct a physically bordered table from the source-image grid.

    The image supplies row/column boundaries; OCR supplies cell text. This
    prevents OCR row grouping from changing a real physical table into an
    arbitrary row/column layout.
    """
    bbox = image_region.get("bbox") if isinstance(image_region, dict) else None

    small_grid = (
        image_region.get("horizontal_lines", 0) < 10
        or image_region.get("vertical_lines", 0) < 5
    ) if isinstance(image_region, dict) else False

    xs, ys = _grid_line_positions(
        image_path,
        bbox,
        small_grid=small_grid
    )

    xs = _prune_spurious_grid_lines(xs)
    ys = _prune_spurious_grid_lines(ys)

    import numpy as np

    projected_ys = _project_horizontal_grid_lines(
        image_path, bbox, small_grid=small_grid
    )
    if len(projected_ys) >= 2:
        # Projection is useful for faint rules, but it can also mistake a
        # filled UI band or scan artifact for a horizontal rule.  Compare the
        # normalized projection with the raw physical result instead of
        # blindly replacing one with the other. Prefer the candidate with the
        # cleaner spacing structure; use projection when it contributes more
        # coherent physical boundaries.
        projected_clean = _prune_spurious_grid_lines(projected_ys)
        raw_clean = _prune_spurious_grid_lines(ys)

        def _spacing_quality(values):
            if len(values) < 3:
                return (-1.0, 0)
            import numpy as np
            gaps = np.diff(sorted(float(v) for v in values))
            gaps = gaps[gaps > 3.0]
            if len(gaps) < 2:
                return (-1.0, len(gaps))
            med = float(np.median(gaps))
            if med <= 0:
                return (-1.0, len(gaps))
            deviation = float(np.median(np.abs(gaps - med)) / med)
            return (1.0 - min(1.0, deviation * 6.0), len(gaps))

        projected_quality = _spacing_quality(projected_clean)
        raw_quality = _spacing_quality(raw_clean)

        if projected_quality[0] > raw_quality[0] + 0.05:
            ys = projected_clean
        elif raw_quality[0] > projected_quality[0] + 0.05:
            ys = raw_clean
        else:
            # When quality is close, retain the candidate with more physical
            # intervals; this avoids losing a faint but genuine rule.
            ys = (
                projected_clean
                if projected_quality[1] >= raw_quality[1]
                else raw_clean
            )

    # If only a few physical horizontal rules survived image detection, OCR
    # row geometry can recover the missing repeated separators. Do not replace
    # a rich physical grid with OCR-derived rows.
    if small_grid and blocks and len(ys) <= 3:
        derived_ys = _derive_small_grid_row_boundaries(
            blocks, bbox, ys
        )
        if len(derived_ys) > len(ys):
            ys = derived_ys

    if len(xs) < 2 or len(ys) < 2:
        return None

    # If a nominally strong grid has only a few vertical rules and one or
    # more of those rules form tiny duplicate-edge intervals, use repeated OCR
    # column geometry to recover the stable cell boundaries. This is only a
    # repair for sparse/irregular physical X geometry; dense, coherent grids
    # remain entirely image-driven.
    if blocks and len(xs) <= 5 and len(xs) >= 3:
        xgaps = [
            float(b - a)
            for a, b in zip(xs, xs[1:])
            if b - a > 0
        ]
        if xgaps:
            typical_xgap = float(np.median(xgaps))
            has_tiny_xgap = any(
                gap < 0.25 * typical_xgap
                for gap in xgaps
            )
            sparse_small_table = (
                len(ys) <= 4
                and len(xs) <= 3
            )
            if has_tiny_xgap or sparse_small_table:
                derived_xs = _derive_small_grid_column_boundaries(
                    blocks,
                    [min(xs), max(xs)],
                    bbox
                )
                if len(derived_xs) >= 3 and len(derived_xs) != len(xs):
                    xs = derived_xs

    # Compact UI/report tables are especially vulnerable to text-stroke
    # fragments being mistaken for vertical rules. When OCR contains repeated
    # visual rows, derive the internal columns from those repeated positions
    # rather than trusting every detected vertical line.
    #
    # IMPORTANT: paragraph OCR can collapse an entire visual row into one
    # block. If the source image exposes only the two outer vertical rules,
    # that block geometry cannot recover the missing columns. In that case run
    # ONE region-level OCR pass before deriving columns. Its word-level boxes
    # provide the missing horizontal positions, while the image still supplies
    # the physical table extent.
    if (
        small_grid
        and isinstance(image_region, dict)
        and blocks
    ):
        physical_xs = list(xs)

        if len(physical_xs) <= 2:
            column_evidence_blocks = list(blocks)

            try:
                region_ocr_blocks = _ocr_region_blocks(
                    image_path,
                    bbox
                )
            except Exception:
                region_ocr_blocks = []

            if region_ocr_blocks:
                column_evidence_blocks.extend(region_ocr_blocks)

            derived_xs = _derive_small_grid_column_boundaries(
                column_evidence_blocks,
                physical_xs,
                bbox
            )

            if len(derived_xs) >= 3:
                xs = derived_xs
            else:
                xs = physical_xs
        else:
            xs = physical_xs

    # If only the two outer vertical rules are visible, use a conservative
    # two-column split only when OCR gives strong two-sided evidence.
    if small_grid and len(xs) == 2 and blocks:
        centers = []

        for block in blocks:
            bb = normalize_bbox(block.get("bbox"))

            if bb is not None:
                centers.append(
                    (bb[0] + bb[2]) / 2.0
                )

        centers = sorted(centers)

        if len(centers) >= 2:
            gaps = [
                b - a
                for a, b in zip(
                    centers,
                    centers[1:]
                )
                if b > a
            ]

            if gaps:
                split_index = max(
                    range(len(gaps)),
                    key=lambda i: gaps[i]
                )

                largest_gap = gaps[split_index]
                region_width = max(
                    1.0,
                    xs[1] - xs[0]
                )

                if largest_gap >= 0.15 * region_width:
                    left_center = centers[split_index]
                    right_center = centers[split_index + 1]
                    boundary = (
                        left_center + right_center
                    ) / 2.0

                    if (
                        xs[0] + 0.15 * region_width
                        < boundary
                        < xs[1] - 0.15 * region_width
                    ):
                        xs = [
                            xs[0],
                            boundary,
                            xs[1]
                        ]

    xs = _prune_empty_spurious_columns(
        xs,
        blocks
    )

    ys = _extend_open_bottom_grid(
        image_path,
        bbox,
        ys,
        xs=xs
    )

    # Repair a genuinely missing repeated horizontal rule after the physical
    # geometry has been normalized.  The repair function only inserts a rule
    # when an interval is approximately doubled and both sides contain
    # content/ink, so it remains document-independent.
    ys = _repair_missing_horizontal_grid_lines(
        ys,
        blocks,
        bbox,
        image_path=image_path
    )

    if len(xs) < 2 or len(ys) < 2:
        return None

    # Avoid pathological detections while remaining document-independent.
    if len(xs) > 80 or len(ys) > 200:
        return None

    # Lazily run one region-level OCR pass only if a cell is actually empty.
    # If the pass was already needed above to recover missing column geometry,
    # reuse its result rather than invoking Tesseract again.
    region_fallback_blocks = (
        region_ocr_blocks
        if (
            small_grid
            and isinstance(image_region, dict)
            and blocks
            and "region_ocr_blocks" in locals()
            and region_ocr_blocks
        )
        else None
    )

    rows = []

    for ri in range(len(ys) - 1):
        row_cells = []

        for ci in range(len(xs) - 1):
            left = xs[ci]
            right = xs[ci + 1]
            top = ys[ri]
            bottom = ys[ri + 1]

            inside = []

            for block in blocks:
                bb = normalize_bbox(block.get("bbox"))

                if bb is None:
                    continue

                cx = (bb[0] + bb[2]) / 2.0
                cy = (bb[1] + bb[3]) / 2.0

                if (
                    left <= cx <= right
                    and top <= cy <= bottom
                ):
                    inside.append(block)

            cell_bbox = (
                combine_cell_bbox(inside)
                or [left, top, right, bottom]
            )

            cell_text = combine_cell_text(inside)

            cell_confidence = (
                combine_cell_confidence(inside)
                if inside
                else 0.0
            )

            cell_status = (
                "FALLBACK_OCR"
                if inside
                else "MISSING_OCR"
            )

            # If paragraph OCR missed a cell, use one region-level source-image
            # OCR pass and assign its words by geometric cell containment. This
            # is equivalent in purpose to cell OCR but avoids hundreds of
            # repeated Tesseract invocations on noisy/over-segmented grids.
            if not cell_text:
                if region_fallback_blocks is None:
                    region_fallback_blocks = _ocr_region_blocks(
                        image_path,
                        bbox
                    )

                fallback_inside = []
                for fallback_block in region_fallback_blocks:
                    fbb = normalize_bbox(fallback_block.get("bbox"))
                    if fbb is None:
                        continue
                    fcx = (fbb[0] + fbb[2]) / 2.0
                    fcy = (fbb[1] + fbb[3]) / 2.0
                    if left <= fcx <= right and top <= fcy <= bottom:
                        fallback_inside.append(fallback_block)

                if fallback_inside:
                    cell_text = combine_cell_text(fallback_inside)
                    cell_confidence = combine_cell_confidence(fallback_inside)
                    cell_status = "IMAGE_OCR_FALLBACK"

            row_cells.append({
                "row": ri + 1,
                "column": ci + 1,
                "text": cell_text,
                "bbox": cell_bbox,
                "confidence": cell_confidence,
                "status": cell_status,
            })

        rows.append({
            "row": ri + 1,
            "cells": row_cells
        })

    return {
        "row_count": len(rows),
        "cell_count": sum(
            len(r["cells"])
            for r in rows
        ),
        "rows": rows,
        "status": "FALLBACK_OCR",
        "source": "source_image_grid",
        "column_centers": [
            (xs[i] + xs[i + 1]) / 2.0
            for i in range(len(xs) - 1)
        ],
        "bbox": [
            xs[0],
            ys[0],
            xs[-1],
            ys[-1]
        ],
    }


def _grid_candidate_quality(image_path, region):
    """
    Score an overlapping source-image grid candidate by internal geometry.

    When multiple detectors see the same physical table, one candidate can
    accidentally include nearby UI/footer geometry.  Prefer the candidate
    whose recovered row spacing is internally consistent and whose vertical
    boundaries are supported.  No document names, page numbers, labels, or
    expected row/column counts are used.
    """
    if not region or not image_path:
        return (-1.0, 0.0, 0.0)

    bbox = region.get("bbox")
    if not bbox:
        return (-1.0, 0.0, 0.0)

    small_grid = (
        region.get("horizontal_lines", 0) < 10
        or region.get("vertical_lines", 0) < 5
    )

    xs, ys = _grid_line_positions(
        image_path,
        bbox,
        small_grid=small_grid
    )

    if len(xs) < 2 or len(ys) < 3:
        return (-1.0, 0.0, 0.0)

    # Include a recoverable faint internal rule before comparing candidates.
    ys_repaired = _repair_missing_horizontal_grid_lines(
        ys,
        [],
        bbox,
        image_path=image_path
    )

    if len(ys_repaired) < 3:
        return (-1.0, 0.0, 0.0)

    import numpy as np

    gaps = np.diff(sorted(float(v) for v in ys_repaired))
    gaps = gaps[gaps > 3.0]
    if len(gaps) < 2:
        return (-1.0, 0.0, 0.0)

    median_gap = float(np.median(gaps))
    if median_gap <= 0:
        return (-1.0, 0.0, 0.0)

    # Median absolute deviation is robust to one anomalous footer/status-bar
    # interval and therefore distinguishes a clean table grid from a grid
    # that accidentally absorbed nearby geometry.
    deviation = float(
        np.median(np.abs(gaps - median_gap))
        / median_gap
    )

    # Penalize a terminal interval that is materially different from the
    # repeated table-row spacing. This is useful for open/adjacent UI areas,
    # but it does not assume any particular number of rows.
    terminal_ratio = float(gaps[-1] / median_gap)
    terminal_penalty = 0.0
    if terminal_ratio > 1.35 or terminal_ratio < 0.65:
        terminal_penalty = min(1.0, abs(terminal_ratio - 1.0))

    # More supported physical vertical rules are useful, but geometry
    # regularity is deliberately weighted more heavily than raw line count.
    vertical_support = min(
        1.0,
        float(region.get("vertical_lines", 0)) / 4.0
    )

    regularity = max(
        0.0,
        1.0 - min(1.0, deviation * 8.0)
    )

    score = (
        regularity * 0.75
        + vertical_support * 0.25
        - terminal_penalty * 0.45
    )

    # Return area separately so exact-quality ties can prefer the more
    # precise nested region without making smallness the primary criterion.
    area = max(
        1.0,
        (float(bbox[2]) - float(bbox[0]))
        * (float(bbox[3]) - float(bbox[1]))
    )

    return (
        score,
        -area,
        float(len(ys_repaired))
    )


def _select_overlapping_grid_regions(image_path, regions):
    """
    Deduplicate overlapping physical-grid detections without blindly taking
    the union of their bounding boxes.

    Different threshold scales can produce one precise table grid and one
    larger connected component that includes nearby UI/footer geometry.  The
    candidates are compared by their own physical grid regularity instead of
    by document-specific assumptions.
    """
    if not regions:
        return []

    def overlap_ratio(a, b):
        return _bbox_overlap_ratio(a, b)

    selected = []
    for region in sorted(
        regions,
        key=lambda r: (
            r.get("bbox", [0, 0, 0, 0])[1],
            r.get("bbox", [0, 0, 0, 0])[0]
        )
    ):
        rb = region.get("bbox")
        if not rb:
            continue

        duplicate_index = None
        best_overlap = 0.0
        for index, current in enumerate(selected):
            cb = current.get("bbox")
            if not cb:
                continue
            overlap = overlap_ratio(rb, cb)
            if overlap >= 0.50 and overlap > best_overlap:
                best_overlap = overlap
                duplicate_index = index

        if duplicate_index is None:
            selected.append(dict(region))
            continue

        current = selected[duplicate_index]
        current_score = _grid_candidate_quality(image_path, current)
        new_score = _grid_candidate_quality(image_path, region)

        # When one candidate is substantially contained by the other, prefer
        # the more precise nested grid unless the nested candidate has clearly
        # failed to recover a physical grid. A larger connected component is
        # often caused by nearby UI/footer geometry. Missing table boundaries
        # are repaired later from the smaller physical region, so selecting the
        # larger union here is unsafe.
        cb = current.get("bbox")
        rb = region.get("bbox")
        current_area = max(1.0, (cb[2] - cb[0]) * (cb[3] - cb[1])) if cb else 1.0
        new_area = max(1.0, (rb[2] - rb[0]) * (rb[3] - rb[1])) if rb else 1.0
        current_inside_new = (
            cb is not None and rb is not None
            and rb[0] <= cb[0] and rb[1] <= cb[1]
            and rb[2] >= cb[2] and rb[3] >= cb[3]
        )
        new_inside_current = (
            cb is not None and rb is not None
            and cb[0] <= rb[0] and cb[1] <= rb[1]
            and cb[2] >= rb[2] and cb[3] >= rb[3]
        )

        # For nested candidates, do not automatically prefer the smaller
        # region. A smaller connected component can end before the real table
        # ends (for example when the last row is visually weaker), while the
        # larger candidate may contain the complete physical grid. Compare
        # the actual grid quality first, then the number of recovered
        # horizontal boundaries. Only when those are effectively tied do we
        # prefer the smaller region to avoid unrelated surrounding content.
        def _prefer_candidate(candidate_score, candidate_inside_other,
                              other_score, candidate_region, other_region):
            if candidate_score[0] < 0:
                return False
            if other_score[0] < 0:
                return True

            primary_delta = candidate_score[0] - other_score[0]
            candidate_lines = candidate_score[2]
            other_lines = other_score[2]

            # A clearly better internal grid wins regardless of area.
            if primary_delta > 0.04:
                return True
            if primary_delta < -0.04:
                return False

            # When regularity is effectively tied, prefer the candidate that
            # contains more consistently spaced physical row boundaries. This
            # recovers complete tables whose final rows are absent from a
            # smaller threshold-connected component, without using an
            # expected/document-specific row count.
            if candidate_lines > other_lines:
                return True
            if candidate_lines < other_lines:
                return False

            # Finally, if the grid evidence is genuinely equivalent, prefer
            # the more precise (smaller) region.
            return candidate_inside_other and not (
                other_region.get("bbox") and candidate_region.get("bbox")
                and (candidate_region["bbox"][2] - candidate_region["bbox"][0])
                    * (candidate_region["bbox"][3] - candidate_region["bbox"][1])
                < (other_region["bbox"][2] - other_region["bbox"][0])
                    * (other_region["bbox"][3] - other_region["bbox"][1])
            )

        if new_inside_current:
            if _prefer_candidate(
                new_score, True, current_score,
                region, current
            ):
                selected[duplicate_index] = dict(region)
        elif current_inside_new:
            if _prefer_candidate(
                current_score, True, new_score,
                current, region
            ) is False:
                selected[duplicate_index] = dict(region)
        elif _prefer_candidate(
            new_score, False, current_score,
            region, current
        ):
            selected[duplicate_index] = dict(region)

    return selected

def _split_grid_region_by_horizontal_gaps(
    image_path,
    image_region
):
    """
    Split an oversized connected grid into independent physical regions.

    A screenshot can contain several vertically separated tables that become
    one connected image component. This function uses only physical
    horizontal-rule spacing to separate them.
    """
    if not image_path or not isinstance(image_region, dict):
        return [image_region]

    bbox = image_region.get("bbox")

    if not bbox:
        return [image_region]

    small_grid = (
        image_region.get("horizontal_lines", 0) < 10
        or image_region.get("vertical_lines", 0) < 5
    )

    try:
        _, ys = _grid_line_positions(
            image_path,
            bbox,
            small_grid=small_grid
        )
    except Exception:
        return [image_region]

    if len(ys) < 4:
        return [image_region]

    import numpy as np

    ys = sorted(float(value) for value in ys)
    gaps = np.diff(ys)

    positive_gaps = [
        float(gap)
        for gap in gaps
        if gap > 3.0
    ]

    if len(positive_gaps) < 3:
        return [image_region]

    median_gap = float(np.median(positive_gaps))

    if median_gap <= 0:
        return [image_region]

    # A gap substantially larger than the repeated row interval is treated
    # as a structural separation between physical tables/sections.
    split_threshold = max(
        2.25 * median_gap,
        median_gap + 20.0
    )

    split_positions = []

    for index, gap in enumerate(gaps):
        if gap >= split_threshold:
            split_positions.append(
                (ys[index], ys[index + 1])
            )

    if not split_positions:
        return [image_region]

    output = []
    start_y = float(bbox[1])

    for upper, lower in split_positions:
        end_y = (upper + lower) / 2.0

        if end_y - start_y >= 10.0:
            child = dict(image_region)
            child["bbox"] = [
                float(bbox[0]),
                float(start_y),
                float(bbox[2]),
                float(end_y)
            ]
            child["split_from_parent"] = True
            output.append(child)

        start_y = end_y

    if float(bbox[3]) - start_y >= 10.0:
        child = dict(image_region)
        child["bbox"] = [
            float(bbox[0]),
            float(start_y),
            float(bbox[2]),
            float(bbox[3])
        ]
        child["split_from_parent"] = True
        output.append(child)

    return output or [image_region]


def _split_grid_region_by_vertical_gaps(image_path, image_region):
    """Split a wide connected grid when independent tables are side-by-side."""
    if not image_path or not isinstance(image_region, dict):
        return [image_region]

    bbox = image_region.get("bbox")
    if not bbox:
        return [image_region]

    try:
        import cv2
        import numpy as np

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return [image_region]

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return [image_region]

        crop = image[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if w < 200 or h < 30:
            return [image_region]

        # A side-by-side table pair normally contains a broad vertical band
        # with almost no ink between the two independent grid structures.
        # Evaluate several thresholds so light screenshot backgrounds do not
        # determine the split by themselves.
        candidate_runs = []
        for threshold in (220, 235, 245):
            dark_ratio = np.mean(crop < threshold, axis=0)
            active = dark_ratio >= 0.01
            runs = []
            start = None
            for index, is_active in enumerate(active):
                if not is_active and start is None:
                    start = index
                elif is_active and start is not None:
                    if index - start >= max(30, int(0.10 * w)):
                        runs.append((start, index - 1))
                    start = None
            if start is not None and w - start >= max(30, int(0.10 * w)):
                runs.append((start, w - 1))

            for a, b in runs:
                gap_width = b - a + 1
                if gap_width < 0.12 * w:
                    continue

                # Require substantial visual content on both sides.
                left_ink = float(np.mean(crop[:, :a] < threshold)) if a > 0 else 0.0
                right_ink = float(np.mean(crop[:, b + 1:] < threshold)) if b + 1 < w else 0.0
                if left_ink < 0.003 or right_ink < 0.003:
                    continue

                candidate_runs.append((gap_width, a, b))

        if not candidate_runs:
            return [image_region]

        _, gap_left, gap_right = max(candidate_runs, key=lambda item: item[0])
        split_x = x1 + (gap_left + gap_right + 1) / 2.0

        # Keep a meaningful margin on both sides; this prevents page margins
        # or a nearly empty visual area from becoming a synthetic table.
        if split_x <= x1 + 0.15 * (x2 - x1) or split_x >= x2 - 0.15 * (x2 - x1):
            return [image_region]

        left_region = dict(image_region)
        right_region = dict(image_region)
        left_region["bbox"] = [float(x1), float(y1), float(split_x), float(y2)]
        right_region["bbox"] = [float(split_x), float(y1), float(x2), float(y2)]
        left_region["split_from_vertical_parent"] = True
        right_region["split_from_vertical_parent"] = True

        return [left_region, right_region]
    except Exception:
        return [image_region]


def detect_fallback_tables(
    page_data,
    image_path=None
):
    """Detect and reconstruct tables using physical + OCR evidence."""
    primary_blocks = get_paragraph_blocks(page_data)
    blocks = _effective_ocr_blocks(primary_blocks, image_path)
    if not blocks and image_path is None:
        return []

    # Visual context is handled at two different levels.  A captioned
    # figure/diagram is excluded from table reconstruction because OCR text
    # inside the figure is not automatically a document table.  A large
    # visual container without figure-caption evidence (for example a
    # screenshot containing real UI tables), however, must remain eligible
    # for physical table detection.
    #
    # This distinction is deliberately generic: no document name, page
    # number, screenshot type, or known table dimensions are used.
    visual_regions = _embedded_visual_regions(image_path, blocks)
    captioned_visual_exclusions = [
        region
        for region in visual_regions
        if region.get("kind") == "figure_caption_region"
    ]

    # For ordinary paragraph-based fallback, keep the stronger exclusion and
    # remove both captioned figures and large visual containers.
    usable_blocks = _blocks_outside_regions(blocks, visual_regions)

    # For source-image table detection, only captioned figures are excluded.
    # A screenshot may contain genuine tables, so its OCR blocks must remain
    # available to the grid/horizontal-rule detectors.
    image_table_blocks = _blocks_outside_regions(
        blocks,
        captioned_visual_exclusions
    )

    # --------------------------------------------------------
    # 1) FULLY BORDERED / GRID-BACKED TABLES
    # --------------------------------------------------------
    image_regions = _image_grid_regions(image_path) + _small_grid_regions(image_path)
    tables = []

    if image_regions:

        image_regions = _select_overlapping_grid_regions(
            image_path,
            image_regions
        )

        # A connected screenshot region can contain several independent
        # physical tables. Split them horizontally and vertically before OCR
        # cell reconstruction. Both operations use only image geometry.
        split_regions = []

        for region in image_regions:
            horizontal_parts = _split_grid_region_by_horizontal_gaps(
                image_path,
                region
            )
            for part in horizontal_parts:
                split_regions.extend(
                    _split_grid_region_by_vertical_gaps(
                        image_path,
                        part
                    )
                )

        image_regions = split_regions

        # Preserve compact nested candidates that the large connected visual
        # region may have swallowed. These are considered separately only if
        # they contain exactly two OCR blocks in one visual row; the builder
        # still requires source-image border/separator evidence.
        compact_nested_candidates = []
        for candidate in (_small_grid_regions(image_path) or []):
            cb = candidate.get("bbox") if isinstance(candidate, dict) else None
            if not cb:
                continue
            cblocks = _blocks_inside_image_region(image_table_blocks, cb)
            crows = group_blocks_into_visual_rows(cblocks)
            if len(crows) == 1 and len(crows[0]) == 2:
                compact_nested_candidates.append(candidate)

        for candidate in compact_nested_candidates:
            if not any(
                _bbox_overlap_ratio(candidate.get("bbox"), existing.get("bbox")) >= 0.98
                for existing in image_regions
            ):
                image_regions.append(candidate)

        for component_index, image_region in enumerate(
            image_regions,
            start=1
        ):
            # A grid inside a captioned figure belongs to the figure, not to
            # the document's table structure. Uncaptioned screenshots remain
            # eligible because they can contain genuine tables.
            if any(
                _bbox_overlap_ratio(
                    image_region.get("bbox"),
                    r.get("bbox")
                ) >= 0.50
                for r in captioned_visual_exclusions
            ):
                continue

            expanded_bbox = _expand_grid_bbox_by_vertical_edges(
                image_path, image_region.get("bbox")
            )
            if expanded_bbox != image_region.get("bbox"):
                image_region = dict(image_region)
                image_region["bbox"] = expanded_bbox

            selected_blocks = _blocks_inside_image_region(
                image_table_blocks,
                image_region["bbox"]
            )

            strong_grid = (
                image_region.get("horizontal_lines", 0) >= 6
                and image_region.get("vertical_lines", 0) >= 4
                and image_region.get("intersections", 0) >= 20
            )

            small_grid = (
                image_region.get("horizontal_lines", 0) >= 2
                and image_region.get("vertical_lines", 0) >= 2
                and image_region.get("intersections", 0) >= 4
            )

            if not (strong_grid or small_grid):
                continue

            # A single compact bordered control/button can look like a table
            # to line detectors. Require actual repeated OCR row structure for
            # the degenerate one-column case; genuine multi-column tables are
            # still allowed to proceed to image-cell OCR. This avoids turning
            # UI controls into document tables without naming any control type.
            if small_grid and not strong_grid:
                visual_rows = group_blocks_into_visual_rows(
                    selected_blocks
                )
                raw_xs, raw_ys = _grid_line_positions(
                    image_path,
                    image_region.get("bbox"),
                    small_grid=True
                )
                normalized_xs = _prune_spurious_grid_lines(raw_xs)
                normalized_ys = _prune_spurious_grid_lines(raw_ys)

                # A compact one-row box with only outer vertical rules is not
                # sufficient table evidence. This generic guard removes
                # toolbar/button-like rectangles while preserving true
                # one-row tables that expose an internal column separator or
                # belong to a stronger multi-row grid component.
                # Do not reject yet: a one-row candidate may be a genuine
                # compact 1x2 table. _build_single_row_two_column_table()
                # below will require exactly two OCR blocks plus physical
                # border/separator evidence. Button-like candidates with more
                # text blocks will still fall through to the normal table
                # evidence checks and be rejected.

            visual_rows = group_blocks_into_visual_rows(selected_blocks)

            # A compact 1x2 table can have only one OCR row. Prefer the
            # specialized one-row reconstruction in that case; otherwise the
            # generic rectangular-grid builder can mistake the surrounding UI
            # container for additional rows.
            if len(visual_rows) == 1:
                table = _build_single_row_two_column_table(
                    selected_blocks,
                    image_region,
                    image_path
                )
            else:
                table = _build_grid_table_from_image(
                    selected_blocks,
                    image_region,
                    image_path
                )

            # A compact 1x2 UI table can have only one OCR row, so the normal
            # multi-row reconstruction may intentionally return None. Recover
            # it only when exactly two OCR blocks and source-image border/
            # separator evidence support a real two-column table.
            if table is None and len(visual_rows) != 1:
                table = _build_single_row_two_column_table(
                    selected_blocks,
                    image_region,
                    image_path
                )

            if table is None:
                candidate_rows = group_blocks_into_visual_rows(
                    selected_blocks
                )

                if len(candidate_rows) < MIN_TABLE_ROWS:
                    continue

                candidate_region = _trim_fallback_region_to_sustained_structure(
                    candidate_rows
                )

                is_table, evidence = _fallback_region_is_table(
                    candidate_region,
                    image_path=image_path
                )

                if not is_table:
                    continue

                table = build_fallback_table(candidate_region)

            else:
                evidence = {
                    "accepted": True,
                    "reason": "SOURCE_IMAGE_GRID",
                    "image": image_region,
                }

            if table is not None:

                region_h = max(
                    1.0,
                    image_region["bbox"][3]
                    - image_region["bbox"][1]
                )

                selected_heights = []

                for block in selected_blocks:
                    bb = normalize_bbox(block.get("bbox"))

                    if bb is not None:
                        selected_heights.append(
                            bb[3] - bb[1]
                        )

                if (
                    table.get("source") != "source_image_grid"
                    and selected_heights
                    and sum(
                        h >= 0.80 * region_h
                        for h in selected_heights
                    ) > max(
                        1,
                        len(selected_heights) // 2
                    )
                ):
                    continue

                if (
                    table.get("row_count", 0) < 1
                    or table.get("cell_count", 0) < 2
                ):
                    continue

                evidence["image_grid_region"] = image_region
                evidence["image_component_index"] = component_index
                evidence["image_component_count"] = len(image_regions)
                evidence["candidate_row_count"] = table.get(
                    "row_count",
                    0
                )
                evidence["reconstructed_row_count"] = table.get(
                    "row_count",
                    0
                )
                evidence["physical_grid_reconstruction"] = (
                    table.get("source") == "source_image_grid"
                )
                evidence["adaptive_small_grid"] = not strong_grid

                populated_per_row = [
                    sum(
                        1
                        for cell in row.get("cells", [])
                        if str(
                            cell.get("text", "")
                        ).strip()
                    )
                    for row in table.get("rows", [])
                ]

                nonempty_cells = sum(populated_per_row)
                row_count = table.get("row_count", 0)

                # A physically verified source-image grid is valid table
                # structure even when OCR failed to populate some/all cells.
                # The image geometry must not be discarded merely because the
                # OCR engine returned sparse text. Paragraph-derived fallback
                # tables still require populated-row evidence.
                if table.get("source") != "source_image_grid":
                    if row_count <= 0 or nonempty_cells < 2:
                        continue

                    if row_count >= 2:
                        populated_rows = sum(
                            1
                            for count in populated_per_row
                            if count >= 2
                        )

                        if populated_rows < 2:
                            continue

                table["detection_evidence"] = evidence
                tables.append(table)

    # --------------------------------------------------------
    # 2) HORIZONTALLY RULED TABLES
    # --------------------------------------------------------
    # This handles tables such as invoices that have horizontal separators
    # but no vertical cell borders. The image rules locate the physical area;
    # OCR geometry decides whether that area is actually a table.
    horizontal_regions = _image_horizontal_rule_regions(image_path, image_table_blocks)
    if horizontal_regions:
        for component_index, item in enumerate(horizontal_regions, start=1):
            # A source-image horizontal-rule region already defines a physical
            # table boundary. Do NOT reduce it to the strongest dense OCR run.
            # In invoices and similar ruled layouts, section/header rows,
            # continuation rows, subtotal rows, and sparse summary rows are
            # legitimate table content even when they contain fewer OCR blocks
            # than the main item rows. The previous trimming step collapsed the
            # test invoice to only its five densest item rows.
            candidate_region = item["rows"]
            is_table, evidence = _fallback_region_is_table(
                candidate_region, image_path=image_path
            )
            if not is_table or not evidence.get("horizontal_rule_table", False):
                continue

            table = build_fallback_table(candidate_region)
            if table is None:
                continue

            evidence["horizontal_rule_region"] = item
            evidence["image_component_index"] = component_index
            evidence["image_component_count"] = len(horizontal_regions)
            evidence["borderless_fallback"] = False
            evidence["horizontal_rule_fallback"] = True
            evidence["candidate_row_count"] = len(item["rows"])
            evidence["reconstructed_row_count"] = len(candidate_region)
            table["detection_evidence"] = evidence

            # Prefer an already reconstructed physical grid when the two
            # detections describe the same region. Otherwise retain both
            # independent table regions.
            table_bbox = _table_bbox(table)
            duplicate = False
            if table_bbox is not None:
                for existing in tables:
                    existing_bbox = _table_bbox(existing)
                    if existing_bbox is not None and _bbox_overlap_ratio(
                        table_bbox, existing_bbox
                    ) >= 0.50:
                        duplicate = True
                        break
            if not duplicate:
                tables.append(table)

    # --------------------------------------------------------
    # 3) COMPLETELY BORDERLESS TABLES
    # --------------------------------------------------------
    visual_rows = group_blocks_into_visual_rows(usable_blocks)
    if len(visual_rows) < MIN_TABLE_ROWS:
        return tables

    regions = find_table_regions(visual_rows)

    for region in regions:
        components = _horizontal_region_components(
            region, image_path=image_path
        )

        for component_index, component in enumerate(components, start=1):
            candidate_region = _trim_fallback_region_to_sustained_structure(
                component
            )
            is_table, evidence = _fallback_region_is_table(
                candidate_region, image_path=image_path
            )
            if not is_table:
                continue

            # If an image exists and no physical rule/grid supports the
            # candidate, require one of the explicit adaptive borderless rules.
            if image_path is not None:
                try:
                    import cv2
                    image_available = (
                        cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) is not None
                    )
                except Exception:
                    image_available = False

                if image_available and not any(
                    evidence.get(name, False)
                    for name in (
                        "borderless_3plus",
                        "small_borderless",
                        "borderless_2col",
                    )
                ):
                    continue

            table = build_fallback_table(candidate_region)
            if table is None:
                continue

            evidence["image_grid_region"] = None
            evidence["image_component_index"] = component_index
            evidence["image_component_count"] = len(components)
            evidence["borderless_fallback"] = True
            evidence["candidate_row_count"] = len(visual_rows)
            evidence["reconstructed_row_count"] = len(candidate_region)

            # Do not reconstruct a second OCR-geometry table over a region
            # already established by physical source-image evidence.  This is
            # especially important when a bordered table also looks like a
            # borderless repeated structure to OCR.
            table_bbox = _table_bbox(table)
            duplicate = False
            if table_bbox is not None:
                for existing in tables:
                    existing_bbox = _table_bbox(existing)
                    if existing_bbox is not None and _bbox_overlap_ratio(
                        table_bbox, existing_bbox
                    ) >= 0.20:
                        duplicate = True
                        break
            if duplicate:
                continue

            table["detection_evidence"] = evidence
            tables.append(table)

    return tables


def process_page(
    page_data,
    image_path=None
):

    # --------------------------------------------------------
    # Paddle tables
    # --------------------------------------------------------

    paddle_tables = []
    page_blocks = get_paragraph_blocks(page_data)
    visual_exclusions = _embedded_visual_regions(
        image_path,
        page_blocks
    )

    source_tables = page_data.get(
        "tables",
        []
    )

    if isinstance(
        source_tables,
        list
    ):

        for table in source_tables:

            reconstructed = (
                reconstruct_paddle_table(
                    table
                )
            )

            if reconstructed is not None:

                table_bbox = _table_bbox(reconstructed)
                if table_bbox and any(
                    _bbox_overlap_ratio(table_bbox, r.get("bbox")) >= 0.50
                    for r in visual_exclusions
                ):
                    # The OCR may detect table-like regions inside an embedded
                    # figure/screenshot. Keep those for the future visual
                    # pipeline rather than misclassifying them as document tables.
                    continue

                paddle_tables.append(
                    reconstructed
                )

    # --------------------------------------------------------
    # Paragraph OCR fallback
    # --------------------------------------------------------

    fallback_tables = (
        detect_fallback_tables(
            page_data,
            image_path=image_path
        )
    )

    # --------------------------------------------------------
    # Remove duplicates.
    # --------------------------------------------------------

    fallback_tables = (
        filter_duplicate_fallbacks(
            fallback_tables,
            paddle_tables
        )
    )

    # --------------------------------------------------------
    # Final tables.
    # --------------------------------------------------------

    reconstructed_tables = (
        paddle_tables
        + fallback_tables
    )

    result = dict(
        page_data
    )

    result[
        "reconstructed_tables"
    ] = reconstructed_tables

    return result


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(
    document_dir
):

    document_name = (
        document_dir.name
    )

    page_files = sorted(
        document_dir.glob(
            "page_*.json"
        )
    )

    output_dir = (
        OUTPUT_DIR
        / document_name
    )

    # Keep generated page outputs synchronized with current inputs.
    # Remove only stale page_*.json files; preserve document.json and
    # every unrelated file in the output directory.
    current_page_names = {
        page_file.name
        for page_file in page_files
    }

    if output_dir.exists():
        for old_page_output in output_dir.glob("page_*.json"):
            if old_page_output.name not in current_page_names:
                try:
                    old_page_output.unlink()
                    print(
                        f"Removed stale reconstructed page: "
                        f"{old_page_output}"
                    )
                except OSError as error:
                    raise RuntimeError(
                        f"Could not remove stale reconstructed page "
                        f"{old_page_output}: {error}"
                    ) from error

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not page_files:
        raise ValueError(
            f"No page_*.json inputs found in {document_dir}"
        )

    pages = []

    document_tables = 0
    document_cells = 0

    print()
    print("=" * 80)
    print(
        f"DOCUMENT: {document_name}"
    )
    print("=" * 80)

    print(
        f"Pages found: {len(page_files)}"
    )

    for page_file in page_files:

        print()
        print(
            f"Processing: {page_file}"
        )

        with open(
            page_file,
            "r",
            encoding="utf-8"
        ) as file:

            page_data = json.load(
                file
            )

        # The page image is derived from the current document/page
        # being processed. No document name or page number is
        # hard-coded; this works for future documents as well.
        page_number = page_file.stem
        image_path = (
            Path("output/pages")
            / document_name
            / f"{page_number}.png"
        )

        reconstructed_page = (
            process_page(
                page_data,
                image_path=image_path
            )
        )

        tables = (
            reconstructed_page.get(
                "reconstructed_tables",
                []
            )
        )

        page_cells = sum(
            table.get(
                "cell_count",
                0
            )
            for table in tables
        )

        print(
            f"Tables: {len(tables)}"
        )

        for table_index, table in enumerate(
            tables,
            start=1
        ):

            print(
                f"  Table {table_index}: "
                f"{table.get('row_count', 0)} rows | "
                f"{table.get('cell_count', 0)} cells | "
                f"Source={table.get('source', 'unknown')}"
            )

        document_tables += len(
            tables
        )

        document_cells += (
            page_cells
        )

        pages.append(
            reconstructed_page
        )

        output_page = (
            output_dir
            / page_file.name
        )

        with open(
            output_page,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                reconstructed_page,
                file,
                indent=2,
                ensure_ascii=False
            )

    document_result = {
        "document": document_name,
        "page_count": len(pages),
        "table_count": document_tables,
        "cell_count": document_cells,
        "pages": pages
    }

    document_output = (
        output_dir
        / "document.json"
    )

    with open(
        document_output,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            document_result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Document tables: {document_tables}"
    )

    print(
        f"Document cells : {document_cells}"
    )

    return document_result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "TABLE RECONSTRUCTION + OCR FALLBACK"
    )
    print("=" * 80)

    if not INPUT_DIR.exists():

        print()
        print(
            "ERROR: Input directory does not exist:"
        )

        print(
            INPUT_DIR
        )

        return

    document_dirs = sorted(
        [
            path
            for path in INPUT_DIR.iterdir()
            if path.is_dir()
        ]
    )

    if not document_dirs:

        print()
        print(
            "No document directories found."
        )

        return

    total_documents = 0
    total_pages = 0
    total_tables = 0
    total_cells = 0

    all_documents = []

    for document_dir in document_dirs:

        result = process_document(
            document_dir
        )

        total_documents += 1

        total_pages += (
            result["page_count"]
        )

        total_tables += (
            result["table_count"]
        )

        total_cells += (
            result["cell_count"]
        )

        all_documents.append(
            result
        )

    # ========================================================
    # SAVE ALL DOCUMENTS
    # ========================================================

    all_documents_path = (
        OUTPUT_DIR
        / "all_documents.json"
    )

    with open(
        all_documents_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TABLE RECONSTRUCTION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Documents processed:  "
        f"{total_documents}"
    )

    print(
        f"Pages processed:      "
        f"{total_pages}"
    )

    print(
        f"Tables reconstructed: "
        f"{total_tables}"
    )

    print(
        f"Cells reconstructed:  "
        f"{total_cells}"
    )

    print()

    print(
        f"Output: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"All documents: "
        f"{all_documents_path}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()