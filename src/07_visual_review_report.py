import json
import os
from pathlib import Path
from html import escape


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent

REVIEW_DIR = (
    BASE
    / "output"
    / "verified_data"
    / "review_required"
)

OUTPUT_DIR = (
    BASE
    / "output"
    / "verified_data"
    / "visual_review"
)


# ============================================================
# JSON
# ============================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ============================================================
# CROP PATH
# ============================================================

def resolve_crop_path(crop_value):
    """
    Resolve a Stage-06 review-crop path robustly.

    Stage 06 may store:
      1. an absolute path from the current runtime,
      2. a path relative to the project directory, or
      3. an absolute path from an older/different Colab runtime.

    The third case is important: an old absolute path can be valid
    logically but no longer point to the crop in the current runtime.
    In that situation, recover the crop by its filename from the
    current review_crops tree instead of reporting a false missing crop.
    """

    if not crop_value:
        return None

    raw = str(crop_value).strip()
    if not raw:
        return None

    crop_path = Path(raw)

    # ------------------------------------------------------------
    # 1. Direct path
    # ------------------------------------------------------------
    if crop_path.exists() and crop_path.is_file():
        return crop_path

    # ------------------------------------------------------------
    # 2. Project-relative path
    # ------------------------------------------------------------
    if not crop_path.is_absolute():
        candidate = BASE / crop_path
        if candidate.exists() and candidate.is_file():
            return candidate

    # ------------------------------------------------------------
    # 3. Recover stale absolute paths.
    #
    # Search only inside Stage-06's review_crops directory.
    # This avoids accidentally picking an unrelated file elsewhere.
    # ------------------------------------------------------------
    filename = crop_path.name
    if filename:
        crop_root = (
            BASE
            / "output"
            / "verified_data"
            / "review_crops"
        )

        if crop_root.exists():
            matches = [
                p for p in crop_root.rglob(filename)
                if p.is_file()
            ]

            if len(matches) == 1:
                return matches[0]

            # If multiple files have the same basename, prefer a
            # path whose trailing directory structure resembles the
            # stored path.
            if len(matches) > 1:
                raw_parts = set(crop_path.parts)
                scored = sorted(
                    matches,
                    key=lambda p: sum(
                        1 for part in p.parts
                        if part in raw_parts
                    ),
                    reverse=True
                )
                if scored:
                    return scored[0]

    return None


# ============================================================
# HTML PATH
# ============================================================

def make_html_crop_path(
    crop_path,
    html_file
):
    """
    Create a relative path from the individual HTML report
    to the review crop.

    Example:

        HTML:
        output/verified_data/visual_review/test2.html

        Crop:
        output/verified_data/review_crops/test2/crop.png

        Result:
        ../review_crops/test2/crop.png
    """

    try:

        relative = (
            crop_path
            .relative_to(
                BASE
            )
        )

    except ValueError:

        return ""

    try:

        html_relative = (
            html_file
            .relative_to(
                BASE
            )
        )

    except ValueError:

        return ""

    html_directory = (
        html_relative.parent
    )

    try:

        # Compute the path relative to the HTML file's directory,
        # not relative to BASE. Counting all directories from BASE
        # incorrectly points outside the project output tree.
        result = Path(
            os.path.relpath(
                crop_path,
                start=html_file.parent
            )
        )

        return str(
            result
        ).replace(
            "\\",
            "/"
        )

    except Exception:

        return ""


# ============================================================
# BUILD ONE REPORT
# ============================================================

def build_report(
    review_file
):

    data = load_json(
        review_file
    )

    items = data.get(
        "items",
        []
    )

    document_name = str(
        data.get(
            "document",
            review_file.stem
        )
    )

    output_file = (
        OUTPUT_DIR
        / f"{review_file.stem}.html"
    )

    rows = []

    valid_crops = 0
    missing_crops = 0

    # ========================================================
    # BUILD TABLE ROWS
    # ========================================================

    for index, item in enumerate(
        items,
        start=1
    ):

        verification = item.get(
            "verification",
            {}
        )

        original_text = str(
            item.get(
                "original_text",
                ""
            )
        )

        source_text = str(
            item.get(
                "source_text",
                ""
            )
        )

        decision = str(
            verification.get(
                "decision",
                ""
            )
        )

        reason = str(
            verification.get(
                "reason",
                ""
            )
        )

        page = str(
            item.get(
                "page",
                ""
            )
        )

        table = str(
            item.get(
                "table",
                ""
            )
        )

        row = str(
            item.get(
                "row",
                ""
            )
        )

        column = str(
            item.get(
                "column",
                ""
            )
        )

        object_type = str(
            item.get(
                "object_type",
                ""
            )
        )

        # ====================================================
        # CROP
        # ====================================================

        crop_value = str(
            item.get(
                "review_crop",
                ""
            )
        )

        crop_html = (
            '<span class="missing">'
            'No crop available'
            '</span>'
        )

        if crop_value:

            crop_path = resolve_crop_path(
                crop_value
            )

            if (
                crop_path is not None
                and crop_path.exists()
            ):

                relative_crop = (
                    make_html_crop_path(
                        crop_path,
                        output_file
                    )
                )

                if relative_crop:

                    valid_crops += 1

                    safe_crop = escape(
                        relative_crop
                    )

                    crop_html = (
                        f'<a href="{safe_crop}" '
                        f'target="_blank">'
                        f'<img '
                        f'src="{safe_crop}" '
                        f'alt="Original source image crop">'
                        f'</a>'
                    )

                else:

                    missing_crops += 1

            else:

                missing_crops += 1

        else:

            missing_crops += 1

        # ====================================================
        # ROW
        # ====================================================

        rows.append(
            """
            <tr>

                <td>INDEX</td>

                <td>PAGE</td>

                <td>OBJECT_TYPE</td>

                <td>TABLE</td>

                <td>ROW</td>

                <td>COLUMN</td>

                <td class="ocr">
                    ORIGINAL_TEXT
                </td>

                <td class="ocr">
                    SOURCE_TEXT
                </td>

                <td>
                    <span class="decision">
                        DECISION
                    </span>
                </td>

                <td class="reason">
                    REASON
                </td>

                <td class="image">
                    CROP
                </td>

            </tr>
            """
            .replace(
                "INDEX",
                escape(str(index))
            )
            .replace(
                "PAGE",
                escape(page)
            )
            .replace(
                "OBJECT_TYPE",
                escape(object_type)
            )
            .replace(
                "TABLE",
                escape(table)
            )
            .replace(
                "ROW",
                escape(row)
            )
            .replace(
                "COLUMN",
                escape(column)
            )
            .replace(
                "ORIGINAL_TEXT",
                escape(original_text)
            )
            .replace(
                "SOURCE_TEXT",
                escape(source_text)
            )
            .replace(
                "DECISION",
                escape(decision)
            )
            .replace(
                "REASON",
                escape(reason)
            )
            .replace(
                "CROP",
                crop_html
            )
        )

    # ========================================================
    # HTML TEMPLATE
    # ========================================================

    html = """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>OCR Visual Review - DOCUMENT_NAME</title>

<style>

body {
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 24px;

    background: #f4f4f4;

    color: #222;
}

h1 {
    margin-bottom: 6px;
}

.subtitle {
    margin-bottom: 20px;

    color: #555;
}

.summary {
    padding: 14px;

    margin-bottom: 20px;

    background: white;

    border: 1px solid #ccc;

    line-height: 1.6;
}

.document {
    font-weight: bold;
}

table {
    width: 100%;

    border-collapse: collapse;

    background: white;
}

th,
td {
    border: 1px solid #ccc;

    padding: 10px;

    vertical-align: top;
}

th {
    background: #e8e8e8;

    position: sticky;

    top: 0;

    z-index: 2;

    white-space: nowrap;
}

tbody tr:nth-child(even) {
    background: #fafafa;
}

.ocr {
    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 17px;

    white-space: pre-wrap;

    min-width: 180px;
}

.decision {
    font-weight: bold;

    white-space: nowrap;
}

.reason {
    max-width: 420px;

    min-width: 220px;
}

.image {
    min-width: 280px;

    text-align: center;
}

.image img {
    max-width: 280px;

    max-height: 140px;

    border: 1px solid #888;

    padding: 2px;

    background: white;

    cursor: zoom-in;

    transition:
        max-width 0.2s ease,
        max-height 0.2s ease;
}

.image img:hover {
    max-width: 700px;

    max-height: 500px;
}

.missing {
    color: #a00;

    font-weight: bold;
}

</style>

</head>

<body>

<h1>
    OCR Visual Review
</h1>

<div class="subtitle">
    Review uncertain OCR against the original
    source-image crop.
</div>

<div class="summary">

    <div class="document">
        Document: DOCUMENT_NAME
    </div>

    <div>
        Review items:
        <strong>ITEM_COUNT</strong>
    </div>

    <div>
        Valid crops:
        <strong>VALID_CROPS</strong>
    </div>

    <div>
        Missing crops:
        <strong>MISSING_CROPS</strong>
    </div>

    <br>

    <div>
        The original OCR is preserved unless a separate,
        explicitly verified correction is made.
    </div>

</div>

<table>

<thead>

<tr>
    <th>#</th>
    <th>Page</th>
    <th>Object Type</th>
    <th>Table</th>
    <th>Row</th>
    <th>Column</th>
    <th>Original OCR</th>
    <th>Source OCR</th>
    <th>Decision</th>
    <th>Reason</th>
    <th>Original Image Crop</th>
</tr>

</thead>

<tbody>

ROWS

</tbody>

</table>

</body>

</html>
"""

    # ========================================================
    # INSERT VALUES
    # ========================================================

    html = (
        html
        .replace(
            "DOCUMENT_NAME",
            escape(document_name)
        )
        .replace(
            "ITEM_COUNT",
            str(len(items))
        )
        .replace(
            "VALID_CROPS",
            str(valid_crops)
        )
        .replace(
            "MISSING_CROPS",
            str(missing_crops)
        )
        .replace(
            "ROWS",
            "".join(rows)
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    return (
        output_file,
        len(items),
        valid_crops,
        missing_crops
    )


# ============================================================
# MAIN
# ============================================================

def get_active_document_names():
    """Return document stems for PDFs currently present in input/."""
    input_dir = BASE / "input"
    if not input_dir.exists():
        return set()
    return {
        path.stem
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    }


def cleanup_stale_visual_reports(active_document_names):
    """Remove only stale Stage-07 HTML reports.

    Review JSON, crops, final verified data, and learning memory are never
    deleted here. HTML reports are reproducible presentation artifacts, so
    reports for documents that are no longer active may safely be removed.
    """
    if not OUTPUT_DIR.exists():
        return 0

    removed = 0
    active = {str(name).strip() for name in active_document_names if str(name).strip()}

    for html_file in OUTPUT_DIR.glob("*.html"):
        if html_file.stem in active:
            continue
        try:
            html_file.unlink()
            removed += 1
            print(f"Removing stale visual report: {html_file}")
        except OSError as error:
            print(f"WARNING: Could not remove stale visual report {html_file}: {error}")

    return removed


def main():

    print("=" * 70)
    print("VISUAL REVIEW REPORT")
    print("PER-PDF VISUAL REPORT GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # STALE REPRODUCIBLE REPORT CLEANUP
    # --------------------------------------------------------

    active_document_names = get_active_document_names()
    cleanup_stale_visual_reports(active_document_names)

    # --------------------------------------------------------
    # REVIEW DIRECTORY
    # --------------------------------------------------------

    if not REVIEW_DIR.exists():

        print()
        print(
            "Review directory not found:"
        )

        print(
            REVIEW_DIR
        )

        print()
        print(
            "Nothing to generate."
        )

        print("=" * 70)

        return

    # --------------------------------------------------------
    # FIND REVIEW FILES
    # --------------------------------------------------------

    # --------------------------------------------------------
    # CURRENT ACTIVE DOCUMENTS ONLY
    # --------------------------------------------------------
    # Stage 06 keeps prior review JSON files as audit/history artifacts.
    # Stage 07 must not mix those stale files into the current run.
    # Select only review files belonging to PDFs currently present in input/.
    active_document_names = get_active_document_names()

    review_files = []

    for review_file in sorted(REVIEW_DIR.glob("*.json")):

        if review_file.stem in active_document_names:
            review_files.append(review_file)
            continue

        # Some Stage 06 versions store the document name inside the JSON.
        # Use that metadata as a safe fallback without hard-coding filenames.
        try:
            review_data = load_json(review_file)
            review_document = (
                str(review_data.get("document", "")).strip()
                if isinstance(review_data, dict) else ""
            )
        except Exception:
            review_document = ""

        if review_document in active_document_names:
            review_files.append(review_file)

    if not review_files:

        print()
        print(
            "No review JSON files found."
        )

        print(
            f"Directory: {REVIEW_DIR}"
        )

        print()
        print(
            "No visual reports generated."
        )

        print("=" * 70)

        return

    print()

    print(
        f"Review files found: "
        f"{len(review_files)}"
    )

    # ========================================================
    # PROCESS EACH PDF
    # ========================================================

    total_items = 0
    total_valid_crops = 0
    total_missing_crops = 0

    for review_file in review_files:

        print()
        print("-" * 70)

        print(
            f"PROCESSING: "
            f"{review_file.stem}"
        )

        (
            output_file,
            item_count,
            valid_crops,
            missing_crops
        ) = build_report(
            review_file
        )

        total_items += item_count

        total_valid_crops += valid_crops

        total_missing_crops += missing_crops

        print(
            f"Review items : "
            f"{item_count}"
        )

        print(
            f"Valid crops  : "
            f"{valid_crops}"
        )

        print(
            f"Missing crops: "
            f"{missing_crops}"
        )

        print(
            f"Saved report : "
            f"{output_file}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)

    print(
        "VISUAL REVIEW GENERATION COMPLETE"
    )

    print(
        f"Documents processed: "
        f"{len(review_files)}"
    )

    print(
        f"Total review items : "
        f"{total_items}"
    )

    print(
        f"Valid crops        : "
        f"{total_valid_crops}"
    )

    print(
        f"Missing crops      : "
        f"{total_missing_crops}"
    )

    print()

    print(
        f"Output directory   : "
        f"{OUTPUT_DIR}"
    )

    print()

    print(
        "No OCR data was modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()