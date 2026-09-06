import json
from pathlib import Path
from html import escape


BASE = Path(__file__).resolve().parent.parent

REVIEW_FILE = (
    BASE
    / "output"
    / "verified_data"
    / "review_required.json"
)

OUTPUT_FILE = (
    BASE
    / "output"
    / "verified_data"
    / "visual_review.html"
)

REVIEW_CROP_DIR = (
    BASE
    / "output"
    / "verified_data"
    / "review_crops"
)


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def html_path(path):
    """
    Convert a filesystem path into a path that can be used
    by visual_review.html.

    Both the HTML report and review_crops are inside:

        output/verified_data/

    Therefore the correct relative form is simply:

        review_crops/<filename>
    """

    try:
        relative = path.relative_to(
            OUTPUT_FILE.parent
        )

        return str(
            relative
        ).replace(
            "\\",
            "/"
        )

    except ValueError:

        return ""


def main():

    print("=" * 70)
    print("VISUAL REVIEW REPORT")
    print("=" * 70)

    if not REVIEW_FILE.exists():

        raise FileNotFoundError(
            f"Review file not found:\n"
            f"{REVIEW_FILE}"
        )

    data = load_json(
        REVIEW_FILE
    )

    items = data.get(
        "items",
        []
    )

    rows = []

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

        # ----------------------------------------------------
        # REVIEW CROP
        # ----------------------------------------------------

        crop_value = str(
            item.get(
                "review_crop",
                ""
            )
        )

        crop_html = (
            "<span class=\"missing\">"
            "No crop available"
            "</span>"
        )

        if crop_value:

            crop_path = Path(
                crop_value
            )

            # Handle absolute and relative paths.
            if not crop_path.is_absolute():

                crop_path = (
                    BASE
                    / crop_path
                )

            if crop_path.exists():

                relative = html_path(
                    crop_path
                )

                if relative:

                    crop_html = (
                        f'<a href="{escape(relative)}" '
                        f'target="_blank">'
                        f'<img '
                        f'src="{escape(relative)}" '
                        f'alt="Original image crop">'
                        f'</a>'
                    )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # HTML ROW
        # ----------------------------------------------------

        rows.append(
            f"""
            <tr>

                <td>{index}</td>

                <td>{escape(page)}</td>

                <td>{escape(table)}</td>

                <td>{escape(row)}</td>

                <td>{escape(column)}</td>

                <td class="ocr">
                    {escape(original_text)}
                </td>

                <td class="ocr">
                    {escape(source_text)}
                </td>

                <td>
                    <span class="decision">
                        {escape(decision)}
                    </span>
                </td>

                <td class="reason">
                    {escape(reason)}
                </td>

                <td class="image">
                    {crop_html}
                </td>

            </tr>
            """
        )

    # --------------------------------------------------------
    # COMPLETE HTML
    # --------------------------------------------------------

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>OCR Visual Review</title>

<style>

body {{
    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 24px;

    background: #f4f4f4;

    color: #222;
}}

h1 {{
    margin-bottom: 6px;
}}

.subtitle {{
    margin-bottom: 20px;
    color: #555;
}}

.summary {{
    padding: 12px;
    margin-bottom: 20px;
    background: white;
    border: 1px solid #ccc;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
}}

th,
td {{
    border: 1px solid #ccc;
    padding: 10px;
    vertical-align: top;
}}

th {{
    background: #e8e8e8;
    position: sticky;
    top: 0;
    z-index: 2;
}}

tbody tr:nth-child(even) {{
    background: #fafafa;
}}

.ocr {{
    font-family:
        Consolas,
        "Courier New",
        monospace;

    font-size: 17px;

    white-space: pre-wrap;
}}

.decision {{
    font-weight: bold;
}}

.reason {{
    max-width: 420px;
}}

.image {{
    min-width: 280px;
    text-align: center;
}}

.image img {{
    max-width: 280px;
    max-height: 140px;
    border: 1px solid #888;
    padding: 2px;
    background: white;
    cursor: zoom-in;
}}

.image img:hover {{
    max-width: 700px;
    max-height: 500px;
}}

.missing {{
    color: #a00;
}}

</style>

</head>

<body>

<h1>OCR Visual Review</h1>

<div class="subtitle">
    Review uncertain OCR against the original source-image crop.
</div>

<div class="summary">

    <strong>
        Review items: {len(items)}
    </strong>

    <br>

    The original OCR is preserved unless a separate,
    explicitly verified correction is made.

</div>

<table>

<thead>

<tr>
    <th>#</th>
    <th>Page</th>
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

{''.join(rows)}

</tbody>

</table>

</body>

</html>
"""

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    valid_crops = 0

    for item in items:

        crop_value = str(
            item.get(
                "review_crop",
                ""
            )
        )

        if not crop_value:
            continue

        crop_path = Path(
            crop_value
        )

        if not crop_path.is_absolute():

            crop_path = (
                BASE
                / crop_path
            )

        if crop_path.exists():

            valid_crops += 1

    print()

    print(
        f"Review items : {len(items)}"
    )

    print(
        f"Valid crops  : {valid_crops}"
    )

    print(
        f"Missing crops: "
        f"{len(items) - valid_crops}"
    )

    print()

    print(
        f"Saved report : "
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        "No OCR data was modified."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()