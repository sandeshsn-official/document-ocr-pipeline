#!/usr/bin/env python3
"""
STAGE 11 — JSON → DOCUMENT RECONSTRUCTION TEST

Purpose:
    Evaluate whether each per-PDF final JSON can reconstruct the
    document's content and spatial layout.

This stage does NOT perform new OCR.

It reads:
    output/verified_data/final_verified/<document>.json

Each PDF is reconstructed independently.

Outputs:
    output/reconstructed_from_json/<document>/
        page_XXXX.png
        page_XXXX_debug.png
        <document>_reconstructed.pdf

    output/reconstructed_from_json/reconstruction_report.json

IMPORTANT:
    This is a CONTENT/LAYOUT reconstruction test, not a pixel-perfect
    scan recreation. Original scanned pixels are NOT copied into the
    reconstructed document.
"""

import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "verified_data"
    / "final_verified"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "reconstructed_from_json"
)


# ---------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def safe_filename(name: str) -> str:
    return (
        re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            str(name)
        ).strip("_")
        or "document"
    )


def get_text(obj: Dict[str, Any]) -> str:
    """
    Prefer the most final text available.
    """
    for key in (
        "final_verified_text",
        "verified_text",
        "text"
    ):
        value = obj.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def get_bbox(obj: Dict[str, Any]):
    bbox = obj.get("bbox")

    if not isinstance(
        bbox,
        (list, tuple)
    ):
        return None

    if len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]
    except Exception:
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


# ---------------------------------------------------------------------
# FONT
# ---------------------------------------------------------------------

def font_candidates() -> List[str]:
    return [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]


def load_font(
    size: int,
    bold: bool = False
):
    size = max(
        8,
        min(int(size), 80)
    )

    candidates = []

    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]

    candidates += font_candidates()

    for path in candidates:
        try:
            return ImageFont.truetype(
                path,
                size=size
            )
        except Exception:
            continue

    return ImageFont.load_default()


# ---------------------------------------------------------------------
# PAGE IMAGE RESOLUTION
# ---------------------------------------------------------------------

def resolve_page_image(
    image_ref: str
) -> Path:
    """
    Resolve the page image reference stored by the pipeline.

    Supports:
      - project-relative paths
      - absolute paths
      - paths beginning with output/...
    """
    if not image_ref:
        return Path("")

    raw = Path(str(image_ref))

    if raw.is_absolute() and raw.exists():
        return raw

    # Try the reference relative to the project root first.
    candidate = PROJECT_ROOT / raw
    if candidate.exists():
        return candidate.resolve()

    # Some job-scoped JSON files store paths relative to their job folder,
    # e.g. output\pages\<document>\page_0001.png. Check job folders
    # without changing or moving any source images.
    jobs_dir = PROJECT_ROOT / "output" / "jobs"
    if jobs_dir.is_dir():
        for job_dir in sorted(
            (item for item in jobs_dir.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            candidate = job_dir / raw
            if candidate.exists():
                return candidate.resolve()

            # Also support a path stored relative to the job's output folder.
            candidate = job_dir / "output" / Path(*raw.parts[1:]) if raw.parts and raw.parts[0].lower() == "output" else job_dir / raw
            if candidate.exists():
                return candidate.resolve()

    # Handle references containing ./ or equivalent.
    candidate = PROJECT_ROOT / str(raw).lstrip("/\\")
    if candidate.exists():
        return candidate.resolve()

    return raw


# ---------------------------------------------------------------------
# PAGE SIZE
# ---------------------------------------------------------------------

def normalize_page_size(
    page: Dict[str, Any],
    image_path: Path
):
    """
    Prefer the source page image dimensions.

    This keeps OCR bbox coordinates in the same coordinate system
    used during extraction.
    """
    if image_path.exists():
        try:
            with Image.open(image_path) as im:
                return im.size
        except Exception:
            pass

    # Support dimensions stored directly in JSON.
    for width_key, height_key in (
        ("width", "height"),
        ("image_width", "image_height"),
        ("page_width", "page_height")
    ):
        try:
            width = float(page.get(width_key))
            height = float(page.get(height_key))

            if width > 0 and height > 0:
                return (
                    int(width),
                    int(height)
                )
        except (
            TypeError,
            ValueError
        ):
            pass

    return (
        2480,
        3508
    )


# ---------------------------------------------------------------------
# TEXT RENDERING
# ---------------------------------------------------------------------

def _wrap_text_to_width(draw, text, font, max_width):
    """Wrap text without deleting any OCR characters."""
    paragraphs = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue

        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            box = draw.textbbox((0, 0), candidate, font=font)
            if (box[2] - box[0]) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

    return lines


def draw_text_fitted(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: Tuple[float, float, float, float],
    bold: bool = False
):
    """
    Draw all extracted text inside its OCR bounding box.

    Important: text is never silently discarded because it is too wide.
    Existing line breaks are preserved and long lines are wrapped inside
    the canonical bbox. The font is reduced until the complete text fits
    the available width/height, subject to a small minimum font size.
    """
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    raw_text = str(text)
    if not raw_text.strip():
        return False

    font_size = max(8, min(int(height * 0.72), 80))
    best_lines = None
    best_font = None
    best_line_height = None

    while font_size >= 6:
        font = load_font(font_size, bold=bold)
        lines = _wrap_text_to_width(draw, raw_text, font, width * 0.98)
        line_box = draw.textbbox((0, 0), "Ag", font=font)
        line_height = max(1, line_box[3] - line_box[1])
        total_height = line_height * len(lines)

        if total_height <= height * 0.98:
            best_lines = lines
            best_font = font
            best_line_height = line_height
            break

        font_size -= 1

    if best_lines is None:
        # Last-resort rendering: keep every line/word rather than silently
        # dropping text. It may extend slightly beyond a pathological bbox,
        # but the reconstruction remains content-complete.
        best_font = load_font(6, bold=bold)
        best_lines = _wrap_text_to_width(draw, raw_text, best_font, width)
        line_box = draw.textbbox((0, 0), "Ag", font=best_font)
        best_line_height = max(1, line_box[3] - line_box[1])

    total_height = best_line_height * len(best_lines)
    ty = y1 + max(0, (height - total_height) / 2)

    for line in best_lines:
        box = draw.textbbox((0, 0), line, font=best_font)
        line_width = box[2] - box[0]
        tx = x1 + max(0, (width - line_width) / 2)
        draw.text((tx, ty - box[1]), line, font=best_font, fill="black")
        ty += best_line_height

    return True


# ---------------------------------------------------------------------
# OBJECT EXTRACTION
# ---------------------------------------------------------------------

def extract_objects(
    page: Dict[str, Any]
):
    """
    Extract paragraph blocks and table cells separately.

    Table cells are never rendered as paragraph blocks when a paragraph
    occupies >=90% of the same cell bbox.
    """
    paragraphs = []
    table_cells = []

    paragraph = page.get(
        "paragraph"
    ) or {}

    for idx, block in enumerate(
        paragraph.get(
            "text_blocks"
        ) or [],
        start=1
    ):

        if not isinstance(
            block,
            dict
        ):
            continue

        paragraphs.append({
            "object_type": "paragraph",
            "object_id": (
                f"paragraph_{idx:04d}"
            ),
            "text": get_text(block),
            "bbox": get_bbox(block),
            "verification_status": (
                block.get(
                    "final_verification_status"
                )
                or block.get(
                    "verification_status"
                )
                or ""
            )
        })

    for table_idx, table in enumerate(
        page.get("tables") or [],
        start=1
    ):

        rows = table.get(
            "rows"
        ) or []

        for row_idx, row in enumerate(
            rows,
            start=1
        ):

            cells = row.get(
                "cells"
            ) or []

            for col_idx, cell in enumerate(
                cells,
                start=1
            ):

                if not isinstance(
                    cell,
                    dict
                ):
                    continue

                table_cells.append({
                    "object_type": "table_cell",
                    "object_id": (
                        f"table_{table_idx:02d}"
                        f"_r{row_idx:02d}"
                        f"_c{col_idx:02d}"
                    ),
                    "table": table_idx,
                    "row": row_idx,
                    "column": col_idx,
                    "text": get_text(cell),
                    "bbox": get_bbox(cell),
                    "verification_status": (
                        cell.get(
                            "final_verification_status"
                        )
                        or cell.get(
                            "verification_status"
                        )
                        or ""
                    )
                })

    return (
        paragraphs,
        table_cells
    )


# ---------------------------------------------------------------------
# BBOX DEDUPLICATION
# ---------------------------------------------------------------------

def bbox_overlap_ratio(a, b):
    if not a or not b:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

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
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    return (
        intersection
        / area_a
    )


def remove_duplicate_paragraphs(
    paragraphs,
    table_cells
):
    filtered = []

    for paragraph in paragraphs:

        if not paragraph["bbox"]:
            filtered.append(
                paragraph
            )
            continue

        duplicate = False

        for cell in table_cells:

            if not cell["bbox"]:
                continue

            if (
                bbox_overlap_ratio(
                    paragraph["bbox"],
                    cell["bbox"]
                )
                >= 0.90
                and re.sub(r"\s+", " ", paragraph["text"].strip()).casefold()
                == re.sub(r"\s+", " ", cell["text"].strip()).casefold()
            ):
                # Only remove a paragraph when it is spatially overlapping
                # AND contains the same text as the table cell. A geometry
                # overlap alone is not enough; otherwise legitimate text
                # can disappear from the reconstruction.
                duplicate = True
                break

        if not duplicate:
            filtered.append(
                paragraph
            )

    return filtered


# ---------------------------------------------------------------------
# BUILD ONE PAGE
# ---------------------------------------------------------------------

def build_page(
    page: Dict[str, Any],
    page_image_path: Path,
    output_png: Path,
    output_debug_png: Path
):
    # Extract and deduplicate objects BEFORE creating the reconstruction
    # canvas so the canvas can be audited against the full OCR coordinate
    # extent. A source image can occasionally have dimensions that do not
    # cover every OCR bbox (for example, if an image was resized or a stale
    # page image is referenced). Never clip a verified OCR object silently.
    paragraphs, table_cells = extract_objects(
        page
    )

    paragraphs = remove_duplicate_paragraphs(
        paragraphs,
        table_cells
    )

    all_objects = (
        paragraphs
        + table_cells
    )

    source_width, source_height = normalize_page_size(
        page,
        page_image_path
    )

    # Determine the maximum valid OCR extent. Empty cells/objects without a
    # bbox do not contribute. The reconstruction canvas must be at least as
    # large as every non-empty object's bbox, otherwise lower/right content
    # can be clipped even though draw_text_fitted() succeeds.
    max_bbox_x2 = 0.0
    max_bbox_y2 = 0.0
    min_bbox_x1 = None
    min_bbox_y1 = None

    for obj in all_objects:
        if not obj.get("text") or not obj.get("bbox"):
            continue
        x1, y1, x2, y2 = obj["bbox"]
        max_bbox_x2 = max(max_bbox_x2, float(x2))
        max_bbox_y2 = max(max_bbox_y2, float(y2))
        min_bbox_x1 = float(x1) if min_bbox_x1 is None else min(min_bbox_x1, float(x1))
        min_bbox_y1 = float(y1) if min_bbox_y1 is None else min(min_bbox_y1, float(y1))

    width = max(float(source_width), max_bbox_x2 + 1.0)
    height = max(float(source_height), max_bbox_y2 + 1.0)

    source_bounds_issues = []
    for obj in all_objects:
        if not obj.get("text") or not obj.get("bbox"):
            continue
        x1, y1, x2, y2 = obj["bbox"]
        issues = []
        if x1 < 0 or y1 < 0:
            issues.append("NEGATIVE_COORDINATE")
        if x2 > source_width:
            issues.append("X2_EXCEEDS_SOURCE_WIDTH")
        if y2 > source_height:
            issues.append("Y2_EXCEEDS_SOURCE_HEIGHT")
        if issues:
            item = {
                "object_type": obj["object_type"],
                "object_id": obj["object_id"],
                "text": obj["text"],
                "bbox": list(obj["bbox"]),
                "issues": issues
            }
            if obj.get("table") is not None:
                item.update({
                    "table": obj.get("table"),
                    "row": obj.get("row"),
                    "column": obj.get("column")
                })
            source_bounds_issues.append(item)

    canvas_expanded = (
        int(width) > int(source_width)
        or int(height) > int(source_height)
    )

    image = Image.new(
        "RGB",
        (
            int(width),
            int(height)
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        image
    )

    usable_bbox = 0
    empty_text = 0
    nonempty_without_bbox = 0
    rendered_nonempty = 0
    unresolved = 0
    missing_bbox_objects = []
    render_failures = []

    paragraph_count = len(
        paragraphs
    )

    table_cells_count = len(
        table_cells
    )

    # --------------------------------------------------------
    # PARAGRAPHS
    # --------------------------------------------------------

    for obj in paragraphs:

        if not obj["text"]:
            empty_text += 1
            continue

        if not obj["bbox"]:
            nonempty_without_bbox += 1
            missing_bbox_objects.append({
                "object_type": obj["object_type"],
                "object_id": obj["object_id"],
                "text": obj["text"],
                "verification_status": obj.get("verification_status", ""),
                "problem": "NONEMPTY_WITHOUT_BBOX"
            })
            continue

        usable_bbox += 1

        if "REVIEW" in str(
            obj["verification_status"]
        ):
            unresolved += 1

        rendered = draw_text_fitted(
            draw,
            obj["text"],
            obj["bbox"],
            bold=False
        )
        if rendered:
            rendered_nonempty += 1
        else:
            render_failures.append({
                "object_type": obj["object_type"],
                "object_id": obj["object_id"],
                "text": obj["text"],
                "bbox": list(obj["bbox"]),
                "problem": "RENDER_FAILED"
            })

    # --------------------------------------------------------
    # TABLE CELLS
    # --------------------------------------------------------

    for obj in table_cells:

        if not obj["text"]:
            empty_text += 1
            continue

        if not obj["bbox"]:
            nonempty_without_bbox += 1
            missing_bbox_objects.append({
                "object_type": obj["object_type"],
                "object_id": obj["object_id"],
                "table": obj.get("table"),
                "row": obj.get("row"),
                "column": obj.get("column"),
                "text": obj["text"],
                "verification_status": obj.get("verification_status", ""),
                "problem": "NONEMPTY_WITHOUT_BBOX"
            })
            continue

        usable_bbox += 1

        if "REVIEW" in str(
            obj["verification_status"]
        ):
            unresolved += 1

        rendered = draw_text_fitted(
            draw,
            obj["text"],
            obj["bbox"],
            bold=False
        )
        if rendered:
            rendered_nonempty += 1
        else:
            render_failures.append({
                "object_type": obj["object_type"],
                "object_id": obj["object_id"],
                "table": obj.get("table"),
                "row": obj.get("row"),
                "column": obj.get("column"),
                "text": obj["text"],
                "bbox": list(obj["bbox"]),
                "problem": "RENDER_FAILED"
            })

    image.save(
        output_png
    )

    # --------------------------------------------------------
    # DEBUG IMAGE
    # --------------------------------------------------------

    debug = image.copy()

    debug_draw = ImageDraw.Draw(
        debug
    )

    for obj in all_objects:

        bbox = obj["bbox"]

        if not bbox:
            continue

        x1, y1, x2, y2 = bbox

        debug_draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="black",
            width=2
        )

    debug.save(
        output_debug_png
    )

    return {
        "page": page.get(
            "page"
        ),
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "max_nonempty_bbox_x2": max_bbox_x2,
        "max_nonempty_bbox_y2": max_bbox_y2,
        "min_nonempty_bbox_x1": min_bbox_x1,
        "min_nonempty_bbox_y1": min_bbox_y1,
        "canvas_expanded_for_bbox_extent": canvas_expanded,
        "objects_exceeding_source_bounds": source_bounds_issues,
        "objects_exceeding_source_bounds_count": len(source_bounds_issues),
        "paragraph_blocks_reconstructed": (
            paragraph_count
        ),
        "table_cells_reconstructed": (
            table_cells_count
        ),
        "total_objects_reconstructed": (
            paragraph_count
            + table_cells_count
        ),
        "objects_with_bbox": (
            usable_bbox
        ),
        "empty_text_objects": (
            empty_text
        ),
        "nonempty_objects_without_bbox": (
            nonempty_without_bbox
        ),
        "rendered_nonempty_objects": (
            rendered_nonempty
        ),
        "objects_still_marked_for_review": (
            unresolved
        ),
        "nonempty_objects_without_bbox": (
            nonempty_without_bbox
        ),
        "missing_bbox_objects": missing_bbox_objects,
        "render_failures": render_failures,
        "content_loss_detected": bool(
            missing_bbox_objects or render_failures
        ),
        "coordinate_extent_audit_passed": bool(
            not missing_bbox_objects
            and not render_failures
            and not any(
                item.get("issues")
                for item in source_bounds_issues
            )
            and max_bbox_x2 <= width
            and max_bbox_y2 <= height
        ),
        "coordinate_extent_audit": {
            "canvas_contains_all_nonempty_bboxes": bool(
                max_bbox_x2 <= width and max_bbox_y2 <= height
            ),
            "source_image_contains_all_nonempty_bboxes": bool(
                not source_bounds_issues
            ),
            "source_bounds_issues": source_bounds_issues
        }
    }


# ---------------------------------------------------------------------
# PDF CREATION
# ---------------------------------------------------------------------

def make_pdf(
    page_pngs: List[Path],
    pdf_path: Path
):
    """
    Put each reconstructed PNG onto a PDF page at native dimensions.
    """
    pdf = None

    for png in page_pngs:

        with Image.open(
            png
        ) as image:

            width, height = image.size

        if pdf is None:

            pdf = canvas.Canvas(
                str(pdf_path),
                pagesize=(
                    width,
                    height
                )
            )

        else:

            pdf.setPageSize(
                (
                    width,
                    height
                )
            )

        pdf.drawImage(
            ImageReader(
                str(png)
            ),
            0,
            0,
            width=width,
            height=height,
            preserveAspectRatio=False,
            mask="auto"
        )

        pdf.showPage()

    if pdf is not None:
        pdf.save()


# ---------------------------------------------------------------------
# DOCUMENT NAME
# ---------------------------------------------------------------------

def document_name_from_json(
    data: Dict[str, Any],
    json_path: Path
):
    name = (
        data.get("document")
        or data.get("document_id")
        or data.get("name")
    )

    if name:
        return str(name)

    return json_path.stem


# ---------------------------------------------------------------------
# RECONSTRUCT ONE PDF
# ---------------------------------------------------------------------

def reconstruct_document(
    input_json: Path
):
    data = load_json(
        input_json
    )

    doc_name = document_name_from_json(
        data,
        input_json
    )

    safe_doc = safe_filename(
        doc_name
    )

    print("=" * 70)
    print(
        f"DOCUMENT: {doc_name}"
    )
    print("=" * 70)

    document_output_dir = (
        OUTPUT_DIR
        / safe_doc
    )

    document_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pages = data.get(
        "pages"
    ) or []

    # Compatibility with a single nested document,
    # without requiring or generating a combined file.
    if not pages:

        documents = data.get(
            "documents"
        ) or []

        if documents:

            first_document = documents[0]

            if isinstance(
                first_document,
                dict
            ):
                pages = (
                    first_document.get(
                        "pages"
                    )
                    or []
                )

    page_pngs = []
    page_reports = []

    for page in pages:

        if not isinstance(
            page,
            dict
        ):
            continue

        page_no = page.get(
            "page",
            len(page_reports) + 1
        )

        try:
            page_no_int = int(
                page_no
            )
        except (
            TypeError,
            ValueError
        ):
            page_no_int = (
                len(page_reports) + 1
            )

        image_ref = str(
            page.get(
                "image",
                ""
            )
            or ""
        )

        page_image_path = resolve_page_image(
            image_ref
        )

        png_path = (
            document_output_dir
            / f"page_{page_no_int:04d}.png"
        )

        debug_png_path = (
            document_output_dir
            / f"page_{page_no_int:04d}_debug.png"
        )

        result = build_page(
            page,
            page_image_path,
            png_path,
            debug_png_path
        )

        result["source_image_exists"] = (
            page_image_path.exists()
        )

        result["source_image"] = str(
            page_image_path
        )

        page_reports.append(
            result
        )

        page_pngs.append(
            png_path
        )

        print(
            f"Page {page_no_int}: "
            f"paragraphs={result['paragraph_blocks_reconstructed']}, "
            f"table_cells={result['table_cells_reconstructed']}, "
            f"bbox={result['objects_with_bbox']}, "
            f"rendered={result['rendered_nonempty_objects']}, "
            f"nonempty_without_bbox={result['nonempty_objects_without_bbox']}, "
            f"render_failures={len(result['render_failures'])}, "
            f"review_flags={result['objects_still_marked_for_review']}, "
            f"canvas={int(result['width'])}x{int(result['height'])}, "
            f"max_bbox=({result['max_nonempty_bbox_x2']:.1f},{result['max_nonempty_bbox_y2']:.1f}), "
            f"source_bounds_issues={result['objects_exceeding_source_bounds_count']}"
        )

        if result["canvas_expanded_for_bbox_extent"]:
            print(
                "  INFO: reconstruction canvas was expanded to contain "
                "the full non-empty OCR bbox extent."
            )

        if result["content_loss_detected"]:
            print("  WARNING: non-empty content could not be reconstructed:")
            for item in result["missing_bbox_objects"]:
                print(f"    - {item}")
            for item in result["render_failures"]:
                print(f"    - {item}")

        pdf_filename = f"{safe_doc}_reconstructed.pdf"

        if len(str(document_output_dir / pdf_filename)) > 240:
            short_hash = hashlib.sha256(
                safe_doc.encode("utf-8")
            ).hexdigest()[:8]

            pdf_filename = (
                f"{safe_doc[:20]}_{short_hash}_reconstructed.pdf"
            )

        pdf_path = document_output_dir / pdf_filename

    if page_pngs:
        make_pdf(
            page_pngs,
            pdf_path
        )

    document_report = {
        "document": doc_name,
        "input_json": str(input_json),
        "page_count": len(pages),
        "pages": page_reports,
        "reconstructed_pdf": (
            str(pdf_path)
            if page_pngs
            else None
        )
    }

    print()

    if page_pngs:
        print(
            f"Reconstructed PDF: "
            f"{pdf_path}"
        )
    else:
        print(
            "No pages found in this "
            "per-PDF JSON."
        )

    print()

    return document_report


# ---------------------------------------------------------------------
# RUN OUTPUT CLEANUP
# ---------------------------------------------------------------------

def clean_reconstruction_output():
    """Preserve prior Stage 11 results.

    Kept for compatibility with older callers. Ordinary reconstruction
    runs must never delete outputs belonging to previously processed
    documents.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def resolve_input_directory():
    """Resolve the input directory without requiring one fixed layout.

    Priority:
      1. Directory passed as the first command-line argument.
      2. OCR_STAGE11_INPUT_DIR environment variable.
      3. Most recent job's verified_data/final_verified or verified_data.
      4. Project-wide output/verified_data/final_verified.
      5. Project-wide output/verified_data.
    """
    import os

    explicit = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1
        else None
    )
    if explicit is None:
        configured = os.environ.get("OCR_STAGE11_INPUT_DIR", "").strip()
        if configured:
            explicit = Path(configured).expanduser()

    if explicit is not None:
        resolved = explicit if explicit.is_absolute() else PROJECT_ROOT / explicit
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Stage 11 input directory does not exist: {resolved}"
            )
        return resolved.resolve()

    jobs_dir = PROJECT_ROOT / "output" / "jobs"
    job_candidates = []
    if jobs_dir.is_dir():
        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            for relative in (
                Path("output") / "verified_data" / "final_verified",
                Path("output") / "verified_data",
                Path("verified_data") / "final_verified",
                Path("verified_data"),
            ):
                candidate = job_dir / relative
                if candidate.is_dir() and any(
                    item.is_file() and item.suffix.lower() == ".json"
                    and not item.name.endswith(".backup")
                    and "backup" not in item.name.lower()
                    and item.name != "reconstruction_report.json"
                    for item in candidate.glob("*.json")
                ):
                    job_candidates.append((job_dir.stat().st_mtime, candidate))

    # Prefer the newest job input when it is available, so a job-scoped
    # run does not accidentally use unrelated project-wide test fixtures.
    for _, candidate in sorted(job_candidates, key=lambda item: item[0], reverse=True):
        return candidate.resolve()

    for candidate in (INPUT_DIR, PROJECT_ROOT / "output" / "verified_data"):
        if candidate.is_dir() and any(
            item.is_file() and item.suffix.lower() == ".json"
            and not item.name.endswith(".backup")
            and "backup" not in item.name.lower()
            and item.name != "reconstruction_report.json"
            for item in candidate.glob("*.json")
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        "No Stage 11 input directory with JSON files was found. "
        "Pass a directory as the first argument or set "
        "OCR_STAGE11_INPUT_DIR."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    input_dir = resolve_input_directory()

    print("=" * 70)
    print(
        "STAGE 11 — JSON → DOCUMENT RECONSTRUCTION TEST"
    )
    print("=" * 70)
    print()

    print(
        "PER-PDF RECONSTRUCTION"
    )

    print(
        "Input directory:"
    )

    print(
        input_dir
    )

    print()

    input_files = sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name != "reconstruction_report.json"
        and "backup" not in path.name.lower()
        and path.name.endswith(".json")
    )

    if not input_files:
        raise FileNotFoundError(
            "No per-PDF JSON files found in:\n"
            f"{input_dir}"
        )

    # Validate all inputs before writing or replacing any reconstruction.
    for input_json in input_files:
        try:
            with input_json.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception as exc:
            raise ValueError(
                f"Invalid JSON input; no reconstruction was started: {input_json}\n{exc}"
            ) from exc

    print(
        f"Per-PDF JSON files found: "
        f"{len(input_files)}"
    )

    for path in input_files:
        print(
            f"  {path.name}"
        )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = OUTPUT_DIR / "reconstruction_report.json"
    if report_path.is_file():
        try:
            report = load_json(report_path)
        except Exception:
            report = {}
    else:
        report = {}

    if not isinstance(report, dict):
        report = {}
    previous_documents = report.get("documents", [])
    if not isinstance(previous_documents, list):
        previous_documents = []

    # Keep prior report entries for documents not included in this run.
    merged_documents = {
        str(item.get("document", "")): item
        for item in previous_documents
        if isinstance(item, dict) and item.get("document")
    }

    current_run_documents = []
    for input_json in input_files:
        document_report = reconstruct_document(input_json)
        document_key = str(document_report.get("document") or input_json.stem)
        document_report["last_processed_input"] = str(input_json.resolve())
        merged_documents[document_key] = document_report
        current_run_documents.append(document_key)

    report.update({
        "stage": 11,
        "purpose": "JSON content/layout reconstruction evaluation",
        "input_directory": str(input_dir),
        "last_input_directory": str(input_dir),
        "last_run_documents": current_run_documents,
        "last_run_document_count": len(current_run_documents),
        "retained_document_count": len(merged_documents),
        "documents": list(merged_documents.values()),
    })

    save_json(report_path, report)

    print("=" * 70)
    print(
        "STAGE 11 COMPLETE"
    )
    print("=" * 70)
    print()

    print(
        f"Documents processed in this run: "
        f"{len(current_run_documents)}"
    )
    print(
        f"Documents retained in cumulative report: "
        f"{len(merged_documents)}"
    )

    print(
        f"Output folder: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )

    print()

    for document in report[
        "documents"
    ]:

        pdf = document.get(
            "reconstructed_pdf"
        )

        if pdf:
            print(
                f"PDF: {pdf}"
            )

    print()

    # Completeness metrics describe this run only, not historical report entries.
    current_run_reports = [
        merged_documents[key]
        for key in current_run_documents
        if key in merged_documents
    ]
    total_missing = sum(
        len(page.get("missing_bbox_objects", []))
        for document in current_run_reports
        for page in document.get("pages", [])
    )
    total_render_failures = sum(
        len(page.get("render_failures", []))
        for document in current_run_reports
        for page in document.get("pages", [])
    )

    print("CONTENT COMPLETENESS CHECK")
    print("=" * 70)
    print(f"Non-empty objects without bbox : {total_missing}")
    print(f"Render failures                : {total_render_failures}")
    print(
        "Content loss detected         : "
        + ("YES" if (total_missing or total_render_failures) else "NO")
    )
    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The reconstructed PDF contains "
        "OCR-extracted text positioned "
        "using JSON bounding boxes."
    )

    print(
        "It is NOT expected to be "
        "pixel-identical to the original scan."
    )

    print()

    print(
        "Use the reconstruction + report "
        "to evaluate:"
    )

    print(
        "  - content completeness"
    )

    print(
        "  - table structure"
    )

    print(
        "  - spatial placement"
    )

    print(
        "  - missing/empty objects"
    )

    print(
        "  - unresolved verification items"
    )


if __name__ == "__main__":
    main()
