import os
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import pytesseract
from paddleocr import PaddleOCR


# ============================================================
# CPU / PADDLE SETTINGS
# ============================================================

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"


# ============================================================
# DIRECTORIES
# ============================================================

INPUT_DIR = Path("input")
INPUT_PAGES = Path("output/pages")
OUTPUT_DIR = Path("output/final")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove common Tesseract border artifacts
    text = text.replace("|", " ")
    text = text.replace("¦", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


def similarity(text1, text2):

    a = normalize_text(text1)
    b = normalize_text(text2)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# CELL CROPPING
# ============================================================

def crop_from_box(
    image,
    box,
    padding=8
):

    # Format:
    # [x1, y1, x2, y2]

    if (
        len(box) == 4
        and not isinstance(
            box[0],
            (list, tuple)
        )
    ):

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

    else:

        xs = [
            int(point[0])
            for point in box
        ]

        ys = [
            int(point[1])
            for point in box
        ]

        x1 = min(xs)
        y1 = min(ys)

        x2 = max(xs)
        y2 = max(ys)

    x1 = max(
        0,
        x1 - padding
    )

    y1 = max(
        0,
        y1 - padding
    )

    x2 = min(
        image.shape[1],
        x2 + padding
    )

    y2 = min(
        image.shape[0],
        y2 + padding
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# TESSERACT
# ============================================================

def run_tesseract(crop):

    if crop is None or crop.size == 0:
        return ""

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    processed = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    return clean_text(text)


# ============================================================
# PADDLE OCR RESULT EXTRACTION
# ============================================================

def extract_paddle_result(
    results
):

    texts = []
    scores = []

    for result in results:

        try:

            result_data = result.json

        except Exception:

            continue

        if isinstance(
            result_data,
            str
        ):

            try:

                result_data = json.loads(
                    result_data
                )

            except Exception:

                continue

        if not isinstance(
            result_data,
            dict
        ):

            continue

        res = result_data.get(
            "res",
            result_data
        )

        if not isinstance(
            res,
            dict
        ):

            continue

        rec_texts = res.get(
            "rec_texts",
            []
        )

        rec_scores = res.get(
            "rec_scores",
            []
        )

        for text in rec_texts:

            text = clean_text(text)

            if text:

                texts.append(text)

        for score in rec_scores:

            try:

                scores.append(
                    float(score)
                )

            except Exception:

                pass

    final_text = clean_text(
        " ".join(texts)
    )

    if scores:

        confidence = (
            sum(scores)
            /
            len(scores)
        )

    else:

        confidence = 0.0

    return (
        final_text,
        confidence
    )


# ============================================================
# OCR ONE IMAGE
# ============================================================

# ============================================================
# OCR ONE IMAGE
# ============================================================

def run_page_ocr(
    ocr,
    image_path
):

    print(
        "Running paragraph OCR..."
    )

    results = ocr.predict(
        str(image_path)
    )

    text_parts = []
    text_blocks = []
    scores = []

    for result in results:

        try:

            result_data = result.json

        except Exception:

            continue

        if isinstance(
            result_data,
            str
        ):

            try:

                result_data = json.loads(
                    result_data
                )

            except Exception:

                continue

        if not isinstance(
            result_data,
            dict
        ):

            continue

        res = result_data.get(
            "res",
            result_data
        )

        if not isinstance(
            res,
            dict
        ):

            continue

        rec_texts = res.get(
            "rec_texts",
            []
        )

        rec_scores = res.get(
            "rec_scores",
            []
        )

        rec_boxes = (
            res.get("rec_boxes")
            or res.get("rec_polys")
            or res.get("dt_polys")
            or []
        )

        for index, text in enumerate(
            rec_texts
        ):

            text = clean_text(
                text
            )

            if not text:
                continue

            # --------------------------------------------
            # CONFIDENCE
            # --------------------------------------------

            confidence = 0.0

            if index < len(rec_scores):

                try:

                    confidence = float(
                        rec_scores[index]
                    )

                except Exception:

                    confidence = 0.0

            scores.append(
                confidence
            )

            # --------------------------------------------
            # BOUNDING BOX
            # --------------------------------------------

            bbox = None

            if index < len(rec_boxes):

                try:

                    box = rec_boxes[index]

                    # Polygon:
                    # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

                    if (
                        isinstance(box, list)
                        and len(box) > 0
                        and isinstance(box[0], list)
                    ):

                        xs = [
                            float(point[0])
                            for point in box
                        ]

                        ys = [
                            float(point[1])
                            for point in box
                        ]

                        bbox = [
                            min(xs),
                            min(ys),
                            max(xs),
                            max(ys)
                        ]

                    # Already [x1,y1,x2,y2]

                    elif (
                        isinstance(box, list)
                        and len(box) >= 4
                    ):

                        bbox = [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3])
                        ]

                except Exception:

                    bbox = None

            # --------------------------------------------
            # STORE PARAGRAPH BLOCK
            # --------------------------------------------

            text_blocks.append({

                "text":
                    text,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "bbox":
                    bbox

            })

            text_parts.append(
                text
            )

    # --------------------------------------------
    # AVERAGE CONFIDENCE
    # --------------------------------------------

    average_confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    return {

        "text":
            "\n".join(text_parts),

        "text_blocks":
            text_blocks,

        "average_confidence":
            average_confidence
    }

# ============================================================
# FIND EXISTING TABLE JSON
# ============================================================

def find_table_json(
    document_name,
    page_name
):

    # ========================================================
    # TABLE JSON IS STORED PER DOCUMENT
    # ========================================================

    table_json = (
        Path("output/table_only")
        / document_name
        / (
            Path(page_name).stem
            + "_res.json"
        )
    )

    if table_json.exists():

        return table_json

    return None

    candidates = [

        Path(
            "output/table_only"
        ) / document_name / (
            Path(page_name).stem
            + "_res.json"
        ),

        Path(
            "output/table_only"
        ) / (
            Path(page_name).stem
            + "_res.json"
        ),

        Path(
            "output/table_only"
        ) / document_name / (
            Path(page_name).stem
            + ".json"
        )

    ]

    for path in candidates:

        if path.exists():

            return path

    return None


# ============================================================
# VERIFY TABLE CELLS
# ============================================================

def process_tables(
    ocr,
    image,
    table_json_path,
    document_name,
    page_name
):

    if table_json_path is None:
        return []

    print(
        f"Table JSON: "
        f"{table_json_path}"
    )

    try:

        with open(
            table_json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception as error:

        print(
            f"Could not read table JSON: "
            f"{error}"
        )

        return []

    # ========================================================
    # GET TABLE LIST
    # ========================================================

    res = data.get(
        "res",
        {}
    )

    if not isinstance(res, dict):

        print(
            "Invalid JSON: 'res' is not a dictionary."
        )

        return []

    tables = res.get(
        "table_res_list",
        []
    )

    if not tables:

        print(
            "No tables found."
        )

        return []

    print(
        f"Tables found: {len(tables)}"
    )

    final_tables = []

    for table_index, table in enumerate(
        tables,
        start=1
    ):

        cell_boxes = table.get(
            "cell_box_list",
            []
        )

        print()
        print(
            f"Table {table_index}: "
            f"{len(cell_boxes)} cells"
        )

        cells = []

        for cell_index, box in enumerate(
            cell_boxes,
            start=1
        ):

            crop = crop_from_box(
                image,
                box
            )

            if crop is None or crop.size == 0:

                cells.append({

                    "cell_index":
                        cell_index,

                    "bbox":
                        box,

                    "paddle_text":
                        "",

                    "paddle_confidence":
                        0.0,

                    "tesseract_text":
                        "",

                    "similarity":
                        0.0,

                    "status":
                        "EMPTY_CROP",

                    "final_text":
                        ""
                })

                continue

            # ------------------------------------------------
            # PADDLE ON CELL
            # ------------------------------------------------

            try:

                results = ocr.predict(
                    crop
                )

                paddle_text, paddle_confidence = (
                    extract_paddle_result(
                        results
                    )
                )

            except Exception as error:

                print(
                    f"Cell {cell_index}: "
                    f"Paddle error: {error}"
                )

                paddle_text = ""
                paddle_confidence = 0.0

            # ------------------------------------------------
            # TESSERACT
            # ------------------------------------------------

            try:

                tesseract_text = run_tesseract(
                    crop
                )

            except Exception as error:

                print(
                    f"Cell {cell_index}: "
                    f"Tesseract error: {error}"
                )

                tesseract_text = ""

            # ------------------------------------------------
            # COMPARE
            # ------------------------------------------------

            score = similarity(
                paddle_text,
                tesseract_text
            )

            if (
                not paddle_text
                and
                not tesseract_text
            ):

                status = "EMPTY"

            elif score >= 0.95:

                status = "HIGH_AGREEMENT"

            elif score >= 0.80:

                status = "CLOSE_AGREEMENT"

            elif (
                paddle_confidence < 0.85
                and
                tesseract_text
            ):

                status = "PADDLE_LOW_CONFIDENCE"

            else:

                status = "DISAGREEMENT"

            # ------------------------------------------------
            # CURRENT FINAL TEXT
            # ------------------------------------------------

            # Paddle remains primary.
            # We are not blindly replacing it.

            if paddle_text:

                final_text = paddle_text

            else:

                final_text = tesseract_text

            cells.append({

                "cell_index":
                    cell_index,

                "bbox":
                    box,

                "paddle_text":
                    paddle_text,

                "paddle_confidence":
                    round(
                        paddle_confidence,
                        4
                    ),

                "tesseract_text":
                    tesseract_text,

                "similarity":
                    round(
                        score,
                        4
                    ),

                "status":
                    status,

                "final_text":
                    final_text
            })

            print(
                f"  Cell {cell_index}: "
                f"{status}"
            )

        final_tables.append({

            "table_index":
                table_index,

            "cell_count":
                len(cells),

            "cells":
                cells
        })

    return final_tables


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("DOCUMENT OCR MAIN PROCESSOR")
print("=" * 70)

print()
print(
    f"Input pages : {INPUT_PAGES}"
)

print(
    f"Output      : {OUTPUT_DIR}"
)


# ============================================================
# LOAD PADDLE OCR
# ============================================================

print()
print("=" * 70)
print("LOADING PADDLE OCR")
print("=" * 70)


ocr = PaddleOCR(
    lang="en",
    engine="paddle",
    device="cpu",

    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


print()
print(
    "PaddleOCR loaded successfully."
)


# ============================================================
# FIND ACTIVE DOCUMENTS FROM INPUT
# ============================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(
        f"Input directory not found:\n"
        f"{INPUT_DIR.resolve()}"
    )

active_pdfs = sorted(
    [
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".pdf"
    ]
)

active_document_names = {
    path.stem
    for path in active_pdfs
}

# ------------------------------------------------------------
# REMOVE STALE GENERATED DOCUMENT FOLDERS
# ------------------------------------------------------------
#
# If a PDF was deleted from input\, its old generated data
# must not remain active in the pipeline.
# ------------------------------------------------------------

for root_dir in [
    INPUT_PAGES,
    OUTPUT_DIR,
]:

    if not root_dir.exists():
        continue

    for child in root_dir.iterdir():

        if not child.is_dir():
            continue

        if child.name not in active_document_names:

            print(
                f"Removing stale document output: "
                f"{child}"
            )

            import shutil

            shutil.rmtree(
                child,
                ignore_errors=True
            )

# ------------------------------------------------------------
# USE ONLY CURRENT INPUT DOCUMENTS
# ------------------------------------------------------------

document_dirs = sorted(
    [
        INPUT_PAGES / document_name
        for document_name in active_document_names
        if (
            INPUT_PAGES
            / document_name
        ).is_dir()
    ]
)

# ============================================================
# PROCESS EACH DOCUMENT
# ============================================================

all_documents = []


for document_dir in document_dirs:

    document_name = document_dir.name

    print()
    print("=" * 70)
    print(
        f"DOCUMENT: {document_name}"
    )
    print("=" * 70)


    page_files = sorted(
        document_dir.glob(
            "*.png"
        )
    )


    print(
        f"Pages found: "
        f"{len(page_files)}"
    )


    document_pages = []


    # ========================================================
    # PROCESS EACH PAGE
    # ========================================================

    for page_number, image_path in enumerate(
        page_files,
        start=1
    ):

        page_name = image_path.name

        print()
        print("-" * 70)

        print(
            f"PAGE {page_number} "
            f"OF {len(page_files)}"
        )

        print(
            f"Processing: "
            f"{image_path}"
        )

        print("-" * 70)


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "ERROR: Could not load image."
            )

            continue


        # ----------------------------------------------------
        # PARAGRAPH OCR
        # ----------------------------------------------------

        try:

            paragraph_data = run_page_ocr(
                ocr,
                image_path
            )

        except Exception as error:

            print(
                f"Paragraph OCR failed: "
                f"{error}"
            )

            paragraph_data = {

                "text":
                    "",

                "text_blocks":
                    [],

                "average_confidence":
                    0.0
            }


        # ----------------------------------------------------
        # TABLE JSON
        # ----------------------------------------------------

        table_json = find_table_json(
            document_name,
            page_name
        )


        # ----------------------------------------------------
        # TABLE PROCESSING
        # ----------------------------------------------------

        tables = process_tables(
            ocr,
            image,
            table_json,
            document_name,
            page_name
        )


        # ----------------------------------------------------
        # PAGE RESULT
        # ----------------------------------------------------

        page_result = {

            "document":
                document_name,

            "page":
                page_number,

            "image":
                str(image_path),

            "paragraph":
                paragraph_data,

            "tables":
                tables
        }


        document_pages.append(
            page_result
        )


        # ----------------------------------------------------
        # SAVE PAGE JSON
        # ----------------------------------------------------

        document_output_dir = (
            OUTPUT_DIR /
            document_name
        )

        document_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        page_output = (
            document_output_dir /
            f"page_{page_number:04d}.json"
        )


        with open(
            page_output,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                page_result,
                file,
                indent=2,
                ensure_ascii=False
            )


        print()
        print(
            f"Saved: {page_output}"
        )


    # ========================================================
    # DOCUMENT JSON
    # ========================================================

    document_result = {

        "document":
            document_name,

        "page_count":
            len(document_pages),

        "pages":
            document_pages
    }


    document_output_dir = (
        OUTPUT_DIR /
        document_name
    )


    document_json = (
        document_output_dir /
        "document.json"
    )


    with open(
        document_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            document_result,
            file,
            indent=2,
            ensure_ascii=False
        )


    all_documents.append(
        document_result
    )


    print()
    print(
        f"Document saved: "
        f"{document_json}"
    )


# ============================================================
# COMBINED JSON
# ============================================================

combined_json = (
    OUTPUT_DIR /
    "all_documents.json"
)


combined_result = {

    "document_count":
        len(all_documents),

    "documents":
        all_documents
}


with open(
    combined_json,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        combined_result,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)

print()
print(
    f"Documents processed: "
    f"{len(all_documents)}"
)

print()
print(
    f"Combined output:"
)

print(
    combined_json
)

print()
print("=" * 70)