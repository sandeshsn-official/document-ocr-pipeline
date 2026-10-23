import json
import shutil
from pathlib import Path


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    BASE_DIR
    / "input"
)

RECONSTRUCTED_DIR = (
    BASE_DIR
    / "output"
    / "reconstructed"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "final_data"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# EXTRACT PARAGRAPH DATA
# ============================================================

def extract_paragraph(page_data):

    paragraph = page_data.get(
        "paragraph",
        {}
    )

    if not isinstance(
        paragraph,
        dict
    ):

        return {
            "text": "",
            "text_blocks": [],
            "average_confidence": 0.0
        }

    text = paragraph.get(
        "text",
        ""
    )

    text_blocks = paragraph.get(
        "text_blocks",
        []
    )

    confidence = paragraph.get(
        "average_confidence",
        0.0
    )

    return {

        "text":
            text,

        "text_blocks":
            text_blocks,

        "average_confidence":
            confidence
    }


# ============================================================
# EXTRACT TABLE DATA
# ============================================================

def extract_tables(page_data):

    tables = page_data.get(
        "reconstructed_tables",
        []
    )

    if not isinstance(
        tables,
        list
    ):

        return []

    return tables


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(page_file):

    page_data = load_json(
        page_file
    )

    return {

        "document":
            page_data.get(
                "document",
                ""
            ),

        "page":
            page_data.get(
                "page",
                0
            ),

        "image":
            page_data.get(
                "image",
                ""
            ),

        "paragraph":
            extract_paragraph(
                page_data
            ),

        "tables":
            extract_tables(
                page_data
            )
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FINAL DOCUMENT DATA BUILDER")
print("PER-PDF JSON OUTPUT")
print("=" * 70)


# ============================================================
# CHECK INPUT DIRECTORY
# ============================================================

if not INPUT_DIR.exists():

    raise FileNotFoundError(
        f"Input directory not found:\n"
        f"{INPUT_DIR.resolve()}"
    )


# ============================================================
# FIND ACTIVE PDFs
# ============================================================

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


# Do not clean existing outputs when the input folder is empty.
# An empty folder can mean PDFs were moved temporarily or the wrong
# project directory is being used; treating it as a successful run
# would otherwise delete previously generated data.
if not active_pdfs:

    raise FileNotFoundError(
        "No PDF files found in the input directory. "
        f"Existing reconstructed/final data was left untouched. Input: {INPUT_DIR.resolve()}"
    )


print()

print(
    f"Active PDFs in input: "
    f"{len(active_pdfs)}"
)


for pdf in active_pdfs:

    print(
        f"  {pdf.name}"
    )


# ============================================================
# CLEAN STALE RECONSTRUCTED DATA
# ============================================================

if RECONSTRUCTED_DIR.exists():

    for child in RECONSTRUCTED_DIR.iterdir():

        if not child.is_dir():
            continue

        if child.name not in active_document_names:

            print()

            print(
                f"Removing stale reconstructed data: "
                f"{child}"
            )

            shutil.rmtree(
                child,
                ignore_errors=True
            )


# ============================================================
# CLEAN STALE FINAL DATA
#
# Every active PDF gets its own JSON file.
# Old combined all_documents.json is removed.
# ============================================================

if OUTPUT_DIR.exists():

    # --------------------------------------------------------
    # REMOVE OLD COMBINED JSON
    # --------------------------------------------------------

    combined_file = (
        OUTPUT_DIR
        / "all_documents.json"
    )

    if combined_file.exists():

        print()

        print(
            "Removing old combined JSON:"
        )

        print(
            combined_file
        )

        combined_file.unlink()


    # --------------------------------------------------------
    # REMOVE STALE DOCUMENT OUTPUTS
    # --------------------------------------------------------

    for child in OUTPUT_DIR.iterdir():

        if child.is_file():

            document_name = child.stem

        elif child.is_dir():

            document_name = child.name

        else:

            continue


        if document_name not in active_document_names:

            print()

            print(
                f"Removing stale final data: "
                f"{child}"
            )

            if child.is_dir():

                shutil.rmtree(
                    child,
                    ignore_errors=True
                )

            else:

                try:

                    child.unlink()

                except FileNotFoundError:

                    pass


# ============================================================
# PROCESS EACH PDF INDEPENDENTLY
# ============================================================

documents_processed = 0


for pdf in active_pdfs:

    document_name = pdf.stem

    document_dir = (
        RECONSTRUCTED_DIR
        / document_name
    )


    print()

    print("=" * 70)

    print(
        f"DOCUMENT: "
        f"{document_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # RECONSTRUCTED DATA MUST EXIST
    # --------------------------------------------------------

    if not document_dir.exists():

        print()

        print(
            "WARNING: Reconstructed data not found:"
        )

        print(
            document_dir
        )

        print(
            "Skipping document."
        )

        continue


    # --------------------------------------------------------
    # FIND PAGES
    # --------------------------------------------------------

    page_files = sorted(
        document_dir.glob(
            "page_*.json"
        )
    )


    print()

    print(
        f"Pages found: "
        f"{len(page_files)}"
    )


    if not page_files:

        print()
        print(
            "ERROR: No reconstructed page JSON files found; "
            "existing final output for this document was not overwritten."
        )
        continue


    pages = []
    page_errors = []


    # --------------------------------------------------------
    # PROCESS PAGES
    # --------------------------------------------------------

    for page_file in page_files:

        print()

        print(
            f"Processing: "
            f"{page_file}"
        )


        try:

            page_result = process_page(
                page_file
            )

        except Exception as error:

            print()

            print(
                "ERROR reading page:"
            )

            print(
                error
            )
            page_errors.append((page_file, error))
            continue


        pages.append(
            page_result
        )


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        paragraph = page_result[
            "paragraph"
        ]

        text_blocks = paragraph.get(
            "text_blocks",
            []
        )

        tables = page_result.get(
            "tables",
            []
        )


        print(
            f"Paragraph text: "
            f"{len(paragraph.get('text', ''))} characters"
        )


        print(
            f"Paragraph blocks: "
            f"{len(text_blocks)}"
        )


        print(
            f"Tables: "
            f"{len(tables)}"
        )


        for table_index, table in enumerate(
            tables,
            start=1
        ):

            rows = table.get(
                "rows",
                []
            )

            print(
                f"  Table {table_index}: "
                f"{len(rows)} rows"
            )


    # Never publish a partial document as if every page succeeded.
    # Keep any existing final JSON untouched so it can be investigated.
    if page_errors:

        print()
        print(
            f"ERROR: {len(page_errors)} page(s) failed for {document_name}. "
            "Skipping final JSON write to avoid silently producing incomplete data."
        )
        continue

    # --------------------------------------------------------
    # DOCUMENT OBJECT
    # --------------------------------------------------------

    document_result = {

        "document":
            document_name,

        "document_id":
            document_name,

        "source_pdf":
            pdf.name,

        "page_count":
            len(pages),

        "pages":
            pages
    }


    # --------------------------------------------------------
    # SAVE ONLY THIS DOCUMENT
    # --------------------------------------------------------

    document_output = (
        OUTPUT_DIR
        / f"{document_name}.json"
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


    documents_processed += 1


    print()

    print(
        f"Saved individual JSON:"
    )

    print(
        document_output
    )


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "FINAL DATA BUILD COMPLETE"
)

print("=" * 70)

print()

print(
    f"Active PDFs: "
    f"{len(active_pdfs)}"
)

print(
    f"Documents processed: "
    f"{documents_processed}"
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Each PDF now has its own JSON file."
)

print(
    "No all_documents.json is generated."
)

print()

print("=" * 70)