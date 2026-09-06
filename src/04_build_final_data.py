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
# ============================================================

if OUTPUT_DIR.exists():

    for child in OUTPUT_DIR.iterdir():

        if child.name == "all_documents.json":
            continue

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
# PROCESS ONLY ACTIVE INPUT DOCUMENTS
# ============================================================

all_documents = []


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


    pages = []


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


    # --------------------------------------------------------
    # DOCUMENT OBJECT
    # --------------------------------------------------------

    document_result = {

        "document":
            document_name,

        "document_id":
            document_name,

        "page_count":
            len(pages),

        "pages":
            pages
    }


    all_documents.append(
        document_result
    )


    # --------------------------------------------------------
    # SAVE INDIVIDUAL DOCUMENT
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


    print()

    print(
        f"Saved: "
        f"{document_output}"
    )


# ============================================================
# SAVE COMBINED DOCUMENT
# ============================================================

combined_output = {

    "document_count":
        len(all_documents),

    "documents":
        all_documents
}


combined_file = (
    OUTPUT_DIR
    / "all_documents.json"
)


with open(
    combined_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        combined_output,
        file,
        indent=2,
        ensure_ascii=False
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
    f"Documents: "
    f"{len(all_documents)}"
)

print()

print(
    "Combined JSON:"
)

print(
    combined_file
)

print()

print("=" * 70)