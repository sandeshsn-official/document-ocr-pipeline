import fitz
from pathlib import Path


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output/pages")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# CONVERT ONE PDF
# --------------------------------------------------

def convert_pdf_to_images(pdf_path):

    pdf_name = pdf_path.stem

    # Create separate folder for each PDF
    pdf_output_dir = OUTPUT_DIR / pdf_name

    pdf_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    document = fitz.open(pdf_path)

    print()
    print("=" * 70)
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(document)}")
    print("=" * 70)

    for page_number, page in enumerate(
        document,
        start=1
    ):

        # 300 DPI
        dpi = 300
        zoom = dpi / 72

        matrix = fitz.Matrix(
            zoom,
            zoom
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        output_file = (
            pdf_output_dir /
            f"page_{page_number:04d}.png"
        )

        pix.save(output_file)

        print(
            f"Page {page_number} "
            f"-> {output_file}"
        )

    document.close()


# --------------------------------------------------
# PROCESS ALL PDFs
# --------------------------------------------------

def main():

    pdf_files = sorted(
        INPUT_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print()
        print("ERROR: No PDF files found.")
        print("Put your PDFs inside the input folder.")
        return

    print()
    print("=" * 70)
    print(f"Found {len(pdf_files)} PDF file(s)")
    print("=" * 70)

    for pdf_path in pdf_files:

        try:

            convert_pdf_to_images(
                pdf_path
            )

        except Exception as error:

            print()
            print(
                f"ERROR processing "
                f"{pdf_path.name}"
            )

            print(error)

    print()
    print("=" * 70)
    print("PDF CONVERSION COMPLETE")
    print("=" * 70)


# --------------------------------------------------
# START
# --------------------------------------------------

if __name__ == "__main__":
    main()