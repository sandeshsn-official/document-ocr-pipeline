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

    # Ensure the PDF closes even if rendering a page raises an error.
    with fitz.open(pdf_path) as document:

        page_count = len(document)
        if page_count == 0:
            raise ValueError("The PDF contains no pages.")

        # Remove only this PDF's old page images after opening and validating
        # the PDF, preventing stale pages when a rerun has fewer pages.
        for old_page_image in pdf_output_dir.glob("page_*.png"):
            if old_page_image.is_file():
                old_page_image.unlink()

        print()
        print("=" * 70)
        print(f"PDF: {pdf_path.name}")
        print(f"Pages: {page_count}")
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

    # Confirm the generated page count matches the source PDF.
    generated_images = sorted(pdf_output_dir.glob("page_*.png"))
    if len(generated_images) != page_count:
        raise RuntimeError(
            f"Expected {page_count} page images, "
            f"but found {len(generated_images)}."
        )


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
        return 1

    print()
    print("=" * 70)
    print(f"Found {len(pdf_files)} PDF file(s)")
    print("=" * 70)

    failed_files = []

    for pdf_path in pdf_files:

        try:

            convert_pdf_to_images(
                pdf_path
            )

        except Exception as error:

            failed_files.append(pdf_path.name)

            print()
            print(
                f"ERROR processing "
                f"{pdf_path.name}"
            )

            print(f"{type(error).__name__}: {error}")

    print()
    print("=" * 70)

    if failed_files:
        print("PDF CONVERSION FINISHED WITH ERRORS")
        print(f"Successful PDFs: {len(pdf_files) - len(failed_files)}")
        print(f"Failed PDFs: {len(failed_files)}")
        for failed_name in failed_files:
            print(f" - {failed_name}")
        print("=" * 70)
        return 1

    print("PDF CONVERSION COMPLETE")
    print(f"Successfully converted: {len(pdf_files)} PDF file(s)")
    print("=" * 70)
    return 0


# --------------------------------------------------
# START
# --------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
