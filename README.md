# Document OCR Pipeline

A Python-based OCR pipeline for extracting text, technical symbols, identifiers, measurements, and tables from PDF documents.

The pipeline is designed for technical documents where **source-faithful extraction is important**, especially for values such as:

- `Ω`
- `±`
- `≤`
- `≥`
- `°C`
- `µm`
- `→`
- `∂`
- `∇`
- `∫`
- `∈`
- `∉`
- `≠`
- `⊂`
- `⊃`
- `∞`

The pipeline uses OCR, document structure extraction, table reconstruction, validation, and an additional verification/review stage.

---

# 1. System Requirements

## Operating System

The pipeline is currently developed and tested on:

- Windows 10/11
- PowerShell

Linux can potentially be supported, but the commands in this README are written for Windows.

---

# 2. Software That Must Be Installed

On a completely new computer, install the following:

### Required

1. Python 3.11 (64-bit)
2. Git
3. Tesseract OCR

### Python packages

The Python packages are installed using:

powershell

pip install -r requirements.txt
OCR models

PaddleOCR downloads/caches the required OCR models when they are first used.

The models do NOT need to be committed to this GitHub repository.

3. Install Python

Install Python 3.11 (64-bit).

During Python installation, make sure:

Add Python to PATH

is enabled.

After installation, open a NEW PowerShell window and check:

python --version

Expected:

Python 3.11.x

Also check:

python -m pip --version
4. Install Git

Check whether Git is installed:

git --version

If Git is not recognized, install Git for Windows and reopen PowerShell.

5. Install Tesseract OCR

Tesseract is required by the verification stage of the pipeline.

After installing Tesseract, check:

tesseract --version

Expected output will look similar to:

tesseract 5.x.x

If PowerShell says:

tesseract : The term 'tesseract' is not recognized...

Tesseract is either not installed or its installation directory is not in PATH.

A common installation directory is:

C:\Program Files\Tesseract-OCR

If necessary, add that directory to the Windows PATH.

Then open a NEW PowerShell window and run:

tesseract --version

again.

6. Clone the Repository

Open PowerShell and run:

git clone https://github.com/sandeshsn-official/document-ocr-pipeline.git

Enter the project:

cd document-ocr-pipeline

Check the files:

dir

You should see the project files and folders.

7. Create a Python Virtual Environment

Inside the project directory:

python -m venv .venv

Activate it:

.venv\Scripts\activate

The PowerShell prompt should now contain:

(.venv)

Example:

(.venv) PS C:\...\document-ocr-pipeline>
8. Upgrade pip

Run:

python -m pip install --upgrade pip
9. Install Python Dependencies

The repository contains:

requirements.txt

Install the dependencies:

pip install -r requirements.txt

This may take some time because OCR-related packages can be large.

10. Verify Python Dependencies

Check PaddleOCR:

python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

Expected:

PaddleOCR OK

Check Paddle:

python -c "import paddle; print('Paddle:', paddle.__version__)"

Check the other major dependencies:

python -c "import cv2, fitz, numpy, PIL, pytesseract; print('Dependencies OK')"

Expected:

Dependencies OK
11. PaddleOCR Models

PaddleOCR requires OCR models.

The first time the OCR pipeline is executed, PaddleOCR may download the required models.

You may see messages similar to:

Downloading model...

or:

Model files already exist. Using cached files.

The second message means that the models have already been downloaded on that computer.

The models are normally cached outside the GitHub repository.

Therefore:

GitHub repository
    |
    +-- Python source code
    +-- requirements.txt
    +-- configuration/files
    |
    +-- NOT the downloaded OCR model cache

Each new computer may need to download the models once.

12. Important: Internet Access

The first PaddleOCR run may require internet access to download models.

Therefore, on a new computer:

Internet available
        |
        v
PaddleOCR downloads models
        |
        v
Models are cached locally
        |
        v
Future runs can use the cached models

If the computer has NO internet access, the required PaddleOCR model files must be transferred to that computer separately.

Do not assume that cloning the GitHub repository transfers the model cache.

13. Project Structure

The main pipeline scripts are:

src/
│
├── 01_pdf_to_image.py
├── 02_structure_ocr.py
├── 03_reconstruct_tables.py
├── 04_build_final_data.py
├── 05_validate_final_data.py
├── 06_accuracy_verification.py
├── 07_visual_review_report.py
├── 08_manual_review.py
├── 09_build_learning_memory.py
├── 10_correct_review.py
└── main.py

The general pipeline is:

PDF
 |
 v
01_pdf_to_image
 |
 v
Page Images
 |
 v
02_structure_ocr
 |
 v
OCR Text + Structure
 |
 v
03_reconstruct_tables
 |
 v
Tables
 |
 v
04_build_final_data
 |
 v
Structured JSON
 |
 v
05_validate_final_data
 |
 v
Validation
 |
 v
06_accuracy_verification
 |
 v
Source/Image Verification
 |
 v
08_manual_review
 |
 v
Human Verification
 |
 v
09_build_learning_memory
 |
 v
Reusable OCR Corrections
14. Input Documents

Place PDF documents in the project's input directory.

Example:

input/
    document1.pdf
    document2.pdf

Do NOT commit confidential ISRO documents to GitHub.

The input/ directory should remain local.

15. Output

The pipeline creates output files and directories containing:

page images
OCR results
structured data
reconstructed tables
validation results
verification results
review information
learning memory
final verified data

Generated output should normally remain local and should not be committed to GitHub unless intentionally required.

16. Running the Pipeline
Step 1 - Convert PDF to Images
python src\01_pdf_to_image.py

This converts PDF pages into images.

Step 2 - OCR and Structure Extraction
python src\02_structure_ocr.py

This performs OCR and extracts document structure.

PaddleOCR is used during this stage.

Step 3 - Reconstruct Tables
python src\03_reconstruct_tables.py

This reconstructs detected tables and their cells.

Step 4 - Build Final Structured Data
python src\04_build_final_data.py

This combines the extracted information into structured JSON.

Step 5 - Validate the Data
python src\05_validate_final_data.py

This performs structural/data validation.

Step 6 - Accuracy Verification
python src\06_accuracy_verification.py

This performs an additional verification stage.

It compares OCR results and uses source-image crops/secondary OCR to identify potentially incorrect text, symbols, numbers, and identifiers.

Technical disagreements such as:

Ω
±
≤
≥
µm
→

are treated as high-risk because a small OCR error can change the meaning of a technical document.

Step 7 - Manual Review
python src\08_manual_review.py

Items that cannot be safely verified automatically can be reviewed manually.

The original source image should be treated as the ground truth.

For example:

OCR result:       48.62 O
Source document:  48.62 Ω

The correct value should be taken from the source image/document, not blindly from another OCR engine.

Step 8 - Build Learning Memory
python src\09_build_learning_memory.py

This builds reusable correction patterns from previous manual decisions.

For example:

Ω -> O -> Ω
± -> + -> ±
≥ -> > -> ≥

These patterns can help reduce repeated manual review.

However, learning memory must not be treated as proof that every future occurrence is correct.

For source-faithful extraction, uncertain/high-risk cases should still be verified against the source.

17. Important Accuracy Principle

The goal of this project is NOT simply:

"Get a high OCR confidence score"

The important goal is:

Source Document
       =
Extracted Data

as closely as possible.

For example:

Source:
≤ 50.00 Ω

Incorrect OCR:
<= 50.00 O

These are not considered identical for a technical document.

Likewise:

±0.30

must not silently become:

+0.30

and:

≥ 15 min

must not silently become:

> 15 min

The original source image/PDF is the final reference for uncertain values.

18. Do Not Assume OCR Is Always Correct

OCR can confuse:

O / 0
I / 1 / l
Ω / O
µ / u
± / +
≤ / <
≥ / >
→ / -
∫ / f
∂ / o

Technical identifiers can also be corrupted.

Example:

PRB-O0I7

must not automatically be changed to another value without checking the source.

19. Recommended Fresh-Machine Test

Before processing real documents, test the installation.

Run:

python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

Then:

tesseract --version

Then run the first two pipeline stages on a small test PDF:

python src\01_pdf_to_image.py

and:

python src\02_structure_ocr.py

If these complete successfully, the core OCR environment is working.

20. First-Time Setup Checklist

On a new computer:

[ ] Windows available
[ ] Python 3.11 64-bit installed
[ ] Python added to PATH
[ ] Git installed
[ ] Tesseract installed
[ ] Tesseract added to PATH
[ ] Repository cloned
[ ] Project directory opened
[ ] Virtual environment created
[ ] Virtual environment activated
[ ] pip upgraded
[ ] requirements.txt installed
[ ] PaddleOCR import works
[ ] Paddle import works
[ ] OpenCV/PyMuPDF/PIL dependencies work
[ ] PaddleOCR models downloaded/cached
[ ] Test PDF processed
21. Quick Setup Commands

For a computer where Python, Git, and Tesseract are already installed:

git clone https://github.com/sandeshsn-official/document-ocr-pipeline.git

cd document-ocr-pipeline

python -m venv .venv

.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"

tesseract --version

Then place the test PDF in:

input/

and start the pipeline.

22. Troubleshooting
Python not recognized

If:

python --version

does not work:

Install Python
Enable "Add Python to PATH"
Restart PowerShell
Git not recognized

If:

git --version

does not work:

Install Git for Windows
Restart PowerShell
Tesseract not recognized

If:

tesseract --version

does not work:

Install Tesseract OCR
Add the Tesseract installation directory to PATH
Restart PowerShell
PaddleOCR import error

If:

python -c "from paddleocr import PaddleOCR"

fails:

Check that the virtual environment is activated:

.venv\Scripts\activate

Then reinstall:

pip install -r requirements.txt
PaddleOCR model download failure

If the models cannot be downloaded:

Check internet connectivity.
Check firewall/proxy restrictions.
Check whether the machine allows Python to access the internet.
If the machine is offline, transfer the required model cache separately.
Dependency installation problems

The current requirements.txt was generated from the development environment using pip freeze.

It may therefore contain packages that are:

machine-specific
GPU-specific
CUDA-specific
unnecessary for the core OCR pipeline

If installation fails on another machine, do not randomly install packages one by one.

Instead, identify the failing package and create a clean/minimal requirements file for the target machine.

23. GPU / CPU

The pipeline can depend on the installed Paddle/PaddleOCR runtime.

A new machine may not have the same GPU/CUDA environment as the development machine.

Therefore:

Development PC GPU environment
          !=
ISRO PC environment

The Python/Paddle installation should be selected according to the target computer's CPU/GPU environment.

If no compatible GPU is available, use a CPU-compatible installation.

24. Security and Confidential Documents

Do NOT upload confidential documents to the public GitHub repository.

The repository should contain:

Source code
Configuration
Documentation
Requirements

The repository should NOT contain:

Confidential PDFs
ISRO documents
Sensitive extracted data
Credentials
Passwords
API keys
Private model files
Large generated outputs

Use the local:

input/

directory for documents.

25. Important Current Limitation

The repository contains the OCR pipeline code, but a completely fresh computer still requires:

Python
Git
Tesseract
Python dependencies
PaddleOCR models

Therefore:

git clone

alone is NOT sufficient.

The first-time setup must be completed before processing documents.

26. Recommended Production Flow

For an actual technical document:

1. Place PDF in input/
        ↓
2. Convert PDF to page images
        ↓
3. Run PaddleOCR
        ↓
4. Extract paragraphs/text
        ↓
5. Detect/reconstruct tables
        ↓
6. Build structured JSON
        ↓
7. Validate JSON
        ↓
8. Verify uncertain OCR
        ↓
9. Compare against source image
        ↓
10. Manually verify unresolved high-risk cases
        ↓
11. Build learning memory
        ↓
12. Use FINAL VERIFIED data

The chatbot/RAG system should use the verified output, not blindly use raw OCR output.

27. Source of Truth

For technical documents:

Original PDF / Original Page Image
              ↓
          SOURCE OF TRUTH

OCR is an extraction mechanism.

It should NOT be considered the source of truth.

If OCR says:

48.62 O

and the source clearly says:

48.62 Ω

the final verified value should be:

48.62 Ω
28. Final Verification Before Deployment

Before using the pipeline on important documents, verify:

[ ] OCR completed
[ ] No missing pages
[ ] Text extracted
[ ] Tables reconstructed
[ ] JSON generated
[ ] Validation passed
[ ] High-risk symbols checked
[ ] Numeric values checked
[ ] Technical identifiers checked
[ ] Manual review completed where required
[ ] Final verified JSON generated

Only after these checks should the verified data be supplied to the downstream chatbot/RAG system.

29. Repository

GitHub repository:

https://github.com/sandeshsn-official/document-ocr-pipeline.git


### One thing I strongly recommend before tomorrow

Your README is now the **setup guide**, but I would make **one more change to the project before relying on it at ISRO**:

**Make `main.py` actually run the required stages with one command**, so the final workflow becomes:

powershell
python src\main.py

→ put PDF in input/
→ pipeline runs
→ final verified JSON is produced.
