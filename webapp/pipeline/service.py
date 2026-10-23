from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# PROJECT PATHS
# ============================================================================

WEBAPP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEBAPP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

JOBS_DIR = PROJECT_ROOT / "output" / "jobs"

PYTHON = sys.executable


# ============================================================================
# STAGE DEFINITIONS
# ============================================================================

STAGES = {
    1: "01_pdf_to_image.py",
    2: "02_structure_ocr.py",
    3: "03_reconstruct_tables.py",
    4: "04_build_final_data.py",
    5: "05_validate_final_data.py",
    6: "06_accuracy_verification.py",
    7: "07_visual_review_report.py",
    8: "08_manual_visual_review.py",
    9: "09_build_learning_memory.py",
    10: "10_correct_review.py",
    11: "11_reconstruct_from_json.py",
    12: "12_restore_historical_review_evidence.py",
}


# ============================================================================
# JOB PATHS
# ============================================================================

def create_job() -> dict:
    """
    Create an isolated job directory.

    No OCR stage is executed here.
    """

    job_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    job_dir = JOBS_DIR / job_id

    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    logs_dir = job_dir / "logs"

    input_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)

    state = {
        "job_id": job_id,
        "status": "created",
        "current_stage": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }

    _write_state(job_dir, state)

    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "logs_dir": logs_dir,
    }


def get_job(job_id: str) -> dict:
    """
    Return paths for an existing job.
    """

    job_dir = JOBS_DIR / job_id

    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job not found: {job_id}")

    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "input_dir": job_dir / "input",
        "output_dir": job_dir / "output",
        "logs_dir": job_dir / "logs",
    }


# ============================================================================
# UPLOAD
# ============================================================================

def save_uploaded_pdf(job: dict, uploaded_file) -> Path:
    """
    Save a Django UploadedFile into the job input directory.

    The uploaded filename is preserved where possible.
    """

    original_name = Path(uploaded_file.name).name

    if not original_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    destination = job["input_dir"] / original_name

    with destination.open("wb") as handle:
        for chunk in uploaded_file.chunks():
            handle.write(chunk)

    return destination


# ============================================================================
# STATE
# ============================================================================

def _state_path(job_dir: Path) -> Path:
    return job_dir / "job_state.json"


def _write_state(job_dir: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    _state_path(job_dir).write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_job_status(job_id: str) -> dict:
    job = get_job(job_id)
    path = _state_path(job["job_dir"])

    if not path.exists():
        raise FileNotFoundError(f"Job state does not exist: {job_id}")

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _set_status(
    job: dict,
    status: str,
    stage: int | None = None,
    error: str | None = None,
) -> None:

    state_path = _state_path(job["job_dir"])

    if state_path.exists():
        state = json.loads(
            state_path.read_text(encoding="utf-8")
        )
    else:
        state = {
            "job_id": job["job_id"],
        }

    state["status"] = status
    state["current_stage"] = stage
    state["error"] = error

    _write_state(job["job_dir"], state)


# ============================================================================
# LOGGING
# ============================================================================

def _stage_log_path(job: dict, stage: int) -> Path:
    return job["logs_dir"] / f"stage_{stage:02d}.log"


def _write_stage_log(
    job: dict,
    stage: int,
    stdout: str,
    stderr: str,
) -> None:

    log = [
        f"STAGE {stage}",
        "=" * 80,
        "",
        "STDOUT",
        "-" * 80,
        stdout,
        "",
        "STDERR",
        "-" * 80,
        stderr,
    ]

    _stage_log_path(job, stage).write_text(
        "\n".join(log),
        encoding="utf-8",
    )


# ============================================================================
# STAGE EXECUTION
# ============================================================================

def run_stage(
    job: dict,
    stage: int,
    *,
    script_root: Path | None = None,
) -> subprocess.CompletedProcess:
    """
    Execute one existing OCR stage.

    Stage output is streamed live to the terminal and written
    to the stage log while the process is running.
    """

    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")

    if script_root is None:
        script_root = SRC_DIR

    script = script_root / STAGES[stage]

    if not script.exists():
        raise FileNotFoundError(
            f"Stage {stage} does not exist: {script}"
        )

    _set_status(
        job,
        status="running",
        stage=stage,
    )

    log_path = _stage_log_path(job, stage)

    command = [
        PYTHON,
        "-u",
        str(script),
    ]

    output_lines = []

    with log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as log_file:

        log_file.write(
            f"STAGE {stage}\n"
            + "=" * 80
            + "\n\n"
        )
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(job["job_dir"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="", flush=True)

                log_file.write(line)
                log_file.flush()

                output_lines.append(line)

            return_code = process.wait()

        except KeyboardInterrupt:
            process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            log_file.write(
                "\n\nPROCESS INTERRUPTED BY USER\n"
            )
            log_file.flush()

            _set_status(
                job,
                status="interrupted",
                stage=stage,
                error="Stage interrupted by user.",
            )

            raise

    stdout = "".join(output_lines)

    if return_code != 0:
        error = (
            f"Stage {stage} failed with exit code "
            f"{return_code}."
        )

        _set_status(
            job,
            status="failed",
            stage=stage,
            error=error,
        )

        raise RuntimeError(
            error
            + "\n\n"
            + stdout[-4000:]
        )

    return subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout=stdout,
        stderr="",
    )


# ============================================================================
# JOB SOURCE PREPARATION
# ============================================================================


def prepare_job_source(job: dict) -> Path:
    """
    Prepare an isolated source copy for a job.

    Preserve an existing job source directory instead of deleting
    and recreating it. This protects job-specific files and changes.
    """
    job_src = job["job_dir"] / "src"

    # Never destroy an existing job's source directory.
    if job_src.is_dir():
        return job_src

    if job_src.exists():
        raise RuntimeError(
            f"Job source path exists but is not a directory: {job_src}"
        )

    if not SRC_DIR.is_dir():
        raise FileNotFoundError(
            f"Original pipeline source directory not found: {SRC_DIR}"
        )

    shutil.copytree(
        SRC_DIR,
        job_src,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
        ),
    )

    return job_src



# ============================================================================
# JOB PIPELINE
# ============================================================================

def run_initial_pipeline(job: dict) -> None:
    """
    Run the automatic portion of the OCR pipeline.

    Stage 8 is deliberately stopped before human review.
    """

    job_src = prepare_job_source(job)

    try:
        for stage in range(1, 8):
            run_stage(
                job,
                stage,
                script_root=job_src,
            )

        _set_status(
            job,
            status="awaiting_review",
            stage=8,
        )

    except Exception:
        raise


# ============================================================================
# REVIEW CHECK
# ============================================================================


def has_pending_review(job: dict) -> bool:
    """
    Check whether unresolved Stage 8 review items remain.
    """
    review_dir = (
        job["output_dir"]
        / "verified_data"
        / "review_required"
    )

    if not review_dir.exists():
        return False

    for path in review_dir.glob("*.json"):
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(data, dict):
            continue

        items = data.get("items", [])
        if isinstance(items, list) and items:
            return True

    return False



# ============================================================================
# CONTINUE AFTER REVIEW
# ============================================================================

def run_after_review(job: dict) -> None:
    """
    Continue the job after Stage 8 human review.

    Stage 8 itself is controlled through stage8_review_service.py.
    """

    try:
        job_src = job["job_dir"] / "src"

        if not job_src.exists():
            job_src = prepare_job_source(job)

        # Stage 9 and later are intentionally kept separate from the
        # normal per-document OCR execution for now.
        #
        # Their shared/global learning behavior will be connected
        # explicitly after the job-isolation layer is verified.

        _set_status(
            job,
            status="review_completed",
            stage=8,
        )

    except Exception as exc:
        _set_status(
            job,
            status="failed",
            stage=8,
            error=str(exc),
        )
        raise


# ============================================================================
# RESULT HELPERS
# ============================================================================


def get_job_output(job_id: str) -> Path:
    job = get_job(job_id)
    return job["output_dir"]



def get_job_log(job_id: str, stage: int) -> str:
    job = get_job(job_id)

    path = _stage_log_path(job, stage)

    if not path.exists():
        raise FileNotFoundError(
            f"Stage {stage} log does not exist for job {job_id}."
        )

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


# ============================================================================
# SIMPLE PIPELINE STATUS
# ============================================================================

def describe_job(job_id: str) -> dict:
    """
    Return a compact description suitable for Django JSON responses.
    """

    job = get_job(job_id)
    state = get_job_status(job_id)

    return {
        "job_id": job_id,
        "status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "error": state.get("error"),
        "job_dir": str(job["job_dir"]),
        "output_dir": str(job["output_dir"]),
    }