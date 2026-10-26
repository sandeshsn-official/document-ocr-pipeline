import json
import tempfile
from pathlib import Path
from pipeline.stage_artifact_history import (
    begin_stage_capture,
    finish_stage_capture,
    read_artifact,
)

with tempfile.TemporaryDirectory() as temp:
    job_dir = Path(temp) / "test_job"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True)

    job = {
        "job_id": "history_repeat_test",
        "job_dir": job_dir,
        "output_dir": output_dir,
    }

    target = output_dir / "pages" / "Calibration_Report" / "page_0001.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first-run-artifact")

    first_token = begin_stage_capture(job, 1)
    finish_stage_capture(job, first_token, return_code=0)

    manifest_path = job_dir / "stage_history" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_ref = manifest["runs"][-1]["after_state"][
        "pages/Calibration_Report/page_0001.png"
    ]

    second_token = begin_stage_capture(job, 1)
    target.write_bytes(b"second-run-artifact")
    finish_stage_capture(job, second_token, return_code=0)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second_ref = manifest["runs"][-1]["after_state"][
        "pages/Calibration_Report/page_0001.png"
    ]

    assert read_artifact(job_dir, first_ref) == b"first-run-artifact"
    assert read_artifact(job_dir, second_ref) == b"second-run-artifact"
    assert first_ref["sha256"] != second_ref["sha256"]
    assert len(manifest["runs"]) == 2

    print("FIRST_ARTIFACT_PRESERVED=PASS")
    print("SECOND_ARTIFACT_CAPTURED=PASS")
    print("TWO_RUNS_RECORDED=PASS")
