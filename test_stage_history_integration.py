import json
import shutil
from pathlib import Path
from pipeline.service import create_job, prepare_job_source, run_stage

source_pdf = Path.cwd() / "input" / "Calibration_Report.pdf"
if not source_pdf.is_file():
    raise FileNotFoundError(source_pdf)

job = create_job()
print(f"TEST_JOB_ID={job['job_id']}")

shutil.copy2(source_pdf, job["input_dir"] / source_pdf.name)
job_src = prepare_job_source(job)

run_stage(job, 1, script_root=job_src)

manifest_path = job["job_dir"] / "stage_history" / "manifest.json"
print(f"MANIFEST_EXISTS={manifest_path.is_file()}")

if not manifest_path.is_file():
    raise RuntimeError("Stage-history manifest was not created")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"RUN_COUNT={len(manifest['runs'])}")
print(f"LAST_RUN={json.dumps(manifest['runs'][-1], indent=2)}")

assert manifest["runs"][-1]["stage"] == 1
assert manifest["runs"][-1]["status"] == "succeeded"
assert manifest["runs"][-1]["after_state"], "No output artifacts were captured"

print("STAGE_HISTORY_INTEGRATION=PASS")
