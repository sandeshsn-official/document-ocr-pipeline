import json
import tempfile
import unittest
from pathlib import Path

from webapp.pipeline.stage_artifact_history import (
    begin_stage_capture,
    finish_stage_capture,
    load_stage_history,
    read_artifact,
)


class StageArtifactHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.job_dir = Path(self.temp.name) / "job_test"
        self.output_dir = self.job_dir / "output"
        self.output_dir.mkdir(parents=True)
        self.job = {
            "job_id": "job_test",
            "job_dir": self.job_dir,
            "output_dir": self.output_dir,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_preserves_pre_stage_file_when_stage_overwrites_it(self):
        target = self.output_dir / "final" / "doc.json"
        target.parent.mkdir()
        target.write_text('{"value":"before"}', encoding="utf-8")

        token = begin_stage_capture(self.job, 2)
        target.write_text('{"value":"after"}', encoding="utf-8")
        finish_stage_capture(self.job, token, return_code=0)

        manifest = load_stage_history(self.job_dir)
        baseline_ref = manifest["baseline"]["state"]["final/doc.json"]
        after_ref = manifest["runs"][0]["after_state"]["final/doc.json"]
        self.assertNotEqual(baseline_ref["sha256"], after_ref["sha256"])
        self.assertEqual(read_artifact(self.job_dir, baseline_ref), b'{"value":"before"}')
        self.assertEqual(read_artifact(self.job_dir, after_ref), b'{"value":"after"}')
        self.assertEqual(manifest["runs"][0]["changes"]["modified"], ["final/doc.json"])

    def test_records_added_and_removed_files(self):
        (self.output_dir / "old.txt").write_text("old", encoding="utf-8")
        token = begin_stage_capture(self.job, 1)
        (self.output_dir / "old.txt").unlink()
        (self.output_dir / "new.txt").write_text("new", encoding="utf-8")
        finish_stage_capture(self.job, token, return_code=0)

        manifest = load_stage_history(self.job_dir)
        changes = manifest["runs"][0]["changes"]
        self.assertEqual(changes["added"], ["new.txt"])
        self.assertEqual(changes["removed"], ["old.txt"])

    def test_failed_stage_attempt_is_still_recorded(self):
        token = begin_stage_capture(self.job, 3)
        (self.output_dir / "partial.txt").write_text("partial", encoding="utf-8")
        finish_stage_capture(self.job, token, return_code=1, error="synthetic failure")

        manifest = load_stage_history(self.job_dir)
        self.assertEqual(manifest["runs"][0]["status"], "failed")
        self.assertEqual(manifest["runs"][0]["error"], "synthetic failure")
        self.assertIn("partial.txt", manifest["runs"][0]["after_state"])

    def test_content_objects_are_deduplicated(self):
        (self.output_dir / "a.txt").write_text("same", encoding="utf-8")
        token = begin_stage_capture(self.job, 1)
        finish_stage_capture(self.job, token, return_code=0)
        (self.output_dir / "b.txt").write_text("same", encoding="utf-8")
        token = begin_stage_capture(self.job, 2)
        finish_stage_capture(self.job, token, return_code=0)

        manifest = load_stage_history(self.job_dir)
        a_ref = manifest["runs"][0]["after_state"]["a.txt"]
        b_ref = manifest["runs"][1]["after_state"]["b.txt"]
        self.assertEqual(a_ref["sha256"], b_ref["sha256"])
        self.assertEqual(a_ref["object"], b_ref["object"])


if __name__ == "__main__":
    unittest.main()
