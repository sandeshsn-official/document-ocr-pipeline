
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from pipeline import service


class JobOutputPathTests(SimpleTestCase):
    def test_get_job_output_returns_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            fake_job = {
                "output_dir": output_dir,
            }

            with patch.object(
                service, "get_job", return_value=fake_job
            ):
                self.assertEqual(
                    service.get_job_output("test-job"),
                    output_dir,
                )


class PendingReviewTests(SimpleTestCase):
    def test_returns_false_when_no_review_directory_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job = {
                "output_dir": Path(temp_dir) / "output",
            }

            self.assertFalse(service.has_pending_review(job))

    def test_returns_true_when_review_items_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            review_dir = (
                Path(temp_dir)
                / "output"
                / "verified_data"
                / "review_required"
            )
            review_dir.mkdir(parents=True)

            review_file = review_dir / "test_review.json"
            review_file.write_text(
                json.dumps({"items": [{"text": "example"}]}),
                encoding="utf-8",
            )

            job = {
                "output_dir": Path(temp_dir) / "output",
            }

            self.assertTrue(service.has_pending_review(job))
