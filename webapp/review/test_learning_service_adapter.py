import json
import tempfile
import unittest
from pathlib import Path

from pipeline.learning_service import inspect_learning_ledger


class TestLearningServiceAdapter(unittest.TestCase):

    def test_inspection_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            learning_dir = root / "output" / "learning"
            learning_dir.mkdir(parents=True)

            decisions_path = (
                learning_dir / "manual_review_decisions.json"
            )
            memory_path = (
                learning_dir / "ocr_learning_memory.json"
            )

            decisions_data = {
                "decisions": [
                    {"decision": "USER_CONFIRMED_ORIGINAL"},
                    {"decision": "USER_SKIPPED"},
                ],
                "review_outcomes": [
                    {"outcome": "USER_SKIPPED"},
                ],
            }
            memory_data = {
                "version": 7,
                "exact_patterns": {},
            }

            decisions_path.write_text(
                json.dumps(decisions_data),
                encoding="utf-8",
            )
            memory_path.write_text(
                json.dumps(memory_data),
                encoding="utf-8",
            )

            original_decisions = decisions_path.read_bytes()
            original_memory = memory_path.read_bytes()

            result = inspect_learning_ledger(root)

            self.assertEqual(result["decision_count"], 2)
            self.assertEqual(result["review_outcome_count"], 1)
            self.assertEqual(result["trusted_decision_count"], 1)
            self.assertEqual(result["memory_version"], 7)
            self.assertTrue(result["read_only"])

            self.assertEqual(
                decisions_path.read_bytes(),
                original_decisions,
            )
            self.assertEqual(
                memory_path.read_bytes(),
                original_memory,
            )


if __name__ == "__main__":
    unittest.main()
