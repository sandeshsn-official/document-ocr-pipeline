import json
import tempfile
import unittest
from pathlib import Path

from pipeline.learning_service import (
    get_learning_paths,
    inspect_learning_ledger,
    update_learning_json,
)


class TestLearningServiceIntegration(unittest.TestCase):

    def test_update_uses_safe_utility_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = get_learning_paths(root)
            paths["learning_dir"].mkdir(parents=True)

            original = {
                "decisions": [
                    {"decision": "USER_CONFIRMED_ORIGINAL"}
                ],
                "review_outcomes": [
                    {"outcome": "USER_SKIPPED"}
                ],
            }

            paths["decisions_file"].write_text(
                json.dumps(original),
                encoding="utf-8",
            )

            def add_metadata(data):
                data["test_metadata"] = {"checked": True}
                return data

            update_learning_json(
                paths["decisions_file"],
                add_metadata,
                project_root=root,
            )

            result = json.loads(
                paths["decisions_file"].read_text(encoding="utf-8")
            )

            self.assertEqual(result["decisions"], original["decisions"])
            self.assertEqual(
                result["review_outcomes"],
                original["review_outcomes"],
            )
            self.assertTrue(result["test_metadata"]["checked"])

    def test_update_rejects_non_learning_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outside_file = root / "unrelated.json"
            outside_file.write_text('{"keep": true}', encoding="utf-8")

            with self.assertRaises(ValueError):
                update_learning_json(
                    outside_file,
                    lambda data: {"changed": True},
                    project_root=root,
                )

            self.assertEqual(
                json.loads(outside_file.read_text(encoding="utf-8")),
                {"keep": True},
            )

    def test_inspection_remains_read_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = get_learning_paths(root)
            paths["learning_dir"].mkdir(parents=True)

            paths["decisions_file"].write_text(
                json.dumps({
                    "decisions": [
                        {"decision": "USER_CONFIRMED_SOURCE"}
                    ],
                    "review_outcomes": [],
                }),
                encoding="utf-8",
            )

            before = paths["decisions_file"].read_bytes()
            result = inspect_learning_ledger(root)

            self.assertTrue(result["read_only"])
            self.assertEqual(result["trusted_decision_count"], 1)
            self.assertEqual(paths["decisions_file"].read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
