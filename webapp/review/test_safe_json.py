import json
import tempfile
import unittest
from pathlib import Path

from pipeline.safe_json import (
    LockTimeoutError,
    atomic_write_json,
    exclusive_file_lock,
    update_json_safely,
)


class TestSafeJson(unittest.TestCase):

    def test_atomic_write_produces_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.json"
            payload = {"version": 1, "items": ["a", "b"]}

            atomic_write_json(path, payload)

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                payload,
            )
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tmp")),
                [],
            )

    def test_update_preserves_existing_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.json"
            original = {
                "decisions": [
                    {"decision": "USER_CONFIRMED_ORIGINAL"}
                ],
                "review_outcomes": [
                    {"outcome": "USER_SKIPPED"}
                ],
            }
            atomic_write_json(path, original)

            def updater(data):
                data["new_metadata"] = {"checked": True}
                return data

            result = update_json_safely(path, updater)

            self.assertEqual(
                result["decisions"],
                original["decisions"],
            )
            self.assertEqual(
                result["review_outcomes"],
                original["review_outcomes"],
            )
            self.assertTrue(result["new_metadata"]["checked"])

    def test_updater_failure_preserves_original_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.json"
            atomic_write_json(path, {"keep": True})
            original_bytes = path.read_bytes()

            def failing_updater(data):
                raise ValueError("Simulated failure")

            with self.assertRaises(ValueError):
                update_json_safely(path, failing_updater)

            self.assertEqual(path.read_bytes(), original_bytes)

    def test_lock_timeout_when_another_lock_is_held(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ledger.json"

            with exclusive_file_lock(path):
                with self.assertRaises(LockTimeoutError):
                    with exclusive_file_lock(
                        path,
                        timeout=0.15,
                        poll_interval=0.02,
                    ):
                        self.fail("Second lock should not be acquired")


if __name__ == "__main__":
    unittest.main()
