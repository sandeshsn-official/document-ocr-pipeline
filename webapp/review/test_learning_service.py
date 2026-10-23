# test_learning_service.py

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


# Locate the existing Stage 9 script without modifying it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE9_PATH = PROJECT_ROOT / "src" / "09_build_learning_memory.py"

spec = importlib.util.spec_from_file_location(
    "stage9_learning_memory_test_module",
    STAGE9_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load Stage 9 from: {STAGE9_PATH}")

stage9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage9)


class TestLearningService(unittest.TestCase):

    def make_decision(
        self,
        decision_type,
        document="document_a.pdf",
        page=1,
        original_text="O",
        source_text="0",
        verified_text="O",
    ):
        return {
            "decision": decision_type,
            "document": document,
            "page": page,
            "object_type": "table_cell",
            "table": 1,
            "row": 1,
            "column": 1,
            "original_text": original_text,
            "source_text": source_text,
            "verified_text": verified_text,
        }

    def test_only_trusted_human_decisions_enter_learning(self):
        trusted = self.make_decision(
            "USER_CONFIRMED_ORIGINAL"
        )
        skipped = self.make_decision(
            "USER_SKIPPED",
            document="document_b.pdf",
        )
        automatic = self.make_decision(
            "AUTO_RESOLVED",
            document="document_c.pdf",
        )

        memory = stage9.build_memory(
            [trusted, skipped, automatic]
        )

        self.assertEqual(
            memory["trusted_human_decisions"],
            1,
        )

    def test_skipped_and_automatic_outcomes_are_not_human_evidence(self):
        outcomes = [
            {
                "document": "document_a.pdf",
                "page": 1,
                "outcome": "USER_SKIPPED",
            },
            {
                "document": "document_b.pdf",
                "page": 1,
                "outcome": "AUTO_RESOLVED",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            decisions_file = Path(temp_dir) / "decisions.json"
            decisions_file.write_text(
                json.dumps({"decisions": [], "review_outcomes": outcomes}),
                encoding="utf-8",
            )

            loaded = stage9.load_review_outcomes(decisions_file)

        self.assertEqual(len(loaded), 2)
        self.assertFalse(
            any(
                item.get("outcome")
                in stage9.TRUSTED_HUMAN_DECISIONS
                for item in loaded
            )
        )

    def test_existing_decision_file_is_not_modified_by_tests(self):
        original_data = {
            "decisions": [
                self.make_decision(
                    "USER_CONFIRMED_ORIGINAL"
                )
            ],
            "review_outcomes": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            decisions_file = Path(temp_dir) / "decisions.json"
            original_bytes = json.dumps(
                original_data,
                ensure_ascii=False,
            ).encode("utf-8")

            decisions_file.write_bytes(original_bytes)

            loaded = stage9.load_review_outcomes(decisions_file)
            stage9.build_memory(original_data["decisions"])

            self.assertTrue(loaded)
            self.assertEqual(
                decisions_file.read_bytes(),
                original_bytes,
            )

    def test_conflicting_evidence_is_not_promoted_to_reusable_rule(self):
        decisions = [
            self.make_decision(
                "USER_CONFIRMED_ORIGINAL",
                document="document_a.pdf",
                original_text="O",
                source_text="0",
                verified_text="O",
            ),
            self.make_decision(
                "USER_CONFIRMED_SOURCE",
                document="document_b.pdf",
                original_text="O",
                source_text="0",
                verified_text="0",
            ),
        ]

        memory = stage9.build_memory(decisions)

        # Conflicting evidence must not create a reusable pattern
        # that selects one verified value for this exact pair.
        matching_patterns = [
            pattern
            for pattern in memory["exact_patterns"].values()
            if isinstance(pattern, dict)
            and pattern.get("original_text") == "O"
            and pattern.get("source_text") == "0"
        ]

        for pattern in matching_patterns:
            self.assertNotIn(
                pattern.get("verified_text"),
                {"O", "0"},
            )


if __name__ == "__main__":
    unittest.main()
