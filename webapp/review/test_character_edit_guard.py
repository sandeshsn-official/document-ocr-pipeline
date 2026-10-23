
import ast
import re
import unittest
from pathlib import Path


SOURCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "06_accuracy_verification.py"
)


class CharacterEditGuardSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_FILE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

        cls.function = next(
            node
            for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "find_learned_match"
        )

        cls.function_source = ast.get_source_segment(
            cls.source, cls.function
        )

    def test_character_edit_branch_exists(self):
        self.assertIn(
            'family == "character_edit"',
            self.function_source,
        )

    def test_invalid_position_guard_exists(self):
        self.assertRegex(
            self.function_source,
            r"position\s*<\s*0\s+or\s+position\s*>=\s*"
            r"len\(original_match_text\)",
        )

    def test_unicode_lookup_handles_invalid_mapping(self):
        self.assertIn("unicodedata.lookup", self.function_source)
        self.assertRegex(
            self.function_source,
            r"except\s*\(KeyError,\s*TypeError\)",
        )

    def test_unrelated_original_character_is_rejected(self):
        self.assertRegex(
            self.function_source,
            r"original_char\s+not\s+in\s+\{\s*"
            r"witness_char\s*,\s*target_char\s*\}",
        )

    def test_already_correct_character_is_skipped(self):
        self.assertRegex(
            self.function_source,
            r"original_char\s*==\s*target_char",
        )

    def test_result_must_change_original_text(self):
        self.assertRegex(
            self.function_source,
            r"verified_value\s*==\s*original_match_text",
        )


if __name__ == "__main__":
    unittest.main()