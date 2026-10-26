import unittest
from stage_output_adapter import adapt_stage_document
from comparison_engine import compare_tables


def stage_doc(doc, values, status="FALLBACK_OCR"):
    rows = [{"row": 1, "cells": [
        {"column": 1, "text": "Sample", "confidence": 1.0, "status": status},
        {"column": 2, "text": "Voltage (V)", "confidence": 1.0, "status": status},
    ]}]
    for i, (sample, voltage) in enumerate(values, start=2):
        rows.append({"row": i, "cells": [
            {"column": 1, "text": sample, "confidence": .99, "status": status, "bbox": [1, 2, 3, 4]},
            {"column": 2, "text": voltage, "confidence": .98, "status": status, "bbox": [5, 6, 7, 8]},
        ]})
    return {"document": doc, "page_count": 1, "pages": [{
        "document": doc, "page": 1, "tables": [],
        "reconstructed_tables": [{"rows": rows, "status": status, "source": "source_image_grid"}],
    }]}


class StageOutputAdapterTests(unittest.TestCase):
    def test_converts_reconstructed_rows_and_keeps_provenance(self):
        adapted = adapt_stage_document(stage_doc("DocA", [("A-01", "5.00")]), stage="3")
        self.assertEqual(adapted["table_count"], 1)
        table = adapted["tables"][0]
        self.assertEqual(table["header"], ["Sample", "Voltage (V)"])
        cell = table["rows"][1]["cells"][1]
        self.assertEqual(cell["text"], "5.00")
        self.assertEqual(cell["provenance"]["page"], 1)
        self.assertEqual(cell["provenance"]["source_ocr_confidence"], .98)
        self.assertEqual(cell["provenance"]["verification_status"], "UNVERIFIED")

    def test_fallback_ocr_comparison_remains_ambiguous(self):
        left = adapt_stage_document(stage_doc("DocA", [("A-01", "5.00")]), stage="2")["tables"][0]
        right = adapt_stage_document(stage_doc("DocA", [("A-01", "5.10")]), stage="3")["tables"][0]
        result = compare_tables(left, right, row_key="Sample")
        row = next(r for r in result["rows"] if r["row_key"] == "a-01")
        voltage = next(c for c in row["cells"] if c["field"] == "Voltage (V)")
        self.assertEqual(voltage["status"], "AMBIGUOUS")
        self.assertEqual(voltage["left"]["value"], "5.00")
        self.assertEqual(voltage["right"]["value"], "5.10")

    def test_empty_tables_falls_back_to_tables_array(self):
        doc = {"document": "DocB", "pages": [{"page": 1, "tables": [{
            "status": "SOURCE_VERIFIED", "rows": [
                {"row": 1, "cells": [{"column": 1, "text": "ID"}, {"column": 2, "text": "Value"}]},
                {"row": 2, "cells": [{"column": 1, "text": "X1"}, {"column": 2, "text": "42"}]},
            ]}], "reconstructed_tables": []}]}
        adapted = adapt_stage_document(doc, stage="2")
        self.assertEqual(adapted["table_count"], 1)
        self.assertEqual(adapted["tables"][0]["rows"][1]["cells"][1]["text"], "42")

    def test_adapts_document_without_stage_argument(self):
        adapted = adapt_stage_document(
            stage_doc("GenericDocument", [("A-01", "5.00")])
        )

        self.assertEqual(adapted["table_count"], 1)
        self.assertIsNone(adapted["stage"])

        table = adapted["tables"][0]
        cell = table["rows"][1]["cells"][1]

        self.assertEqual(cell["text"], "5.00")
        self.assertIsNone(cell["provenance"]["stage"])
        self.assertEqual(
            cell["provenance"]["verification_status"],
            "UNVERIFIED",
        )

    def test_missing_page_or_table_structure_fails_explicitly(self):
        with self.assertRaises(ValueError):
            adapt_stage_document({"document": "NoTables"}, stage="3")


if __name__ == "__main__":
    unittest.main()
