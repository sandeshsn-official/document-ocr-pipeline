import unittest

from comparison_engine import compare_tables


def make_table(table_id, headers, rows, doc="sample"):
    table_rows = [{
        "source_row_index": 1,
        "source_row_value": 1,
        "row_type": "header",
        "cells": [
            {"column": i + 1, "text": h, "normalized_text": h.casefold(),
             "provenance": {"verification_status": "SOURCE_VERIFIED"}}
            for i, h in enumerate(headers)
        ],
    }]
    for row_number, values in enumerate(rows, start=2):
        table_rows.append({
            "source_row_index": row_number,
            "source_row_value": row_number,
            "row_type": "data",
            "cells": [
                {"column": i + 1, "text": value, "normalized_text": str(value).casefold(),
                 "provenance": {
                     "document_id": doc, "page": 1, "table_index": 1,
                     "row": row_number, "column": i + 1,
                     "verification_status": "SOURCE_VERIFIED",
                     "source_ocr_confidence": 0.95,
                 }}
                for i, value in enumerate(values)
            ],
        })
    return {
        "table_id": table_id,
        "table_index": 1,
        "header": headers,
        "rows": table_rows,
        "table_provenance": {"document_id": doc, "page": 1, "table_index": 1},
    }


def make_index(table, identifiers):
    records = []
    for row_index, identifier in identifiers:
        records.append({
            "table_id": table["table_id"],
            "source_row_index": row_index,
            "identifier_candidates": [{
                "header": "Sensor ID", "value": identifier,
                "normalized_value": identifier.casefold(), "score": 1.0,
            }],
        })
    return {"tables": [table], "records": records}


class ComparisonEngineTests(unittest.TestCase):
    def test_reordered_columns_align_by_header(self):
        left = make_table("L", ["Sensor ID", "Temperature"], [["SEN-104", "24.4"]])
        right = make_table("R", ["Temperature", "Sensor ID"], [["25.1", "SEN-104"]])
        li = make_index(left, [(2, "SEN-104")])
        ri = make_index(right, [(2, "SEN-104")])
        result = compare_tables(left, right, left_index=li, right_index=ri)
        row = next(r for r in result["rows"] if r["row_key"] == "sen-104")
        cells = {c["field"]: c["status"] for c in row["cells"]}
        self.assertEqual(cells["Temperature"], "CHANGED")
        self.assertEqual(cells["Sensor ID"], "UNCHANGED")

    def test_uncertain_ocr_is_ambiguous(self):
        left = make_table("L", ["Sensor ID", "Temperature"], [["SEN-104", "24.4"]])
        right = make_table("R", ["Sensor ID", "Temperature"], [["SEN-104", "24.5"]])
        right["rows"][1]["cells"][1]["provenance"]["verification_status"] = "HIGH_RISK_AGREEMENT_REVIEW"
        li = make_index(left, [(2, "SEN-104")])
        ri = make_index(right, [(2, "SEN-104")])
        result = compare_tables(left, right, left_index=li, right_index=ri)
        row = next(r for r in result["rows"] if r["row_key"] == "sen-104")
        temperature = next(c for c in row["cells"] if c["field"] == "Temperature")
        self.assertEqual(temperature["status"], "AMBIGUOUS")
        self.assertEqual(temperature["value_relation"], "DIFFERENT")
        self.assertEqual(temperature["right"]["verification_status"], "HIGH_RISK_AGREEMENT_REVIEW")

    def test_numeric_disagreement_is_ambiguous(self):
        left = make_table(
            "L", ["Sensor ID", "Temperature"], [["SEN-104", "23.8"]]
        )
        right = make_table(
            "R", ["Sensor ID", "Temperature"], [["SEN-104", "24.4"]]
        )
        right["rows"][1]["cells"][1]["provenance"].update({
            "verification_status": "NUMERIC_DISAGREEMENT",
            "verification_reason": "Conflicting OCR values require review.",
            "source_ocr_confidence": 0.555,
        })

        li = make_index(left, [(2, "SEN-104")])
        ri = make_index(right, [(2, "SEN-104")])
        result = compare_tables(left, right, left_index=li, right_index=ri)

        row = next(r for r in result["rows"] if r["row_key"] == "sen-104")
        temperature = next(
            c for c in row["cells"] if c["field"] == "Temperature"
        )

        self.assertEqual(temperature["status"], "AMBIGUOUS")
        self.assertEqual(temperature["value_relation"], "DIFFERENT")
        self.assertTrue(temperature["right"]["verification_uncertain"])
        self.assertEqual(
            temperature["right"]["verification_status"],
            "NUMERIC_DISAGREEMENT",
        )
        self.assertEqual(temperature["left"]["value"], "23.8")
        self.assertEqual(temperature["right"]["value"], "24.4")


    def test_same_value_is_unchanged(self):
        left = make_table("L", ["Sensor ID", "Voltage"], [["SEN-104", "5.01"]])
        right = make_table("R", ["Sensor ID", "Voltage"], [["SEN-104", "5.01"]])
        li = make_index(left, [(2, "SEN-104")])
        ri = make_index(right, [(2, "SEN-104")])
        result = compare_tables(left, right, left_index=li, right_index=ri)
        row = next(r for r in result["rows"] if r["row_key"] == "sen-104")
        voltage = next(c for c in row["cells"] if c["field"] == "Voltage")
        self.assertEqual(voltage["status"], "UNCHANGED")
        self.assertEqual(voltage["value_relation"], "MATCH")



    def test_unkeyed_rows_are_not_matched_by_position(self):
        left = make_table("L", ["Description"], [["alpha"]])
        right = make_table("R", ["Description"], [["beta"]])
        result = compare_tables(left, right)
        self.assertEqual(len(result["rows"]), 2)
        self.assertTrue(all(r["row_status"] in {"ADDED", "REMOVED"} for r in result["rows"]))


if __name__ == "__main__":
    unittest.main()
