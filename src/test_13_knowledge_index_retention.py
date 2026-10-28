import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("13_build_knowledge_index.py")
SPEC = importlib.util.spec_from_file_location(
    "knowledge_index_retention_under_test",
    MODULE_PATH,
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def make_document(document_id):
    return {
        "document_id": document_id,
        "pages": [{"page": 1}],
    }


def make_record(document_id, record_id, value):
    return {
        "record_id": record_id,
        "document_id": document_id,
        "page": 1,
        "table_id": f"table-{document_id}",
        "table_index": 1,
        "source_row_index": 1,
        "fields": {
            "column_1": {
                "column_index": 1,
                "header": "Test ID",
                "value": value,
                "normalized_value": value.casefold(),
            }
        },
    }


def make_table(document_id, table_id):
    return {
        "document_id": document_id,
        "table_id": table_id,
        "table_index": 1,
        "header_detection": {"detected": True},
        "cell_count": 1,
    }


def make_value_occurrence(document_id, table_id, row_index, value):
    return {
        "document_id": document_id,
        "page": 1,
        "table_id": table_id,
        "table_index": 1,
        "source_row_index": row_index,
        "value": value,
    }


def make_index(documents):
    index = {
        "documents": [],
        "tables": [],
        "records": [],
        "text_items": [],
        "value_index": {},
    }

    for document_id, records, text_items in documents:
        index["documents"].append(make_document(document_id))

        for table_id in sorted({
            record["table_id"] for record in records
        }):
            index["tables"].append(make_table(document_id, table_id))

        index["records"].extend(records)
        index["text_items"].extend(text_items)

        for record in records:
            for field in record.get("fields", {}).values():
                value = field.get("value")
                if value is None:
                    continue
                key = str(value).casefold()
                index["value_index"].setdefault(key, []).append(
                    make_value_occurrence(
                        document_id,
                        record["table_id"],
                        record.get("source_row_index", 1),
                        str(value),
                    )
                )

    return index


class KnowledgeIndexRetentionTests(unittest.TestCase):

    def test_missing_document_retains_all_evidence_and_current_document(self):
        old_paragraph = {
            "text_item_id": "text-old-1",
            "document_id": "doc-old",
            "document": "Old_Report",
            "source_pdf": "Old_Report.pdf",
            "kind": "paragraph",
            "page": 1,
            "text": "Historical paragraph evidence",
            "searchable_text": "historical paragraph evidence",
            "verification_status": "verified",
            "provenance": {
                "source_file": "Old_Report.json",
                "source_field": "paragraph.text",
            },
        }
        old_text_block = {
            "text_item_id": "block-old-1",
            "document_id": "doc-old",
            "document": "Old_Report",
            "source_pdf": "Old_Report.pdf",
            "kind": "text_block",
            "page": 1,
            "block_index": 2,
            "original_text": "Original OCR text",
            "verified_text": "Verified historical text",
            "final_verified_text": "Verified historical text",
            "searchable_text": "verified historical text",
            "verification_status": "verified",
            "provenance": {
                "source_file": "Old_Report.json",
                "source_field": "text_block.final_verified_text",
                "block_index": 2,
            },
        }
        old_records = [
            make_record("doc-old", "old-record-1", "OLD-A"),
            make_record("doc-old", "old-record-2", "OLD-B"),
        ]
        old_records[1]["table_id"] = "table-doc-old-2"

        previous = make_index([
            ("doc-old", old_records, [old_paragraph, old_text_block]),
            (
                "doc-current",
                [make_record("doc-current", "current-record-1", "LIVE-X")],
                [],
            ),
        ])

        current = make_index([
            (
                "doc-current",
                [make_record("doc-current", "current-record-1", "LIVE-X")],
                [],
            )
        ])

        merged = BUILDER.merge_knowledge_indexes(previous, current)

        docs = {
            item["document_id"]: item
            for item in merged["documents"]
        }
        self.assertEqual(
            docs["doc-old"]["source_status"],
            "SOURCE_MISSING_HISTORICAL",
        )
        self.assertEqual(docs["doc-current"]["source_status"], "CURRENT")

        old_records_after = [
            item for item in merged["records"]
            if item["document_id"] == "doc-old"
        ]
        self.assertEqual(len(old_records_after), 2)
        self.assertTrue(all(
            item["source_status"] == "SOURCE_MISSING_HISTORICAL"
            for item in old_records_after
        ))

        old_tables = [
            item for item in merged["tables"]
            if item["document_id"] == "doc-old"
        ]
        self.assertEqual(len(old_tables), 2)
        self.assertTrue(all(
            item["source_status"] == "SOURCE_MISSING_HISTORICAL"
            for item in old_tables
        ))

        self.assertEqual(len(merged["text_items"]), 2)

        text_items_by_id = {
            item["text_item_id"]: item
            for item in merged["text_items"]
        }
        paragraph_after = text_items_by_id["text-old-1"]
        block_after = text_items_by_id["block-old-1"]

        for item in (paragraph_after, block_after):
            self.assertEqual(
                item["source_status"],
                "SOURCE_MISSING_HISTORICAL",
            )
            self.assertEqual(item["document_id"], "doc-old")
            self.assertEqual(
                item["provenance"]["source_file"],
                "Old_Report.json",
            )

        self.assertEqual(
            paragraph_after["searchable_text"],
            "historical paragraph evidence",
        )
        self.assertEqual(
            paragraph_after["provenance"]["source_field"],
            "paragraph.text",
        )
        self.assertEqual(
            block_after["final_verified_text"],
            "Verified historical text",
        )
        self.assertEqual(block_after["block_index"], 2)
        self.assertEqual(
            block_after["provenance"]["source_field"],
            "text_block.final_verified_text",
        )

        self.assertEqual(
            merged["value_index"]["old-a"][0]["source_status"],
            "SOURCE_MISSING_HISTORICAL",
        )
        self.assertEqual(
            merged["value_index"]["live-x"][0]["source_status"],
            "CURRENT",
        )

    def test_repeated_empty_source_merge_does_not_duplicate_evidence(self):
        records = [
            make_record("doc-old", "record-1", "A"),
            make_record("doc-old", "record-2", "B"),
        ]
        original = make_index([
            (
                "doc-old",
                records,
                [{
                    "text_item_id": "text-1",
                    "document_id": "doc-old",
                    "page": 1,
                    "text": "Paragraph",
                }],
            )
        ])

        once = BUILDER.merge_knowledge_indexes(original, None)
        twice = BUILDER.merge_knowledge_indexes(once, None)

        for section in ("documents", "tables", "records", "text_items"):
            self.assertEqual(
                len(twice[section]),
                len(once[section]),
                f"Unexpected duplication in {section}",
            )

        for value in once["value_index"]:
            self.assertEqual(
                len(twice["value_index"][value]),
                len(once["value_index"][value]),
                f"Unexpected value-index duplication for {value}",
            )

    def test_restoration_replaces_old_evidence_and_marks_current(self):
        old = make_index([
            (
                "doc-A",
                [make_record("doc-A", "record-A", "OLD-VALUE")],
                [],
            )
        ])
        historical = BUILDER.merge_knowledge_indexes(old, None)

        fresh = make_index([
            (
                "doc-A",
                [make_record("doc-A", "record-A", "NEW-VALUE")],
                [],
            )
        ])
        restored = BUILDER.merge_knowledge_indexes(historical, fresh)

        self.assertEqual(len(restored["documents"]), 1)
        self.assertEqual(len(restored["records"]), 1)
        self.assertEqual(restored["records"][0]["source_status"], "CURRENT")
        self.assertEqual(
            restored["records"][0]["fields"]["column_1"]["value"],
            "NEW-VALUE",
        )
        self.assertNotIn("old-value", restored["value_index"])
        self.assertEqual(
            restored["value_index"]["new-value"][0]["source_status"],
            "CURRENT",
        )

    def test_two_empty_indexes_are_rejected(self):
        with self.assertRaises(ValueError):
            BUILDER.merge_knowledge_indexes(None, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
