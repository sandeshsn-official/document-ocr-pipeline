import importlib.util
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parent


def load_stage(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, SRC / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE_13C = load_stage(
    "13c_entity_linking.py", "stage_13c_under_test"
)
STAGE_13D = load_stage(
    "13d_entity_retrieval.py", "stage_13d_under_test"
)
STAGE_13E = load_stage(
    "13e_query_api.py", "stage_13e_under_test"
)
STAGE_13F = load_stage(
    "13f_cross_document_query.py", "stage_13f_under_test"
)
STAGE_13G = load_stage(
    "13g_generic_knowledge_query.py", "stage_13g_under_test"
)


class Stage13HistoricalStatusIntegrationTests(unittest.TestCase):
    def test_historical_and_current_status_survive_13c_and_13d(self):
        knowledge_index = {
            "records": [
                {
                    "record_id": "record-old",
                    "document_id": "doc-old",
                    "source_status": "SOURCE_MISSING_HISTORICAL",
                    "page": 1,
                    "table_id": "table-old",
                    "table_index": 1,
                    "source_row_index": 2,
                    "fields": {
                        "column_1": {
                            "column_index": 1,
                            "header": "Sensor ID",
                            "value": "SEN-OLD",
                            "normalized_value": "sen-old",
                        },
                    },
                },
                {
                    "record_id": "record-current",
                    "document_id": "doc-current",
                    "source_status": "CURRENT",
                    "page": 1,
                    "table_id": "table-current",
                    "table_index": 1,
                    "source_row_index": 2,
                    "fields": {
                        "column_1": {
                            "column_index": 1,
                            "header": "Sensor ID",
                            "value": "SEN-CURRENT",
                            "normalized_value": "sen-current",
                        },
                    },
                },
            ]
        }

        entity_candidates = {
            "entity_candidates": [
                {
                    "entity_candidate_id": "candidate-old",
                    "canonical_value": "SEN-OLD",
                    "display_value": "SEN-OLD",
                    "occurrences": [
                        {
                            "record_id": "record-old",
                            "document_id": "doc-old",
                            "source_status": "SOURCE_MISSING_HISTORICAL",
                            "column_index": 1,
                            "value": "SEN-OLD",
                            "normalized_value": "sen-old",
                            "page": 1,
                            "table_index": 1,
                            "source_row_index": 2,
                        }
                    ],
                },
                {
                    "entity_candidate_id": "candidate-current",
                    "canonical_value": "SEN-CURRENT",
                    "display_value": "SEN-CURRENT",
                    "occurrences": [
                        {
                            "record_id": "record-current",
                            "document_id": "doc-current",
                            "source_status": "CURRENT",
                            "column_index": 1,
                            "value": "SEN-CURRENT",
                            "normalized_value": "sen-current",
                            "page": 1,
                            "table_index": 1,
                            "source_row_index": 2,
                        }
                    ],
                },
            ]
        }

        linked_entities = STAGE_13C.build_entities(
            knowledge_index, entity_candidates
        )
        entity_links = {"entities": linked_entities}

        record_map = STAGE_13D.build_record_map(knowledge_index)
        retrieved_entities, exact_lookup = (
            STAGE_13D.build_retrieval_entities(
                entity_links, record_map
            )
        )

        retrieved_by_value = {
            entity["canonical_value"]: entity
            for entity in retrieved_entities
        }

        old_record = retrieved_by_value["sen-old"]["records"][0]
        current_record = retrieved_by_value["sen-current"]["records"][0]

        self.assertEqual(
            old_record["source_status"],
            "SOURCE_MISSING_HISTORICAL",
        )
        self.assertEqual(old_record["fields"][0]["value"], "SEN-OLD")

        self.assertEqual(current_record["source_status"], "CURRENT")
        self.assertEqual(
            current_record["fields"][0]["value"], "SEN-CURRENT"
        )

        self.assertIn("sen-old", exact_lookup)
        self.assertIn("sen-current", exact_lookup)

        retrieval_index = {"entities": retrieved_entities}

        old_query = STAGE_13E.query_entity(
            retrieval_index, "SEN-OLD"
        )
        current_query = STAGE_13E.query_entity(
            retrieval_index, "SEN-CURRENT"
        )

        self.assertTrue(old_query["found"])
        self.assertTrue(current_query["found"])

        queried_old_record = old_query["entities"][0]["records"][0]
        queried_current_record = current_query["entities"][0]["records"][0]

        self.assertEqual(
            queried_old_record["source_status"],
            "SOURCE_MISSING_HISTORICAL",
        )
        self.assertEqual(
            queried_current_record["source_status"],
            "CURRENT",
        )
        self.assertEqual(
            queried_old_record["fields"][0]["value"],
            "SEN-OLD",
        )
        self.assertEqual(
            queried_current_record["fields"][0]["value"],
            "SEN-CURRENT",
        )

        cross_document_index = STAGE_13F.build_cross_document_index(
            "synthetic-test-job",
            entity_links,
            retrieval_index,
        )

        cross_by_value = {
            entity["normalized_value"]: entity
            for entity in cross_document_index["entities"]
        }

        old_occurrences = cross_by_value["sen-old"]["occurrences"]
        current_occurrences = cross_by_value["sen-current"]["occurrences"]

        self.assertEqual(len(old_occurrences), 1)
        self.assertEqual(len(current_occurrences), 1)
        self.assertEqual(
            old_occurrences[0]["source_status"],
            "SOURCE_MISSING_HISTORICAL",
        )
        self.assertEqual(
            current_occurrences[0]["source_status"],
            "CURRENT",
        )
        self.assertEqual(
            old_occurrences[0]["fields"]["column_1"]["value"],
            "SEN-OLD",
        )
        self.assertEqual(
            current_occurrences[0]["fields"]["column_1"]["value"],
            "SEN-CURRENT",
        )

        generic_knowledge_index = {
            "records": [
                {
                    "record_id": "record-old",
                    "document_id": "doc-old",
                    "source_status": "SOURCE_MISSING_HISTORICAL",
                    "page": 1,
                    "fields": {
                        "column_1": {
                            "header": "Sensor ID",
                            "value": "SEN-OLD",
                        },
                    },
                },
                {
                    "record_id": "record-current",
                    "document_id": "doc-current",
                    "source_status": "CURRENT",
                    "page": 1,
                    "fields": {
                        "column_1": {
                            "header": "Sensor ID",
                            "value": "SEN-CURRENT",
                        },
                    },
                },
            ],
        }

        old_generic_result = STAGE_13G.query_index(
            cross_document_index,
            generic_knowledge_index,
            "SEN-OLD",
        )
        current_generic_result = STAGE_13G.query_index(
            cross_document_index,
            generic_knowledge_index,
            "SEN-CURRENT",
        )

        self.assertEqual(old_generic_result["match_type"], "EXACT_VALUE")
        self.assertEqual(current_generic_result["match_type"], "EXACT_VALUE")

        old_generic_occurrence = (
            old_generic_result["results"][0]["occurrences"][0]
        )
        current_generic_occurrence = (
            current_generic_result["results"][0]["occurrences"][0]
        )

        self.assertEqual(
            old_generic_occurrence["source_status"],
            "SOURCE_MISSING_HISTORICAL",
        )
        self.assertEqual(
            current_generic_occurrence["source_status"],
            "CURRENT",
        )
        self.assertEqual(
            old_generic_occurrence["fields"]["column_1"]["value"],
            "SEN-OLD",
        )
        self.assertEqual(
            current_generic_occurrence["fields"]["column_1"]["value"],
            "SEN-CURRENT",
        )


if __name__ == "__main__":
    unittest.main()
