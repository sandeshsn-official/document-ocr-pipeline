"""

Stage 13G - Generic Knowledge Query Layer



Purpose

\-------

Provide a deterministic, reusable query interface over the Stage 13F

cross-document aggregation index.



Supported query forms include:

    A-01

    "A-01"

    show everything about A-01

    everything about A-01

    details for A-01

    information on A-01



The query layer first resolves an exact normalized entity value. It then

returns all linked occurrences and their complete fields/provenance.



No fuzzy matching, embeddings, vector search, semantic linking, or LLM

reasoning is performed here.

"""



from __future__ import annotations



import argparse

import json

import re

from datetime import datetime, timezone

from pathlib import Path

from typing import Any





SCHEMA_VERSION = "13g.1"





def utc_now() -> str:

    return datetime.now(timezone.utc).isoformat()





def normalize(value: Any) -> str:

    if value is None:

        return ""

    text = str(value).strip().casefold()

    text = re.sub(r"\s+", " ", text)

    return text





def load_json(path: Path) -> dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as f:

        return json.load(f)





def knowledge_dir(job_id: str, project_root: Path) -> Path:

    return (

        project_root

        / "output"

        / "jobs"

        / job_id

        / "output"

        / "knowledge"

    )





def extract_query_value(query: str) -> str:

    """

    Deterministically remove common command-like wrappers while preserving

    the actual entity value. This is deliberately conservative.



    If no known wrapper is present, the complete query is treated as the

    candidate entity value.

    """

    text = str(query).strip()



    # Remove matching surrounding quotes.

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:

        return text[1:-1].strip()



    patterns = [

        r"^\s*show\s+everything\s+about\s+(.+?)\s*$",

        r"^\s*find\s+all\s+information\s+about\s+(.+?)\s*$",

        r"^\s*find\s+information\s+about\s+(.+?)\s*$",
        r"^\s*all\s+information\s+about\s+(.+?)\s*$",

        r"^\s*show\s+all\s+about\s+(.+?)\s*$",

        r"^\s*everything\s+about\s+(.+?)\s*$",

        r"^\s*details\s+for\s+(.+?)\s*$",

        r"^\s*information\s+on\s+(.+?)\s*$",

        r"^\s*information\s+about\s+(.+?)\s*$",

        r"^\s*details\s+about\s+(.+?)\s*$",

        r"^\s*data\s+for\s+(.+?)\s*$",

    ]



    for pattern in patterns:

        match = re.match(pattern, text, flags=re.IGNORECASE)

        if match:

            value = match.group(1).strip()

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:

                value = value[1:-1].strip()

            return value



    return text





def iter_entities(index: dict[str, Any]):

    entities = index.get("entities", [])



    if isinstance(entities, dict):

        for entity_id, entity in entities.items():

            if isinstance(entity, dict):

                item = dict(entity)

                item.setdefault("entity_id", entity_id)

                yield item

        return



    if isinstance(entities, list):

        for entity in entities:

            if isinstance(entity, dict):

                yield entity





def build_entity_maps(index: dict[str, Any]):

    by_id: dict[str, dict[str, Any]] = {}

    by_normalized: dict[str, list[dict[str, Any]]] = {}



    for entity in iter_entities(index):

        entity_id = entity.get("entity_id")

        if not entity_id:

            continue



        entity_id = str(entity_id)

        by_id[entity_id] = entity



        normalized = normalize(

            entity.get("normalized_value")

            or entity.get("canonical_value")

            or entity.get("display_value")

            or ""

        )



        if normalized:

            by_normalized.setdefault(normalized, []).append(entity)



    return by_id, by_normalized





def query_knowledge_values(

    knowledge_index,

    query,

):

    """

    Generic deterministic value lookup across every indexed record field.



    This is intentionally independent of entity candidates. Any value

    present in the knowledge index can be queried.



    No domain-specific field names, column counts, or identifier assumptions.

    """



    query_text = str(query or "").strip()



    if not query_text:

        return []



    query_normalized = normalize(query_text)



    matches = []



    records = knowledge_index.get(

        "records",

        [],

    )



    if not isinstance(records, list):

        return []



    for record in records:



        if not isinstance(record, dict):

            continue



        fields = record.get(

            "fields",

            {},

        )



        if isinstance(fields, dict):

            field_items = fields.items()



        elif isinstance(fields, list):

            field_items = []



            for position, field in enumerate(fields):



                if not isinstance(field, dict):

                    continue



                field_key = (

                    field.get("field_key")

                    or field.get("key")

                    or f"field_{position + 1}"

                )



                field_items.append(

                    (

                        field_key,

                        field,

                    )

                )



        else:

            continue



        matched_fields = []



        for field_key, field in field_items:



            if not isinstance(field, dict):

                continue



            value = field.get(

                "value"

            )



            if value is None:

                value = field.get(

                    "text"

                )



            if value is None:

                continue



            value_text = str(

                value

            ).strip()



            if not value_text:

                continue



            value_normalized = normalize(value_text)



            if value_normalized != query_normalized:

                continue



            matched_fields.append(

                {

                    "field_key": field_key,

                    **field,

                }

            )



        if not matched_fields:

            continue



        occurrence = {

            "record_id": record.get(

                "record_id"

            ),

            "document_id": record.get(

                "document_id"

            ),

            "page": record.get(

                "page"

            ),

            "table_id": record.get(

                "table_id"

            ),

            "table_index": record.get(

                "table_index"

            ),

            "source_row_index": record.get(

                "source_row_index"

            ),

            "source_row_value": record.get(

                "source_row_value"

            ),

            "identifier_candidates": record.get(

                "identifier_candidates",

                [],

            ),

            "fields": fields,

            "matched_fields": matched_fields,

            "provenance": record.get(

                "provenance",

                {},

            ),

        }



        matches.append(

            occurrence

        )



    return matches





def query_index(
    index: dict[str, Any],
    knowledge_index: dict[str, Any],
    query: str,
) -> dict[str, Any]:

    requested_value = extract_query_value(query)
    normalized_query = normalize(requested_value)

    by_id, by_normalized = build_entity_maps(index)

    # Exact entity lookup first.
    if requested_value in by_id:
        matches = [by_id[requested_value]]
        match_type = "EXACT_ENTITY_ID"
    else:
        matches = by_normalized.get(
            normalized_query,
            [],
        )
        match_type = (
            "EXACT_VALUE"
            if matches
            else "NO_MATCH"
        )

    results = []

    for entity in matches:

        occurrences = entity.get(
            "occurrences",
            [],
        )

        if not isinstance(occurrences, list):
            occurrences = []

        documents = []
        tables = []

        for occurrence in occurrences:

            if not isinstance(occurrence, dict):
                continue

            document_id = occurrence.get("document_id")

            if (
                document_id is not None
                and document_id not in documents
            ):
                documents.append(document_id)

            table_key = {
                "document_id": occurrence.get("document_id"),
                "table_id": occurrence.get("table_id"),
                "table_index": occurrence.get("table_index"),
            }

            if table_key not in tables:
                tables.append(table_key)

        results.append(
            {
                "entity_id": entity.get("entity_id"),
                "canonical_value": entity.get("canonical_value"),
                "display_value": entity.get("display_value"),
                "normalized_value": entity.get("normalized_value"),
                "link_status": entity.get("link_status"),
                "cross_document": entity.get(
                    "cross_document",
                    len(documents) > 1,
                ),
                "statistics": {
                    "occurrence_count": len(occurrences),
                    "document_count": len(documents),
                    "table_count": len(tables),
                    "documents": documents,
                },
                "occurrences": occurrences,
            }
        )

    # Generic exact-value fallback. This searches every record and
    # every field dynamically; it does not depend on entity candidates.
    if not results:

        value_matches = query_knowledge_values(
            knowledge_index,
            requested_value,
        )

        if value_matches:

            documents = []
            tables = []

            for occurrence in value_matches:

                document_id = occurrence.get("document_id")

                if (
                    document_id is not None
                    and document_id not in documents
                ):
                    documents.append(document_id)

                table_key = {
                    "document_id": occurrence.get("document_id"),
                    "table_id": occurrence.get("table_id"),
                    "table_index": occurrence.get("table_index"),
                }

                if table_key not in tables:
                    tables.append(table_key)

            results = [
                {
                    "entity_id": None,
                    "canonical_value": normalized_query,
                    "display_value": requested_value,
                    "normalized_value": normalized_query,
                    "link_status": "value_match",
                    "cross_document": len(documents) > 1,
                    "statistics": {
                        "occurrence_count": len(value_matches),
                        "document_count": len(documents),
                        "table_count": len(tables),
                        "documents": documents,
                    },
                    "occurrences": value_matches,
                }
            ]

            match_type = "VALUE_MATCH"

    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "13G",
        "stage_name": "Generic Knowledge Query Layer",
        "generated_at_utc": utc_now(),
        "method": "deterministic_exact_entity_or_value_query",
        "llm_used": False,
        "semantic_search_performed": False,
        "query": {
            "original": query,
            "extracted_value": requested_value,
            "normalized_value": normalized_query,
        },
        "match_type": match_type,
        "match_count": len(results),
        "results": results,
    }


def print_result(result: dict[str, Any]) -> None:

    query = result["query"]

    print("=" * 72)

    print("STAGE 13G - GENERIC KNOWLEDGE QUERY")

    print("=" * 72)

    print(f"Query:          {query['original']}")

    print(f"Entity value:   {query['extracted_value']}")

    print(f"Match type:     {result['match_type']}")

    print(f"Matches:        {result['match_count']}")

    print()



    if not result["results"]:

        print("No exact entity or value match found.")

        print("13G does not perform fuzzy or semantic matching.")

        return



    for entity in result["results"]:

        print(f"Entity:         {entity.get('display_value')}")

        print(f"Entity ID:      {entity.get('entity_id')}")

        print(f"Link status:    {entity.get('link_status')}")

        print(f"Cross-document: {entity.get('cross_document')}")

        print(

            f"Documents:      "

            f"{entity['statistics']['document_count']}"

        )

        print(

            f"Occurrences:    "

            f"{entity['statistics']['occurrence_count']}"

        )

        print()



        for occurrence in entity.get("occurrences", []):

            print(

                f"Document={occurrence.get('document_id')} | "

                f"Page={occurrence.get('page')} | "

                f"Table={occurrence.get('table_index')} | "

                f"Row={occurrence.get('source_row_index')}"

            )



            fields = occurrence.get("fields", {})

            if isinstance(fields, dict):

                for field_key, field in fields.items():

                    if not isinstance(field, dict):

                        continue



                    header = field.get("header", field_key)

                    value = field.get("value", "")

                    column = field.get("column_index")



                    verification_status = field.get("verification_status")
                    verification_reason = (
                        field.get("verification_reason")
                        or field.get("reason")
                    )
                    source_ocr_confidence = field.get("source_ocr_confidence")

                    # Some producers keep verification metadata under provenance.
                    provenance = field.get("provenance", {})
                    if isinstance(provenance, dict):
                        if verification_status is None:
                            verification_status = provenance.get("verification_status")
                        if verification_reason is None:
                            verification_reason = (
                                provenance.get("verification_reason")
                                or provenance.get("reason")
                            )
                        if source_ocr_confidence is None:
                            source_ocr_confidence = provenance.get("source_ocr_confidence")

                    print(

                        f"  {header!r} = {value!r} "

                        f"(column={column})"

                    )

                    # Surface recorded verification metadata; never infer a
                    # status when the source data does not provide one.
                    if verification_status is not None:
                        print(f"    Verification: {verification_status}")
                    if source_ocr_confidence is not None:
                        print(f"    Source OCR confidence: {source_ocr_confidence}")
                    if verification_reason is not None:
                        print(f"    Verification note: {verification_reason}")



            print()





def main() -> int:

    parser = argparse.ArgumentParser(

        description="Stage 13G - Generic deterministic knowledge query"

    )

    parser.add_argument("--job-id", required=True)

    parser.add_argument("--query", required=True)

    parser.add_argument(

        "--project-root",

        default=".",

        help="OCR project root containing output/jobs",

    )

    parser.add_argument(

        "--output",

        default=None,

        help="Optional JSON result path",

    )



    args = parser.parse_args()



    project_root = Path(args.project_root).resolve()

    directory = knowledge_dir(args.job_id, project_root)



    index_path = directory / "cross_document_query_index.json"
    index = load_json(index_path)

    knowledge_index_path = (
        directory / "knowledge_index.json"
    )
    knowledge_index = load_json(
        knowledge_index_path
    )

    result = query_index(
        index,
        knowledge_index,
        args.query,
    )

    print_result(result)



    if args.output:

        output_path = Path(args.output).resolve()

        output_path.parent.mkdir(parents=True, exist_ok=True)



        with output_path.open("w", encoding="utf-8") as f:

            json.dump(result, f, indent=2, ensure_ascii=False)



        print(f"JSON output:    {output_path}")



    return 0





if __name__ == "__main__":

    raise SystemExit(main())


