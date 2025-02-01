# src/get_gong_data/refresh_gong_from_bq.py
from google.cloud import bigquery
from dotenv import load_dotenv
import os
from queries import transcript_query, attributes
from helper import (
    chunk_text,
    embed_text,
    clean_attribute_value,
    clean_attributes_for_row,
)
from typing import List, Dict
import json
import uuid
from openai import OpenAI
import turbopuffer as tpuf


def fetch_transcripts_from_bigquery(limit):
    load_dotenv()
    gcp_project_id = os.getenv("GCP_PROJECT_ID")

    client = bigquery.Client(project=gcp_project_id)

    query = transcript_query + f"LIMIT {limit};"
    rows = client.query_and_wait(query)

    return rows


def process_and_embed_transcripts(rows: List[Dict]) -> Dict:
    """
    Processes each row to embed the combined_transcript and prepares data for upsert.
    """
    doc_ids: List[str] = []
    doc_vectors: List[List[float]] = []
    upsert_attributes = {key: [] for key in attributes.keys()}

    for row in rows:
        call_id = row.get("gong_call_id_c")
        call_title = row.get("name")

        # Process transcript
        combined_transcript = row.get("combined_transcript", "")
        if not combined_transcript:
            print(f"Skipping call {call_title} with empty combined_transcript.")
            continue

        try:
            transcript_list = json.loads(combined_transcript)
            entire_transcript_text = " ".join(
                item.get("text", "") for item in transcript_list
            )
        except json.JSONDecodeError as e:
            print(
                f"Error parsing combined_transcript for call {call_title}-{call_id}: {e}"
            )
            continue

        if not entire_transcript_text.strip() or len(entire_transcript_text) < 10:
            print(
                f"Skipping call {call_title}-{call_id} with empty transcript text after parsing."
            )
            continue
        if row.get("gong_call_duration_sec_c") < 10:
            print(
                f"Skipping call {call_title}-{call_id} with call duration less than 10 seconds."
            )
            continue

        # TODO: Chunk the text
        # chunks = chunk_text(entire_text, chunk_size=2000)
        # for idx, chunk in enumerate(chunks):
        #     if not chunk.strip():
        #         continue
        #     doc_id = f"{call_title}-{call_id}-{idx}"
        #     vector = embed_text(chunk)
        #     if not vector:
        #         continue
        #     doc_ids.append(doc_id)
        #     doc_vectors.append(vector)

        vector = embed_text(entire_transcript_text)
        if not vector:
            print(f"Embedding failed for call {call_title}-{call_id}.")
            continue

        # Create a unique document id.
        doc_id = call_id
        doc_ids.append(doc_id)
        doc_vectors.append(vector)

        # Clean attributes for this row.
        cleaned_attrs = clean_attributes_for_row(row, list(upsert_attributes.keys()))
        # Append each cleaned value into the column-oriented attributes dict.
        for attr_key, value in cleaned_attrs.items():
            upsert_attributes[attr_key].append(value)

    return {
        "doc_ids": doc_ids,
        "doc_vectors": doc_vectors,
        "attributes": upsert_attributes,
    }


def upsert_to_tpuf(
    namespace: str, doc_ids: List[str], doc_vectors: List[List[float]], attributes: Dict
):
    load_dotenv()

    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"
    ns = tpuf.Namespace(namespace)

    ns.upsert(ids=doc_ids, vectors=doc_vectors, attributes=attributes)


def refresh_gong_transcripts(
    limit: int = 100, chunk_size: int = 2000, namespace: str = "tay-test"
):
    """
    Get the transcript data from Gong
    """
    rows = fetch_transcripts_from_bigquery(limit)
    vector_and_attributes = process_and_embed_transcripts(rows)
    upsert_to_tpuf(namespace, **vector_and_attributes)


if __name__ == "__main__":
    refresh_gong_transcripts(2)
