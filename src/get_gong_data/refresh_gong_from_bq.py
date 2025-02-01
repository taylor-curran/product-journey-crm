# src/get_gong_data/refresh_gong_from_bq.py
import os
from typing import Dict, List

import turbopuffer as tpuf
from dotenv import load_dotenv
from google.cloud import bigquery
from helper import (
    clean_attributes_for_row,
    embed_text,
    process_combined_transcript,
    chunk_text,
)
from queries import attributes, transcript_query


def fetch_transcripts_from_bigquery(limit):
    load_dotenv()
    gcp_project_id = os.getenv("GCP_PROJECT_ID")

    client = bigquery.Client(project=gcp_project_id)

    query = transcript_query + f"LIMIT {limit};"
    rows = client.query_and_wait(query)

    return rows


def process_and_embed_transcripts(
    rows: List[Dict], chunk_size: int = 2000, overlap: int = 200
) -> Dict:
    """
    Processes each row to embed the combined_transcript and prepares data for upsert.
    """
    doc_ids: List[str] = []
    doc_vectors: List[List[float]] = []
    upsert_attributes = {key: [] for key in attributes.keys()}

    for row in rows:
        call_id = row.get("gong_call_id_c")
        call_title = row.get("name")

        # Skip if call duration is less than 10 seconds
        if row.get("gong_call_duration_sec_c") < 10:
            print(
                f"Skipping call {call_title}-{call_id} with call duration less than 10 seconds."
            )
            continue

        # Process transcript using the new helper function
        combined_transcript = row.get("combined_transcript", "")
        entire_transcript_text = process_combined_transcript(
            combined_transcript, call_title, call_id
        )

        # Chunk the transcript text.
        chunks = chunk_text(
            entire_transcript_text, chunk_size=chunk_size, overlap=overlap
        )

        # Process each chunk.
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            # Create a stable chunk ID: original call_id plus a chunk index.
            chunk_id = f"{call_id}-{idx}"
            vector = embed_text(chunk)
            if not vector:
                print(f"Embedding failed for call {call_title}-{call_id} chunk {idx}.")
                continue

        # Append the call_id and vector to the lists
        doc_ids.append(chunk_id)
        doc_vectors.append(vector)

        # Clean attributes for this row to meet tpuf attribute dtype requirements
        cleaned_attrs = clean_attributes_for_row(row, list(upsert_attributes.keys()))
        # And append attributes
        for attr_key, value in cleaned_attrs.items():
            upsert_attributes[attr_key].append(value)

        if "chunk_index" not in upsert_attributes:
            upsert_attributes["chunk_index"] = []
        upsert_attributes["chunk_index"].append(f"{idx} of {len(chunks)}")

    return {
        "doc_ids": doc_ids,
        "doc_vectors": doc_vectors,
        "attributes": upsert_attributes,
    }


def upsert_to_tpuf(
    namespace: str, doc_ids: List[str], doc_vectors: List[List[float]], attributes: Dict
):
    load_dotenv()  # TODO: Ask nate -- do i need this everywhere

    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"
    ns = tpuf.Namespace(namespace)

    ns.upsert(ids=doc_ids, vectors=doc_vectors, attributes=attributes)


def refresh_gong_transcripts(
    limit: int = 100,
    namespace: str = "tay-test",
    chunk_size: int = 2000,
    overlap: int = 200,
):
    """
    Get the transcript data from Gong
    """
    rows = fetch_transcripts_from_bigquery(limit)
    vector_and_attributes = process_and_embed_transcripts(rows)
    upsert_to_tpuf(namespace, **vector_and_attributes)


if __name__ == "__main__":
    refresh_gong_transcripts(2)
