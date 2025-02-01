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
from prefect import task, flow
from prefect.cache_policies import TASK_SOURCE, INPUTS


@task
def fetch_transcripts_from_bigquery(limit_n_calls):
    load_dotenv()
    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    client = bigquery.Client(project=gcp_project_id)

    if limit_n_calls == 0:
        query = transcript_query + ";"
    else:
        query = transcript_query + f"LIMIT {limit_n_calls};"

    rows = client.query_and_wait(query)

    return list(rows)


@task
def process_and_embed_transcripts(
    rows: List[Dict], chunk_size: int = 2000, overlap: int = 200
) -> Dict:
    """
    Processes each row to embed the combined_transcript (chunked) and prepares data for upsert.
    """
    doc_ids: List[str] = []
    doc_vectors: List[List[float]] = []
    # Start with the original attribute keys; note that "chunk_index" is not in attributes.
    upsert_attributes = {key: [] for key in attributes.keys()}
    # Initialize a new key for the chunk index.
    upsert_attributes["chunk_index"] = []
    upsert_attributes["transcript_text"] = []

    for row in rows:
        call_id = row.get("gong_call_id_c")
        call_title = row.get("name")
        print(f"💼 Processing call {call_title}-{call_id}")

        # Skip if call duration is less than 10 seconds
        if row.get("gong_call_duration_sec_c") < 10:
            print(
                f"Skipping call {call_title}-{call_id} with duration less than 10 sec."
            )
            continue

        # Process transcript using the helper function
        combined_transcript = row.get("combined_transcript", "")
        entire_transcript_text = process_combined_transcript(
            combined_transcript, call_title, call_id
        )

        # Chunk the transcript text.
        chunks = chunk_text(
            entire_transcript_text, chunk_size=chunk_size, overlap=overlap
        )

        # Compute the cleaned attributes once per row (they’re the same for every chunk)
        cleaned_attrs = clean_attributes_for_row(row, list(attributes.keys()))

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

            # Append each chunk's data.
            doc_ids.append(chunk_id)
            doc_vectors.append(vector)
            for attr_key, value in cleaned_attrs.items():
                upsert_attributes[attr_key].append(value)
            upsert_attributes["chunk_index"].append(f"-{idx}- of {len(chunks)}")
            upsert_attributes["transcript_text"].append(chunk)

    return {
        "doc_ids": doc_ids,
        "doc_vectors": doc_vectors,
        "attributes": upsert_attributes,
    }


@task
def batch_upsert(
    namespace: str,
    doc_ids: List[str],
    doc_vectors: List[List[float]],
    attributes: Dict,
    batch_size: int = 50,
):
    """
    Upsert documents in smaller batches.
    """
    load_dotenv()
    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"
    ns = tpuf.Namespace(namespace)

    for i in range(0, len(doc_ids), batch_size):
        print(
            f"🐡 Upserting batch {i // batch_size + 1} of {(len(doc_ids) // batch_size) + 1}"
        )
        batch_ids = doc_ids[i : i + batch_size]
        batch_vectors = doc_vectors[i : i + batch_size]
        # For attributes, slice each list so that every attribute list is the same length as the batch.
        batch_attributes = {k: v[i : i + batch_size] for k, v in attributes.items()}
        ns = tpuf.Namespace(namespace)
        ns.upsert(ids=batch_ids, vectors=batch_vectors, attributes=batch_attributes)


@flow(log_prints=True, persist_result=False)
def refresh_gong_transcripts(
    namespace: str = "tay-test",
    limit_n_calls: int = 50,
    chunk_size: int = 2000,
    overlap: int = 200,
):
    """
    Get the transcript data from Gong
    """
    rows = fetch_transcripts_from_bigquery(limit_n_calls)
    vector_and_attributes = process_and_embed_transcripts(rows)
    batch_upsert(
        namespace,
        vector_and_attributes["doc_ids"],
        vector_and_attributes["doc_vectors"],
        vector_and_attributes["attributes"],
    )


if __name__ == "__main__":
    # refresh_gong_transcripts(namespace="tay-sales-calls", limit_n_calls=0)
    refresh_gong_transcripts(namespace="tay-test", limit_n_calls=20)
