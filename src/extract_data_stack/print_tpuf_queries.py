import turbopuffer as tpuf
import os
import datetime
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


def embed_text(text: str) -> List[float]:
    """
    Generates an embedding vector for the provided text using OpenAI's API.
    """

    load_dotenv()

    # Initialize the OpenAI client once
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []


def query_namespace(
    namespace: str = "tay-sales-calls",
    query_text: str = "find me a call where data orchestration was discussed",
    top_k: int = 3,
    include_attributes: list[str] = ["name", "gong_call_id_c"],
    n_characters: int = 500,
    gong_primary_opportunity_c: str = "006Rm00000QuHC6IAN",
) -> list:
    # Convert query into a vector
    query_vector = embed_text(query_text)

    # Configure your API key and base URL
    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"

    # Query the namespace using the vector
    ns = tpuf.Namespace(namespace)

    # Define a filter to restrict results to documents with the specified opportunity.
    filters = ["gong_primary_opportunity_c", "Eq", gong_primary_opportunity_c]

    results = ns.query(
        vector=query_vector,
        distance_metric="cosine_distance",
        top_k=top_k,
        include_attributes=include_attributes,
        filters=filters,
    )

    if n_characters > len(results[0].attributes["transcript_text"]):
        n_characters = len(results[0].attributes["transcript_text"])

    for result in results:
        print("\nResult:")
        print(f"  ID: {result.id}")
        print(f"  Distance: {result.dist:.4f}")
        print("  Attributes:")
        for attr, value in result.attributes.items():
            if attr == "transcript_text":
                print("")
                print("-----")
                print(f"Transcript Text ({n_characters} characters):")
                print("")
                print(f"{value[:n_characters]}...")

                print("")
                print("Last 400 characters:")
                print("")
                print(f"{value[-400:]}")
                print("")
                print(f"Chunk Length: {len(value)}")
                print("-----")
                print("")
            else:
                print(f"    {attr}: {value}")

    return results


def print_namespace_schema(namespace: str = "tay-test"):
    # Configure your API key and base URL
    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"

    # Instantiate your namespace object
    ns = tpuf.Namespace(namespace)

    schema = ns.schema()

    for attr_name, attr_schema in schema.items():
        # Assuming the AttributeSchema objects expose these properties:
        attr_type = getattr(attr_schema, "type", "unknown")
        filterable = getattr(attr_schema, "filterable", None)
        full_text_search = getattr(attr_schema, "full_text_search", None)

        print(f"Attribute: {attr_name}")
        print(f"  Type: {attr_type}")
        print(f"  Filterable: {filterable}")
        print(f"  Full-text search: {full_text_search}")
        print()

    return schema


def get_unique_gong_primary_opportunities(
    namespace: str = "tay-sales-calls",
    top_k: int = 1000,  # adjust based on your dataset size
) -> list:
    """
    Query the vectorstore for documents in the specified namespace,
    and return a list of unique 'gong_primary_opportunity_c' values.
    """
    load_dotenv()

    # Configure the Turbopuffer API
    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"

    # Instantiate the namespace
    ns = tpuf.Namespace(namespace)

    # Query the namespace.
    # Here we don't provide a vector or rank_by, so the API should return
    # documents without reordering by relevance.
    results = ns.query(top_k=top_k, include_attributes=["gong_primary_opportunity_c"])

    # Use a set to collect unique values
    unique_opportunities = set()
    for result in results:
        opportunity = result.attributes.get("gong_primary_opportunity_c")
        if opportunity:
            unique_opportunities.add(opportunity)

    return list(unique_opportunities)


if __name__ == "__main__":
    # schema = print_namespace_schema("tay-test")

    # results = query_namespace(
    #     "tay-sales-calls",
    #     "Find sections that talk about the cloud provider",
    #     top_k=3,
    #     include_attributes=[
    #         "gong_title_c",
    #         "gong_call_id_c",
    #         "chunk_index",
    #         "gong_participants_emails_c",
    #         "transcript_text",
    #         "gong_primary_opportunity_c",
    #     ],
    #     n_characters=2000,
    # )

    unique_ops = get_unique_gong_primary_opportunities()
    print("Unique gong_primary_opportunity_c values:")
    for op in unique_ops:
        print(op)


# Interesting Queries

"Find me a call about sports data."

# ------------------------------------------------------------------------------------------------

# Note: For Full-Text Search (BM25)
# If you have configured an attribute (say, combined_transcript or another text field) in your schema with full‑text search enabled, you can use BM25. In this case you don’t need to compute an embedding for your query. Instead, you pass your query text to the BM25 ranker using the rank_by parameter:

# import turbopuffer as tpuf

# ns = tpuf.Namespace("your_namespace_name")
# results = ns.query(
#     rank_by=["combined_transcript", "BM25", "data orchestration"],
#     top_k=10,
#     include_attributes=["name", "gong_call_id_c"]
# )

# for result in results:
#     print(result.id, result.attributes, result.dist)
# This tells Turbopuffer to score documents based on BM25 relevance of the combined_transcript field against the query text "data orchestration".
