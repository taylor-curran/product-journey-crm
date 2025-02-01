import turbopuffer as tpuf
from helper import embed_text
import os


def query_namespace(
    namespace: str = "tay-test",
    query_text: str = "find me a call where data orchestration was discussed",
    top_k: int = 3,
    include_attributes: list[str] = ["name", "gong_call_id_c"],
) -> list:
    # Convert query into a vector
    query_vector = embed_text(query_text)

    # Configure your API key and base URL
    tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
    tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"

    # Query the namespace using the vector
    ns = tpuf.Namespace(namespace)
    results = ns.query(
        vector=query_vector,
        distance_metric="cosine_distance",
        top_k=top_k,
        include_attributes=include_attributes,
    )

    for result in results:
        print("\nResult:")
        print(f"  ID: {result.id}")
        print(f"  Distance: {result.dist:.4f}")
        print("  Attributes:")
        for attr, value in result.attributes.items():
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


if __name__ == "__main__":
    # schema = print_namespace_schema("tay-test")

    results = query_namespace(
        "tay-test",
        "Find me a call where migration and upgrade were discussed.",
        top_k=1,
        include_attributes=["gong_title_c"],
    )


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
