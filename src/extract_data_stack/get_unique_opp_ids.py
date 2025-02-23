import turbopuffer as tpuf
import os
from dotenv import load_dotenv


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
    results = ns.query(top_k=top_k, include_attributes=["gong_primary_opportunity_c"])

    # Use a set to collect unique values
    unique_opportunities = set()
    for result in results:
        # Check if result and its attributes exist
        if result and result.attributes:
            opportunity = result.attributes.get("gong_primary_opportunity_c")
            if opportunity:
                unique_opportunities.add(opportunity)
        else:
            print("Warning: result or its attributes is None.")

    return list(unique_opportunities)


if __name__ == "__main__":
    unique_ops = get_unique_gong_primary_opportunities()
    print("Unique gong_primary_opportunity_c values:")
    for op in unique_ops:
        print(op)
