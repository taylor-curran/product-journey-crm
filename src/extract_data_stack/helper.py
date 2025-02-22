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


def consolidate_and_print_metadata(results):
    """
    Consolidates metadata from results and prints them in a formatted manner.
    """
    transcript_metadata = [
        {
            k: (v[:300] if isinstance(v, str) and len(v) > 300 else v)
            for k, v in result.attributes.copy().items()
            if k != "transcript_text"
        }
        for result in results
    ]

    # Define keys for aggregation.
    list_keys = ["name", "gong_call_id_c", "gong_title_c", "gong_call_brief_c", "gong_call_start_c"]
    unique_keys = ["gong_primary_opportunity_c", "gong_participants_emails_c"]

    # Initialize consolidated structure.
    consolidated = {key: [] for key in list_keys}
    for key in unique_keys:
        consolidated[key] = set()

    # Accumulate values from all metadata entries.
    for metadata in transcript_metadata:
        for key in list_keys:
            if key in metadata:
                consolidated[key].append(metadata[key])
        for key in unique_keys:
            if key in metadata:
                consolidated[key].add(metadata[key])

    # Print consolidated data.
    print("\nConsolidated Results:")
    for key in list_keys:
        print(f"{key}:")
        for value in consolidated[key]:
            print(f"- {value}")
        print()

    for key in unique_keys:
        print(f"{key}:")
        for value in consolidated[key]:
            print(f"- {value}")
        print()


# TODO: Ask nate how do i avoid needing to make a copy of this file in every directory?
