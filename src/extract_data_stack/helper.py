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


# TODO: Ask nate how do i avoid needing to make a copy of this file in every directory?
