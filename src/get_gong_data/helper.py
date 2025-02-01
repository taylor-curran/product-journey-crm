# src/get_gong_data/helper.py
import datetime
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


def chunk_text(s: str, chunk_size: int = 2000) -> List[str]:
    """
    Splits a large string into smaller chunks of specified size.
    Adjust chunk_size based on OpenAI's token limits and your requirements.
    """
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


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


def clean_attribute_value(key: str, value: Any) -> Any:
    """
    Cleans a single attribute value based on its key.
    Adjust the rules below to fit your data and desired types.
    """
    if value is None:
        return None

    # For duration and probability, convert floats to ints.
    if key in ("gong_call_duration_sec_c", "gong_opp_probability_time_of_call_c"):
        try:
            # If the value is a float (or even a numeric string), convert to int.
            return int(float(value))
        except (ValueError, TypeError):
            return None

    # For datetime/date fields, return ISO format.
    if key in (
        "gong_call_start_c",
        "gong_opp_close_date_time_of_call_c",
        "gong_scheduled_c",
    ):
        if isinstance(value, (datetime.datetime, datetime.date)):
            return value.isoformat()
        # If already a string, try to parse and re-format it.
        try:
            parsed = datetime.datetime.fromisoformat(value)
            return parsed.isoformat()
        except (ValueError, TypeError):
            return value  # leave as is if parsing fails

    # For keys that are supposed to hold JSON data.
    if "json" in key:
        # If it's a string, try to load and then dump to normalize formatting.
        if isinstance(value, str):
            try:
                loaded = json.loads(value)
                return json.dumps(loaded)
            except (json.JSONDecodeError, TypeError):
                return value  # leave as is if not valid JSON
        elif isinstance(value, (dict, list)):
            try:
                return json.dumps(value)
            except Exception:
                return str(value)

    # For the boolean field, ensure a bool.
    if key == "gong_is_private_c":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1"):
                return True
            elif value.lower() in ("false", "no", "0"):
                return False
        return bool(value)

    # For all other fields, return the value unchanged.
    return value


def clean_attributes_for_row(
    row: Dict[str, Any], attribute_keys: List[str]
) -> Dict[str, Any]:
    """
    Returns a dictionary of cleaned attribute values for the given row.
    """
    cleaned = {}
    for key in attribute_keys:
        cleaned[key] = clean_attribute_value(key, row.get(key))
    return cleaned


def process_combined_transcript(
    combined_transcript: str, call_title: str, call_id: str
) -> str:
    """
    Processes the combined transcript and returns the entire transcript text.
    Raises a ValueError if the transcript is invalid or empty.
    """
    if not combined_transcript:
        print(f"Skipping call {call_title} with empty combined_transcript.")
        return ""

    try:
        transcript_list = json.loads(combined_transcript)
        entire_transcript_text = " ".join(
            item.get("text", "") for item in transcript_list
        )
    except json.JSONDecodeError as e:
        print(f"Error parsing combined_transcript for call {call_title}-{call_id}: {e}")
        return ""

    if not entire_transcript_text.strip() or len(entire_transcript_text) < 10:
        print(
            f"Skipping call {call_title}-{call_id} with empty transcript text after parsing."
        )
        return ""

    return entire_transcript_text


# test

# def openai_or_rand_vector(text: str) -> list[float]:
#     if not os.getenv("OPENAI_API_KEY"): print("OPENAI_API_KEY not set, using random vectors"); return [__import__('random').random()]*2
#     try: return __import__('openai').embeddings.create(model="text-embedding-3-small",input=text).data[0].embedding
#     except ImportError: return [__import__('random').random()]*2

# ns.upsert(
#     ids=[1, 2],
#     vectors=[openai_or_rand_vector("walrus narwhal"), openai_or_rand_vector("elephant walrus rhino")],
#     attributes={"name": ["foo", "foo"], "public": [1, 0], "text": ["walrus narwhal", "elephant walrus rhino"]},
#     distance_metric='cosine_distance',
#     schema={
#         "text": { # Configure FTS/BM25, other attribtues have inferred types (name: str, public: int)
#             "type": "string",
#              # More schema & FTS options https://turbopuffer.com/docs/schema
#             "full_text_search": True,
#         }
#     }
# )


# print(ns.query(
#   vector=openai_or_rand_vector("walrus narwhal"),
#   top_k=10,
#   distance_metric="cosine_distance",
#   filters=["And", [["name", "Eq", "foo"], ["public", "Eq", 1]]],
#   include_attributes=["name"],
#   include_vectors=False,
# ))
# ))
