from google.cloud import bigquery
from dotenv import load_dotenv
import os
from queries import transcript_query

def refresh_gong_transcripts(limit: int = 100):
    """
    Get the transcript data from Gong
    """
    load_dotenv()
    gcp_project_id = os.getenv("GCP_PROJECT_ID")

    client = bigquery.Client(project=gcp_project_id)

    query = transcript_query + f"LIMIT {limit};"
    rows = client.query_and_wait(query)

    for row in rows:
        print(dict(row))


if __name__ == "__main__":
    refresh_gong_transcripts(2)
