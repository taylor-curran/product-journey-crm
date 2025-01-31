from prefect import task, flow
from google.cloud import bigquery
from dotenv import load_dotenv
import os
from queries import transcript_query


@flow
def refresh_gong_transcripts(limit: int = 100):
    """
    Get the transcript data from Gong
    """
    load_dotenv()
    gcp_project_id = os.getenv("GCP_PROJECT_ID")

    client = bigquery.Client(project=gcp_project_id)

    transcript_query = """
    -- First, find all relevant call IDs from Salesforce
    WITH calls_of_interest AS (
    SELECT
    sf.gong_call_id_c AS call_id
    FROM `prefect-data-warehouse.salesforce_ft.gong_gong_call_c` AS sf
    WHERE sf.gong_scope_c = 'External'
    AND EXISTS (
        SELECT 1
        FROM UNNEST(JSON_EXTRACT_ARRAY(sf.gong_related_participants_json_c)) AS participant
        WHERE JSON_EXTRACT_SCALAR(participant, '$.Gong__Gong_Participant_Title__c') LIKE '%Account Executive%'
    )
    ),

    -- Now filter transcripts to only those call_ids we actually need
    combined_transcripts AS (
    SELECT
    t.call_id,
    TO_JSON_STRING(
        ARRAY_CONCAT_AGG(
        JSON_EXTRACT_ARRAY(t.sentence) 
        ORDER BY CAST(t.index AS INT64)
        )
    ) AS combined_transcript
    FROM `prefect-data-warehouse.gongio_ft.transcript` t
    INNER JOIN calls_of_interest coi
    ON t.call_id = coi.call_id
    GROUP BY t.call_id
    )

    -- Finally, join it back to Salesforce to select the fields
    SELECT 
    sf.id AS salesforce_record_id,
    sf.name,
    sf.gong_call_duration_sec_c,
    sf.gong_call_id_c,
    sf.gong_call_start_c,
    sf.gong_participants_emails_c,
    sf.gong_related_participants_json_c,
    sf.gong_primary_account_c,
    sf.gong_primary_opportunity_c,
    sf.gong_related_contacts_json_c,
    sf.gong_related_leads_json_c,
    sf.gong_related_opportunities_json_c,
    sf.gong_opp_close_date_time_of_call_c,
    sf.gong_opp_probability_time_of_call_c,
    sf.gong_opp_stage_time_of_call_c,
    sf.gong_title_c,
    sf.gong_scheduled_c,
    sf.gong_is_private_c,
    sf.gong_call_brief_c,
    sf.gong_call_highlights_next_steps_c,
    sf.gong_call_key_points_c,
    sf.gong_scope_c,
    ct.combined_transcript

    FROM `prefect-data-warehouse.salesforce_ft.gong_gong_call_c` AS sf
    LEFT JOIN combined_transcripts AS ct 
    ON sf.gong_call_id_c = ct.call_id
    WHERE sf.gong_scope_c = 'External'
    AND EXISTS (
    SELECT 1
    FROM UNNEST(JSON_EXTRACT_ARRAY(sf.gong_related_participants_json_c)) AS participant
    WHERE JSON_EXTRACT_SCALAR(participant, '$.Gong__Gong_Participant_Title__c') LIKE '%Account Executive%'
    )
    ORDER BY sf.gong_call_start_c DESC
    """

    query = transcript_query + f"LIMIT {limit};"
    rows = client.query_and_wait(query)


if __name__ == "__main__":
    refresh_gong_transcripts(10)
