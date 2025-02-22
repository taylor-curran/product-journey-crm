from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Literal
import turbopuffer as tpuf
from helper import embed_text
import os
from typing import Annotated


class OrchestrationTool(str, Enum):
    DAGSTER = "Dagster"
    HOME_GROWN_ADVANCED = "Home-Grown Advanced Orchestration Tool"
    HOME_GROWN_BASIC = "Home-Grown Basic Orchestration Tool"
    AIRFLOW_MWAA = "Airflow (MWAA)"
    AIRFLOW_ASTRONOMER = "Airflow (Astronomer)"
    AIRFLOW_AZURE = "Airflow (Azure Managed)"
    AIRFLOW_GCP = "Airflow (GCP Managed)"
    AIRFLOW_OSS = "Airflow (OSS)"
    AIRFLOW_NOT_SPECIFIED = "Airflow (Not Specified)"
    ACTIVEBATCH = "ActiveBatch"
    TEMPORAL = "Temporal"
    CONTROL_M = "Control-M (BMC)"
    INFORMATICA = "Informatica PowerCenter"
    ALTERYX = "Alteryx"
    SQL_SERVER_JOBS = "SQL Server Jobs"
    AWS_STEP_FUNCTIONS = "AWS Step Functions"
    AWS_LAMBDA_FUNCTIONS = "AWS Lambda Functions"
    AZURE_FUNCTIONS = "Azure Functions"
    AZURE_DATA_FACTORY = "Azure Data Factory"
    GCP_CLOUD_FUNCTIONS = "GCP Cloud Functions"
    IBM_WORKLOAD_SCHEDULER = "IBM Workload Scheduler"
    MATILLION = "Matillion"
    AUTOSYS = "AutoSys"
    TALEND = "Talend"
    DATASTAGE = "DataStage (IBM)"
    SSIS = "SQL Server Integration Services (SSIS)"
    BOOMI = "Boomi"
    SNAPLOGIC = "SnapLogic"
    MULESOFT = "MuleSoft"
    OTHER_LEGACY_SYSTEM = "Other Legacy System"
    CAMUNDA = "Camunda"
    OTHER = "Other"


class TechStack(BaseModel):
    previous_solution: Optional[OrchestrationTool] = Field(
        None,
        description="The account's previous, current, or legacy data orchestration solution",
    )
    cloud_provider: str = Field(
        description="The account's cloud provider; often aws, azure, gcp, oci, on-prem, etc."
    )


class TechStackResult(BaseModel):
    """Result model for tech stack extraction with confidence scores"""

    tech_stack: TechStack
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score of the extraction"
    )
    previous_solution_snippet: Optional[str] = Field(
        None,
        description="Relevant transcript snippet for previous solution",
    )
    cloud_provider_snippet: Optional[str] = Field(
        None,
        description="Relevant transcript snippet for cloud provider",
    )


class OpportunityContext(BaseModel):
    """Context for tech stack extraction"""

    gong_primary_opportunity_c: str = Field(
        description="The Gong primary opportunity ID"
    )


# Define the agent with proper typing and configuration
tech_stack_agent = Agent(
    model="openai:gpt-4o",
    deps_type=OpportunityContext,
    result_type=TechStackResult,
    system_prompt="""
    You are a technical analyst specialized in understanding customer's data infrastructure.
    Your task is to:
    1. Extract information about the customer's data stack from call transcripts
    2. Identify their previous/current orchestration tools and cloud providers
    3. Provide confidence scores and relevant snippets to support your analysis
    
    Use the query_transcript_vector_db_for_opportunity tool to find relevant information.
    Always explain your reasoning and provide evidence from the transcripts.
    """,
)


@tech_stack_agent.tool
async def query_transcript_vector_db_for_opportunity(
    ctx: RunContext[OpportunityContext],
    query_text: str = "What is the customer's data stack?",
    top_k: int = 3,
) -> List[dict]:
    """Query the vector database for relevant transcript snippets"""
    query_vector = embed_text(query_text)
    namespace = "tay-sales-calls"
    ns = tpuf.Namespace(namespace)

    filters = [
        "gong_primary_opportunity_c",
        "Eq",
        ctx.deps.gong_primary_opportunity_c,
    ]

    results = ns.query(
        vector=query_vector,
        distance_metric="cosine_distance",
        top_k=top_k,
        filters=filters,
        include_attributes=[
            "transcript_text",
            "name",
            "gong_call_id_c",
            "gong_participants_emails_c",
            "gong_primary_opportunity_c",
            "gong_title_c",
            "gong_call_brief_c",
            "gong_opp_close_date_time_of_call_c",
        ],
    )
    # transcript_text is what the agent needs to process, so we extract it here
    transcript_texts = [result.attributes["transcript_text"] for result in results]
    # but other attributes are important for interpretability, verfication, and evaluation of the agent

    # the transcript attribute is too long, i want to print all attributes except transcript_text
    # Print all attributes except transcript_text for each result
    transcript_metadata = [
        result.attributes.pop("transcript_text", None) for result in results
    ]

    for result in results:
        print("\nResult Metadata:")
        for attr, value in result.attributes.items():
            if attr != "transcript_text":
                print(f"{attr}:")
                print(f"{value}")
                print()

    return results


def extract_data_stack_task(opp_id: str) -> TechStackResult:
    """
    Extract information about the data stack from call transcripts
    filtered by the provided opportunity ID.

    Args:
        opp_id: The Gong primary opportunity ID

    Returns:
        TechStackResult containing the extracted tech stack information,
        confidence score, and supporting evidence
    """
    context = OpportunityContext(gong_primary_opportunity_c=opp_id)
    result = tech_stack_agent.run_sync(
        "Analyze the customer's data stack and identify their orchestration tools and cloud providers.",
        deps=context,
    )
    return result.data


tech_stack = extract_data_stack_task("006Rm00000OG8LZIA1")

print(f"""
Tech Stack:
    Previous Solution: {tech_stack.tech_stack.previous_solution}
    Cloud Provider: {tech_stack.tech_stack.cloud_provider}

Confidence Score: {tech_stack.confidence_score:.2f}
___ ___ ___ ___

Previous Solution Snippet:
• {tech_stack.previous_solution_snippet}
___ ___ ___

Cloud Provider Snippet:
• {tech_stack.cloud_provider_snippet}
___ ___ ___

""")
