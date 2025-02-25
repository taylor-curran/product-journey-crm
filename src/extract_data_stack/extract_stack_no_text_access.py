# src/extract_data_stack/main_no_text_access.py

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Literal
import turbopuffer as tpuf
from helper import embed_text, consolidate_and_print_metadata
import os
from typing import Annotated
from tech_stack_enums import OrchestrationTool


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
    
    Use the query_transcript_vector_db_for_transcripts tool to find relevant information.
    Always explain your reasoning and provide evidence from the transcripts.
    """,
)


@tech_stack_agent.tool
async def query_transcript_vector_db_for_transcripts(
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
        include_attributes=[],
    )

    return results


def extract_data_stack_no_data(opp_id: str) -> TechStackResult:
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
    print(f"""
    Tech Stack:
        Previous Solution: {result.data.tech_stack.previous_solution}
        Cloud Provider: {result.data.tech_stack.cloud_provider}

    Confidence Score: {result.data.confidence_score:.2f}
    ___ ___ ___ ___

    Previous Solution Snippet:
    • {result.data.previous_solution_snippet}
    ___ ___ ___

    Cloud Provider Snippet:
    • {result.data.cloud_provider_snippet}
    ___ ___ ___

    """)

    return result.data


if __name__ == "__main__":
    tech_stack = extract_data_stack_no_data("006Rm00000OG8LZIA1")
