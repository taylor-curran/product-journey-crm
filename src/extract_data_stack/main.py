import controlflow as cf
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Literal
from tech_stack_enums import OrchestrationTool

import turbopuffer as tpuf
from helper import embed_text
import os

import marvin

class TechStack(BaseModel):
    previous_solution: Optional[OrchestrationTool] = Field(
        None, description="The account's previous, current, or legacy data orchestration solution"
    )
    cloud_provider: str = Field(description="The account's cloud provider; often aws, azure, gcp, oci, on-prem, etc.")

def query_transcript_vector_db_for_opportunity(
    query_text: str = "What is the customer's data stack?",
    top_k: int = 3,
    include_attributes: list[str] = ["name", "gong_call_id_c", "gong_call_brief_c"],
    gong_primary_opportunity_c: str = "006Rm00000QuHC6IAN"
) -> list:

    query_vector = embed_text(query_text)
    namespace = "tay-sales-calls"
    ns = tpuf.Namespace(namespace)

    # Define a filter to restrict results to documents with the specified opportunity.
    filters = ['gong_primary_opportunity_c', 'Eq', gong_primary_opportunity_c]

    # Execute the query with the vector, filter, and other parameters.
    results = ns.query(
        vector=query_vector,
        distance_metric="cosine_distance",
        top_k=top_k,
        include_attributes=include_attributes,
        filters=filters
    )
    
    return results



sales_engineer = marvin.Agent(
    name="Sales Engineer",
    description="An AI agent specialized in extracting technical information about the customer from sales calls",
    instructions="Extract information about the customer's data stack from the transcript",
    tools=[query_transcript_vector_db_for_opportunity],
    model="openai:gpt-4",
)

task = marvin.Task(
    instructions="Extract information about the opp ID 006Rm00000OG8LZIA1 data stack from the transcript",
    result_type=TechStack,
    agents=[sales_engineer]
)


result = task.run()
breakpoint()
print(result)