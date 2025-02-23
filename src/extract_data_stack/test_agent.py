# test_o1_agent_performance.py

import pytest
from unittest.mock import AsyncMock, patch
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, TextPart
from pydantic_ai import models

from main import tech_stack_agent, OpportunityContext

@pytest.mark.anyio("asyncio")  # only run with asyncio, avoids the 'trio' error
async def test_no_transcripts_halts_confidence():
    models.ALLOW_MODEL_REQUESTS = False  # block real LLM calls

    async def fn_model(messages_list: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # 1st pass: agent tries to call the 'query_transcript_vector_db_for_transcripts' tool
        if len(messages_list) == 1:
            return ModelResponse(parts=[
                ToolCallPart(
                    tool_name='query_transcript_vector_db_for_transcripts',
                    args={"query_text": "What is the customer's data stack?", "top_k": 3}
                )
            ])
        else:
            # 2nd pass: we expect an empty list from the tool
            tool_return = messages_list[-1].parts[0]
            # Verify it sees the empty list
            assert tool_return.content == "[]", f"Expected empty list, got: {tool_return.content!r}"

            # Return a final answer with low confidence
            final_json = """
            {
                "tech_stack": {
                    "previous_solution": null,
                    "cloud_provider": "unknown"
                },
                "confidence_score": 0.2,
                "previous_solution_snippet": null,
                "cloud_provider_snippet": null
            }
            """
            return ModelResponse(parts=[TextPart(content=final_json)])

    # Override the agent's model so there's no real LLM call
    with tech_stack_agent.override(model=FunctionModel(fn_model)):
        # Patch the *agent's internal* reference to the tool
        with patch.object(
            tech_stack_agent._function_tools['query_transcript_vector_db_for_transcripts'],
            'function',
            new=AsyncMock(return_value=[])
        ):
            context = OpportunityContext(gong_primary_opportunity_c="fake-opp-id")
            result = await tech_stack_agent.run(
                "Analyze the customer's data stack and identify their orchestration tools and cloud providers.",
                deps=context,
            )

    # Assert the final outcome
    assert result.data.confidence_score < 0.5, "Confidence should be low if no data was found."
    assert result.data.tech_stack.previous_solution is None
    assert result.data.cloud_provider_snippet is None
    assert result.data.previous_solution_snippet is None
