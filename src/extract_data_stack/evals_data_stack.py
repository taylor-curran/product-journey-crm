# src/extract_data_stack/evals_data_stack.py

from main_no_text_access import extract_data_stack_no_data


def eval_no_text():
    """
    Evaluation for the scenario where the vector query returns no transcript text.
    In this case we want the agent to not hallucinate a strong answer.

    Expected behavior:
      - Confidence score should be low (e.g. ≤ 0.5)
      - No transcript snippets should be present.
    """
    opp_id = "006Rm00000OG8LZIA1"
    result = extract_data_stack_no_data(opp_id)
    score = 0

    # Penalize high confidence when there is no evidence.
    # For instance, if confidence > 0.5, subtract points proportional to the excess.
    if result.confidence_score > 0.1:
        excess = result.confidence_score - 0.1
        penalty = excess * 100  # adjust the multiplier as needed
        score -= penalty

    # Also penalize if any snippet is present since we expect no supporting evidence.
    if result.previous_solution_snippet:
        score -= 50
    if result.cloud_provider_snippet:
        score -= 50

    return score


def main():
    # Run both eval cases.
    score_no_text = eval_no_text()

    # Report individual scores
    print(f"Eval score for no-text scenario: {score_no_text}")


if __name__ == "__main__":
    main()
