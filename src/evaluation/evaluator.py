"""Evaluate a single test case against the agent."""

from src.evaluation.metrics import EvaluationResult
from src.evaluation.test_cases import EvaluationCase
from src.utils.logging import get_logger

logger = get_logger(__name__)


def determine_failure_reason(
    category_correct: bool,
    tools_correct: bool,
    execution_success: bool,
    final_answer_generated: bool,
) -> str:
    """Determine the primary cause of an evaluation failure (priority order)."""
    if not category_correct:
        return "category_mismatch"
    if not tools_correct:
        return "tool_selection_mismatch"
    if not execution_success:
        return "execution_failure"
    if not final_answer_generated:
        return "no_final_answer"
    return "success"


def evaluate_case(
    case: EvaluationCase,
    agent_app=None,
    final_state: dict | None = None,
) -> EvaluationResult:
    """Evaluate a test case.

    Pass a pre-computed `final_state` to avoid executing the agent twice.
    Otherwise `agent_app` is invoked (note: it will stop at a human-approval
    interrupt; use ProductionEvaluator to auto-resume approvals).
    """
    if final_state is None:
        if agent_app is None:
            raise ValueError("Either 'agent_app' or 'final_state' must be provided.")
        try:
            final_state = agent_app.invoke(
                {"messages": [("user", case.query)], "retry_count": 0, "tool_iterations": 0},
                config={"configurable": {"thread_id": case.thread_id}},
            )
        except Exception:
            logger.exception("Execution error on query: '%s'", case.query)
            final_state = {}

    execution_success = (
        final_state.get("execution_status") == "completed"
        and final_state.get("error") is None
    )

    predicted_category = final_state.get("category") or ""
    predicted_tools = final_state.get("selected_tools") or []

    # Final text: the 'response' key, falling back to the last message
    response = final_state.get("response")
    if not response and final_state.get("messages"):
        last_msg = final_state["messages"][-1]
        response = getattr(last_msg, "content", str(last_msg))

    category_correct = predicted_category == case.expected_category
    tools_correct = set(case.expected_tools) == set(predicted_tools)
    final_answer_generated = bool(response and str(response).strip())

    overall_success = (
        category_correct
        and tools_correct
        and execution_success
        and final_answer_generated
    )

    return EvaluationResult(
        query=case.query,
        expected_category=case.expected_category,
        predicted_category=predicted_category,
        category_correct=category_correct,
        expected_tools=case.expected_tools,
        predicted_tools=predicted_tools,
        tools_correct=tools_correct,
        execution_success=execution_success,
        final_answer_generated=final_answer_generated,
        overall_success=overall_success,
        failure_reason=determine_failure_reason(
            category_correct, tools_correct, execution_success, final_answer_generated
        ),
    )
