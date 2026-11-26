"""Evaluation result schema, metric computation, and report printing."""

from typing import Any, Dict, List

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    query: str
    expected_category: str
    predicted_category: str
    category_correct: bool

    expected_tools: List[str]
    predicted_tools: List[str]
    tools_correct: bool

    execution_success: bool
    final_answer_generated: bool
    overall_success: bool
    failure_reason: str = "success"


def calculate_metrics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Compute summary metrics across all evaluation cases."""
    total = len(results)
    if total == 0:
        return {
            "total_cases": 0,
            "category_accuracy": 0.0,
            "tool_accuracy": 0.0,
            "execution_success_rate": 0.0,
            "final_answer_rate": 0.0,
            "overall_success_rate": 0.0,
        }

    return {
        "total_cases": total,
        "category_accuracy": (sum(r.category_correct for r in results) / total) * 100,
        "tool_accuracy": (sum(r.tools_correct for r in results) / total) * 100,
        "execution_success_rate": (sum(r.execution_success for r in results) / total) * 100,
        "final_answer_rate": (sum(r.final_answer_generated for r in results) / total) * 100,
        "overall_success_rate": (sum(r.overall_success for r in results) / total) * 100,
    }


def print_evaluation_report(
    results: List[EvaluationResult],
    metrics: Dict[str, Any],
) -> None:
    """Format and print the evaluation summary report."""
    print("=" * 40)
    print("        AGENT EVALUATION REPORT")
    print("=" * 40 + "\n")
    print(f"Total Cases: {metrics['total_cases']}\n")
    print(f"Category Accuracy:       {metrics['category_accuracy']:>5.1f}%")
    print(f"Tool Selection Accuracy: {metrics['tool_accuracy']:>5.1f}%")
    print(f"Execution Success Rate:  {metrics['execution_success_rate']:>5.1f}%")
    print(f"Final Answer Rate:       {metrics['final_answer_rate']:>5.1f}%")
    print(f"Overall Success Rate:    {metrics['overall_success_rate']:>5.1f}%")
    print("\n" + "-" * 40)
    print("CASE RESULTS")
    print("-" * 40 + "\n")

    for r in results:
        status_symbol = "✓" if r.overall_success else "✗"
        print(f"{status_symbol} {r.query}")
        print(f"  Category:  {'PASS' if r.category_correct else 'FAIL'}")
        print(f"  Tools:     {'PASS' if r.tools_correct else 'FAIL'}")
        print(f"  Execution: {'PASS' if r.execution_success else 'FAIL'}")
        print(f"  Overall:   {'PASS' if r.overall_success else 'FAIL'}")
        if not r.overall_success:
            print(f"  Reason:    {r.failure_reason}")
        print()
