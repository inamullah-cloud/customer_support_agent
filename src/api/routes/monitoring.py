"""Observability and evaluation/reliability endpoints."""

from fastapi import APIRouter, HTTPException

from src.agent.graph import app as agent_app
from src.api.dependencies import get_container
from src.api.schemas import EvaluationRunResponse
from src.evaluation.metrics import calculate_metrics
from src.evaluation.reliability import ProductionLevelReliabilityReport
from src.evaluation.test_cases import EVAL_DATASET
from src.observability.models import AgentObservabilityDashboard, AgentObservabilityReport

observability_router = APIRouter(prefix="/observability", tags=["observability"])
evaluation_router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# ---------------- observability (live chat runs) ----------------

@observability_router.get("/dashboard", response_model=AgentObservabilityDashboard)
def observability_dashboard() -> AgentObservabilityDashboard:
    c = get_container()
    return c.obs_aggregator.build_dashboard(
        c.obs_store.get_all(), threshold_ms=c.slow_node_threshold_ms
    )


@observability_router.get("/threads/{thread_id}", response_model=AgentObservabilityReport)
def thread_report(thread_id: str) -> AgentObservabilityReport:
    report = get_container().obs_store.get(thread_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"No report for thread '{thread_id}'.")
    return report


# ---------------- evaluation / reliability ----------------

def _build_evaluation_response() -> EvaluationRunResponse:
    c = get_container()
    results = c.run_store.get_latest_batch()
    if not results:
        raise HTTPException(status_code=404, detail="No evaluation runs yet. POST /evaluation/run first.")

    return EvaluationRunResponse(
        metrics=calculate_metrics([r.evaluation for r in results]),
        average_latency_ms=c.run_store.get_average_latency(results),
        results=results,
        dashboard=c.failure_analyzer.build_dashboard(results, c.run_store.get_previous_batch()),
    )


@evaluation_router.post("/run", response_model=EvaluationRunResponse)
def run_evaluation() -> EvaluationRunResponse:
    """Run the evaluation dataset against the live agent (slow: makes LLM calls)."""
    c = get_container()
    results = [
        c.evaluator.evaluate_run(case, agent_app, thread_id=c.evaluator.unique_thread_id(case))
        for case in EVAL_DATASET
    ]
    c.run_store.save_batch(results)
    return _build_evaluation_response()


@evaluation_router.get("/dashboard", response_model=EvaluationRunResponse)
def evaluation_dashboard() -> EvaluationRunResponse:
    """Latest evaluation batch, compared with the previous one for regressions."""
    return _build_evaluation_response()


@evaluation_router.get("/reliability", response_model=ProductionLevelReliabilityReport)
def reliability_report() -> ProductionLevelReliabilityReport:
    c = get_container()
    results = c.run_store.get_latest_batch()
    if not results:
        raise HTTPException(status_code=404, detail="No evaluation runs yet. POST /evaluation/run first.")
    return c.reliability_analyzer.analyze(results)
