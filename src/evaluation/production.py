"""Production-style evaluation: evaluation + observability for each run."""

import threading
from typing import Optional
from uuid import uuid4

from langgraph.types import Command
from pydantic import BaseModel

from src.evaluation.evaluator import evaluate_case
from src.evaluation.metrics import EvaluationResult
from src.evaluation.test_cases import EvaluationCase
from src.observability.models import AgentObservabilityReport
from src.observability.store import ObservabilityStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ProductionRunResult(BaseModel):
    """Combined evaluation and observability result for one agent run."""

    evaluation: EvaluationResult
    observability: AgentObservabilityReport | None = None


class ProductionEvaluator:
    """Run a case once, then combine its evaluation with its observability report.

    auto_approve: True/False automatically answers human-approval interrupts;
    None leaves the run paused at the interrupt.
    """

    MAX_APPROVAL_RESUMES = 3

    def __init__(
        self,
        store: Optional[ObservabilityStore] = None,
        auto_approve: bool | None = True,
    ) -> None:
        self.store = store or ObservabilityStore()
        self.auto_approve = auto_approve

    def _run(self, agent_app, case: EvaluationCase, thread_id: str) -> dict:
        """Stream the agent once, auto-resuming approval interrupts."""
        config = {"configurable": {"thread_id": thread_id}}
        graph_input = {"messages": [("user", case.query)], "retry_count": 0, "tool_iterations": 0}
        final_state: dict = {}

        for _ in range(self.MAX_APPROVAL_RESUMES + 1):
            interrupted = False

            for chunk in agent_app.stream(graph_input, config=config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupted = True
                    elif isinstance(update, dict):
                        final_state.update(update)

            if interrupted and self.auto_approve is not None:
                graph_input = Command(resume=self.auto_approve)
                continue
            break

        return final_state

    def evaluate_run(
        self,
        case: EvaluationCase,
        agent_app,
        thread_id: str | None = None,
    ) -> ProductionRunResult:
        thread_id = thread_id or case.thread_id

        try:
            final_state = self._run(agent_app, case, thread_id)
        except Exception:
            logger.exception("Execution error on query: '%s'", case.query)
            final_state = {"execution_status": "failed", "error": "agent execution raised an exception"}

        report = final_state.get("observability_report")
        if report is not None:
            self.store.save(thread_id, report)

        return ProductionRunResult(
            evaluation=evaluate_case(case, final_state=final_state),
            observability=self.store.get(thread_id),
        )

    @staticmethod
    def unique_thread_id(case: EvaluationCase) -> str:
        """A fresh thread id so repeated runs don't inherit earlier conversation state."""
        return f"{case.thread_id}-{uuid4().hex[:8]}"


class ProductionRunStore:
    """Store production evaluation results, grouped in batches (one batch per run)."""

    def __init__(self) -> None:
        self._batches: list[list[ProductionRunResult]] = []
        self._lock = threading.RLock()

    def save(self, result: ProductionRunResult) -> None:
        """Save a single result as its own batch."""
        self.save_batch([result])

    def save_batch(self, results: list[ProductionRunResult]) -> None:
        with self._lock:
            self._batches.append(list(results))

    def get_all(self) -> list[ProductionRunResult]:
        with self._lock:
            return [r for batch in self._batches for r in batch]

    def get_latest_batch(self) -> list[ProductionRunResult]:
        with self._lock:
            return list(self._batches[-1]) if self._batches else []

    def get_previous_batch(self) -> list[ProductionRunResult]:
        with self._lock:
            return list(self._batches[-2]) if len(self._batches) >= 2 else []

    def get_success_rate(self, results: list[ProductionRunResult]) -> float:
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.evaluation.overall_success)
        return round((passed / len(results)) * 100, 2)

    def get_average_latency(self, results: list[ProductionRunResult]) -> float:
        durations = [r.observability.total_duration_ms for r in results if r.observability is not None]
        if not durations:
            return 0.0
        return round(sum(durations) / len(durations), 2)
