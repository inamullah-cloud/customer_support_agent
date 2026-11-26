"""Storage and aggregation of observability reports across runs."""

import threading
from typing import Dict, List

from src.observability.models import (
    AgentObservabilityDashboard,
    AgentObservabilityReport,
    ObservabilityHealth,
    ObservabilitySummary,
)


class ObservabilityStore:
    """In-memory store of the latest observability report per thread."""

    def __init__(self) -> None:
        self._store: dict[str, AgentObservabilityReport] = {}
        self._lock = threading.RLock()

    def save(self, thread_id: str, report: AgentObservabilityReport) -> None:
        with self._lock:
            self._store[thread_id] = report

    def get(self, thread_id: str) -> AgentObservabilityReport | None:
        with self._lock:
            return self._store.get(thread_id)

    def get_all(self) -> list[AgentObservabilityReport]:
        with self._lock:
            return list(self._store.values())


class ObservabilityAggregator:
    """Analyze observability reports across multiple agent runs."""

    def average_execution_time(self, reports: list[AgentObservabilityReport]) -> float:
        if not reports:
            return 0.0
        durations = [r.total_duration_ms for r in reports]
        return round(sum(durations) / len(durations), 2)

    def slow_node_frequency(self, reports: list[AgentObservabilityReport]) -> dict[str, int]:
        counts: Dict[str, int] = {}
        for report in reports:
            for node_name in report.slow_nodes:
                counts[node_name] = counts.get(node_name, 0) + 1
        return counts

    def build_summary(self, reports: list[AgentObservabilityReport]) -> ObservabilitySummary:
        valid: List[AgentObservabilityReport] = [r for r in reports if r is not None]
        return ObservabilitySummary(
            total_runs=len(valid),
            average_execution_time_ms=self.average_execution_time(valid),
            slow_node_frequency=self.slow_node_frequency(valid),
        )

    def health_check(
        self,
        reports: list[AgentObservabilityReport],
        threshold_ms: float = 1000,
    ) -> ObservabilityHealth:
        valid = [r for r in reports if r is not None]
        average = self.average_execution_time(valid)
        slow = self.slow_node_frequency(valid)

        if slow:
            node, count = max(slow.items(), key=lambda item: item[1])
        else:
            node, count = None, 0

        status = "healthy" if average <= threshold_ms and count == 0 else "degraded"

        return ObservabilityHealth(
            average_execution_time_ms=average,
            slowest_node=node,
            slow_node_count=count,
            status=status,
        )

    def build_dashboard(
        self,
        reports: list[AgentObservabilityReport],
        threshold_ms: float = 1000,
    ) -> AgentObservabilityDashboard:
        return AgentObservabilityDashboard(
            summary=self.build_summary(reports),
            health=self.health_check(reports, threshold_ms=threshold_ms),
        )
