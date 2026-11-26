"""Trace recording (NodeTimer) and trace analysis (TraceAnalyzer)."""

from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from src.observability.models import AgentObservabilityReport, TraceEvent


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NodeTimer:
    """Records start/finish trace events for one node execution.

    Usage inside a node:
        timer = NodeTimer(state, "agent")
        ...
        return {..., "execution_trace": timer.finish("details")}
    """

    def __init__(self, state: dict, node: str, reset: bool = False) -> None:
        self.node = node
        existing = [] if reset else (state.get("execution_trace") or [])
        self.trace: List[TraceEvent] = list(existing)
        self._start = perf_counter()
        self.trace.append(TraceEvent(event="node_started", node=node, timestamp=_now()))

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._start) * 1000, 2)

    def add(self, event: str, details: str | None = None, duration_ms: float | None = None) -> None:
        """Append a custom event (e.g. a single tool call)."""
        self.trace.append(
            TraceEvent(
                event=event,
                node=self.node,
                details=details,
                timestamp=_now(),
                duration_ms=duration_ms,
            )
        )

    def finish(self, details: str | None = None) -> List[TraceEvent]:
        """Close the node and return the full updated trace."""
        self.add("node_completed", details=details, duration_ms=self.elapsed_ms())
        return self.trace


def _events(trace: List[TraceEvent] | None) -> List[TraceEvent]:
    return [TraceEvent.model_validate(e) if isinstance(e, dict) else e for e in (trace or [])]


class TraceAnalyzer:
    """Analyze structured agent execution traces."""

    TOOL_EVENTS = {"tool_completed", "tool_failed"}

    def get_node_durations(self, trace: List[TraceEvent]) -> Dict[str, float]:
        """Total time spent in each node (summed if a node ran several times)."""
        durations: Dict[str, float] = {}
        for e in _events(trace):
            if e.event == "node_completed":
                durations[e.node] = round(durations.get(e.node, 0.0) + (e.duration_ms or 0.0), 2)
        return durations

    def get_tool_durations(self, trace: List[TraceEvent]) -> Dict[str, float]:
        """Total time spent in each tool."""
        durations: Dict[str, float] = {}
        for e in _events(trace):
            if e.event in self.TOOL_EVENTS and e.details:
                durations[e.details] = round(durations.get(e.details, 0.0) + (e.duration_ms or 0.0), 2)
        return durations

    def get_total_duration(self, trace: List[TraceEvent]) -> float:
        """Active processing time in ms (sum of node durations).

        Time spent waiting for a human approval is deliberately excluded.
        """
        return round(sum(self.get_node_durations(trace).values()), 2)

    def get_slow_nodes(self, trace: List[TraceEvent], threshold_ms: float) -> Dict[str, float]:
        return {
            node: dur
            for node, dur in self.get_node_durations(trace).items()
            if dur > threshold_ms
        }

    def get_slowest_node(self, trace: List[TraceEvent]) -> Optional[Tuple[str, float]]:
        durations = self.get_node_durations(trace)
        if not durations:
            return None
        return max(durations.items(), key=lambda item: item[1])

    def build_summary(self, trace: List[TraceEvent]) -> Dict:
        return {
            "total_duration_ms": self.get_total_duration(trace),
            "node_durations": self.get_node_durations(trace),
            "tool_durations": self.get_tool_durations(trace),
        }

    def build_report(self, trace: List[TraceEvent], threshold_ms: float) -> AgentObservabilityReport:
        return AgentObservabilityReport(
            total_duration_ms=self.get_total_duration(trace),
            node_durations=self.get_node_durations(trace),
            tool_durations=self.get_tool_durations(trace),
            slow_nodes=self.get_slow_nodes(trace, threshold_ms),
            slowest_node=self.get_slowest_node(trace),
        )
