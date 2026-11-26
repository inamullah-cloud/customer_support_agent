"""Pydantic models for execution traces and observability reports."""

from datetime import datetime

from pydantic import BaseModel


class TraceEvent(BaseModel):
    """A single event in an agent execution trace."""

    event: str
    node: str
    details: str | None = None
    timestamp: datetime
    duration_ms: float | None = None


class AgentObservabilityReport(BaseModel):
    """Structured observability report for one agent execution."""

    total_duration_ms: float
    node_durations: dict[str, float]
    tool_durations: dict[str, float]
    slow_nodes: dict[str, float]
    slowest_node: tuple[str, float] | None = None


class ObservabilitySummary(BaseModel):
    """Aggregated observability summary across agent runs."""

    total_runs: int
    average_execution_time_ms: float
    slow_node_frequency: dict[str, int]


class ObservabilityHealth(BaseModel):
    """Health status derived from aggregated observability data."""

    average_execution_time_ms: float
    slowest_node: str | None
    slow_node_count: int
    status: str


class AgentObservabilityDashboard(BaseModel):
    """Complete observability information for the agent."""

    summary: ObservabilitySummary
    health: ObservabilityHealth
