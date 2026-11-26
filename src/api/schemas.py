"""Request/response models for the HTTP API."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.agent.memory import MemoryType
from src.evaluation.production import ProductionRunResult
from src.evaluation.reliability import ProductionFailureDashboard
from src.observability.models import AgentObservabilityReport


# ---------------- Chat ----------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(
        default=None,
        description="Conversation id. Omit to start a new conversation.",
    )
    user_id: str | None = Field(
        default=None,
        max_length=100,
        description="Enables memory: stored memories are recalled and new ones may be saved.",
    )


class ApprovalDecision(BaseModel):
    approved: bool


class PendingToolCall(BaseModel):
    name: str
    args: dict[str, Any] = {}


class ApprovalRequest(BaseModel):
    """Returned when the agent is paused waiting for a human decision."""

    type: str = "approval_request"
    message: str
    category: str | None = None
    tools: list[str] = []
    pending_tool_calls: list[PendingToolCall] = []


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    response: str | None = None
    category: str | None = None
    selected_tools: list[str] = []
    plan: list[str] = []
    executed_tools: list[str] = []
    fallback_used: bool = False
    memories_used: list[str] = []
    memories_saved: list[str] = []
    error: str | None = None
    approval: ApprovalRequest | None = None
    observability: AgentObservabilityReport | None = None


class ThreadMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[PendingToolCall] = []


class ThreadState(ChatResponse):
    messages: list[ThreadMessage] = []


# ---------------- Memory ----------------

class MemoryCreate(BaseModel):
    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=2000)


class MemoryUpdate(BaseModel):
    memory_type: MemoryType | None = None
    content: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _at_least_one_field(self):
        if self.memory_type is None and self.content is None:
            raise ValueError("Provide at least one of 'memory_type' or 'content'.")
        return self


# ---------------- Evaluation ----------------

class EvaluationRunResponse(BaseModel):
    metrics: dict[str, Any]
    average_latency_ms: float
    results: list[ProductionRunResult]
    dashboard: ProductionFailureDashboard


# ---------------- Health ----------------

class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
