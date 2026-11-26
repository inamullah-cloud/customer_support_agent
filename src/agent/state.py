"""Agent state definition, status types, and structured-output schemas."""

from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agent.memory import MemoryType
from src.observability.models import AgentObservabilityReport, TraceEvent


# ============================================================
# Status types
# ============================================================

ReflectionStatus = Literal[
    "not_checked",
    "approved",
    "needs_correction",
]

ExecutionStatus = Literal[
    "classifying",
    "selecting_tools",
    "executing",
    "completed",
    "failed",
]

ApprovalStatus = Literal[
    "not_required",
    "pending",
    "approved",
    "rejected",
]


# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    user_id: str | None
    memories_used: List[str]
    memories_saved: List[str]

    category: str | None
    selected_tools: List[str]

    plan: List[str]

    executed_tools: List[str]
    execution_status: ExecutionStatus
    tool_iterations: int

    reflection_status: ReflectionStatus
    reflection_count: int
    reflection_feedback: str

    approval_required: bool
    approval_status: ApprovalStatus

    response: str | None
    error: str | None

    retry_count: int
    fallback_used: bool

    execution_trace: List[TraceEvent]
    observability_report: AgentObservabilityReport | None


# ============================================================
# Structured LLM output schemas
# ============================================================

class Classification(BaseModel):
    """Structured classification output."""

    category: str


class ToolSelection(BaseModel):
    """Structured tool-selection output."""

    tools: List[str] = Field(
        description="Names of the tools required to answer the user's query."
    )


class Plan(BaseModel):
    """Structured planning output."""

    steps: List[str] = Field(
        description="Ordered steps required to answer the user's request."
    )


class Reflection(BaseModel):
    """Structured reflection output."""

    status: Literal[
        "approved",
        "needs_correction",
    ]

    feedback: str = Field(
        description="Explain why the answer is acceptable or what needs correction."
    )


class NewMemory(BaseModel):
    """One piece of information worth remembering about the user."""

    memory_type: MemoryType
    content: str = Field(
        description="A short third-person statement, e.g. 'Prefers SMS updates'."
    )


class MemoryExtraction(BaseModel):
    """Structured memory-extraction output."""

    memories: List[NewMemory] = Field(
        default_factory=list,
        description="Durable items worth remembering; empty if there is nothing.",
    )
