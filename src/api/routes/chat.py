"""Chat endpoints: send messages, stream progress, resolve approvals, read threads."""

import json
from typing import Any, Iterator
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from src.agent.graph import app as agent_app
from src.api.dependencies import get_container
from src.api.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    PendingToolCall,
    ThreadMessage,
    ThreadState,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

ROLES = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}
STREAM_FIELDS = (
    "category", "selected_tools", "plan", "execution_status", "executed_tools",
    "approval_status", "reflection_status", "memories_used", "memories_saved",
    "response", "error",
)


# ---------------- helpers ----------------

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _text(content: Any) -> str | None:
    if content is None:
        return None
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _pending_approval(snapshot) -> ApprovalRequest | None:
    interrupts = getattr(snapshot, "interrupts", None) or ()
    if not interrupts:
        return None
    return ApprovalRequest.model_validate(interrupts[0].value)


def _build_response(thread_id: str, record: bool = True) -> ChatResponse:
    """Build the API response from the thread's latest checkpoint."""
    snapshot = agent_app.get_state(_config(thread_id))
    values = snapshot.values or {}
    approval = _pending_approval(snapshot)
    report = values.get("observability_report")

    if record and report is not None and approval is None:
        get_container().obs_store.save(thread_id, report)

    return ChatResponse(
        thread_id=thread_id,
        status="awaiting_approval" if approval else "completed",
        response=_text(values.get("response")),
        category=values.get("category"),
        selected_tools=values.get("selected_tools") or [],
        plan=values.get("plan") or [],
        executed_tools=values.get("executed_tools") or [],
        fallback_used=bool(values.get("fallback_used")),
        memories_used=values.get("memories_used") or [],
        memories_saved=values.get("memories_saved") or [],
        error=values.get("error"),
        approval=approval,
        observability=report,
    )


def _new_turn_input(request: ChatRequest) -> dict:
    return {
        "messages": [("user", request.message)],
        "user_id": request.user_id,
        "retry_count": 0,
        "tool_iterations": 0,
    }


def _ensure_not_paused(thread_id: str) -> None:
    if _pending_approval(agent_app.get_state(_config(thread_id))) is not None:
        raise HTTPException(
            status_code=409,
            detail="This thread is waiting for an approval decision.",
        )


def _ensure_paused(thread_id: str) -> None:
    if _pending_approval(agent_app.get_state(_config(thread_id))) is None:
        raise HTTPException(status_code=409, detail="This thread has no pending approval.")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _stream_response(thread_id: str, graph_input: Any) -> StreamingResponse:
    """Stream a graph run as Server-Sent Events: thread, node*, final | error."""

    def events() -> Iterator[str]:
        yield _sse("thread", {"thread_id": thread_id})
        try:
            for chunk in agent_app.stream(graph_input, config=_config(thread_id), stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__" or not isinstance(update, dict):
                        continue
                    view = {k: update[k] for k in STREAM_FIELDS if k in update}
                    yield _sse("node", {"node": node, "update": view})
        except Exception:
            logger.exception("Agent stream failed (thread %s)", thread_id)
            yield _sse("error", {"detail": "Agent execution failed."})
            return

        yield _sse("final", _build_response(thread_id).model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- endpoints ----------------

@router.post("", response_model=ChatResponse)
def send_message(request: ChatRequest) -> ChatResponse:
    """Send a message. Returns `awaiting_approval` if a human decision is needed."""
    thread_id = request.thread_id or uuid4().hex
    _ensure_not_paused(thread_id)

    try:
        agent_app.invoke(_new_turn_input(request), config=_config(thread_id))
    except Exception:
        logger.exception("Agent execution failed (thread %s)", thread_id)
        raise HTTPException(status_code=500, detail="Agent execution failed.")

    return _build_response(thread_id)


@router.post("/stream")
def stream_message(request: ChatRequest) -> StreamingResponse:
    """Same as POST /chat, streamed as Server-Sent Events."""
    thread_id = request.thread_id or uuid4().hex
    _ensure_not_paused(thread_id)
    return _stream_response(thread_id, _new_turn_input(request))


@router.post("/{thread_id}/approval", response_model=ChatResponse)
def resolve_approval(thread_id: str, decision: ApprovalDecision) -> ChatResponse:
    """Approve or reject the action a paused thread is waiting on."""
    _ensure_paused(thread_id)

    try:
        agent_app.invoke(Command(resume=decision.approved), config=_config(thread_id))
    except Exception:
        logger.exception("Agent resume failed (thread %s)", thread_id)
        raise HTTPException(status_code=500, detail="Agent execution failed.")

    return _build_response(thread_id)


@router.post("/{thread_id}/approval/stream")
def stream_approval(thread_id: str, decision: ApprovalDecision) -> StreamingResponse:
    """Same as POST /chat/{thread_id}/approval, streamed as Server-Sent Events."""
    _ensure_paused(thread_id)
    return _stream_response(thread_id, Command(resume=decision.approved))


@router.get("/{thread_id}", response_model=ThreadState)
def get_thread(thread_id: str) -> ThreadState:
    """Return a thread's status and message history (e.g. to restore a chat on reload)."""
    snapshot = agent_app.get_state(_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    base = _build_response(thread_id, record=False)
    messages = [
        ThreadMessage(
            role=ROLES.get(m.type, m.type),
            content=_text(m.content) or "",
            tool_calls=[
                PendingToolCall(name=c["name"], args=c["args"])
                for c in (getattr(m, "tool_calls", None) or [])
            ],
        )
        for m in snapshot.values.get("messages", [])
    ]
    return ThreadState(**base.model_dump(), messages=messages)
