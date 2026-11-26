"""Memory endpoints (per-user long-term memory)."""

from fastapi import APIRouter, Query, Response

from src.agent.memory import AgentMemory, MemoryType
from src.api.dependencies import get_container
from src.api.schemas import MemoryCreate, MemoryUpdate

router = APIRouter(prefix="/users/{user_id}/memories", tags=["memory"])


@router.post("", response_model=AgentMemory, status_code=201)
def create_memory(user_id: str, body: MemoryCreate) -> AgentMemory:
    return get_container().memory_manager.add_memory(user_id, body.memory_type, body.content)


@router.get("", response_model=list[AgentMemory])
def list_memories(
    user_id: str,
    memory_type: MemoryType | None = None,
    q: str | None = Query(default=None, description="Substring search in memory content."),
) -> list[AgentMemory]:
    manager = get_container().memory_manager
    memories = manager.search_memories(user_id, q) if q else manager.get_user_memories(user_id)
    if memory_type is not None:
        memories = [m for m in memories if m.memory_type == memory_type]
    return memories


@router.get("/{memory_id}", response_model=AgentMemory)
def get_memory(user_id: str, memory_id: str) -> AgentMemory:
    return get_container().memory_manager.get_memory(memory_id, user_id)


@router.patch("/{memory_id}", response_model=AgentMemory)
def update_memory(user_id: str, memory_id: str, body: MemoryUpdate) -> AgentMemory:
    return get_container().memory_manager.update_memory(
        memory_id, user_id, content=body.content, memory_type=body.memory_type
    )


@router.delete("/{memory_id}", status_code=204)
def delete_memory(user_id: str, memory_id: str) -> Response:
    get_container().memory_manager.delete_memory(memory_id, user_id)
    return Response(status_code=204)
