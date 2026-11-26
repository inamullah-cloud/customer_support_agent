"""Long-term / context memory for the agent (in-memory implementation)."""

import re
import threading
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

MemoryType = Literal["preference", "fact", "goal", "context"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


class AgentMemory(BaseModel):
    """Structured long-term and context memory record."""

    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    memory_type: MemoryType
    content: str
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


# ============================================================
# Exceptions
# ============================================================

class AgentMemoryError(Exception):
    """Base exception for agent memory operations."""


class MemoryNotFoundError(AgentMemoryError):
    """Raised when a memory entry does not exist or the user lacks access."""


class DuplicateMemoryError(AgentMemoryError):
    """Raised when attempting to save a duplicate memory record."""


# ============================================================
# Storage layer
# ============================================================

class MemoryStore:
    """Low-level storage with duplicate protection and user isolation."""

    def __init__(self) -> None:
        self._storage: dict[str, AgentMemory] = {}
        self._lock = threading.RLock()

    def save(self, memory: AgentMemory) -> AgentMemory:
        """Save or update a record; identical content+type per user is rejected."""
        with self._lock:
            for existing in self._storage.values():
                if (
                    existing.memory_id != memory.memory_id
                    and existing.user_id == memory.user_id
                    and existing.memory_type == memory.memory_type
                    and existing.content.strip().lower() == memory.content.strip().lower()
                ):
                    raise DuplicateMemoryError(
                        f"Memory with identical content and type '{memory.memory_type}' "
                        f"already exists for user '{memory.user_id}'."
                    )
            self._storage[memory.memory_id] = memory
            return memory

    def get(self, memory_id: str, user_id: str) -> Optional[AgentMemory]:
        with self._lock:
            memory = self._storage.get(memory_id)
            if memory is None or memory.user_id != user_id:
                return None
            return memory

    def delete(self, memory_id: str, user_id: str) -> bool:
        with self._lock:
            if self.get(memory_id, user_id) is None:
                return False
            del self._storage[memory_id]
            return True

    def list_memories(self, user_id: Optional[str] = None) -> list[AgentMemory]:
        with self._lock:
            if user_id is None:
                return list(self._storage.values())
            return [m for m in self._storage.values() if m.user_id == user_id]


# ============================================================
# Management layer
# ============================================================

class MemoryManager:
    """Higher-level memory lifecycle and user-level operations."""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self.store = store or MemoryStore()

    def add_memory(self, user_id: str, memory_type: MemoryType, content: str) -> AgentMemory:
        memory = AgentMemory(user_id=user_id, memory_type=memory_type, content=content)
        return self.store.save(memory)

    def get_memory(self, memory_id: str, user_id: str) -> AgentMemory:
        memory = self.store.get(memory_id=memory_id, user_id=user_id)
        if memory is None:
            raise MemoryNotFoundError(
                f"Memory record '{memory_id}' not found for user '{user_id}'."
            )
        return memory

    def update_memory(
        self,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> AgentMemory:
        existing = self.get_memory(memory_id=memory_id, user_id=user_id)

        updated = existing.model_copy(
            update={
                "content": content if content is not None else existing.content,
                "memory_type": memory_type if memory_type is not None else existing.memory_type,
                "updated_at": _now_iso(),
            }
        )
        return self.store.save(updated)

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        if not self.store.delete(memory_id=memory_id, user_id=user_id):
            raise MemoryNotFoundError(
                f"Memory record '{memory_id}' not found for user '{user_id}'."
            )
        return True

    def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> list[AgentMemory]:
        memories = self.store.list_memories(user_id)
        if memory_type is not None:
            return [m for m in memories if m.memory_type == memory_type]
        return memories

    def search_memories(self, user_id: str, query: str) -> list[AgentMemory]:
        lower_query = query.lower()
        return [m for m in self.get_user_memories(user_id) if lower_query in m.content.lower()]

    def get_relevant_memories(self, user_id: str, query: str, limit: int = 5) -> list[AgentMemory]:
        """Memories most relevant to `query`: word overlap first, then most recently updated."""
        query_tokens = _tokens(query)
        ranked = sorted(
            self.get_user_memories(user_id),
            key=lambda m: (len(query_tokens & _tokens(m.content)), m.updated_at),
            reverse=True,
        )
        return ranked[:limit]


# ============================================================
# Shared instance (used by the agent graph and the API)
# ============================================================

_manager: MemoryManager | None = None
_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MemoryManager()
        return _manager


def reset_memory_manager() -> None:
    """Drop all stored memories (used by tests)."""
    global _manager
    with _manager_lock:
        _manager = None
