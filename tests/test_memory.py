import pytest

from src.agent.memory import DuplicateMemoryError, MemoryManager, MemoryNotFoundError


def test_add_get_update_delete():
    manager = MemoryManager()
    memory = manager.add_memory("u1", "preference", "Likes email updates")

    assert manager.get_memory(memory.memory_id, "u1").content == "Likes email updates"

    updated = manager.update_memory(memory.memory_id, "u1", content="Likes SMS updates")
    assert updated.content == "Likes SMS updates"
    assert updated.updated_at >= memory.updated_at

    assert manager.delete_memory(memory.memory_id, "u1") is True
    with pytest.raises(MemoryNotFoundError):
        manager.get_memory(memory.memory_id, "u1")


def test_user_isolation():
    manager = MemoryManager()
    memory = manager.add_memory("u1", "fact", "Lives in Mingora")

    with pytest.raises(MemoryNotFoundError):
        manager.get_memory(memory.memory_id, "u2")
    assert manager.get_user_memories("u2") == []


def test_duplicate_rejected():
    manager = MemoryManager()
    manager.add_memory("u1", "goal", "Learn Python")
    with pytest.raises(DuplicateMemoryError):
        manager.add_memory("u1", "goal", "  learn python ")
    # same text for another user, or another type, is fine
    manager.add_memory("u2", "goal", "Learn Python")
    manager.add_memory("u1", "fact", "Learn Python")


def test_filter_and_search():
    manager = MemoryManager()
    manager.add_memory("u1", "fact", "Lives in Mingora")
    manager.add_memory("u1", "goal", "Visit Lahore")

    assert len(manager.get_user_memories("u1", memory_type="goal")) == 1
    assert [m.content for m in manager.search_memories("u1", "mingora")] == ["Lives in Mingora"]


def test_relevant_memories_ranked_by_overlap_then_recency():
    manager = MemoryManager()
    manager.add_memory("u1", "preference", "Prefers SMS updates")
    manager.add_memory("u1", "fact", "Lives in Mingora")
    manager.add_memory("u1", "goal", "Learn Python")

    ranked = manager.get_relevant_memories("u1", "what is the weather in Mingora", limit=2)
    assert ranked[0].content == "Lives in Mingora"
    assert len(ranked) == 2
